import torch
from torch.functional import F
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm
import math

def compute_advantages(critic, states, next_states, rewards, config):

    num_actor = config.get("num_actor", 1)
    if num_actor > 1:
        advantanges = []
        targets = []
        traj_len = len(states) // num_actor
        for i in range(0, len(states), traj_len):
            batch_slice = slice(i, i + traj_len)
            s = states[batch_slice]
            next_s = next_states[batch_slice]
            r = rewards[batch_slice]
            adv, tgt = compute_advantage(critic, s, next_s, r, config)
            advantanges.append(adv)
            targets.append(tgt)
        return torch.cat(advantanges, dim=0), torch.cat(targets, dim=0)
    else:
        return compute_advantage(critic, states, next_states, rewards, config)


def compute_advantage(critic, states, next_states, rewards, config):
    is_gae = config.get("is_gae", True)
    if is_gae:
        gm = config.get("gamma", 0.99)
        lam = config.get("lam", 0.95)
        return compute_gae_adv(critic, states, next_states, rewards, gamma=gm, lam=lam)
    else:
        return compute_mean_cost_adv(critic, states, next_states, rewards)

def compute_mean_cost_adv(critic, states, next_states, rewards):
    with torch.no_grad():
        values = critic(states)           # v(s)
        next_values = critic(next_states) # v(s')
        mean_cost = -rewards.mean()

        advantages = rewards + mean_cost + next_values - values
        target = rewards + mean_cost + next_values 
    return advantages, target

def compute_gae_adv(critic, states, next_states, rewards, gamma=0.99, lam=0.95):
    with torch.no_grad():
        advantages = []
        gae = 0
        values = critic(states)
        next_values = critic(next_states)

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_values[t] - values[t]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)
        advantages = torch.tensor(advantages, device=rewards.device)
        target = advantages + values
    return advantages, target

def ppo_update(actor_critic, memory, config):

    if config["lr_decay"]:
        actor, critic, optimizer_actor, optimizer_critic, scheduler_actor, scheduler_critic = actor_critic
    else:
        actor, critic, optimizer_actor, optimizer_critic = actor_critic
    # 将所有数据从 CPU 传输到 GPU
    device = next(actor.parameters()).device
    states, actions, old_log_probs, costs, next_states = memory
    states = states.to(device)
    actions = actions.to(device)
    old_log_probs = old_log_probs.to(device)
    costs = costs.to(device) # 将 costs 也移到设备上
    next_states = next_states.to(device)
    
    rewards = -costs
    batch_size = config.get("batch_size", 64)
    clip_ratio = config["Clipping_parameter"]
    max_norm = config.get("max_norm", 1.0)

    # 计算优势
    advantages, targets = compute_advantages(critic, states, next_states, rewards, config)
    if config["target_value_normlization"]:
        mean = targets.mean()
        std = targets.std()
        eps = 1e-8
        targets = (targets - mean) / (std + eps)
    # update
    for i in tqdm(range(0, len(states), batch_size), desc="PPO Update", leave=False):
       
        #-----1.Critic Update-----
        batch_slice = slice(i, i + batch_size)
        s = states[batch_slice]
        adv = advantages[batch_slice]
        tgt = targets[batch_slice]
        
        v_s = critic(s)
        critic_loss = F.mse_loss(v_s, tgt)
        
        critic.loss.append(critic_loss)
        optimizer_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=max_norm)
        optimizer_critic.step()
        
        #-----2.Actor Update-----
        h_actor = None

        f = actions[batch_slice]
        logp_old = old_log_probs[batch_slice]
        if config["adv_normlization"]:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8) # 归一化adv 稳定计算
        if hasattr(actor, "gru"):
            logits, h_actor = actor(s, h_actor)
            logits = logits.view_as(f)
        else:
            logits = actor(s).view_as(f)
        
        dist = Categorical(logits=logits)
        entropy = dist.entropy().mean()
        entropy_coef = config.get("entropy_coef", 0) 

        logp_new = F.log_softmax(logits, dim=-1)
        log_ratio = ((logp_new - logp_old) * f).sum(dim=(1, 2))
        ratio = torch.exp(log_ratio)
        # print(f"adv range: min={adv.min().item()}, max={adv.max().item()}")
        # print(f"Ratio range: min={ratio.min().item()}, max={ratio.max().item()}")
        surr1 = ratio * adv        
        surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        
        actor_loss = -torch.min(surr1, surr2).mean() 
        actor_loss -= entropy_coef * entropy 
        actor.loss.append(actor_loss)
        optimizer_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=max_norm)
        optimizer_actor.step()
        
        if config["lr_decay"]:
            scheduler_critic.step()
            scheduler_actor.step()

def mlp_sample(env, actor, config,  is_random=False):
    """
    Collect a trajectory by running actor in the environment.

    Args:
        env: environment with .reset() and .step(logits)
        actor: actor network (state -> logits)
        max_steps: number of transitions to collect
        is_random: whether to randomize the environment initial state

    Returns:
        trajectory: list of dicts with keys:
            - state (Tensor)
            - action (Tensor)
            - probs (Tensor or log_probs)
            - reward (float)
            - next_state (Tensor)
            - f (raw atomic action matrix)
    """
    trajectory = []

    reward_scaling = RewardScaling(shape=(1), gamma=config.get("gamma", 0.9))
    state = env.reset()  # Tensor, shape = (state_dim,)
    max_days = config["Simulation_days"]
    num_epochs_per_day = config["num_epochs_per_day"]
    total_iters = max_days * num_epochs_per_day
    actor_device = next(actor.parameters()).device
    h_actor = None
    for i in tqdm(range(total_iters), desc="Sample", leave=False):
        state_tensor = state.float().unsqueeze(0).to(actor_device)  # shape: (1, state_dim)
        with torch.no_grad():
            if hasattr(actor, "gru"):
                if i % 64 == 0:
                    h_actor = None
                logits, h_actor = actor(state_tensor, h_actor)
                logits = logits.view(env.J, env.J)
            else:
                logits = actor(state_tensor).view(env.J, env.J)  # raw logits for each atomic decision

        # env.step: accepts logits, internally samples action and returns next_state, reward, etc.
        next_state, cost, action = env.step(logits.cpu())


        trajectory.append({
            'state': state,               # shape: (state_dim,)
            'logits': logits,    # shape: (J, J)
            'action': action,             # e.g. selected indices or tensor actions
            'cost': cost,             # scalar
            'scaled_cost': reward_scaling(cost),
            'next_state': next_state,      # shape: (state_dim,)
        })

        state = next_state

    return trajectory

def overflow_sample(env, config, is_random=False):
    trajectory = []
    state = env.reset(is_random)  # Tensor, shape = (state_dim,)
    max_days = config["Simulation_days"]
    num_epochs_per_day = config["num_epochs_per_day"]
    total_iters = max_days * num_epochs_per_day
    p = torch.tensor(config["overflow_priority"], dtype=torch.float)
    p[p==0]=0.1
    p[p==-1]=0
    p[p==2]=0.25
    p[p==1]=0.4
    logits = torch.log(p)
    logits[logits==-torch.inf] = -1e9
    for _ in tqdm(range(total_iters), desc="Sample"):
        next_state, cost, action = env.overflow_step()
        p = config["overflow_priority"]

        trajectory.append({
            'state': state,               # shape: (state_dim,)
            'logits': logits,    # shape: (J, J)
            'action': action,             # e.g. selected indices or tensor actions
            'cost': cost,             # scalar
            'next_state': next_state      # shape: (state_dim,)
        })

        state = next_state

    return trajectory

def encode_time_feature(h_value, num_epochs_per_day):

    h_tensor = torch.tensor(h_value, dtype=torch.float)

    # Calculate the angle in radians
    # The factor 2 * math.pi ensures a full cycle (0 to 2*pi) over the period
    angle = 2 * math.pi * h_tensor / num_epochs_per_day

    # Calculate sine and cosine features
    sin_feature = torch.sin(angle)
    cos_feature = torch.cos(angle)

    return sin_feature.unsqueeze(0), cos_feature.unsqueeze(0)

class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = torch.zeros(shape)
        self.S = torch.zeros(shape)
        self.std = torch.sqrt(self.S)

    def update(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.clone().detach()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = torch.sqrt(self.S / self.n)

class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=Flase
        if update:  
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x

class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = torch.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = torch.zeros(self.shape)


def main(env, actor, critic, optimizer_actor, optimizer_critic, trajectory, config):
    states = torch.stack([t['state'] for t in trajectory]).float()
    actions = torch.stack([t['action'] for t in trajectory]).float()
    costs = torch.tensor([t['cost'] for t in trajectory], dtype=torch.float32)
    next_states = torch.stack([t['next_state'] for t in trajectory]).float()

    # Recompute log_probs using current logits for compatibility with PPO update
    logits = torch.stack([t['logits'] for t in trajectory]).float()
    old_probs = F.log_softmax(logits, dim=-1)

    memory = (states, actions, old_probs, costs, next_states)
    ppo_update(actor, critic, memory, optimizer_actor, optimizer_critic, config)

