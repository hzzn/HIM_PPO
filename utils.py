import torch
from torch.functional import F
from tqdm import tqdm

def compute_advantage(critic, states, rewards, next_states, gamma):
    with torch.no_grad():
        values = critic(states)           # v(s)
        next_values = critic(next_states) # v(s')
        
        advantages = rewards - gamma + next_values - values
        returns = advantages + values     # = g - gamma + v(s')

    return advantages.detach(), values.detach(), returns.detach()
def compute_gae_adv(critic, states, next_states, rewards, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    values = critic(states)
    next_values = critic(next_states)

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return torch.tensor(advantages)

def ppo_update(actor, critic, memory, optimizer_actor, optimizer_critic, config):
    device = next(actor.parameters()).device
    states, actions, old_log_probs, costs, next_states = memory
    # 将所有数据从 CPU 传输到 GPU
    states = states.to(device)
    actions = actions.to(device)
    old_log_probs = old_log_probs.to(device)
    costs = costs.to(device) # 将 costs 也移到设备上
    next_states = next_states.to(device)

    rewards = -costs
    batch_size = config.get("batch_size", 64)
    gm = config.get("gamma", 0.99)
    lam = config.get("lam", 0.95)
    clip_ratio = config["Clipping_parameter"]
    # gamma = costs.mean().item()

    # Step 1: update critic
    
    for i in tqdm(range(0, len(states), batch_size), desc="Critic Update", leave=False):
        #1. Critic Update
        batch_slice = slice(i, i + batch_size)
        s = states[batch_slice]
        next_s = next_states[batch_slice]
        r = rewards[batch_slice]

        # g = costs[i:i + batch_size]

        v_s = critic(s)
        # v_s_prime = critic(s_prime)
        adv  = compute_gae_adv(critic, s, next_s, r, gm, lam).to(device)

        # target = g - gamma + - v_s_prime
        target = adv + v_s.detach() # detach 是关键，防止反向图混乱
        
        critic_loss = F.mse_loss(v_s, target)
        
        critic.loss.append(critic_loss)
        optimizer_critic.zero_grad()
        critic_loss.backward()
        max_norm = config.get("max_norm", 1.0)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=max_norm)
        optimizer_critic.step()

    # Step 2: compute advantage
    # advantages, _, _ = compute_advantage(critic, states, costs, next_states, gamma)

    # Step 3: update actor
    # total_batches = (len(states) + batch_size - 1) // batch_size

    advantages = compute_gae_adv(critic, states, next_states, rewards, gm, lam).to(device)
    for i in tqdm(range(0, len(states), batch_size), desc="Actor Updates", leave=False):
        h_actor = None
        batch_slice = slice(i, i + batch_size)
        s = states[batch_slice]
        f = actions[batch_slice]
        logp_old = old_log_probs[batch_slice]
        adv = advantages[batch_slice]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8) # 归一化adv 稳定计算
        if hasattr(actor, "gru"):
            logits, h_actor = actor(s, h_actor)
            logits = logits.view_as(f)
        else:
            logits = actor(s).view_as(f)
        
        logp_new = F.log_softmax(logits, dim=-1)
        log_ratio = ((logp_new - logp_old) * f).sum(dim=(1, 2))
        ratio = torch.exp(log_ratio)
        #print(f"adv range: min={adv.min().item()}, max={adv.max().item()}")
        #print(f"Ratio range: min={ratio.min().item()}, max={ratio.max().item()}")
        surr1 = ratio * adv        
        surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        
        actor_loss = -torch.min(surr1, surr2).mean()
        actor.loss.append(actor_loss)
        optimizer_actor.zero_grad()
        actor_loss.backward()
        max_norm = config.get("max_norm", 1.0)
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=max_norm)
        optimizer_actor.step()

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
    state = env.reset(is_random)  # Tensor, shape = (state_dim,)
    max_days = config["Simulation_days"]
    num_epochs_per_day = config["num_epochs_per_day"]
    total_iters = max_days * num_epochs_per_day
    actor_device = next(actor.parameters()).device
    h_actor = None
    for i in tqdm(range(total_iters), desc="Sample"):
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
            'next_state': next_state      # shape: (state_dim,)
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

