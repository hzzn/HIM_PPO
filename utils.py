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

def ppo_update(actor, critic, memory, optimizer_actor, optimizer_critic, config):
    states, actions, old_log_probs, costs, next_states = memory
    batch_size = config.get("batch_size", 64)
    clip_ratio = config["Clipping_parameter"]
    gamma = costs.mean().item()

    # Step 1: update critic
    for i in tqdm(range(0, len(states), batch_size), desc="Critic Update", leave=False):
        s = states[i:i + batch_size]
        s_prime = next_states[i:i + batch_size]
        g = costs[i:i + batch_size]
        v_s = critic(s)
        v_s_prime = critic(s_prime)
        pred = v_s
        target = g - gamma + - v_s_prime
        critic_loss = F.mse_loss(pred, target)
        optimizer_critic.zero_grad()
        critic_loss.backward()
        optimizer_critic.step()

    # Step 2: compute advantage
    advantages, _, _ = compute_advantage(critic, states, costs, next_states, gamma)

    # Step 3: update actor
    total_batches = (len(states) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(states), batch_size), desc="Actor Updates", leave=False):
        batch_slice = slice(i, i + batch_size)
        s = states[batch_slice]
        f = actions[batch_slice]
        logp_old = old_log_probs[batch_slice]
        adv = advantages[batch_slice]
        logits = actor(s).view_as(f)
        logp_new = F.log_softmax(logits, dim=-1)
        log_ratio = ((logp_new - logp_old) * f).sum(dim=(1, 2))
        ratio = torch.exp(log_ratio)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        actor_loss = -torch.min(surr1, surr2).mean()
        optimizer_actor.zero_grad()
        actor_loss.backward()

        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
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

    for _ in tqdm(range(total_iters), desc="Sample"):
        state_tensor = state.float().unsqueeze(0)  # shape: (1, state_dim)
        with torch.no_grad():
            logits = actor(state_tensor).view(env.J, env.J)  # raw logits for each atomic decision

        # env.step: accepts logits, internally samples action and returns next_state, reward, etc.
        next_state, cost, action = env.step(logits)


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

