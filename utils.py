import torch
from torch.functional import F
from tqdm import tqdm

def compute_advantage(crictic, states, rewards, next_states, gamma):
    values = crictic(states)
    next_values = crictic(next_states)
    returns = rewards + gamma * next_values
    advantages = returns - values
    return advantages.detach(), values, returns

def ppo_update(actor, critic, memory, optimizer_actor, optimizer_critic, gamma=0.99, clip_ratio=0.2, batch_size=32):
    states, actions, old_probs, rewards, next_states, f = memory

    total_batches = (len(states) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(states), batch_size), total=total_batches, desc="PPO Update"):
        batch_slice = slice(i, i + batch_size)
        batch_states = states[batch_slice]
        batch_actions = actions[batch_slice]
        batch_old_probs = old_probs[batch_slice]
        batch_rewards = rewards[batch_slice]
        batch_f = f[batch_slice]
        batch_next_states = next_states[batch_slice]
        batch_advantages, batch_values, batch_returns = compute_advantage(
            critic, batch_states, batch_rewards, batch_next_states, gamma
        )

        # Actor loss
        dist = F.softmax(actor(batch_states), dim=-1)
        new_probs = torch.gather(dist, index=batch_actions.unsqueeze(2), dim=2).squeeze(2)
        batch_old_probs = torch.gather(batch_old_probs, index=batch_actions.unsqueeze(2), dim=2).squeeze(2)
        eps = 1e-10
        log_new_probs = torch.log(new_probs+eps)
        log_old_probs = torch.log(batch_old_probs+eps)

        ratios = torch.sum((log_new_probs - log_old_probs) * batch_f, dim=-1)
        ratios = torch.exp(ratios)

        surr1 = ratios * batch_advantages
        surr2 = torch.clamp(ratios, 1 - clip_ratio, 1 + clip_ratio) * batch_advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # Critic loss
        critic_loss = F.mse_loss(batch_values, batch_returns)

        # 更新
        optimizer_actor.zero_grad()
        actor_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)

        # 检查梯度是否为NaN
        for name, param in actor.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"⚠️ NaN detected in gradient of {name}")
                break

        optimizer_actor.step()

        optimizer_critic.zero_grad()
        critic_loss.backward()
        optimizer_critic.step()

def mlp_sample(env, actor, max_steps=100, is_random=False):
    trajectory = []
    state = env.reset(is_random)
    
    for t in range(max_steps):
        with torch.no_grad():
            logits = actor(state.float()).view(env.J, env.J)
            next_state, reward, probs, action, f = env.step(logits)
            
        trajectory.append({
            'state': state,
            'action': action,
            'probs': probs,
            'reward': reward,
            'f': f,
            'next_state': next_state
        })

        state = next_state

    return trajectory

def rnn_sample(env, actor, max_steps=100, is_random=False):
    trajectory = []
    state = env.reset(is_random)
    h_prev = actor.init_hidden_states()
    
    for t in range(max_steps):
        with torch.no_grad():
            logits, h_prev  = actor(state.float(), h_prev)
            next_state, reward, probs, action, f = env.step(logits.squeeze(0))
            
        trajectory.append({
            'state': state,
            'action': action,
            'probs': probs,
            'reward': reward,
            'f': f,
            'next_state': next_state
        })

        state = next_state

    return trajectory


def arrival_rate_5(t, index):
    assert t >= 0 and t <= 23 and index >= 0 and index <= 4
    if index == 0:
        if t >= 0 and t < 12: return 9 / 12
        else: return 5 / 12
    elif index == 1:
        if t >= 0 and t < 12: return 5 / 12
        else: return 9 / 12
    else: return 7 / 12

def discharge_probability_5(t, index):
    assert t >= 0 and t <= 23 and index >= 0 and index <= 4
    if index == 0:
        if t >= 13 and t <= 18: return 0.5 * (t - 12) / 3 
        else: return 0
    else:
        if t >= 7 and t <= 18: return 0.25 * (t - 7) / 3
        else: return 0




def main(env, actor, critic, optimizer_actor, optimizer_crictic, trajectory, gamma=0.99, clip_ratio=0.2, train_epochs=5, batch_size=32):
        states = torch.tensor([t['state'] for t in trajectory], dtype=torch.float32)
        actions = [t['action'] for t in trajectory] 
        rewards = [t['reward'] for t in trajectory]
        next_states = torch.tensor([t['next_state'] for t in trajectory], dtype=torch.float32)
        f = [t['f'] for t in trajectory]
        old_probs = [t['probs'] for t in trajectory]
        memory = states, actions, old_probs, rewards, next_states, f
        ppo_update(actor, critic, memory, optimizer_actor, optimizer_crictic, gamma, clip_ratio, batch_size)

