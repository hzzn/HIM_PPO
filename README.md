# HIM_PPO
:::

> **Autors：Jingjing Sun， Jim Dai， Pengyi Shi**
>
> **Year：2024**
>
> **Jounal：Preprinted in arXiv**
>
> **前身：Dai JG, Shi P (2019) Inpatient overflow: An approximate dynamic programming approach. Manufacturing  & Service Operations Management （ADP）**
>
> **摘要:**
>
>  将患者溢出至非主要病区可以有效缓解医院的拥堵，但不合理的溢出可能会导致服务质量不匹配等问题。因此，我们需要在缓解拥堵和减少不合理溢出之间进行权衡。我们将这一溢出管理问题建模为一个离散时间马尔可夫决策过程（MDP），其状态空间和动作空间均较大。为克服维度灾难，我们将每个时间步的动作分解为一系列原子动作，并采用一种基于actor-critic架构的算法——近端策略优化（PPO）——来指导这些原子动作。此外，我们针对神经网络的设计进行了优化，以考虑系统流量的日周期模式。在不同规模的医院环境下，PPO策略始终优于常见的最先进策略。  
>
> **关键词:  **多类别多池排队系统；住院床位管理；近端策略优化  
>

### 一、问题背景与建模假设
医院中共有 $ J $ 个科室，每个科室 $ j $ 的最大床位容量为 $ N_j $。将一天划分为 $ m $ 个时间段，每个时刻系统状态由一个 $ 2J+1 $ 维的向量表示：

+ 1维表示当前时间段；
+ J 维向量 $ X = (x_1, x_2, \dots, x_J) $表示各科室当前的在院病人数；
+ J 维向量 $ Y = (y_1, y_2, \dots, y_J) $表示当前时间段准备出院的病人数。

每个科室的病人到达过程符合**泊松分布**，出院过程符合**伯努利分布**。对于某个时间段 $ [h, h'] $，病人到达 $ a_j \sim \text{Poisson} \left( \int_h^{h'} \lambda_j(s) ds \right) $，出院人数$ d_j \sim \text{Bin}(y_j, p_j^h) $，其中：$ p_j^h = \frac{F_j(h') - F_j(h)}{1 - F_j(h)} $。这里 $ F_j(\cdot) $是病人出院时间的分布函数（CDF）。当某科室超出容量（即$ x_j > N_j $）时，多出来的病人将进入该科室的等待队列。

```python
import numpy as np
import gym
from gym import spaces
import torch
from torch.distributions import Multinomial

class HospitalEnv(gym.Env):

    def __init__(self, J=3, N=None, C=None, B=None, mu=None, lam=None, m=24):
        self.J = J
        self.N = N or [10]*J
        self.C = C or [1.0]*J
        self.B = B or np.ones((J, J))  # overflow cost
        self.mu = mu or [0.1]*J  # 出院概率
        self.lam = lam or [1.0]*J  # 到达强度
        self.m = m
        self.time = 0
        self.state = None  # [t, x_1...x_J, y_1...y_J]

        self.action_space = spaces.Box(0, 1, shape=(J, J))  # 归一化概率
        self.observation_space = spaces.Box(0, np.inf, shape=(1 + 2 * J,))

        self.reset()

    def reset(self):
        self.time = 0
        x = torch.zeros(self.J, dtype=int)
        y = torch.zeros(self.J, dtype=int)
        self.state = torch.cat([torch.tensor([self.time]), x, y], dim=0)
        return self.state

    # action[i]=j 表示i病房的患者移至j病房
    def step(self, action):
        t, x, y = self.state[0].item(), self.state[1:1+self.J], self.state[1+self.J:]
        q = torch.clamp(x - torch.tensor(self.N), min=0)
        residual_capacity = torch.clamp(torch.tensor(self.N) - x, min=0)
        
        f = torch.zeros(self.J, dtype=int)
        for i in range(self.J):
            target = action[i]
            if target is None or target == i: continue
            f[i] = torch.min(q[i], residual_capacity[target])
            x[i] -= f[i]
            residual_capacity[target] = max(0, residual_capacity[target] - q[i])
            x[target] = x[target] - residual_capacity[target]


        # 成本计算
        wait_cost = sum(self.C[i] * (q[i] - f[i]) for i in range(self.J))
        overflow_cost = sum(self.B[i][action[i]] * f[i] for i in range(self.J) if action[i])
        reward = - (wait_cost + overflow_cost)

        # 状态更新
        arrivals = torch.poisson(torch.tensor(self.mu) / self.m).to(int)
        if self.time == 0:
            samplers = [Multinomial(total_count=min(x[i], self.N[i]).item(), probs=torch.tensor([self.mu[i], 1 - self.mu[i]])) for i in range(self.J)]
            discharges = torch.tensor([sampler.sample()[0] for sampler in samplers], dtype=int)
            x_new = x + arrivals
            y_new = discharges
        else:
            samplers = [Multinomial(total_count=y[i].item(), probs=torch.tensor([self.discharge_probability(self.time, self.time+1, i), 1 - self.discharge_probability(self.time, self.time+1, i)])) for i in range(self.J)]
            discharges = torch.tensor([sampler.sample()[0] for sampler in samplers],dtype=int)
            x_new = x + arrivals - discharges
            y_new = y - discharges

        self.time = (self.time + 1) % self.m
        self.state = torch.cat([torch.tensor([self.time]), x_new, y_new], dim=0)
        done = self.time == 0
        return self.state, reward, done, {"action_matrix": f}

    def discharge_probability(self, curr_time, next_time, index_pool):
        """
        根据CDF计算出院概率 p_j
        """
        #return ((1 - self.mu_rates[index_pool])**curr_time - (1 - self.mu_rates[index_pool])**next_time) / (1 - self.mu_rates[index_pool])**curr_time
        if curr_time == 7: return 1
        return 1 - (1 - self.mu[index_pool])**(next_time - curr_time)

```

### 二、策略与动作设计
在每个决策时刻，需要将等待队列中的病人进行分配。一个病人若被分配到其原属科室，则表示他继续等待；若被转移到其他科室，则认为是“overflow”。

策略函数由一个神经网络（Actor）表示，网络输出为一个 $ J \times J $ 的矩阵，每个元素代表“来自科室 i 的病人被转移到科室 j”的logits，softmax后得到转移概率分布 $ \kappa(j | s, i) $。

```python
class Actor(nn.Module):
    def __init__(self, J):
        super().__init__()
        self.state_dim = 2 * J + 1
        self.action_dim = J * J

        self.fc1 = nn.Linear(self.state_dim, self.action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        probs = F.softmax(x.view(J, J), dim=1)
        return probs  # 输出形状为 J x J
```

### 三、价值函数（Critic）与优势函数计算
Critic函数采用线性函数逼近，值函数形式如下：

$ \hat{v}_\eta(s) = \hat{\beta}_1(x_1 + x_2) + \hat{\beta}_3(x_1^2 + x_2^2) $，其中 $ \hat{\beta}_1, \hat{\beta}_3 $ 是待学习参数。

对于给定状态 $ s $ 和动作$ f = (f_{1,2}, q_1 - f_{1,2}, 0, 0) $，下一时刻的状态为：

$ x_1' = x_1 - f_{1,2} + A_1 - D_1,\quad x_2' = x_2 + f_{1,2} + A_2 - D_2 $

期望的价值函数为：

$ \mathbb{E}[ \hat{v}_\eta(s') ] = \hat{\beta}_3(1 - \mu)^2[(x_1 - f_{1,2})^2 + (x_2 + f_{1,2})^2] + (\hat{\beta}_1 + \hat{\beta}_3(2\lambda - \mu))(1 - \mu)(x_1 + x_2) + 2\hat{\beta}_1\lambda + 2\hat{\beta}_3(\lambda + \lambda^2)E[v^η(s′)] $

优势函数为：

$ \hat{A}_\eta(s, f) = (B - C)f_{1,2} + \hat{\beta}_3(1 - \mu)^2 [2f_{1,2}^2 - 2(x_1 - x_2)f_{1,2}] + \text{常数项} $

其中，C 为等待成本，B 为overflow成本。

```python
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        # 使用线性函数逼近器: v(s) = beta1 * sum(x) + beta3 * sum(x^2)
        self.beta1 = nn.Parameter(torch.tensor(1.0))
        self.beta3 = nn.Parameter(torch.tensor(1.0))

    def forward(self, state):
        x = state[1:1+J]  # X部分
        x_sum = torch.sum(x)
        x_sq_sum = torch.sum(x**2)
        return self.beta1 * x_sum + self.beta3 * x_sq_sum

# ========== PPO 更新 ==========
def compute_advantage(critic, states, rewards, next_states, dones):
    values = torch.stack([critic(s) for s in states])
    next_values = torch.stack([critic(s) for s in next_states])
    returns = rewards + GAMMA * next_values * (1 - dones)
    advantages = returns - values
    return advantages.detach()
```

### 四、批处理动作建模（Batching）
为提高模拟效率并确保公平性，采取**批处理决策机制**：

+ 在某状态 s 下，统一为等待病人设定一个分配概率分布；
+ 对每类（原始科室为 i）病人，将其分配至 j 科室的概率为 $ \kappa(j|s, i) $；
+ 每类病人的动作组合符合**多项分布**，即：

$ \pi(f | s) = \prod_{i=1}^{J} \frac{q_i!}{\prod_{j=1}^J f_{i,j}!} \prod_{j=1}^J \kappa(j|s, i)^{f_{i,j}} $

该方法无需按顺序逐一处理每个病人，避免了大量组合爆炸，提高了训练效率。

此外，PPO算法中关键的**策略比值计算**：

$ r_{\theta, \eta}(f | s) = \frac{\pi_\theta(f | s)}{\pi_\eta(f | s)} = \prod_{i=1}^{J} \prod_{j=1}^{J} \left( \frac{\kappa_\theta(j|s, i)}{\kappa_\eta(j|s, i)} \right)^{f_{i,j}} $该表达式因组合项抵消，显著简化了比值计算过程。

```python
import torch
from torch.distributions import Multinomial
import numpy as np

def sample_episode(env, actor, critic, max_steps=100):
    trajectory = []
    state = env.reset()
    
    for t in range(max_steps):
        with torch.no_grad():
            logits = actor(torch.tensor(state, dtype=torch.float32))
            probs = torch.softmax(logits, dim=-1)

            # Reshape to [J, J] for κ(j | s, i)
            J = int(np.sqrt(len(probs)))
            probs_matrix = probs.view(J, J)
            actions = []

            # 对每个等待的病人类别（即科室 i）采样分配到 j 的数量
            for i, q_i in enumerate(env.waiting_queue):  # q_i 是当前等待队列中第 i 类病人的数量
                if q_i > 0:
                    dist = Multinomial(total_count=q_i, probs=probs_matrix[i])
                    f_i = dist.sample().int().tolist()
                else:
                    f_i = [0] * J
                actions.append(f_i)

            # f[i][j] 表示从 i 类病人分配到 j 的数量
            f = np.array(actions)

        next_state, reward, done, info = env.step(f)

        trajectory.append({
            'state': state,
            'action': f,
            'reward': reward,
            'next_state': next_state
        })

        state = next_state

        if done:
            break

    return trajectory
```

### 五、算法实现流程（PPO）
整个PPO算法分为三个主要模块：

1. **数据采样（并行仿真）**：使用多个actor并行采样仿真轨迹；
2. **策略评估**：利用线性Critic计算值函数与优势函数；
3. **策略优化**：利用Adam优化器，更新Actor参数 $ \theta $，直到收敛为止（代价差异小于阈值 $ \delta $）。

pytorch 用于神经网络训练，Ray用于分布式数据生成。

```python
# ========== PPO 更新 ==========
def ppo_update(actor, critic, memory, optimizer_actor, optimizer_critic):
    states, actions, old_probs, rewards, next_states, dones = memory
    advantages = compute_advantage(critic, states, rewards, next_states, dones)

    for _ in range(EPOCHS):
        for i in range(0, len(states), BATCH_SIZE):
            batch_slice = slice(i, i + BATCH_SIZE)
            batch_states = states[batch_slice]
            batch_actions = actions[batch_slice]
            batch_old_probs = old_probs[batch_slice]
            batch_advantages = advantages[batch_slice]

            # Actor loss
            new_probs = []
            for s, a in zip(batch_states, batch_actions):
                dist = actor.get_action_dist(s)
                pi = dist[a[0], a[1]]  # a 是(i, j)
                new_probs.append(pi)
            new_probs = torch.stack(new_probs)
            ratios = new_probs / batch_old_probs

            surr1 = ratios * batch_advantages
            surr2 = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            values = torch.stack([critic(s) for s in batch_states])
            critic_loss = F.mse_loss(values, rewards[batch_slice])

            # 更新
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()
```

### 六、算法运行主函数
```python
import torch
from torch.distributions import Multinomial
import numpy as np
from .sample import sample_episode

def main(env, actor, critic, actor_optimizer, critic_optimizer, gamma=0.99, clip_ratio=0.2, train_epochs=5):
    for iteration in range(1000):  # 外层PPO迭代
        trajectory = sample_episode(env, actor, critic)
        
        # 提取轨迹数据
        states = torch.tensor([t['state'] for t in trajectory], dtype=torch.float32)
        actions = [t['action'] for t in trajectory]  # 动作是f[i][j]数组
        rewards = [t['reward'] for t in trajectory]
        next_states = torch.tensor([t['next_state'] for t in trajectory], dtype=torch.float32)

        # 计算 critic 的当前估值
        with torch.no_grad():
            values = critic(states).squeeze()
            next_values = critic(next_states).squeeze()
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32)
            advantages = returns - values

        # 更新策略网络 Actor
        for _ in range(train_epochs):
            logits = actor(states)
            probs = torch.softmax(logits, dim=-1)

            J = int(np.sqrt(logits.shape[-1]))
            probs_matrix = probs.view(-1, J, J)  # [batch_size, J, J]

            log_probs = []
            for i, f in enumerate(actions):  # f 是 JxJ 的分配矩阵
                f = torch.tensor(f, dtype=torch.int32)
                lp = 0.0
                for src in range(J):
                    if f[src].sum() > 0:
                        dist = Multinomial(total_count=f[src].sum(), probs=probs_matrix[i, src])
                        lp += dist.log_prob(f[src].float())
                log_probs.append(lp)
            log_probs = torch.stack(log_probs)

            with torch.no_grad():
                old_log_probs = log_probs.clone()

            ratios = torch.exp(log_probs - old_log_probs)
            surrogate1 = ratios * advantages
            surrogate2 = torch.clamp(ratios, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            actor_optimizer.zero_grad()
            policy_loss.backward()
            actor_optimizer.step()

        # 更新 Critic
        for _ in range(train_epochs):
            value_preds = critic(states).squeeze()
            critic_loss = torch.nn.functional.mse_loss(value_preds, returns)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

        print(f"Iteration {iteration}: Return = {returns.mean().item():.2f}")
```
