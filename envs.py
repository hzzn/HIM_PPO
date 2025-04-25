import numpy as np
import gym
from gym import spaces
import torch
from torch.distributions import Multinomial, Categorical
from torch.functional import F

class HospitalEnv(gym.Env):

    def __init__(self, J=3, N=None, C=None, B=None, mu=None, lam=None, m=8, discharge_probability=None):
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
        self.discharge_probability = discharge_probability
        self.reset()

    def reset(self, is_random=False):
        self.time = 0
        if is_random:
            x = torch.cat([torch.randint(2*n//3, n+1, (1,)) for n in self.N])
        else:
            x = torch.zeros(self.J, dtype=int)
        y = torch.zeros(self.J, dtype=int)
        self.state = torch.cat([torch.tensor([self.time]), x, y], dim=0)
        return self.state

    # action[i]=j 表示i病房的患者移至j病房
    def step(self, logits):
        t, x, y = self.state[0].item(), self.state[1:1+self.J], self.state[1+self.J:]
        # 状态更新
        if callable(self.lam):
            self.lam = [self.lam(t, i) for i in range(self.J)]
        arrivals = torch.poisson(torch.tensor(self.lam)).int()
        if callable(self.mu):
            self.mu = [self.mu(t, i) for i in range(self.J)]
            
        if self.time == 0:
            samplers = [Multinomial(
                total_count=min(x[i].item(), self.N[i]), 
                probs=torch.tensor([self.mu[i], 1 - self.mu[i]])
            ) for i in range(self.J)]

            discharges = torch.tensor(
                [samplers[i].sample()[0].item() if x[i] > 0 else 0 for i in range(self.J)], 
                dtype=int
            )
            x = x + arrivals
            y = discharges
        else:
            if self.J == 2:
                samplers = [Multinomial(
                    total_count=y[i].item(), 
                    probs=torch.tensor(
                        [self.discharge_p2(self.time, self.time+1, i), 
                        1 - self.discharge_p2(self.time, self.time+1, i)]
                    )
                ) for i in range(self.J)]
            else:
                samplers = [Multinomial(
                    total_count=y[i].item(), 
                    probs=torch.tensor(
                        [self.discharge_probability(self.time, i), 
                        1 - self.discharge_probability(self.time, i)]
                    )
                ) for i in range(self.J)]

            discharges = torch.tensor(
                [samplers[i].sample()[0].item() if y[i] > 0 else 0 for i in range(self.J)],
                dtype=int
                )
            x = x + arrivals - discharges
            y = y - discharges

        q = torch.clamp(x - torch.tensor(self.N), min=0)
        residual_capacity = torch.clamp(torch.tensor(self.N) - x, min=0)
        f = torch.zeros(self.J, dtype=int)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        for i in range(self.J):      
            target = action[i]
            if residual_capacity[target] <= 0:
                target = i
                action[i] = i
                f[i] = q[i]
            else:
                f[i] = torch.min(q[i], residual_capacity[target])
                x[i] -= f[i]
                residual_capacity[target] = max(0, residual_capacity[target] - q[i])
                x[target] = x[target] + f[i]


        # 成本计算
        q = torch.clamp(x - torch.tensor(self.N), min=0)
        wait_cost = sum(self.C[i] * (q[i]) for i in range(self.J))
        overflow_cost = sum(self.B[i][action[i]] * f[i] for i in range(self.J) if action[i] != i)
        reward = - (wait_cost + overflow_cost)

        self.time = (self.time + 1) % self.m
        self.state = torch.cat([torch.tensor([self.time]), x, y], dim=0)
        
        return self.state, reward, probs, action, f

    def discharge_p2(self, curr_time, next_time, index_pool):
        """
        根据CDF计算出院概率 p_j
        """
        #return ((1 - self.mu_rates[index_pool])**curr_time - (1 - self.mu_rates[index_pool])**next_time) / (1 - self.mu_rates[index_pool])**curr_time
        if curr_time == self.m - 1: return 1
        return 1 - (1 - self.mu[index_pool])**(next_time - curr_time)
