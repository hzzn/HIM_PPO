import numpy as np
import gym
from gym import spaces
import torch
from pandocfilters import applyJSONFilters
from torch.distributions import Multinomial, Categorical
from torch.functional import F
from win32comext.axscript.client.framework import state_map


class HospitalEnv(gym.Env):

    def __init__(self, config):
        # === Pool structure ===
        self.num_pools = config["num_pools"]  # 𝒥: Number of pools
        self.J = self.num_pools

        # === Server configuration ===
        self.N_j = torch.tensor(config["num_servers"],
                                                 dtype=torch.int)  # N_j: Number of beds per pool

        # === Hourly arrival and discharge rates ===
        self.hourly_arrival_rate = torch.tensor(config["arrival_rate_hourly"], dtype=torch.float)  # λ_j(t)
        self.daily_arrival_rate = torch.sum(self.hourly_arrival_rate, dim=1)  # Λ_j: Total arrival per day

        self.hourly_discharge_rate = torch.tensor(config["discharge_rate_hourly"], dtype=torch.float) # h_dis 
        self.discharge_cdf = torch.cumsum(self.hourly_discharge_rate, dim = 1) # CDF for the discharge time 参照p33 B.1.
        self.daily_discharge_rate = torch.tensor(config["discharge_rate_daily"], dtype=torch.float) # Total discharge per day

        # === Decision epoch ===
        self.num_epochs_per_day = config["num_epochs_per_day"]  # m: Number of epochs per day
        self.epoch_times = torch.tensor(
            [i * (24 / self.num_epochs_per_day) for i in range(self.num_epochs_per_day)], dtype=torch.float)

        # === Day tracking ===
        self.day_count = 0  # t: current day t
        self.epoch_index_today = 0  # H(t): current epoch index within day

        # === State representation (X_j, Y_j, h) ===
        self.X_j = torch.zeros(self.num_pools, dtype=torch.int)  # X_j: current number of customers in pool j
        self.Y_j = torch.zeros(self.num_pools,
                               dtype=torch.int)  # Y_j: number of patients to be discharged today in each pool
        self.state = torch.cat([self.X_j, self.Y_j, torch.tensor([self.epoch_index_today])], dim=0)
        self.overflow = torch.zeros(self.num_pools, dtype=torch.int)  # Overflow patients

        # === Action representation ===
        self.overflow_and_wait_decision = torch.zeros(self.num_pools, self.num_pools,
                                                      dtype=torch.int)  # F(t_k): overflow/wait decisions matrix
        self.mask = torch.tensor(config["mask"], dtype=torch.int)  # Mask for invalid actions

        # === Waiting count and in-service tracking ===
        self.waiting_count_list = torch.zeros(self.num_pools, dtype=torch.int)  # Q_j: Waiting count per class
        self.in_service_count_list = torch.zeros(self.num_pools, dtype=torch.int)  # Z_j: In-service count

        # === Cost structure ===
        self.holding_cost = torch.tensor(config["waiting_cost"],
                                                   dtype=torch.float)  # C_j: Cost per unit of waiting
        self.overflow_cost = torch.tensor(config["overflow_cost"],
                                                 dtype=torch.float)  # B_ij: Overflow cost from class i to pool j

        # === One-epoch cost tracking ===
        self.current_epoch_cost = 0.0  # g(s, f): One-epoch cost under action f

        self.reset()

    def reset(self, is_random=False):
        """
        Reset the environment to the beginning of the simulation (day 0, epoch 0).
        """
        self.day_count = 0  # t = 0
        self.epoch_index_today = 0  # h(t) = 0
        if is_random:
            self.X_j = torch.cat([torch.randint(2*n//3, n+1, (1,)) for n in self.N_j])
        else:
            self.X_j = torch.zeros(self.num_pools, dtype=torch.int)  # X_j: current number of customers in pool j
        self.Y_j = torch.zeros(self.num_pools, dtype=torch.int)  # Y_j: number of patients to be discharged today in each pool
        self.state = torch.cat([self.X_j, self.Y_j, torch.tensor([self.epoch_index_today])], dim=0)

        self.waiting_count_list = torch.zeros(self.num_pools, dtype=torch.int)  # Waiting count per class
        self.in_service_count_list = torch.zeros(self.num_pools, dtype=torch.int)  # In-service count per class
        
        return self.state

    def step(self, logits):
        """
        Performs one transition step in the environment based on the given action.
        """
        self.post_state = self.state.clone()
        action = self.simulated_action(logits)  # Sample an action from the action probabilities

        self.compute_post_action_state(action)
        cost = self.compute_cost(action)

        # Simulate Poisson arrivals and Binomial discharges
        prob = self.simulate_exogenous_events()

        return self.state, cost, action
    
    def overflow_step(self):
        self.post_state = self.state.clone()
        self.overflow = torch.clamp(self.X_j - self.N_j, min=0)
        # 初始化动作矩阵（每个病房的患者分配）
        action = torch.zeros(self.num_pools, self.num_pools, dtype=torch.int)
        for i in range(self.J):
            num_patients = self.overflow[i]
            for _ in range(num_patients):
               print() 
        self.compute_post_action_state(action)
        cost = self.compute_cost(action)

        # Simulate Poisson arrivals and Binomial discharges
        prob = self.simulate_exogenous_events()

        return self.state, cost, action

    def simulated_action(self, logits):
        # 计算溢出患者数
        self.overflow = torch.clamp(self.X_j - self.N_j, min=0)
        # 初始化动作矩阵（每个病房的患者分配）
        action = torch.zeros(self.num_pools, self.num_pools, dtype=torch.int)
        action_prob = F.softmax(logits, dim=-1)  # 计算动作概率
        # 判断并重新采样直到可行
        for i in range(self.J):
            num_patients = self.overflow[i]
            dist = Categorical(action_prob[i, :])  # 创建 Categorical 分布对象
            for _ in range(num_patients):
                num_resample = 0 # 记录重采样次数, 为重采样次数设定阈值, 防止陷入死循环
                while True:  # 重复采样直到得到可行的分配
                    if num_resample >= 5:
                        action[i, i] += 1
                        self.X_j[i] += 1
                        break     

                    target = dist.sample()  # 采样

                    if self.N_j[target] > self.X_j[target]:
                        action[i, target] += 1
                        self.X_j[target] += 1
                        break # 如果分配可行，退出循环, 否则继续循环重新采样

                    num_resample += 1

        return action


    def compute_post_action_state(self, action):
        """
        :param state: shape = (2J + 1,) -> [x_0..x_J-1, y_0..y_J-1, t]
        :param action: shape = (J, J) -> action[i, j] = 类别 i 分配到病房 j 的患者数量
        :return: post-action 状态
        """

        self.X_j = self.X_j - self.overflow
        t = self.state[-1:]

        self.post_state = torch.cat([self.X_j, self.Y_j, t])

        return self.post_state


    def compute_cost(self, action):
        """
        :param post_state: shape = (2J + 1,) -> [x+, y, t+1]
        :param action: shape = (J, J), 表示 overflow 分配
        :return: float scalar cost
        """

        # 1. holding cost: max(x_j^+ - N_j, 0)
        q_post = torch.clamp(self.X_j - self.N_j, min=0)
        holding = torch.sum(self.holding_cost * q_post)

        # 2. overflow cost: sum_{i,j} B_{i,j} * f_{i,j}
        overflow = torch.sum(self.overflow_cost * action)

        return holding + overflow

    def simulate_exogenous_events(self):

        h = int(self.post_state[-1].item())

        # 判断是否为午夜（h == 0）
        is_midnight = (h == 0)

        # 计算 t 对应的小时段（当前 3 小时段的起始小时索引）
        t = h * 3  # h=0 -> t=0, h=1 -> t=3, ..., h=7 -> t=21

        # ======== aj: 到达患者人数（泊松分布）========
        aj = torch.zeros(self.J, dtype=torch.int)
        for j in range(self.J):
            for dt in range(3):
                lam = self.hourly_arrival_rate[j][t + dt]
                aj[j] += torch.poisson(torch.tensor([lam])).int().item()

        if is_midnight:
            # ======== 午夜更新逻辑 ========
            # bj: 计划出院（每日）
            bj = torch.zeros(self.J, dtype=torch.int)
            for j in range(self.J):
                cap = min(self.X_j[j], self.N_j[j])  # 当前病房真实容量下的患者数
                bj[j] = torch.distributions.Binomial(
                    total_count=cap, probs=self.daily_discharge_rate[j]
                ).sample().int()

            self.X_j = self.X_j + aj  # 到达患者加入
            self.Y_j = bj  # 设置当天需要出院的患者

        else:
            # ======== 非午夜更新逻辑 ======== 参照p33 B.1.
            dj = torch.zeros(self.J, dtype=torch.int)
            for j in range(self.J):
                if self.Y_j[j] > 0:
                    eps = 1e-10
                    F_h = self.discharge_cdf[j][3*h]
                    F_h_prime = self.discharge_cdf[j][3*(h-1)]
                    probs = (F_h - F_h_prime + eps) / (1 - F_h_prime + eps) # 参照p33 B.1.
                    out = torch.distributions.Binomial(
                        total_count=self.Y_j[j], probs=probs
                    ).sample().int() # 参照p33 B.1.
                    
                    dj[j] += out

            self.X_j = self.X_j + aj - dj
            self.Y_j = self.Y_j - dj

        # 时间更新
        h = (h + 1) % self.num_epochs_per_day

        if h == 0:
            self.day_count += 1  # 每日递增

        # 拼接新的状态
        self.state = torch.cat([self.X_j, self.Y_j, torch.tensor([h])])

        self.epoch_index_today = h

        transition_prob = 0

        return transition_prob
