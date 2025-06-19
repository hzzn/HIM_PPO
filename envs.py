import numpy as np
import torch
from torch.distributions import Multinomial, Categorical
from torch.functional import F
from utils import Normalization, RewardScaling
import math
import random



class HospitalEnv():

    def __init__(self, config):
        # === Pool structure ===
        self.num_pools = config["num_pools"]  # 𝒥: Number of pools
        self.J = self.num_pools
        self.config = config
        # === Server configuration ===
        self.N_j = torch.tensor(config["num_servers"],
                                                 dtype=torch.int)  # N_j: Number of beds per pool
        max_overall_N_j = torch.max(self.N_j).float()
        self.normalized_N_j = self.N_j.float() / max_overall_N_j.float() # 确保 N_j 和 max_overall_N_j 都是浮点数

        self.priority = torch.tensor(config["overflow_priority"], dtype=torch.int) # overflow优先级, overflow_cost=30时为1, 35为2
        # === Hourly arrival and discharge rates ===
        self.hourly_arrival_rate = torch.tensor(config["arrival_rate_hourly"], dtype=torch.float)  # λ_j(t)
        self.daily_arrival_rate = torch.sum(self.hourly_arrival_rate, dim=1)  # Λ_j: Total arrival per day

        self.hourly_discharge_rate = torch.tensor(config["discharge_rate_hourly"], dtype=torch.float) # h_dis 
        self.discharge_cdf = torch.cumsum(self.hourly_discharge_rate, dim = 1) # CDF for the discharge time 参照p33 B.1.
        self.daily_discharge_rate = torch.tensor(config["discharge_rate_daily"], dtype=torch.float) # Total discharge per day

        # === Decision epoch ===
        self.num_epochs_per_day = config["num_epochs_per_day"]  # m: Number of epochs per day

        self.overflow = torch.zeros(self.num_pools, dtype=torch.int)  # Overflow patients

        # === Action representation ===

        self.mask = torch.tensor(config["mask"], dtype=torch.int)  # Mask for invalid actions

        # === Cost structure ===
        self.holding_cost = torch.tensor(config["waiting_cost"],
                                                   dtype=torch.float)  # C_j: Cost per unit of waiting
        self.overflow_cost = torch.tensor(config["overflow_cost"],
                                                 dtype=torch.float)  # B_ij: Overflow cost from class i to pool j
        self.reset()

    def reset(self, is_random=False):
        """
        Reset the environment to the beginning of the simulation (day 0, epoch 0).
        """
        self.day_count = 0  # t = 0
        self.epoch_index_today = 0  # h(t) = 0
        self.normlization = Normalization(shape=(2 * self.J))
        h = 0
        if is_random:
            # self.X_j = torch.cat([torch.randint(2*n//3, n+1, (1,)) for n in self.N_j])
            self.X_j = torch.tensor([2 * n // 3 for n in self.N_j])
        else:
            self.X_j = torch.zeros(self.num_pools, dtype=torch.int)  # X_j: current number of customers in pool j
        self.Y_j = torch.zeros(self.num_pools, dtype=torch.int)  # Y_j: number of patients to be discharged today in each pool
        
       
        if self.config["running_mean_std_norm"]:
            X_Y = torch.cat([self.X_j, self.Y_j], dim=0)
            X_Y = self.normlization(X_Y)
        else:
            X = self.X_j / self.N_j
            Y = self.Y_j / self.N_j
            X_Y = torch.cat([X, Y], dim=0)
        
        if self.config["normalized_N_j"]:
            X_Y = torch.cat([X_Y, self.normalized_N_j], dim=0)

        if self.config["sin_cos_encode"]:
            sin, cos = self.encode_time_feature(h)
            self.state = torch.cat([X_Y, sin, cos, torch.tensor([h])], dim=0).float()
        else:
            self.state = torch.cat([X_Y, torch.tensor([h])], dim=0).float()
        
        return self.state

    def step(self, logits):
        """
        Performs one transition step in the environment based on the given action.
        """

        action = self.simulated_action(logits)  # Sample an action from the action probabilities

        self.compute_post_action_state(action)
        cost = self.compute_cost(action)

        # Simulate Poisson arrivals and Binomial discharges
        prob = self.simulate_exogenous_events()

        return self.state, cost, action
    
    def overflow_step(self):
        # rule-based policy: completely overflow 
        self.overflow = torch.clamp(self.X_j - self.N_j, min=0)
        
        cap = torch.clamp(self.N_j - self.X_j, min=0)
        # 初始化动作矩阵（每个病房的患者分配）
        action = torch.zeros(self.num_pools, self.num_pools, dtype=torch.int)
        for i in range(self.J):
            # 找到该科室可以overflow的科室, 并按优先级排序
            num_patients = self.overflow[i]
            p_i = self.priority[i]
            target_list = torch.nonzero(p_i > 0, as_tuple=True)[0]
            p_i = p_i[target_list]
            target_list = target_list[torch.argsort(p_i)]
            target_list = torch.cat([target_list, torch.tensor([i])])
            for _ in range(num_patients):
                for j in target_list:
                    if cap[j] > 0:
                        action[i][j] += 1
                        self.X_j[i] -= 1
                        self.X_j[j] += 1
                        cap[j] -= 1
                    elif j == i:
                        action[i][j] += 1
                        self.X_j[j] += 1

        self.compute_post_action_state(action)
        cost = self.compute_cost(action)

        # Simulate Poisson arrivals and Binomial discharges
        self.simulate_exogenous_events()

        return self.state, cost, action

    def simulated_action(self, logits):
        # 计算溢出患者数
        self.overflow = torch.clamp(self.X_j - self.N_j, min=0)
        # 初始化动作矩阵（每个病房的患者分配）
        action = torch.zeros(self.num_pools, self.num_pools, dtype=torch.int)
        action_prob = F.softmax(logits, dim=-1)  # 计算动作概率
        available_capacity = self.N_j - self.X_j # 注意：这里是基于当前 X_j 的

        # 打乱科室序号, 随机抽取科室进行决策
        indices = list(range(self.J))
        random.shuffle(indices)

        for i in indices:
            num_patients = self.overflow[i].item() # 获取整数值
            if num_patients <= 0:
                continue

            # 1. 一次性采样所有病人的目标
            dist = Categorical(action_prob[i, :])
            sampled_targets = dist.sample((num_patients,)) # 一次采样 num_patients 个

            # 2. 遍历采样结果并分配
            for target in sampled_targets:
                target_idx = target.item() # 获取整数索引

                # 3. 检查容量并分配
                # 或者，更简单但可能不完美的做法是：如果目标池 *当前* 有空位，就分配
                # 一个更优但复杂的做法是优先分配给有空位的，再处理没空位的
                
                # 简化版：检查当前临时容量
                if available_capacity[target_idx] > 0:
                    action[i, target_idx] += 1
                    available_capacity[target_idx] -= 1 # 减少可用容量
                else:
                    # 回退策略：分配回原类别 i (表示等待)
                    action[i, i] += 1

        return action
        # 判断并重新采样直到可行

    def compute_post_action_state(self, action):
        """
        :param state: shape = (2J + 1,) -> [x_0..x_J-1, y_0..y_J-1, t]
        :param action: shape = (J, J) -> action[i, j] = 类别 i 分配到病房 j 的患者数量
        :return: post-action 状态
        """

        # 1. 移除所有溢出的病人
        self.X_j = self.X_j - self.overflow
        
        # 2. 计算每个池子的总流入量（包括等待的病人）
        inflows = torch.sum(action, dim=0)         
        # 3. 更新 X_j
        self.X_j = self.X_j + inflows

        t = torch.tensor([self.epoch_index_today])
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

        if self.config["running_mean_std_norm"]:
            X_Y = torch.cat([self.X_j, self.Y_j], dim=0)
            X_Y = self.normlization(X_Y)
        elif["X/N"]:
            X = self.X_j / self.N_j
            Y = self.Y_j / self.N_j
            X_Y = torch.cat([X, Y], dim=0)
        else:
            X_Y = torch.cat([X, Y], dim=0)
            
        if self.config["queue"]:
            X_Y = torch.cat([torch.min(self.X_j, self.N_j), torch.clamp(self.X_j - self.N_j, min=0), self.Y_j])
            # X_Y = self.normlization(X_Y)
            X_Y = torch.cat([X_Y, self.N_j])
        if self.config["normalized_N_j"]:
            X_Y = torch.cat([X_Y, self.normalized_N_j], dim=0)

        if self.config["sin_cos_encode"]:
            sin, cos = self.encode_time_feature(h)
            self.state = torch.cat([X_Y, sin, cos, torch.tensor([h])], dim=0).float()
        elif self.config["position_embedding"]:
            self.state += self.get_positional_embeddings(self.epoch_index_today, self.state.size(0))
        else:
            self.state = torch.cat([X_Y, torch.tensor([h])], dim=0).float()

        self.epoch_index_today = h

    def encode_time_feature(self, h):

        h_tensor = torch.tensor(h, dtype=torch.float)

        # Calculate the angle in radians
        # The factor 2 * math.pi ensures a full cycle (0 to 2*pi) over the period
        angle = 2 * math.pi * h_tensor / self.num_epochs_per_day

        # Calculate sine and cosine features
        sin_feature = torch.sin(angle)
        cos_feature = torch.cos(angle)

        return sin_feature.unsqueeze(0), cos_feature.unsqueeze(0)

    def get_positional_embeddings(self, h, d_model) -> torch.Tensor:

    # 初始化[N, d_model]矩阵
        result = torch.ones(d_model)

    # 对pos和i分别遍历

        for i in range(d_model):
            # 对i的奇偶性进行判断
            if i % 2 == 0:
                term = h / 10000 ** (i / d_model)
                result[h, i] = np.sin(term)
            else:
                term = h / 10000 ** ((i - 1) / d_model)
                result[h, i] = np.cos(term)

        return result