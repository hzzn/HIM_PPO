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

        # === Server configuration ===
        self.N_j = torch.tensor(config["num_servers"],
                                                 dtype=torch.int)  # N_j: Number of beds per pool

        # === Hourly arrival and discharge rates ===
        self.hourly_arrival_rate = torch.tensor(config["arrival_rate_hourly"], dtype=torch.float)  # λ_j(t)
        self.daily_arrival_rate = torch.sum(self.hourly_arrival_rate, dim=1)  # Λ_j: Total arrival per day

        self.hourly_discharge_rate = torch.tensor(config["discharge_rate_hourly"], dtype=torch.float)  # h_dis
        self.daily_discharge_rate = torch.sum(self.hourly_discharge_rate, dim=1)  # Total discharge per day

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
        self.pre_action_state = self.state.clone()    # Pre-action state
        self.post_action_state = self.state.clone()    # Post-action state

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

    def reset(self):
        """
        Reset the environment to the beginning of the simulation (day 0, epoch 0).
        """
        self.day_count = 0  # t = 0
        self.epoch_index_today = 0  # h(t) = 0

        self.X_j = torch.zeros(self.num_pools, dtype=torch.int)  # X_j: current number of customers in pool j
        self.Y_j = torch.zeros(self.num_pools,
                               dtype=torch.int)  # Y_j: number of patients to be discharged today in each pool
        self.state = torch.cat([self.X_j, self.Y_j, torch.tensor([self.epoch_index_today])], dim=0)

        self.waiting_count_list = torch.zeros(self.num_pools, dtype=torch.int)  # Waiting count per class
        self.in_service_count_list = torch.zeros(self.num_pools, dtype=torch.int)  # In-service count per class

        return self.state

    def step(self, action_prob):
        """
        Performs one transition step in the environment based on the given action.
        """
        self.pre_action_state = self.state.clone()
        action = self.simulated_action()  # Sample an action from the action probabilities

        self.compute_post_action_state(action)
        cost = self.compute_cost(action)

        # Simulate Poisson arrivals and Binomial discharges
        prob = self.simulate_exogenous_events()


        return self.state, cost, action, prob

    def simulated_action(self, action_prob):
        # 计算溢出患者数
        self.overflow = torch.max(self.X_j - self.N_j, torch.zeros_like(self.X_j))
        # 初始化动作矩阵（每个病房的患者分配）
        action = torch.zeros(self.num_pools, self.num_pools, dtype=torch.int)
        # 判断并重新采样直到可行
        for i in range(self.J):
            num_patients = self.overflow[i]
            if num_patients > 0:
                while True:  # 重复采样直到得到可行的分配
                    # 使用 Categorical 分布根据 action_prob 来生成患者分配数量
                    dist = Categorical(action_prob[i, :])  # 创建 Categorical 分布对象
                    patient_distribution = dist.sample((num_patients,))  # 采样出每个病房的患者分配
                    # 计算每个病房分配的患者数
                    temp_action = torch.zeros_like(action[i, :])  # 临时记录分配
                    for j in patient_distribution:
                        temp_action[j] += 1  # 将分配到病房 j 的患者数量加 1
                    # 判断分配是否可行
                    if self.check_action_validity(temp_action, self.overflow):
                        action[i, :] = temp_action
                        break # 如果分配可行，退出循环
                    # 否则重新采样

        return action

    def check_action_validity(self, action, overflow):
        """
        判断分配的action是否可行：
        1. 乘以mask，检查每行的总和是否等于溢出患者数overflow[i]。
        2. 纵向加和，检查每个病房接收的患者数是否小于等于床位剩余数。
        """
        # 1. 乘以 mask，确保无效动作的患者数为0
        action_masked = action * self.mask

        # 检查每行的总和是否等于溢出患者数
        row_sums = torch.sum(action_masked, dim=1)
        row_check = torch.all(row_sums == overflow)  # 每行总和是否等于 overflow

        # 2. 检查纵向加和，确保每个病房的分配数不超过其剩余床位
        column_sums = torch.sum(action_masked, dim=0)
        column_check = torch.all(column_sums <= torch.max(self.N_j - self.X_j, torch.zeros_like(self.N_j)))

        return row_check and column_check


    def compute_post_action_state(self, action):
        """
        :param state: shape = (2J + 1,) -> [x_0..x_J-1, y_0..y_J-1, t]
        :param action: shape = (J, J) -> action[i, j] = 类别 i 分配到病房 j 的患者数量
        :return: post-action 状态
        """
        state = self.state.clone()
        x = state[:self.J]
        y = state[self.J:2 * self.J]
        t = int(state[-1].item())


        # Step 2: apply action — action.sum(dim=0) 表示每个病房接收的患者总数
        x_post = x - self.overflow + action.sum(dim=0)

        # Step 3: 拼接新的状态
        post_state = torch.cat([x_post, y, torch.tensor([t], dtype=state.dtype)])
        self.post_action_state = post_state.clone()
        return post_state


    def compute_cost(self,action):
        """
        :param post_state: shape = (2J + 1,) -> [x+, y, t+1]
        :param action: shape = (J, J), 表示 overflow 分配
        :return: float scalar cost
        """
        x_post = self.post_state[:self.J]  # 应用动作后的 x+

        # 1. holding cost: max(x_j^+ - N_j, 0)
        q_post = torch.clamp(x_post - self.N_j, min=0)
        holding = torch.sum(self.holding_cost * q_post)

        # 2. overflow cost: sum_{i,j} B_{i,j} * f_{i,j}
        overflow = torch.sum(self.overflow_cost * action)

        return holding + overflow

    def simulate_exogenous_events(self):
        # 1. clone 状态并拆分
        state = self.state.clone()
        x = state[:self.J]
        y = state[self.J:2 * self.J]
        h = int(state[-1].item())

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
                cap = min(x[j], self.N_j[j])  # 当前病房真实容量下的患者数
                bj[j] = torch.distributions.Binomial(total_count=cap, probs=self.daily_discharge_rate[j]).sample().int()

            x_new = x + aj  # 到达患者加入
            y_new = bj  # 设置当天需要出院的患者
            h_new = torch.tensor([1])
            self.day_count += 1  # 新的一天开始

        else:
            # ======== 非午夜更新逻辑 ========
            dj = torch.zeros(self.J, dtype=torch.int)
            for j in range(self.J):
                remaining = y[j].item()
                total_discharged = 0
                for dt in range(3):
                    rate = self.hourly_discharge_rate[j][t + dt]
                    out = torch.distributions.Binomial(total_count=remaining, probs=rate).sample().int()
                    total_discharged += out
                    remaining -= out
                dj[j] = total_discharged

            x_new = x + aj - dj
            y_new = y - dj

            # 时间更新
            h_new_val = h + 1 if h + 1 < self.epoch_index_today else 0
            h_new = torch.tensor([h_new_val])
            if h_new_val == 0:
                self.day_count += 1  # 每日递增

        # 拼接新的状态
        self.state = torch.cat([x_new, y_new, h_new])
        self.X_j = x_new
        self.Y_j = y_new
        self.epoch_index_today = h_new

        return transition_prob





    '''
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
        
    '''
