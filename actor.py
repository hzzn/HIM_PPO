import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 设置超参数
GAMMA = 0.99
EPS_CLIP = 0.2
EPOCHS = 10
BATCH_SIZE = 64

print("branch zzz")

print("hzzn altered")

# 状态维度：时间段 + 每个科室的病人数 + 当前出院人数  # 每个等待病人从科室i分配到科室j
# ========== Actor ==========
class Actor(nn.Module):
    def __init__(self, J, B):
        super().__init__()
        self.J = J
        self.state_dim = 2 * J + 1
        self.action_dim = J * J
        self.register_buffer("mask", torch.tensor(B))
        self.fc1 = nn.Linear(self.state_dim, self.action_dim)
        self.reset_parameters()

    def reset_parameters(self):
        # 使用 Kaiming 初始化，适用于 ReLU 激活
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0.0)

    def forward(self, state):
        x = self.fc1(state)
        if torch.isnan(x).any():
            print("⚠️ NaN detected in fc1 output!")
        logits = F.relu(x)
        return logits.view(-1, self.J, self.J).masked_fill(self.mask==0, value=-1e9)  # 输出形状为 batch_size x J x J


class Actor_GRU(nn.Module):
    
    def __init__(self, J, B, hidden_dim=64, num_layers=1):
        super().__init__()
        self.J = J
        self.state_dim = 2 * J  # 移除时间维度
        self.action_dim = J * J
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 注册mask用于非法转移的屏蔽
        self.register_buffer("mask", torch.tensor(B))

        # GRU 层：输入是 (batch, seq_len, input_dim)
        # self.gru = nn.GRU(input_size=self.state_dim, hidden_size=hidden_dim,
        #                   num_layers=num_layers, batch_first=True)

        self.h_t = torch.zeros(num_layers, self.hidden_dim)
        self.gru_cells = nn.ModuleList([
            nn.GRUCell(input_size=self.state_dim if i == 0 else hidden_dim, hidden_size=hidden_dim)
            for i in range(num_layers)
        ])

        # 输出映射层：从 GRU 输出 -> action logits
        self.fc = nn.Linear(hidden_dim, self.action_dim)

        self.reset_parameters()

    def reset_parameters(self):
        # 初始化 GRU 和线性层
        for name, param in self.gru_cells.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='relu')
        nn.init.constant_(self.fc.bias, 0)

    def init_hidden_states(self):
        return torch.zeros(self.num_layers, self.hidden_dim)

    def forward(self, state, h_prev=None):
        """
        state: Tensor of shape (batch_size, state_dim) 或
               (batch_size, seq_len, state_dim) 若考虑历史序列
        """
        if h_prev is None:
            h_prev = torch.zeros(self.num_layers, self.hidden_dim)
        
        h_next = [h_prev[i].clone() for i in range(self.num_layers)]
        h_list = []
        if len(state.shape) == 1: state = state.unsqueeze(0)
        for t in range(state.size(0)):
            x = state[t][1:]
            new_h = []
            for i, gru_cell in enumerate(self.gru_cells):
                h = h_next[i]
                x = gru_cell(x, h)
                new_h.append(x)
            h_next = new_h    
            h_list.append(x)

        h_next = torch.stack(h_next)
        h_list = torch.stack(h_list)

        x = self.fc(h_list)

        logits = F.relu(x)
        return logits.view(-1, self.J, self.J).masked_fill(self.mask == 0, value=-1e9), h_next

# ========== Critic ==========
class Critic(nn.Module):
    def __init__(self, J):
        super(Critic, self).__init__()
        # 使用线性函数逼近器: v(s) = beta1 * sum(x) + beta3 * sum(x^2)
        self.J = J
        self.beta1 = nn.Parameter(torch.tensor(1.0))
        self.beta3 = nn.Parameter(torch.tensor(1.0))

    def forward(self, state):
        x = state[:, 1: 1 + self.J]  # X部分
        x_sum = torch.sum(x, dim=-1)
        x_sq_sum = torch.sum(x**2, dim=-1)
        return self.beta1 * x_sum + self.beta3 * x_sq_sum
