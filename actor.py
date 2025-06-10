import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init



# ========== PartiallySharedActor ==========
class Actor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.J = config["num_pools"]
        self.state_dim = 2 * self.J
        self.action_dim = self.J * self.J
        self.m = config["num_epochs_per_day"]
        self.register_buffer("mask", torch.tensor(config["mask"], dtype=torch.int))
        self.hidden_dim = 34
        # 共享的隐藏层
        self.fc1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.output_layers = nn.ModuleList([nn.Linear(self.hidden_dim, self.action_dim) for _ in range(self.m)])
        self.reset_parameters()
        self.loss = []
    def reset_parameters(self):
        # 使用 Kaiming 初始化，适用于 ReLU 激活
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0.0)

    def forward(self, states):
        logits = []
        for state in states:
            epoch_index = int(state[-1].item())
            state_input = state[:-1]
            x = torch.relu(self.fc1(state_input))  # 经过第一个隐藏层
            logits.append(self.output_layers[epoch_index](x))  # 选择对应时间步的输出层
        logits = torch.stack(logits, dim=0)
        logits = logits.view(-1, self.J, self.J).masked_fill(self.mask == 0, value=-1e9)  # 无效动作的logits设为一个很小的负值
        return logits # 输出形状为 batch_size x J x J

class MLPActor(nn.Module):
    def __init__(self, config, use_bias=True):
        super().__init__()

        self.J = config["num_pools"]
        self.action_dim = self.J * self.J
        self.state_dim = config["actor_input_dim"]
        self.hidden_dims = config.get("actor_hidden_dim", [64])
        self.loss = []
        self.register_buffer('mask', torch.tensor(config["mask"], dtype=torch.int).view(self.J, self.J))
        layers = []
        in_features = self.state_dim
        
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim, bias=use_bias))
            layers.append(nn.ReLU()) 
            in_features = hidden_dim 
        
        # Combine all hidden layers into a sequential module
        self.hidden_layers = nn.Sequential(*layers)
        
        # The final output layer maps from the last hidden_dim to 1 (for value prediction)
        self.output_layer = nn.Linear(in_features, self.action_dim, bias=use_bias)       

        # Initialize weights and biases
        self.reset_parameters(use_bias)


    def reset_parameters(self, use_bias):
        # Initialize hidden layers
        for m in self.hidden_layers:
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if use_bias:
                    init.constant_(m.bias, 0)
    
        init.normal_(self.output_layer.weight, mean=0., std=0.01)
        if use_bias:
            init.constant_(self.output_layer.bias, 0)

    def forward(self, state):
        """
        state: tensor of shape (batch_size, state_dim)
        return: tensor of shape (batch_size,)
        """
        x = state[:, :-1] 
        x = self.hidden_layers(x)
        return self.output_layer(x).view(-1, self.J, self.J).masked_fill(self.mask == 0, value=-1e9)


class Actor_GRU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.J = config["num_pools"]
        self.state_dim = config["actor_input_dim"]
        self.action_dim = self.J * self.J
        self.hidden_dims = config.get("actor_hidden_dim", [64])
        self.register_buffer('mask', torch.tensor(config["mask"], dtype=torch.int).view(self.J, self.J))
        
        self.gru = nn.GRUCell(self.state_dim, self.hidden_dims[0])
        self.output_layer = nn.Linear(self.hidden_dims[0], self.action_dim)
        # self.tanh = nn.Tanh()  # 标准的 nn.GRUCell 内部已经包含了一个 tanh 激活函数

        self.reset_parameters()
        self.loss = []

    def reset_parameters(self):
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, nonlinearity='relu')
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        nn.init.kaiming_normal_(self.output_layer.weight, nonlinearity='relu')
        nn.init.constant_(self.output_layer.bias, 0)

    def forward(self, states, h=None):
        """
        states: shape (T, state_dim + 1), 最后一维是 epoch index
        h: 上一时刻的 hidden state，shape = (batch_size, hidden_dim)
        返回：
            logits: (T, J, J)
            h: 新的 hidden state
        """
        T = states.size(0)
        logits = []
        new_h = h
        for t in range(T):
            state_t = states[t][:-3]  # 去掉 epoch index
            if new_h is None:
                new_h = torch.zeros(self.hidden_dims[0], device=states.device)
           
            new_h = self.gru(state_t, new_h)
            logit_t = self.output_layer(new_h)
            logit_t = logit_t.view(self.J, self.J).masked_fill(self.mask == 0, value=-1e9)
            logits.append(logit_t)

        return torch.stack(logits), new_h


# ========== Critic ==========
class Critic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.J = config["num_pools"]
        self.state_dim = 2 * self.J
        self.hidden_dim = 34
        self.reward_dim = 1
        self.m = config["num_epochs_per_day"]
        # 共享的隐藏层
        self.fc1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.output_layers = nn.ModuleList([nn.Linear(self.hidden_dim, self.reward_dim) for _ in range(self.m)])
        
        self.reset_parameters()

    def reset_parameters(self):
        # 使用 Kaiming 初始化，适用于 ReLU 激活
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0.0)

    def forward(self, state):
        """
        :param state: 当前的状态
        :return: 状态的价值（value）
        """
        # 选择对应输出层的索引（state的最后一个元素决定）
        epoch_index = int(state[-1].item())  # 假设 state[-1] 是整数，决定输出层索引
        # 将 state 中除最后一个元素以外的部分传入神经网络
        state_input = state[:-1]
        # 检查是否存在 NaN 值
        if torch.isnan(state_input).any():
            print(" NaN detected in fc1 output!")
        # 经过共享的隐藏层
        x = torch.relu(self.fc1(state_input))  # 经过第一个隐藏层
        # 选择对应时间步的输出层，输出状态的价值
        value = self.output_layers[epoch_index](x)  # 选择对应时间步的输出层
        return value # 输出形状为 batch_size x J x J

class MLPCritic(nn.Module):
    def __init__(self, config, use_bias=True):
        super().__init__()
        self.state_dim = config["critic_input_dim"]
        self.hidden_dims = config.get("critic_hidden_dim", [64])
        self.loss = []

        layers = []
        in_features = self.state_dim
        
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim, bias=use_bias))
            layers.append(nn.ReLU()) 
            in_features = hidden_dim 
        
        # Combine all hidden layers into a sequential module
        self.hidden_layers = nn.Sequential(*layers)
        
        # The final output layer maps from the last hidden_dim to 1 (for value prediction)
        self.output_layer = nn.Linear(in_features, 1, bias=use_bias)       

        # Initialize weights and biases
        self.reset_parameters(use_bias)


    def reset_parameters(self, use_bias):
        # Initialize hidden layers
        for m in self.hidden_layers:
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if use_bias:
                    init.constant_(m.bias, 0)
    
        init.normal_(self.output_layer.weight, mean=0., std=0.01)
        if use_bias:
            init.constant_(self.output_layer.bias, 0)

    def forward(self, state):
        """
        state: tensor of shape (batch_size, state_dim)
        return: tensor of shape (batch_size,)
        """
        x = state[:, :-1] 
        x = self.hidden_layers(x)
        return self.output_layer(x).squeeze(-1)

class Critic_GRU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.state_dim = 2 * config["num_pools"]
        self.hidden_dim = 64
        self.gru = nn.GRUCell(self.state_dim, self.hidden_dim)
        self.tanh = nn.Tanh()
        self.output_layer = nn.Linear(self.hidden_dim, 1)

    def forward(self, state, h=None):
        state = state[:, :-1]  # 去掉 epoch index
        if h is None:
            h = torch.zeros(state.size(0), self.hidden_dim, device=state.device)
        h = self.tanh(self.gru(state, h))
        value = self.output_layer(h)
        return value, h


