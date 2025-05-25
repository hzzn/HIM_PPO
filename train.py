import torch
from torch.optim import Adam

from actor import Actor, Critic, LinearCritic
from utils import mlp_sample, main
from envs import HospitalEnv as Env
from env_config import ENV_CONFIG as config  # 替代 init_config

# 初始化环境和模型
env = Env(config)
actor = Actor(config)
critic = LinearCritic(config)

optimizer_actor = Adam(actor.parameters(), lr=1e-4)
optimizer_critic = Adam(critic.parameters(), lr=1e-3)

# 单轮 PPO
trajectory = mlp_sample(env, actor, config)
main(env, actor, critic, optimizer_actor, optimizer_critic, trajectory, config)