import torch
from torch import nn
from torch.functional import F
from torch.distributions import Multinomial, Categorical
from torch import optim
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import numpy as np
import importlib as imp
from tqdm import tqdm

import actor
import envs
import utils
from actor import  Critic, MLPCritic
from actor import MLPActor as Actor_GRU
from envs import HospitalEnv
from utils import mlp_sample, ppo_update, overflow_sample, compute_gae_adv
from env_config import ENV_CONFIG as config

if __name__ == "__main__":
    epochs = config["num_epoch"]
    costs_mean = []
    act_loss_mean = []
    crtc_loss_mean = []
    env = HospitalEnv(config)
    act = Actor_GRU(config) 
    crtc = MLPCritic(config)
    optimizer_actor = Adam(act.parameters(), lr=5e-4)
    optimizer_critic = Adam(crtc.parameters(), lr=1e-3)
    scheduler_actor = CosineAnnealingLR(optimizer_actor, T_max=epochs, eta_min=1e-4)
    scheduler_critic = CosineAnnealingLR(optimizer_actor, T_max=5, eta_min=1e-4)
    iters = 0

    for i in range(epochs):
        crtc.loss = []
        act.loss = []
        iters += 1
        trajectory = mlp_sample(env, act, config, is_random=False)
        states = torch.stack([t['state'] for t in trajectory]).float()
        actions = torch.stack([t['action'] for t in trajectory]).float()
        costs = torch.tensor([t['cost'] / 100 for t in trajectory], dtype=torch.float32)
        next_states = torch.stack([t['next_state'] for t in trajectory]).float()

        # Recompute log_probs using current logits for compatibility with PPO update
        logits = torch.stack([t['logits'] for t in trajectory]).float()
        old_probs = F.log_softmax(logits, dim=-1)

        memory = (states, actions, old_probs, costs, next_states)
        ppo_update(act, crtc, memory, optimizer_actor, optimizer_critic, scheduler_actor, scheduler_critic, config)

        costs_mean.append(costs.mean() * 8)
        act_loss_mean.append(sum(act.loss) / len(act.loss))
        crtc_loss_mean.append(sum(crtc.loss) / len(crtc.loss))
        print(f"Daily Average Cost: {costs_mean[-1]:.2f}")
        print(f"Average Critic Loss: {crtc_loss_mean[-1]:0f}")
        print(f"Average Actor Loss: {act_loss_mean[-1]:.3f}")