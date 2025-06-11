import torch
from torch import nn
from torch.functional import F
from torch.distributions import Multinomial, Categorical
from torch import optim
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LinearLR
import numpy as np
import importlib as imp
import matplotlib.pyplot as plt # 用于绘图
from tqdm import tqdm
import os
import math
import json
import ray # 导入 Ray

# 假设这些是你自定义的文件
import actor as actor_module # 重命名以避免与变量名actor冲突
import envs
import utils
imp.reload(actor_module)
imp.reload(envs)
imp.reload(utils)
from actor import  MLPCritic # 确保从 actor_module 导入
from actor import MLPActor as Actor_GRU
# from actor import Actor_GRU
from envs import HospitalEnv
from utils import mlp_sample, ppo_update, compute_gae_adv # 确保 ppo_update 和 mlp_sample 适应Ray Actor
from env_config import ENV_CONFIG as config

# --- Ray Actor for Sampling ---
@ray.remote(num_gpus=0.2)
class SamplingActor:
    def __init__(self, actor_config, rank):
        self.config = actor_config.copy()
        # 为每个actor的环境设置不同的种子，如果需要的话
        self.config["seed"] = actor_config.get("seed", 42) + rank
        
        self.env = HospitalEnv(self.config)
        self.actor_model = Actor_GRU(self.config) # 或者 Actor(self.config)
        
        if torch.cuda.is_available():
            self.actor_model.cuda() # 将模型移动到这个actor分配到的GPU上

    def sample_trajectory(self, is_random=False):
        # mlp_sample 需要能接受 env, actor_model, config 作为参数
        # 并确保在内部正确使用它们 (而不是全局变量)
        trajectory = mlp_sample(self.env, self.actor_model, self.config, is_random)
        return trajectory
        # 将trajectory中的tensor转移到CPU，以便跨进程传输

    def set_weights(self, weights):
        # weights 应该是从CPU发送过来的
        self.actor_model.load_state_dict(weights)
        if torch.cuda.is_available(): # 确保加载后模型仍在GPU上
            self.actor_model.cuda()

    def get_actor_model_device(self): # 用于调试，检查模型设备
        return next(self.actor_model.parameters()).device

def main_ppo_training():
    CHECKPOINT_PATH = 'checkpoint_ray_ppo'
    ray.init(num_gpus=1) # 初始化Ray，指定可用的GPU总数

    epochs = config["num_epoch"] 
    num_sampling_actors = config["num_actor"] # 在GPU上并行运行的actor数量
    batch_size = config["batch_size"]
    Simulation_days = config["Simulation_days"]
    m = config["num_epochs_per_day"]

    costs_mean_epoch = []
    act_loss_mean_epoch = []
    crtc_loss_mean_epoch = []
    # --- 主学习器模型 (在主进程中，也可以移到GPU) ---
    learner_actor = Actor_GRU(config)
    learner_critic = MLPCritic(config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learner_actor.to(device)
    learner_critic.to(device)

    optimizer_actor = Adam(learner_actor.parameters(), lr=config.get("actor_lr", 1e-4), eps=config["adam_eps"])
    optimizer_critic = Adam(learner_critic.parameters(), lr=config.get("critic_lr", 1e-4), eps=config["adam_eps"])
    # 余弦退火
    # scheduler_actor = CosineAnnealingLR(optimizer_actor, T_max=epochs, eta_min=1e-6)
    # scheduler_critic = CosineAnnealingLR(optimizer_critic, T_max=epochs, eta_min=1e-6)
    # 线性降为0
    total_steps = epochs * math.ceil(num_sampling_actors * Simulation_days * m / batch_size )
    scheduler_actor = LinearLR(optimizer_actor, start_factor=1.0, end_factor=0.0, total_iters=total_steps)
    scheduler_critic = LinearLR(optimizer_critic, start_factor=1.0, end_factor=0.0, total_iters=total_steps)
    
    # --- 创建采样 Actors ---
    sampling_actors = [SamplingActor.remote(config, rank=i) for i in range(num_sampling_actors)]
    
    # --- 初始权重同步 ---
    learner_actor_weights_cpu = {k: v.cpu() for k, v in learner_actor.state_dict().items()}
    ray.get([actor.set_weights.remote(learner_actor_weights_cpu) for actor in sampling_actors])

    # --- 检查采样actor模型设备 (可选调试步骤) ---
    
    # --- 3. 加载检查点 ---
    try:
        # 加载状态字典
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False) # map_location确保加载到正确设备

        # 将状态字典加载到模型和优化器中
        learner_actor.load_state_dict(checkpoint['act_state_dict'])
        learner_critic.load_state_dict(checkpoint['crtc_state_dict'])
        optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])

        optimizer_actor.param_groups[0]['lr'] = config.get("actor_lr", 1e-4)
        optimizer_critic.param_groups[0]['lr'] = config.get("actor_lr", 1e-4)

        # 恢复训练步数
        total_update_steps = checkpoint['update_steps']

        print(f"Checkpoint loaded successfully from {CHECKPOINT_PATH}")
        print(f"Resuming training from update step: {total_update_steps}")

    except FileNotFoundError:
        print(f"Checkpoint file not found at {CHECKPOINT_PATH}. Starting from scratch.")
        # 如果文件不存在，你可能需要初始化 total_update_steps = 0 或其他默认值
        total_update_steps = 0
    except Exception as e:
        print(f"Error loading checkpoint: {e}. Starting from scratch.")
        # 处理其他加载错误
        total_update_steps = 0

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # 用于记录这个epoch内的所有loss和cost
        current_epoch_costs = []
        current_epoch_actor_losses = []
        current_epoch_critic_losses = []


        total_update_steps += 1
        learner_critic.loss = [] # 重置学习器的loss列表
        learner_actor.loss = []  # 重置学习器的loss列表

        # --- 并行采样 ---
        trajectory_futures = [actor.sample_trajectory.remote(is_random=False) for actor in sampling_actors]
        all_trajectories_list_cpu = ray.get(trajectory_futures) # List of lists of dicts (all tensors on CPU)

        # --- 聚合轨迹 ---
        # 将所有actor的轨迹合并为一个大的batch
        # 注意：需要确保 mlp_sample 返回的trajectory中的tensor都在CPU上，以便ray传输
        
        # 确保 utils.py 中的 ppo_update 和 mlp_sample 能够处理好 Ray actor返回的 trajectory 格式
        # 并将数据转移到正确的device上进行训练
        
        # 将CPU上的轨迹数据转换为torch tensor并移到learner的device
        aggregated_states_list = []
        aggregated_actions_list = []
        aggregated_costs_list = []
        aggregated_next_states_list = []
        aggregated_logits_list = []
        aggregated_scaled_costs_list = []

        for traj_list_cpu in all_trajectories_list_cpu:
            for t_cpu in traj_list_cpu:
                aggregated_states_list.append(t_cpu['state'].to(device))
                aggregated_actions_list.append(t_cpu['action'].to(device))
                aggregated_costs_list.append(t_cpu['cost']) # cost 是标量，直接用
                aggregated_next_states_list.append(t_cpu['next_state'].to(device))
                aggregated_logits_list.append(t_cpu['logits'].to(device))
                if config["reward_scaling"] :
                    aggregated_scaled_costs_list.append(t_cpu['scaled_cost'].to(device))

        
        if not aggregated_states_list:
            print("No trajectories collected, skipping update.")
            continue

        states = torch.stack(aggregated_states_list).float()
        actions = torch.stack(aggregated_actions_list).float()
        costs = torch.tensor(aggregated_costs_list, dtype=torch.float)

        mean_cost = costs.mean().item()
        avg_cost_iter = mean_cost * 8
        costs = costs / (costs.std() + 1e-8)
        next_states = torch.stack(aggregated_next_states_list).float()
        logits = torch.stack(aggregated_logits_list).float()
        old_log_probs = F.log_softmax(logits, dim=-1) # log_softmax on device
        if config["reward_scaling"] :
            costs = torch.tensor(aggregated_scaled_costs_list, dtype=torch.float)
        memory = (states, actions, old_log_probs, costs, next_states, mean_cost)
        
        # --- PPO 更新 (在主学习器上) ---
        ppo_update(learner_actor, learner_critic, memory, optimizer_actor, optimizer_critic, scheduler_actor, scheduler_critic, config)
        # scheduler_actor.step()
        # scheduler_critic.step()

        # --- 记录损失和成本 ---
        if learner_actor.loss and learner_critic.loss:
            avg_actor_loss_iter = torch.stack(learner_actor.loss).mean().item()
            avg_critic_loss_iter = torch.stack(learner_critic.loss).mean().item()
            costs_mean_epoch.append(avg_cost_iter)
            act_loss_mean_epoch.append(avg_actor_loss_iter)
            crtc_loss_mean_epoch.append(avg_critic_loss_iter)
            
            print(f"  Iter {epoch + 1}: Daily Avg Cost: {avg_cost_iter:.2f}, Critic Loss: {avg_critic_loss_iter:.3f}, Actor Loss: {avg_actor_loss_iter:.3f}")

        # --- 更新采样 Actors 的权重 ---
        updated_learner_weights_cpu = {k: v.cpu() for k, v in learner_actor.state_dict().items()}
        weight_update_futures = [actor.set_weights.remote(updated_learner_weights_cpu) for actor in sampling_actors]
        ray.get(weight_update_futures) # 确保权重更新完成
    # --- 训练结束，关闭 Ray ---
    ray.shutdown()
    from datetime import datetime
    timestamp = datetime.now().strftime("%d_%H%M")
    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", f"results_{timestamp}.json")
    data_to_save = {
        "costs_mean_epoch": costs_mean_epoch,
        "act_loss_mean_epoch": act_loss_mean_epoch,
        "crtc_loss_mean_epoch": crtc_loss_mean_epoch
    }
    with open(filepath, 'w') as f:
        json.dump(data_to_save, f, indent=4) # indent=4 使 JSON 文件更易读

    # --- 绘图 ---
    # 注意：从Ray actor收集的loss需要聚合到主进程中才能绘图
    # 这里假设 learner_actor.loss 和 learner_critic.loss 存储了所有迭代的loss均值
    
    # 绘制每个epoch的平均损失和成本
    plot_epochs = list(range(len(costs_mean_epoch)))
    if plot_epochs: # 确保有数据可画
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 3, 1)
        plt.plot(plot_epochs, costs_mean_epoch, marker='o', label='Avg Cost per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Average Cost')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(plot_epochs, act_loss_mean_epoch, marker='o', label='Avg Actor Loss per Epoch', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Actor Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(plot_epochs, crtc_loss_mean_epoch, marker='o', label='Avg Critic Loss per Epoch', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Critic Loss')
        plt.legend()
        plt.grid(True)
        
        plt.suptitle("Training Curves per Epoch")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    # --- 保存模型检查点 (与原代码类似) ---
    torch.save({
        'update_steps': total_update_steps,
        'act_state_dict': learner_actor.state_dict(),
        'crtc_state_dict': learner_critic.state_dict(),
        'optimizer_actor_state_dict': optimizer_actor.state_dict(),
        'optimizer_critic_state_dict': optimizer_critic.state_dict(), # 原代码误用了optimizer_actor
    }, CHECKPOINT_PATH)
    print(f"Checkpoint saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main_ppo_training()