

# basic parameter settings

import numpy as np


ENV_CONFIG = {
    # === Basic structural parameters ===

    "num_pools": 5,
    # J: Number of inpatient service pools (hospital wards).
    # Each pool corresponds to a specialty: [GeMed, Surg, Ortho, Card, OtMed]

    "num_servers": [60, 64, 67, 62, 62], # 参照p41 : The number of servers per pool  is (N1,..., N5) =  (60, 64, 67, 62, 62);
    # N_j: Number of beds (servers) in each pool j.

    # === Patient arrival configuration ===

    "arrival_rate_hourly": [ # 参照论文p41-figure 8(a)
        [0.75]*12 + [0.41]*12,  # pool 0 (GeMed)
        [0.41]*12 + [0.74]*12,  # pool 1 (Surg)
        [0.59]*24,              # pool 2 (Ortho)
        [0.59]*24,              # pool 3 (Card)
        [0.59]*24               # pool 4 (OtMed)
    ],

    # === Discharge configuration ===

    "discharge_rate_hourly": [ # 参照论文p41-figure 8(b)
        [0.0]*12 + [0.16]*5 + [0.2] + [0.0]*6,  # pool 0 (GeMed)
        [0.0]*6 + [0.08]*11 + [0.12] + [0.0]*6,   # pool 1 (Surg)
        [0.0]*6 + [0.08]*11 + [0.12] + [0.0]*6,   # pool 2 (Ortho)
        [0.0]*6 + [0.08]*11 + [0.12] + [0.0]*6,   # pool 3 (Card)
        [0.0]*6 + [0.08]*11 + [0.12] + [0.0]*6    # pool 4 (OtMed)
    ], 

    "discharge_rate_daily": [0.25, 0.25, 0.25, 0.25, 0.25], # 床数为315, 日到达率为70, 名义占用率为0.889, 
                                                            # 计算得日出院率应为0.25, 参照论文p23-table 2

    "num_epochs_per_day": 8,
    # m: Number of decision epochs per day.
    # Epochs are evenly spaced across the day (e.g., every 3 hours).

    # === Cost structure ===

    "waiting_cost": [6.0, 6.0, 6.0, 6.0, 6.0],
    # C_j: Per-epoch waiting cost for each patient class.

    "overflow_cost": [
        [0,    35,   35,   999,   30],  # Class 0 (GeMed)
        [999,  0,    30,   35,    35],  # Class 1 (Surg)
        [35,   30,   0,    999,   35],  # Class 2 (Ortho)
        [35,   30,   35,   0,    999],  # Class 3 (Card)
        [30,   35,   35,   999,    0],  # Class 4 (OtMed)
    ],
    # B_ij: Overflow cost from class i to pool j.
    # 999 = infeasible assignment (not in admissible set)

    "mask" : [
        [1, 1, 1, 0, 1],  # Class 0 (GeMed)
        [0, 1, 1, 1, 1],  # Class 1 (Surg)
        [1, 1, 1, 0, 1],  # Class 2 (Ortho)
        [1, 1, 1, 1, 0],  # Class 3 (Card)
        [1, 1, 1, 0, 1],  # Class 4 (OtMed)
    ],

    "admissible_pools": [
        [0, 1, 2, 4],  # GeMed
        [1, 2, 3, 4],  # Surg
        [0, 1, 2, 4],  # Ortho
        [0, 1, 2, 3],  # Card
        [0, 1, 2, 4],  # OtMed
    ],
    # ℐ_i: Set of feasible pools for class i.
    "overflow_priority" : [
        [0, 2, 2, -1, 1],
        [-1, 0, 1, 2, 2],
        [2, 1, 0, -1, 2],
        [2, 1, 2, 0, -1],
        [1, 2, 2, -1, 0]
    ],

    # === Simulation control ===

    "seed": 42,
    # Random seed for reproducibility

    # === training parameter ===
    "Simulation_days": 800,
    "num_actor" : 5,
    "num_epoch" : 30,
    "batch_size": 64, 

    "Clipping_parameter": 0.3,

    "Gap_Tol": 0.1,
    
    "max_norm" : 0.5,
    "lam" : 0.95,
    "gamma" : 0.99, 
    "actor_lr" : 1e-4,
    "critic_lr" : 1e-4,
    "adam_eps" : 1e-5, # 默认1e-8
    "entropy_coef" : 0.01,
    "actor_input_dim" : 12,
    "actor_hidden_dim" : [64], 

    "critic_input_dim" : 12,
    "critic_hidden_dim" : [64],

    "is_gae" : True,
    "running_mean_std_norm" : True, 
    "sin_cos_encode": True, 
    "normalized_N_j" : False,
    "reward_scaling" : True,
    "orthogonal_init" : True,
}
