

# basic parameter settings

import numpy as np


ENV_CONFIG = {
    # === Basic structural parameters ===

    "num_pools": 5,
    # J: Number of inpatient service pools (hospital wards).
    # Each pool corresponds to a specialty: [GeMed, Surg, Ortho, Card, OtMed]

    "num_servers": [60, 64, 67, 62, 62],
    # N_j: Number of beds (servers) in each pool j.

    # === Patient arrival configuration ===

    "arrival_rate_hourly": [
        [0.75]*12 + [0.41]*12,  # pool 0 (GeMed)
        [0.41]*12 + [0.74]*12,  # pool 1 (Surg)
        [0.59]*24,              # pool 2 (Ortho)
        [0.59]*24,              # pool 3 (Card)
        [0.59]*24               # pool 4 (OtMed)
    ],

    # === Discharge configuration ===

    "discharge_rate_hourly": [
        [0.0]*12 + [0.16]*7 + [0.0]*5,  # pool 0 (GeMed)
        [0.0]*5 + [0.8]*13 + [0.0]*6,   # pool 1 (Surg)
        [0.0]*5 + [0.8]*13 + [0.0]*6,   # pool 2 (Ortho)
        [0.0]*5 + [0.8]*13 + [0.0]*6,   # pool 3 (Card)
        [0.0]*5 + [0.8]*13 + [0.0]*6    # pool 4 (OtMed)
    ],

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

    "admissible_pools": [
        [0, 1, 2, 4],  # GeMed
        [1, 2, 3, 4],  # Surg
        [0, 1, 2, 4],  # Ortho
        [0, 1, 2, 3],  # Card
        [0, 1, 2, 4],  # OtMed
    ],
    # ℐ_i: Set of feasible pools for class i.

    # === Simulation control ===

    "seed": 42,
    # Random seed for reproducibility

    # === training parameter ===
    "Simulation days": 10000,
    # Simulation days per actor.

    "num_actor": 5,
    #number of actors

    "num_epoch": 15,
    #Number of training epochs

    "Clipping parameter": 0.5,

    "Gap Tol": 0.1,


}
