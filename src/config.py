import os
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================================================================
# 0.  CONFIGURATION
# =============================================================================

class Config:
    OBSERVATION_HOURS   = 80
    TIME_STEP_HOURS     = 4
    MAX_STEPS           = OBSERVATION_HOURS // TIME_STEP_HOURS  # 20

    TRAIN_SPLIT         = 0.80
    Z_CLIP              = 3.3

    IV_LEVELS           = 5
    VASO_LEVELS         = 5
    N_ACTIONS           = IV_LEVELS * VASO_LEVELS   # 25

    N_STATE_VARS        = 47

    REWARD_SURVIVE      = +1.0
    REWARD_DIE          = -1.0
    REWARD_STEP         =  0.0

    GAMMA               = 0.99
    LR                  = 3e-4          # slightly higher than original for faster convergence
    BATCH_SIZE          = 128
    REPLAY_BUFFER_SIZE  = 500_000
    HIDDEN_DIM          = 256

    H_INIT              = 1.0
    H_MAX               = 100.0
    H_STEP              = 2.0

    # Reduced for fast iteration — increase for paper-faithful results
    N_TRAINING_SESSIONS = 10
    N_ITERATIONS        = 50

    EPSILON_START       = 0.3           # lower: we're doing offline RL, not exploration
    EPSILON_END         = 0.01
    EPSILON_DECAY       = 0.99

    CHECKPOINT_DIR      = "checkpoints"
    RESULTS_DIR         = "results"

cfg = Config()
os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(cfg.RESULTS_DIR,    exist_ok=True)
