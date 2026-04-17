import os

class Config:
    """Constants and hyperparameters for the Sepsis RL Clinician."""
    
    # Environment & Time-series
    OBSERVATION_HOURS   = 80
    TIME_STEP_HOURS     = 4
    MAX_STEPS           = OBSERVATION_HOURS // TIME_STEP_HOURS  # 20 steps per trajectory
    TRAIN_SPLIT         = 0.80
    Z_CLIP              = 3.3

    # Action Space (5x5 grid = 25 actions)
    IV_LEVELS           = 5
    VASO_LEVELS         = 5
    N_ACTIONS           = IV_LEVELS * VASO_LEVELS

    # State Space
    N_STATE_VARS        = 48

    # Rewards (Sparse binary outcome)
    REWARD_SURVIVE      = +1.0
    REWARD_DIE          = -1.0
    REWARD_STEP         =  0.0

    # Hyperparameters
    GAMMA               = 0.99
    LR                  = 3e-4
    BATCH_SIZE          = 128
    REPLAY_BUFFER_SIZE  = 500_000
    HIDDEN_DIM          = 256

    # Highlight-DQN Tuning
    H_INIT              = 1.0
    H_MAX               = 100.0
    H_STEP              = 2.0

    # Training Sessions
    N_TRAINING_SESSIONS = 10
    N_ITERATIONS        = 50

    # Exploration Policy (Linear Decay)
    EPSILON_START       = 0.3
    EPSILON_END         = 0.01
    EPSILON_DECAY       = 0.99

    # Artifact Directories
    CHECKPOINT_DIR      = "checkpoints"
    RESULTS_DIR         = "results"
    DATA_DIR            = "data"

cfg = Config()

# Ensure directories exist
os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(cfg.RESULTS_DIR,    exist_ok=True)
os.makedirs(cfg.DATA_DIR,       exist_ok=True)

# ── Feature Definitions ──────────────────────────────────────────────────

# List of the 48 clinical features mapped from EHR data
STATE_FEATURES = [
    "age", "gender", "weight_kg",
    "HR", "sysBP", "diaBP", "meanBP", "RR", "temp_c", "SpO2",
    "paO2", "paCO2", "pao2_fio2", "Arterial_pH", "Arterial_lactate",
    "Hb", "WBC_count", "Platelets_count",
    "PTT", "PT",
    "Potassium", "Sodium", "Chloride", "Glucose", "BUN", "Creatinine",
    "Bicarbonate", "HCO3", "SGOT", "Bilirubin",
    "SOFA", "Shock_Index",
    "input_total", "output_total", "output_4hourly", "cumulated_balance",
    "mechvent",
    "previous_dose", "pre_dose_vaso",
    "GCS", "rass", "sedation",
    "Arterial_BE", "Ionised_Ca", "Magnesium", "Phosphate",
    "ALT", "paO2_FiO2_novent",
]

# Candidate column names for mapping raw datasets to canonical names
SURVIVAL_CANDIDATES = ["hospital_expire_flag", "survival_90", "outcome_90d", "discharged_to_death"]
PATIENT_ID_CANDIDATES = ["subject_id", "patient_id", "patientid"]
ADMISSION_ID_CANDIDATES = ["hadm_id", "admission_id", "stay_id"]
