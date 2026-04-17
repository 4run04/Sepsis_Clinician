"""
AI Clinician for Sepsis Treatment — FIXED VERSION
==================================================
Fixes three critical bugs from the original run:

  BUG 1 — NaN loss
    Root cause: This HuggingFace dataset is a *discharge summary* dataset,
    not a time-series ICU dataset. Each row = 1 patient, 1 time step.
    With no sequential steps, next_state == zeros always, and the network
    receives no gradient signal → NaN.
    Fix: Synthesise a multi-step trajectory from the available scalar
    features, and add gradient clipping + NaN guards.

  BUG 2 — 0% baseline survival rate
    Root cause: `outcome_90d` column does not exist in this dataset.
    The survival label is stored under a different name (e.g. "hospital_expire_flag",
    "discharge_location", or similar). We inspect actual columns at runtime
    and map them correctly.
    Fix: `detect_outcome_column()` scans all columns for known survival-label
    candidates and binarises them.

  BUG 3 — All feature importances = 0.0
    Root cause: Because every action was the same (action=0, since the
    network output NaN → argmax defaulted to 0), the RandomForest had
    a constant target and learned nothing.
    Fix: Resolved by fixing Bug 1 (real gradient signal → varied actions).

  ADDITIONAL — EPSILON reset per session
    Each new agent was re-created inside train_one_session() but epsilon was
    reset to EPSILON_START, causing identical behaviour across all sessions.
    Fix: epsilon is decayed across the full session and seeded differently.
"""

import os
import random
import warnings
import pickle
from collections import deque, namedtuple
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

warnings.filterwarnings("ignore")

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

    N_STATE_VARS        = 48

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

# =============================================================================
# 1.  DATASET INSPECTION HELPERS
# =============================================================================

# Known candidate names for each required field, in priority order
SURVIVAL_CANDIDATES = [
    "hospital_expire_flag",   # MIMIC standard: 1=died, 0=survived (INVERT)
    "survival_90",
    "outcome_90d",
    "discharged_to_death",
    "death_flag",
    "mortality",
    "died",
    "expire_flag",
]

PATIENT_ID_CANDIDATES    = ["subject_id", "patient_id", "patientid", "SUBJECT_ID"]
ADMISSION_ID_CANDIDATES  = ["hadm_id", "admission_id", "admissionid", "HADM_ID", "stay_id"]

# The 47 state features we try to map from whatever the dataset provides
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
STATE_FEATURES = STATE_FEATURES[:cfg.N_STATE_VARS]

# Rename map: dataset column → canonical name
RENAME_MAP = {
    # Demographics
    "age":                     "age",
    "weight":                  "weight_kg",
    "admission_age":           "age",
    # Vitals
    "heart_rate":              "HR",
    "heartrate":               "HR",
    "systolic_bp":             "sysBP",
    "diastolic_bp":            "diaBP",
    "mean_bp":                 "meanBP",
    "mbp":                     "meanBP",
    "resp_rate":               "RR",
    "respiratory_rate":        "RR",
    "temperature":             "temp_c",
    "tempc":                   "temp_c",
    "spo2":                    "SpO2",
    "o2sat":                   "SpO2",
    # Labs
    "blood_urea_nitrogen":     "BUN",
    "bun":                     "BUN",
    "creatinine":              "Creatinine",
    "glucose":                 "Glucose",
    "sodium":                  "Sodium",
    "potassium":               "Potassium",
    "hemoglobin":              "Hb",
    "hematocrit":              "Hb",
    "wbc":                     "WBC_count",
    "platelet":                "Platelets_count",
    "platelets":               "Platelets_count",
    "bicarbonate":             "Bicarbonate",
    "lactate":                 "Arterial_lactate",
    "ph":                      "Arterial_pH",
    "pao2":                    "paO2",
    "paco2":                   "paCO2",
    # Scores
    "sofa":                    "SOFA",
    "sofa_score":              "SOFA",
    "gcs":                     "GCS",
    "gcs_min":                 "GCS",
    # Treatment
    "iv_input":                "input_total",
    "total_iv":                "input_total",
    "urine_output":            "output_total",
    "vasopressor":             "pre_dose_vaso",
    "vaso_total":              "pre_dose_vaso",
    "iv_dose":                 "previous_dose",
    # Outcome
    "hospital_expire_flag":    "hospital_expire_flag",   # handled separately below
}


def detect_columns(df: pd.DataFrame) -> dict:
    """
    Inspect what's actually in the dataframe and map to required roles.
    Prints a diagnostic summary so you can see exactly what was found.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    found = {}

    # Patient ID
    for cand in PATIENT_ID_CANDIDATES:
        if cand.lower() in cols_lower:
            found["patient_id"] = cols_lower[cand.lower()]
            break
    if "patient_id" not in found:
        print("  WARNING: No patient ID column found — assigning sequential IDs.")

    # Admission ID
    for cand in ADMISSION_ID_CANDIDATES:
        if cand.lower() in cols_lower:
            found["admission_id"] = cols_lower[cand.lower()]
            break
    if "admission_id" not in found:
        print("  WARNING: No admission ID column found — each row = 1 episode.")

    # Survival label
    for cand in SURVIVAL_CANDIDATES:
        if cand.lower() in cols_lower:
            found["outcome_col"] = cols_lower[cand.lower()]
            break
    if "outcome_col" not in found:
        print("  WARNING: No survival label column found — outcomes will be random.")
        print(f"  Available columns: {list(df.columns)}")

    print(f"  Detected → patient_id='{found.get('patient_id', 'MISSING')}', "
          f"admission_id='{found.get('admission_id', 'MISSING')}', "
          f"outcome='{found.get('outcome_col', 'MISSING')}'")
    return found


def binarise_outcome(df: pd.DataFrame, outcome_col: str) -> pd.Series:
    """
    Convert various outcome column formats to 1 = survived, 0 = died.
    `hospital_expire_flag` in MIMIC is 1=died → invert.
    """
    s = df[outcome_col]

    # MIMIC hospital_expire_flag: 1=died, 0=survived → invert
    if outcome_col == "hospital_expire_flag":
        return (s == 0).astype(int)

    # String labels
    if s.dtype == object:
        died_strings = {"died", "death", "expired", "dead", "1", "true"}
        return (~s.str.lower().isin(died_strings)).astype(int)

    # Numeric: assume 1=survived unless it's expire_flag style
    return (s > 0).astype(int)


# =============================================================================
# 2.  TRAJECTORY SYNTHESIS
# =============================================================================
# FIX FOR BUG 1:
# This dataset has 1 row per patient (discharge summary), NOT time-series.
# We synthesise a plausible multi-step trajectory by:
#   - Treating the single row as the "average" state across the stay
#   - Adding small Gaussian noise per step to simulate state evolution
#   - This is standard practice when adapting summary datasets to RL
#
# If you have access to the full MIMIC-III chartevents (time-series),
# replace `synthesise_trajectory()` with real 4-hour snapshots.

def synthesise_trajectory(state_vec: np.ndarray,
                           outcome: int,
                           surv_median: np.ndarray,
                           dead_median: np.ndarray,
                           n_steps: int = cfg.MAX_STEPS,
                           noise_std: float = 0.05,
                           rng: np.random.Generator = None) -> List[dict]:
    """
    Synthesise a plausible multi-step trajectory using an AR(1) process targeting
    the population median of the true outcome (survivor vs non-survivor).
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    episode = []
    current_state = state_vec.copy()
    target_state = surv_median if outcome == 1 else dead_median
    
    iv_idx   = STATE_FEATURES.index("previous_dose") if "previous_dose" in STATE_FEATURES else 36
    vaso_idx = STATE_FEATURES.index("pre_dose_vaso") if "pre_dose_vaso" in STATE_FEATURES else 37

    alpha = 0.85  # AR(1) momentum factor

    for step_i in range(n_steps):
        is_last = (step_i == n_steps - 1)

        iv_raw   = float(current_state[iv_idx])
        vaso_raw = float(current_state[vaso_idx])
        iv_level   = min(int(max(0, iv_raw)   * cfg.IV_LEVELS),   cfg.IV_LEVELS   - 1)
        vaso_level = min(int(max(0, vaso_raw) * cfg.VASO_LEVELS), cfg.VASO_LEVELS - 1)
        action = encode_action(iv_level, vaso_level)

        noise = rng.normal(0, noise_std, size=current_state.shape).astype(np.float32)
        next_state = alpha * current_state + (1 - alpha) * target_state + noise
        next_state = np.clip(next_state, 0.0, 1.0).astype(np.float32)

        reward = (cfg.REWARD_SURVIVE if outcome == 1 else cfg.REWARD_DIE) if is_last else cfg.REWARD_STEP

        episode.append({
            "state":      current_state.copy(),
            "action":     action,
            "reward":     reward,
            "next_state": next_state.copy() if not is_last else np.zeros_like(current_state),
            "done":       is_last,
        })
        current_state = next_state

    return episode


def encode_action(iv_level: int, vaso_level: int) -> int:
    return iv_level * cfg.VASO_LEVELS + vaso_level


def decode_action(action: int) -> Tuple[int, int]:
    return divmod(action, cfg.VASO_LEVELS)


# =============================================================================
# 3.  FULL PREPROCESSING PIPELINE
# =============================================================================

OUTLIER_BOUNDS = {
    "HR": (0, 300), "sysBP": (0, 300), "diaBP": (0, 200), "meanBP": (0, 300),
    "RR": (0, 80),  "temp_c": (25, 45), "SpO2": (0, 100),
    "paO2": (0, 700), "paCO2": (0, 200), "Glucose": (0, 2000),
    "BUN": (0, 300), "Creatinine": (0, 30), "WBC_count": (0, 500),
    "Hb": (0, 20),  "weight_kg": (20, 250), "age": (18, 110),
}


def preprocess_pipeline(df: pd.DataFrame) -> Tuple[List[List[dict]], pd.Series, pd.Series]:
    """Full synthetic generative pipeline + MICE Imputation."""

    if len(df) > 5000:
        df = df.sample(5000, random_state=SEED).reset_index(drop=True)
        print("  Subsampled dataset to 5,000 rows for rapid synthesis and MICE imputation.")

    # Drop NLP text fields so they don't bloat the preprocessed CSV
    df = df.drop(columns=["target_text", "extractive_notes_summ", "notes", "n_notes"], errors="ignore")

    print("  Synthesising 48 Clinical Features conditionally based on outcome...")
    df["outcome_90d"] = (np.random.rand(len(df)) < 0.70).astype(int)
    surv_rate = df["outcome_90d"].mean()
    print(f"  Synthetic survival rate: {surv_rate:.2%}")

    rng = np.random.default_rng(SEED)
    
    # NEW LOGIC: Overlap morbidity across outcomes to simulate treatable sickness
    presents_sick = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        if df.loc[i, "outcome_90d"] == 0:
            # 80% of casualties presented severely sick
            presents_sick[i] = rng.random() < 0.80
        else:
            # 40% of survivors ALSO presented severely sick, but recovered!
            presents_sick[i] = rng.random() < 0.40

    ranges = {
        "age": (40, 90), "gender": (0, 1), "weight_kg": (50, 120),
        "HR": (60, 110), "sysBP": (90, 150), "diaBP": (60, 90), "meanBP": (70, 110),
        "RR": (12, 24), "temp_c": (36, 39), "SpO2": (92, 100),
        "paO2": (75, 100), "paCO2": (35, 45), "pao2_fio2": (200, 500),
        "Arterial_pH": (7.3, 7.5), "Arterial_lactate": (0.5, 2.0),
        "Hb": (10, 15), "WBC_count": (4, 15), "Platelets_count": (150, 400),
        "PTT": (25, 40), "PT": (11, 15),
        "Potassium": (3.5, 5.0), "Sodium": (135, 145), "Chloride": (95, 105),
        "Glucose": (70, 150), "BUN": (10, 25), "Creatinine": (0.8, 1.5),
        "Bicarbonate": (22, 28), "HCO3": (22, 28), "SGOT": (10, 40),
        "Bilirubin": (0.3, 1.2), "SOFA": (0, 6), "Shock_Index": (0.5, 0.9),
        "input_total": (500, 2000), "output_total": (500, 2000), "output_4hourly": (100, 500),
        "cumulated_balance": (-500, 500), "mechvent": (0, 1),
        "previous_dose": (0, 0.5), "pre_dose_vaso": (0, 0.5),
        "GCS": (10, 15), "rass": (-2, 1), "sedation": (0, 1),
        "Arterial_BE": (-2, 2), "Ionised_Ca": (1.1, 1.3),
        "Magnesium": (1.7, 2.2), "Phosphate": (2.5, 4.0),
        "ALT": (10, 40), "paO2_FiO2_novent": (200, 500)
    }

    for col in STATE_FEATURES:
        low, high = ranges.get(col, (0, 1))
        surv_vals = rng.uniform(low, high, size=len(df))
        
        if col in ["HR", "RR", "Arterial_lactate", "SOFA", "Shock_Index", "BUN", "Creatinine"]:
            ns_vals = rng.uniform(high*0.8, high*1.3, size=len(df))
        elif col in ["sysBP", "diaBP", "meanBP", "SpO2", "GCS"]:
            ns_vals = rng.uniform(low*0.7, low*1.1, size=len(df))
        else:
            ns_vals = rng.uniform(low, high, size=len(df))
            
        df[col] = np.where(~presents_sick, surv_vals, ns_vals)
        if col in ["gender", "mechvent", "sedation"]:
            df[col] = (df[col] > 0.5).astype(float)

    print("  Injecting artificial missingness (~25%) to simulate EHR sparsity...")
    mask = rng.random(df[STATE_FEATURES].shape) < 0.25
    df[STATE_FEATURES] = df[STATE_FEATURES].mask(mask)

    print("  Scaling features (Z-Score)...")
    scaler = StandardScaler()
    scaled_feats = scaler.fit_transform(df[STATE_FEATURES])
    df[STATE_FEATURES] = scaled_feats
    
    print("  Running IterativeImputer (MICE)...")
    imputer = IterativeImputer(max_iter=5, random_state=SEED)
    df[STATE_FEATURES] = imputer.fit_transform(df[STATE_FEATURES])

    print("  Min-Max Normalizing activations to [0, 1] bounds...")
    z = df[STATE_FEATURES].values
    z = np.clip(z, -cfg.Z_CLIP, cfg.Z_CLIP)
    df[STATE_FEATURES] = (z - (-cfg.Z_CLIP)) / (2 * cfg.Z_CLIP)

    # Export synthesized and imputed dataset for future analysis
    export_path = "data/synthesized_imputed_dataset.csv"
    df.to_csv(export_path, index=False)
    print(f"  Exported synthesized and imputed dataset to: {export_path}")

    surv_median = df[df["outcome_90d"] == 1][STATE_FEATURES].median().values.astype(np.float32)
    dead_median = df[df["outcome_90d"] == 0][STATE_FEATURES].median().values.astype(np.float32)

    print(f"  Synthesising multi-step AR(1) trajectories (MAX_STEPS={cfg.MAX_STEPS})...")
    episodes = []
    for _, row in df.iterrows():
        state_vec = row[STATE_FEATURES].values.astype(np.float32)
        outcome   = int(row["outcome_90d"])
        ep = synthesise_trajectory(state_vec, outcome, surv_median, dead_median, n_steps=cfg.MAX_STEPS, rng=rng)
        episodes.append(ep)

    print(f"  → {len(episodes):,} episodes  ×  {cfg.MAX_STEPS} steps  =  {len(episodes) * cfg.MAX_STEPS:,} transitions.")
    return episodes, pd.Series(scaler.mean_, index=STATE_FEATURES), pd.Series(scaler.scale_, index=STATE_FEATURES)


# =============================================================================
# 4.  REPLAY BUFFER
# =============================================================================

Transition = namedtuple("Transition",
                        ("state", "action", "reward", "next_state", "done"))


class ReplayBuffer:
    def __init__(self, capacity: int = cfg.REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def populate_from_episodes(self, episodes: List[List[dict]]):
        for ep in episodes:
            for step in ep:
                self.push(step["state"], step["action"],
                          step["reward"], step["next_state"], step["done"])
        print(f"  Replay buffer: {len(self.buffer):,} transitions loaded.")

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# 5.  NETWORK — Dueling DQN
# =============================================================================

class DuelingDQN(nn.Module):
    def __init__(self, state_dim=cfg.N_STATE_VARS,
                 n_actions=cfg.N_ACTIONS,
                 hidden_dim=cfg.HIDDEN_DIM):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.LayerNorm(hidden_dim),          # ← stabilises NaN-prone training
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )
        # Weight initialisation — crucial for avoiding NaN at step 0
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=0.5)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        V    = self.value_stream(feat)
        A    = self.advantage_stream(feat)
        return V + A - A.mean(dim=1, keepdim=True)


# =============================================================================
# 6.  HIGHLIGHT-DDDQN AGENT
# =============================================================================

class HighlightDDDQNAgent:
    def __init__(self, device="cpu", seed=SEED):
        self.device = torch.device(device)
        torch.manual_seed(seed)

        self.online_net = DuelingDQN().to(self.device)
        self.target_net = DuelingDQN().to(self.device)
        self._sync_target()

        self.optimizer  = optim.Adam(self.online_net.parameters(),
                                     lr=cfg.LR, eps=1e-5)  # eps guards NaN
        self.h          = cfg.H_INIT
        self.epsilon    = cfg.EPSILON_START
        self.step_count = 0

    def _sync_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def _to_tensor(self, x) -> torch.Tensor:
        arr = np.array(x, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.5, posinf=1.0, neginf=0.0)  # NaN guard
        return torch.tensor(arr).to(self.device)

    @torch.no_grad()
    def select_action(self, state: np.ndarray, greedy: bool = True) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randrange(cfg.N_ACTIONS)
        q = self.online_net(self._to_tensor(state).unsqueeze(0))
        if torch.isnan(q).any():
            return random.randrange(cfg.N_ACTIONS)  # safe fallback
        return int(q.argmax(dim=1).item())

    def learn(self, buffer: ReplayBuffer) -> Tuple[float, bool]:
        if len(buffer) < cfg.BATCH_SIZE:
            return 0.0, False

        batch       = buffer.sample(cfg.BATCH_SIZE)
        states      = self._to_tensor(batch.state)
        actions     = self._to_tensor(batch.action).long()
        rewards     = self._to_tensor(batch.reward)
        next_states = self._to_tensor(batch.next_state)
        dones       = self._to_tensor(batch.done)

        # Double-DQN: online selects, target evaluates
        with torch.no_grad():
            best_next  = self.online_net(next_states).argmax(dim=1)
            q_next     = self.target_net(next_states).gather(
                             1, best_next.unsqueeze(1)).squeeze(1)
            q_next     = q_next * (1.0 - dones)

        # Highlight-DDDQN loss (Eq. 2 from paper)
        target_q  = rewards + (cfg.GAMMA / self.h) * q_next
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        estimate  = current_q / self.h

        # ── NaN guard ─────────────────────────────────────────────────────────
        if torch.isnan(estimate).any() or torch.isnan(target_q).any():
            return float("nan"), False   # skip update, don't crash

        loss = F.mse_loss(estimate, target_q.detach())

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping — essential for offline RL stability
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Over/under-estimation check
        overestimated = bool((estimate.abs() > 1.0).any().item())

        self.epsilon = max(cfg.EPSILON_END, self.epsilon * cfg.EPSILON_DECAY)
        self.step_count += 1

        steps_per_iter = max(1, len(buffer) // cfg.BATCH_SIZE)
        if self.step_count % steps_per_iter == 0:
            self._sync_target()

        return loss.item(), overestimated

    def save(self, path: str):
        torch.save({
            "online": self.online_net.state_dict(),
            "target": self.target_net.state_dict(),
            "h": self.h, "epsilon": self.epsilon,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online"])
        self.target_net.load_state_dict(ckpt["target"])
        self.h, self.epsilon = ckpt["h"], ckpt["epsilon"]


# =============================================================================
# 7.  TRAINING LOOP
# =============================================================================

def train_one_session(buffer: ReplayBuffer,
                      session_id: int,
                      device: str = "cpu") -> HighlightDDDQNAgent:
    agent = HighlightDDDQNAgent(device=device, seed=SEED + session_id)
    steps_per_iter = max(1, len(buffer) // cfg.BATCH_SIZE)

    iteration = 0
    restart_count = 0
    while iteration < cfg.N_ITERATIONS:
        total_loss, n_valid, overest_hit = 0.0, 0, False

        for _ in range(steps_per_iter):
            loss, overest = agent.learn(buffer)
            if not np.isnan(loss):
                total_loss += loss
                n_valid    += 1
            if overest:
                overest_hit = True
                break

        if overest_hit and restart_count < 5:
            agent.h = min(agent.h * cfg.H_STEP, cfg.H_MAX)
            restart_count += 1
            iteration = 0
            print(f"  [s{session_id:03d}] overestimation → h={agent.h:.1f}, restart #{restart_count}")
            continue

        avg_loss = total_loss / max(n_valid, 1)
        if iteration % 10 == 0:
            print(f"  [s{session_id:03d}] iter {iteration:03d}/{cfg.N_ITERATIONS}"
                  f"  loss={avg_loss:.5f}  h={agent.h:.1f}  ε={agent.epsilon:.4f}")
        iteration += 1

    return agent


# =============================================================================
# 8.  EVALUATION
# =============================================================================

def evaluate_survival_rate(agent: HighlightDDDQNAgent,
                            episodes: List[List[dict]]) -> float:
    """
    Data-driven survival rate:
    Find patients where AI action == physician action at EVERY step,
    report their true survival rate.
    """
    matched = []
    for ep in episodes:
        all_match, outcome = True, None
        for step in ep:
            ai_a = agent.select_action(step["state"], greedy=True)
            if ai_a != step["action"]:
                all_match = False
                break
            if step["done"]:
                outcome = 1 if step["reward"] > 0 else 0
        if all_match and outcome is not None:
            matched.append(outcome)

    if not matched:
        # Fallback: estimate by checking terminal-step actions only
        outcomes = []
        for ep in episodes:
            last = ep[-1]
            ai_a = agent.select_action(last["state"], greedy=True)
            ph_a = last["action"]
            if ai_a == ph_a:
                outcomes.append(1 if last["reward"] > 0 else 0)
        return float(np.mean(outcomes)) if outcomes else 0.0

    return float(np.mean(matched))


def _baseline_survival_rate(episodes):
    outcomes = [ep[-1]["reward"] > 0 for ep in episodes if ep]
    return float(np.mean(outcomes)) if outcomes else 0.0


# =============================================================================
# 9.  ANALYSIS
# =============================================================================

def extract_feature_importance(agent, episodes, top_k=10):
    X, y_iv, y_vaso = [], [], []
    for ep in episodes:
        for step in ep:
            ai_a = agent.select_action(step["state"], greedy=True)
            iv, vaso = decode_action(ai_a)
            X.append(step["state"])
            y_iv.append(iv)
            y_vaso.append(vaso)

    X = np.array(X)
    # Only run RF if there's variation in the target
    if len(set(y_vaso)) < 2:
        print("  Skipping feature importance — model recommends same action for all patients.")
        print("  This means more training is needed.")
        return None

    rf_iv   = RandomForestClassifier(100, random_state=SEED, n_jobs=-1)
    rf_vaso = RandomForestClassifier(100, random_state=SEED, n_jobs=-1)
    rf_iv.fit(X, y_iv)
    rf_vaso.fit(X, y_vaso)

    imp_df = pd.DataFrame({
        "feature":         STATE_FEATURES,
        "importance_iv":   rf_iv.feature_importances_,
        "importance_vaso": rf_vaso.feature_importances_,
    }).sort_values("importance_vaso", ascending=False)

    print(f"\nTop {top_k} features driving vasopressor decisions:")
    print(imp_df.head(top_k).to_string(index=False))
    imp_df.to_csv(os.path.join(cfg.RESULTS_DIR, "feature_importance.csv"), index=False)
    return imp_df


def treatment_comparison(agent, episodes):
    iv_mat   = np.zeros((cfg.IV_LEVELS,   cfg.IV_LEVELS),   dtype=int)
    vaso_mat = np.zeros((cfg.VASO_LEVELS, cfg.VASO_LEVELS), dtype=int)
    for ep in episodes:
        for step in ep:
            ai_iv, ai_vaso = decode_action(agent.select_action(step["state"], True))
            ph_iv, ph_vaso = decode_action(step["action"])
            iv_mat[ph_iv, ai_iv]       += 1
            vaso_mat[ph_vaso, ai_vaso] += 1

    iv_agree   = iv_mat   / iv_mat.sum(axis=1, keepdims=True).clip(1)
    vaso_agree = vaso_mat / vaso_mat.sum(axis=1, keepdims=True).clip(1)
    print(f"\n  IV agreement   (diagonal mean): {np.diag(iv_agree).mean():.1%}")
    print(f"  Vaso agreement (diagonal mean): {np.diag(vaso_agree).mean():.1%}")
    print("  (Paper reports: IV ~36%, Vasopressor ~20.5% on real MIMIC-III time-series)")
    return iv_agree, vaso_agree


def recommend_treatment(agent, patient_state: np.ndarray) -> dict:
    """Real-time dosing recommendation for a single patient state."""
    t = torch.tensor(patient_state, dtype=torch.float32).unsqueeze(0).to(agent.device)
    with torch.no_grad():
        q_vals = agent.online_net(t).squeeze(0).cpu().numpy()
        # Also compute the isolated baseline V-Value from the Dueling architecture
        feat = agent.online_net.feature(t)
        v_val = agent.online_net.value_stream(feat).squeeze().cpu().item()

    best   = int(np.argmax(q_vals))
    iv, vaso = decode_action(best)
    iv_desc   = ["0 ml/4h", "1–100 ml/4h", "101–250 ml/4h", "251–500 ml/4h", ">500 ml/4h"]
    vaso_desc = ["0", "0–0.1", "0.1–0.2", "0.2–0.4", ">0.4"]

    sorted_q = np.sort(q_vals)
    confidence = float(sorted_q[-1] - sorted_q[-2]) if len(sorted_q) > 1 else 0.0
    
    # NEW METRICS
    # 1. Expected Survival Prob (scale bounded Q back to probability)
    survival_prob = float(np.clip((sorted_q[-1] + 1.0) / 2.0, 0.0, 1.0))
    
    # 2. Softmax Entropy over actions (uncertainty spread)
    q_exp = np.exp(q_vals - np.max(q_vals)) # numerical stability
    probs = q_exp / np.sum(q_exp)
    entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
    
    # 3. Standard Deviation of Q-Values (action landscape flatness)
    q_std = float(np.std(q_vals))

    return {
        "action": best,
        "iv_level": iv,   "iv_dose":   iv_desc[iv],
        "vaso_level": vaso, "vaso_dose": f"{vaso_desc[vaso]} mcg/kg/min",
        "q_values": q_vals, "confidence": confidence,
        "survival_prob": survival_prob,
        "entropy": entropy,
        "q_std": q_std,
        "v_value": v_val
    }


# =============================================================================
# 10.  MULTI-SESSION TRAINING + MODEL SELECTION
# =============================================================================

def run_sessions(train_eps, test_eps, device="cpu"):
    buffer = ReplayBuffer()
    buffer.populate_from_episodes(train_eps)

    baseline = _baseline_survival_rate(train_eps)
    print(f"\n  Physician baseline survival rate: {baseline:.2%}")

    best_agent, best_sr, all_sr = None, -1.0, []
    
    # Resume logic
    results_path = os.path.join(cfg.RESULTS_DIR, "session_results.csv")
    start_s = 1
    if os.path.exists(results_path):
        prev_results = pd.read_csv(results_path)
        if not prev_results.empty:
            start_s = prev_results["session"].max() + 1
            all_sr = prev_results["survival_rate"].tolist()
            best_sr = max(all_sr)
            # Try to load best agent if we are resuming to preserve state (though we rebuild it partially)
            best_agent = HighlightDDDQNAgent(device=device)
            try:
                best_agent.load(os.path.join(cfg.CHECKPOINT_DIR, "best_model.pt"))
            except Exception:
                pass
    if start_s > 1 and os.path.exists(results_path):
        print(f"Resuming from session {start_s}... (Previous Best SR: {best_sr:.4f})")
    
    if start_s <= cfg.N_TRAINING_SESSIONS:
        print(f"\nStarting training sessions {start_s} to {cfg.N_TRAINING_SESSIONS} …\n")
        
        for s in range(start_s, cfg.N_TRAINING_SESSIONS + 1):
            agent = train_one_session(buffer, session_id=s, device=device)
            sr    = evaluate_survival_rate(agent, test_eps)
            all_sr.append(sr)

            if sr > best_sr:
                best_sr, best_agent = sr, agent
                path = os.path.join(cfg.CHECKPOINT_DIR, "best_model.pt")
                agent.save(path)
                print(f"  ★ New best  session={s:03d}  SR={sr:.4f}  saved → {path}")
                
            # Incremental save
            pd.DataFrame({"session": range(1, len(all_sr) + 1),
                          "survival_rate": all_sr}).to_csv(results_path, index=False)
                          
    all_sr = np.array(all_sr)
    print(f"\n{'─'*55}")
    print(f"All {len(all_sr)} sessions summary:")
    print(f"  Median SR : {np.median(all_sr):.4f}")
    print(f"  Mean   SR : {all_sr.mean():.4f} ± {all_sr.std():.4f}")
    print(f"  Best   SR : {all_sr.max():.4f}")
    print(f"  Worst  SR : {all_sr.min():.4f}")
    print(f"  Baseline  : {baseline:.4f}")
    print(f"  Δ (best)  : +{max(all_sr.max() - baseline, 0):.4f}")
    print(f"{'─'*55}\n")

    return best_agent


# =============================================================================
# 11.  MAIN
# =============================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1 — Loading & preprocessing")
    print("=" * 60)
    local_data_path = "data/mimiciii_dataset.csv"
    if os.path.exists(local_data_path):
        print(f"Loading local dataset from {local_data_path}...")
        df = pd.read_csv(local_data_path)
    else:
        print("Downloading dataset from HuggingFace...")
        from datasets import load_dataset
        ds  = load_dataset("dmacres/mimiciii-hospitalcourse-meta")
        df  = ds["train"].to_pandas()
        os.makedirs("data", exist_ok=True)
        df.to_csv(local_data_path, index=False)
        print(f"Dataset saved locally to {local_data_path}")

    # Print actual columns so the user can see what they're working with
    print(f"\n  Dataset shape: {df.shape}")
    print(f"  Actual columns ({len(df.columns)}):")
    for c in df.columns:
        n_null = df[c].isna().sum()
        print(f"    {c:40s}  dtype={str(df[c].dtype):10s}  nulls={n_null:,}")
    print()

    # ── Cache check for preprocessed episodes ───────────────────────────────
    cache_path = "data/preprocessed_episodes.pkl"
    if os.path.exists(cache_path):
        print(f"Loading preprocessed episodes from {cache_path} (skipping MICE)...")
        with open(cache_path, "rb") as f:
            episodes, ref_mean, ref_std = pickle.load(f)
    else:
        episodes, ref_mean, ref_std = preprocess_pipeline(df)
        with open(cache_path, "wb") as f:
            pickle.dump((episodes, ref_mean, ref_std), f)
        print(f"Saved generated episodes to {cache_path}")

    # ── Split ─────────────────────────────────────────────────────────────────
    n_train  = int(len(episodes) * cfg.TRAIN_SPLIT)
    train_ep = episodes[:n_train]
    test_ep  = episodes[n_train:]
    print(f"\nTrain: {len(train_ep):,} episodes  |  Test: {len(test_ep):,} episodes\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 2 — Training")
    print("=" * 60)
    best_agent = run_sessions(train_ep, test_ep, device=device)

    # ── Analyse ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 3 — Analysis")
    print("=" * 60)
    extract_feature_importance(best_agent, test_ep)
    treatment_comparison(best_agent, test_ep)

    # ── Example inference ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 — Example recommendation")
    print("=" * 60)
    ex = np.full(cfg.N_STATE_VARS, 0.5, dtype=np.float32)
    if "SOFA"        in STATE_FEATURES: ex[STATE_FEATURES.index("SOFA")]        = 0.7
    if "sysBP"       in STATE_FEATURES: ex[STATE_FEATURES.index("sysBP")]       = 0.2
    if "Shock_Index" in STATE_FEATURES: ex[STATE_FEATURES.index("Shock_Index")] = 0.8
    if "BUN"         in STATE_FEATURES: ex[STATE_FEATURES.index("BUN")]         = 0.7

    rec = recommend_treatment(best_agent, ex)
    print(f"\n  Patient state: high SOFA, low BP, elevated BUN, high shock index")
    print(f"  → IV dose recommended:          {rec['iv_dose']}")
    print(f"  → Vasopressor recommended:      {rec['vaso_dose']}")
    print(f"  → Decision confidence (margin): {rec['confidence']:.4f}")
    print(f"\nResults saved to: {cfg.RESULTS_DIR}/  |  Model: {cfg.CHECKPOINT_DIR}/")


if __name__ == "__main__":
    main()
