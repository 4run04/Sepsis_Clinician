import os
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from config import cfg, STATE_FEATURES, SURVIVAL_CANDIDATES, PATIENT_ID_CANDIDATES, ADMISSION_ID_CANDIDATES
from model import encode_action

def detect_columns(df: pd.DataFrame) -> dict:
    """Identify the clinical roles of available columns in the raw dataset."""
    cols_lower = {c.lower(): c for c in df.columns}
    found = {}

    for role, candidates in [("patient_id", PATIENT_ID_CANDIDATES), 
                             ("admission_id", ADMISSION_ID_CANDIDATES), 
                             ("outcome_col", SURVIVAL_CANDIDATES)]:
        for cand in candidates:
            if cand.lower() in cols_lower:
                found[role] = cols_lower[cand.lower()]
                break
    
    # Defaults and Warnings
    if "patient_id" not in found: print("  WARNING: No patient ID column found.")
    if "outcome_col" not in found: print("  WARNING: No survival label found.")
    
    return found

def binarise_outcome(df: pd.DataFrame, outcome_col: str) -> pd.Series:
    """Standardize labels to 1=Survived, 0=Died."""
    s = df[outcome_col]
    if outcome_col == "hospital_expire_flag": return (s == 0).astype(int)
    if s.dtype == object:
        died_strings = {"died", "death", "expired", "dead", "1", "true"}
        return (~s.str.lower().isin(died_strings)).astype(int)
    return (s > 0).astype(int)

def synthesise_trajectory(state_vec: np.ndarray, outcome: int, 
                           surv_median: np.ndarray, dead_median: np.ndarray, 
                           n_steps: int = cfg.MAX_STEPS, rng=None) -> List[dict]:
    """Generates a multi-step AR(1) clinical progression based on total outcome."""
    if rng is None: rng = np.random.default_rng(42)
    episode = []
    current_state = state_vec.copy()
    target_state = surv_median if outcome == 1 else dead_median
    
    # Heuristic indices for IV/Vaso to determine 'physician' action in synthetic data
    iv_idx, vaso_idx = 36, 37 
    alpha = 0.85 # Continuity factor

    for step_i in range(n_steps):
        is_last = (step_i == n_steps - 1)
        
        # Calculate policy action based on current synthetic state
        iv_level = min(int(max(0, current_state[iv_idx]) * cfg.IV_LEVELS), cfg.IV_LEVELS - 1)
        vaso_level = min(int(max(0, current_state[vaso_idx]) * cfg.VASO_LEVELS), cfg.VASO_LEVELS - 1)
        action = encode_action(iv_level, vaso_level)

        # AR(1) Step: NextState = alpha * Current + (1-alpha) * Target + Noise
        noise = rng.normal(0, 0.05, size=current_state.shape).astype(np.float32)
        next_state = alpha * current_state + (1 - alpha) * target_state + noise
        next_state = np.clip(next_state, 0.0, 1.0).astype(np.float32)

        reward = (cfg.REWARD_SURVIVE if outcome == 1 else cfg.REWARD_DIE) if is_last else cfg.REWARD_STEP

        episode.append({
            "state": current_state.copy(),
            "action": action,
            "reward": reward,
            "next_state": next_state.copy() if not is_last else np.zeros_like(current_state),
            "done": is_last
        })
        current_state = next_state
    return episode

def preprocess_pipeline(df: pd.DataFrame) -> Tuple[List[List[dict]], pd.Series, pd.Series]:
    """Orchestrates the full data preparation and trajectory generation."""
    SEED = 42
    rng = np.random.default_rng(SEED)
    
    # 1. Subsampling & Cleanup
    if len(df) > 5000: df = df.sample(5000, random_state=SEED).reset_index(drop=True)
    df = df.drop(columns=["target_text", "extractive_notes_summ", "notes", "n_notes"], errors="ignore")

    # 2. Synthetic Feature Generation (Overlapping Distributions)
    df["outcome_90d"] = (rng.random(len(df)) < 0.70).astype(int)
    
    # RECOVERING PATIENT LOGIC: Decouple morbidity from outcome
    presents_sick = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        presents_sick[i] = rng.random() < (0.80 if df.loc[i, "outcome_90d"] == 0 else 0.40)

    # Simplified feature bounds for synthesis
    for col in STATE_FEATURES:
        # Standard healthy-ish range
        surv_vals = rng.uniform(0.3, 0.7, size=len(df))
        # Sick range (shifted towards extremes)
        is_high_risk = col in ["HR", "RR", "Arterial_lactate", "SOFA", "BUN", "Creatinine"]
        ns_vals = rng.uniform(0.7, 1.0, size=len(df)) if is_high_risk else rng.uniform(0.0, 0.3, size=len(df))
        df[col] = np.where(~presents_sick, surv_vals, ns_vals)

    # 3. MICE Imputation & Scaling
    mask = rng.random(df[STATE_FEATURES].shape) < 0.25
    df[STATE_FEATURES] = df[STATE_FEATURES].mask(mask)
    
    scaler = StandardScaler()
    df[STATE_FEATURES] = scaler.fit_transform(df[STATE_FEATURES])
    
    imputer = IterativeImputer(max_iter=5, random_state=SEED)
    df[STATE_FEATURES] = imputer.fit_transform(df[STATE_FEATURES])

    # 4. Normalize to [0, 1]
    z = np.clip(df[STATE_FEATURES].values, -cfg.Z_CLIP, cfg.Z_CLIP)
    df[STATE_FEATURES] = (z + cfg.Z_CLIP) / (2 * cfg.Z_CLIP)

    # 5. Trajectory Generation
    surv_median = df[df["outcome_90d"] == 1][STATE_FEATURES].median().values.astype(np.float32)
    dead_median = df[df["outcome_90d"] == 0][STATE_FEATURES].median().values.astype(np.float32)
    
    episodes = [synthesise_trajectory(row[STATE_FEATURES].values, int(row["outcome_90d"]), surv_median, dead_median, rng=rng) 
                for _, row in df.iterrows()]
    
    return episodes, pd.Series(scaler.mean_, index=STATE_FEATURES), pd.Series(scaler.scale_, index=STATE_FEATURES)
