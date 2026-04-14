import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from config import cfg, SEED
from data.pipeline import preprocess_pipeline
from data.helpers import STATE_FEATURES
from rl.training import run_sessions
from eval.analysis import extract_feature_importance, treatment_comparison, recommend_treatment

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
    import os
    import pandas as pd
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

    episodes, ref_mean, ref_std = preprocess_pipeline(df)

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