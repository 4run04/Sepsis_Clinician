import os
import pickle
import numpy as np
import pandas as pd
import torch
from config import cfg, STATE_FEATURES
from data_utils import preprocess_pipeline
from agent import ReplayBuffer, train_one_session, HighlightDDDQNAgent
from evaluate import evaluate_survival_rate, extract_feature_importance, treatment_comparison, recommend_treatment

def run_sessions(train_eps, test_eps, device="cpu"):
    """Orchestrates multiple training sessions and selects the best model."""
    buffer = ReplayBuffer()
    buffer.populate_from_episodes(train_eps)

    best_agent, best_sr, all_sr = None, -1.0, []
    results_path = os.path.join(cfg.RESULTS_DIR, "session_results.csv")

    print(f"\nStarting {cfg.N_TRAINING_SESSIONS} training sessions...\n")
    
    for s in range(1, cfg.N_TRAINING_SESSIONS + 1):
        agent = train_one_session(buffer, session_id=s, device=device)
        sr = evaluate_survival_rate(agent, test_eps)
        all_sr.append(sr)

        if sr > best_sr:
            best_sr, best_agent = sr, agent
            agent.save(os.path.join(cfg.CHECKPOINT_DIR, "best_model.pt"))
            print(f"  ★ New Best Session {s:02d}: Survival Rate = {sr:.4f}")

        # Save incremental progress
        pd.DataFrame({"session": range(1, len(all_sr) + 1), "survival_rate": all_sr}).to_csv(results_path, index=False)

    return best_agent

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Sepsis AI Clinician (Modular v2.0) ---\nUsing device: {device}")

    # 1. Data Ingestion
    local_data_path = os.path.join(cfg.DATA_DIR, "mimiciii_dataset.csv")
    if os.path.exists(local_data_path):
        df = pd.read_csv(local_data_path)
    else:
        from datasets import load_dataset
        df = load_dataset("dmacres/mimiciii-hospitalcourse-meta")["train"].to_pandas()
        df.to_csv(local_data_path, index=False)

    # 2. Preprocessing / Synthesis
    cache_path = os.path.join(cfg.DATA_DIR, "preprocessed_episodes.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            episodes, _, _ = pickle.load(f)
    else:
        episodes, _, _ = preprocess_pipeline(df)
        with open(cache_path, "wb") as f:
            pickle.dump((episodes, None, None), f)

    # 3. Train/Test Split
    n_train = int(len(episodes) * cfg.TRAIN_SPLIT)
    train_ep, test_ep = episodes[:n_train], episodes[n_train:]

    # 4. Core Training
    best_agent = run_sessions(train_ep, test_ep, device=device)

    # 5. Final Analytics
    extract_feature_importance(best_agent, test_ep)
    treatment_comparison(best_agent, test_ep)

    # 6. Example Inference
    mock_patient = np.full(cfg.N_STATE_VARS, 0.5, dtype=np.float32)
    # Simulate a sick patient
    if "SOFA" in STATE_FEATURES: mock_patient[STATE_FEATURES.index("SOFA")] = 0.8
    if "sysBP" in STATE_FEATURES: mock_patient[STATE_FEATURES.index("sysBP")] = 0.2
    
    rec = recommend_treatment(best_agent, mock_patient)
    print(f"\n--- Example Recommendation ---")
    print(f"IV: {rec['iv_dose']} | Vaso: {rec['vaso_dose']}")
    print(f"Survival Probability: {rec['survival_prob']:.1%}")

if __name__ == "__main__":
    main()
