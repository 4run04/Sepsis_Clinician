import os
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from config import cfg, SEED
from data.helpers import STATE_FEATURES
from data.synthesis import decode_action

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

    best   = int(np.argmax(q_vals))
    iv, vaso = decode_action(best)
    iv_desc   = ["0 ml/4h", "1–100 ml/4h", "101–250 ml/4h", "251–500 ml/4h", ">500 ml/4h"]
    vaso_desc = ["0", "0–0.1", "0.1–0.2", "0.2–0.4", ">0.4"]

    sorted_q = np.sort(q_vals)
    confidence = float(sorted_q[-1] - sorted_q[-2]) if len(sorted_q) > 1 else 0.0

    return {
        "action": best,
        "iv_level": iv,   "iv_dose":   iv_desc[iv],
        "vaso_level": vaso, "vaso_dose": f"{vaso_desc[vaso]} mcg/kg/min",
        "q_values": q_vals, "confidence": confidence,
    }

