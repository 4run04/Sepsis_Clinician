import os
import torch
import numpy as np
import pandas as pd
from typing import List
from sklearn.ensemble import RandomForestClassifier
from config import cfg, STATE_FEATURES
from model import decode_action

def evaluate_survival_rate(agent, episodes: List[List[dict]]) -> float:
    """Calculates survival rate where AI and physician actions agree perfectly."""
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
    return float(np.mean(matched)) if matched else 0.0

def recommend_treatment(agent, patient_state: np.ndarray) -> dict:
    """Detailed clinical prediction for a single patient state, including confidence metrics."""
    t = torch.tensor(patient_state, dtype=torch.float32).unsqueeze(0).to(agent.device)
    with torch.no_grad():
        q_vals = agent.online_net(t).squeeze(0).cpu().numpy()
        # Dueling decomposition
        feat = agent.online_net.feature(t)
        v_val = agent.online_net.value_stream(feat).squeeze().cpu().item()

    # Recommendation
    best_act_id = int(np.argmax(q_vals))
    iv_lvl, vaso_lvl = decode_action(best_act_id)
    
    iv_desc = ["0 ml/4h", "1–100 ml/4h", "101–250 ml/4h", "251–500 ml/4h", ">500 ml/4h"]
    vaso_desc = ["0", "0–0.1", "0.1–0.2", "0.2–0.4", ">0.4"]

    # Post-hoc confidence calculations
    sorted_q = np.sort(q_vals)
    confidence = float(sorted_q[-1] - sorted_q[-2]) 
    survival_prob = float(np.clip((sorted_q[-1] + 1.0) / 2.0, 0.0, 1.0))
    
    q_exp = np.exp(q_vals - np.max(q_vals))
    probs = q_exp / np.sum(q_exp)
    entropy = float(-np.sum(probs * np.log(probs + 1e-9)))

    return {
        "action": best_act_id,
        "iv_dose": iv_desc[iv_lvl],
        "vaso_dose": f"{vaso_desc[vaso_lvl]} mcg/kg/min",
        "confidence": confidence,
        "survival_prob": survival_prob,
        "entropy": entropy,
        "v_value": v_val
    }

def extract_feature_importance(agent, episodes: List[List[dict]]):
    """Uses a surrogate Random Forest to interpret which features drive AI decisions."""
    X, y_vaso = [], []
    for ep in episodes:
        for step in ep:
            X.append(step["state"])
            _, vaso = decode_action(agent.select_action(step["state"]))
            y_vaso.append(vaso)
    
    if len(set(y_vaso)) < 2: return None
    
    rf = RandomForestClassifier(100, n_jobs=-1, random_state=42)
    rf.fit(X, y_vaso)
    
    importance = pd.DataFrame({
        "feature": STATE_FEATURES,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
    
    print("\nTop 10 features driving vasopressor decisions:")
    print(importance.head(10).to_string(index=False))
    return importance

def treatment_comparison(agent, episodes: List[List[dict]]):
    """Calculates confusion matrices between AI and physician policy."""
    iv_mat = np.zeros((cfg.IV_LEVELS, cfg.IV_LEVELS))
    for ep in episodes:
        for step in ep:
            ai_iv, _ = decode_action(agent.select_action(step["state"]))
            ph_iv, _ = decode_action(step["action"])
            iv_mat[ph_iv, ai_iv] += 1
    
    agreement = np.diag(iv_mat).sum() / iv_mat.sum()
    print(f"\n  Policy Agreement Ratio: {agreement:.1%}")
    return agreement
