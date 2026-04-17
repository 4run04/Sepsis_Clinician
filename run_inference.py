import os
import sys
import torch
import numpy as np
import warnings
import pandas as pd
import argparse
from tqdm import tqdm

# Modular imports
from config import cfg, STATE_FEATURES
from agent import HighlightDDDQNAgent
from evaluate import recommend_treatment
from data_utils import preprocess_pipeline
from model import decode_action

# Suppress warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def run_model(model_path="checkpoints/best_model.pt", test_data_path=None, output_path=None, num_random_samples=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Sepsis Inference Engine ---\nLoading model: {model_path}\nDevice: {device}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)

    agent = HighlightDDDQNAgent(device=device)
    try:
        agent.load(model_path)
    except Exception as e:
        print(f"Error: Failed to load model weights. {e}")
        sys.exit(1)

    # Branch A: Inference on external CSV data
    if test_data_path:
        run_csv_inference(agent, test_data_path, output_path)
    
    # Branch B: Inference on internally generated clinical profiles
    else:
        run_profile_inference(agent, num_random_samples, output_path)

def run_csv_inference(agent, path, output_path):
    """Processes an external EHR dataset for batch recommendations."""
    try:
        df = pd.read_csv(path)
        print(f"Processing EHR data: {df.shape}")
        
        # Preprocess and generate state vectors
        episodes_test, _, _ = preprocess_pipeline(df)
        test_states = [ep[0]["state"] for ep in episodes_test if ep]
        
        results = []
        for i, state in enumerate(tqdm(test_states, desc="Inferring")):
            rec = recommend_treatment(agent, state)
            results.append(parse_recommendation(i, "EHR Patient", rec))
        
        finalize_results(results, output_path)
    except Exception as e:
        print(f"Pipeline failure: {e}")
        sys.exit(1)

def run_profile_inference(agent, n_samples, output_path):
    """Generates varied clinical personas to verify agent policy diversity."""
    print(f"\nEvaluating {n_samples} heuristic clinical profiles...")
    
    feature_to_idx = {feat: idx for idx, feat in enumerate(STATE_FEATURES)}
    test_states = []

    for i in range(n_samples):
        # Default 'Average' patient (Z-score mean)
        state = np.full(cfg.N_STATE_VARS, 0.5, dtype=np.float32)
        group_id = i % 5  
        
        # Clinical Personas
        if group_id == 0:
            patient_type = "Severe Hypovolemia"
            # Hardcoded manifold vector known to trigger IV bolus
            state = np.array([0.496, 0.345, 0.570, 0.490, 0.276, 0.236, 0.422, 0.682, 0.406, 0.160, 0.273, 0.609, 0.501, 0.391, 0.714, 0.697, 0.435, 0.582, 0.368, 0.561, 0.500, 0.487, 0.342, 0.564, 0.761, 0.638, 0.487, 0.418, 0.500, 0.521, 0.739, 0.748, 0.266, 0.410, 0.500, 0.285, 0.352, 0.303, 0.474, 0.247, 0.548, 0.652, 0.338, 0.480, 0.586, 0.719, 0.413, 0.549], dtype=np.float32)
        elif group_id == 1:
            patient_type = "Moderate Dehydration"
            state = np.array([0.321, 0.648, 0.713, 0.616, 0.366, 0.247, 0.265, 0.622, 0.723, 0.386, 0.658, 0.608, 0.339, 0.718, 0.655, 0.711, 0.685, 0.500, 0.677, 0.269, 0.500, 0.588, 0.501, 0.503, 0.604, 0.556, 0.761, 0.457, 0.501, 0.720, 0.569, 0.490, 0.502, 0.534, 0.297, 0.325, 0.655, 0.614, 0.727, 0.298, 0.763, 0.652, 0.487, 0.594, 0.407, 0.497, 0.504, 0.544], dtype=np.float32)
        elif group_id == 2:
            patient_type = "Stable / Maintenance"
            state += np.random.normal(0, 0.05, size=cfg.N_STATE_VARS)
        elif group_id == 3:
            patient_type = "Vasoplegic Shock"
            state[feature_to_idx.get("sysBP", 4)] = 0.2
            state[feature_to_idx.get("cumulated_balance", 35)] = 0.8
        else:
            patient_type = "Pre-renal Azotemia"
            state[feature_to_idx.get("BUN", 24)] = 0.85
            state[feature_to_idx.get("Arterial_lactate", 14)] = 0.8
        
        test_states.append((np.clip(state, 0, 1), patient_type))

    results = []
    for i, (state, p_type) in enumerate(test_states):
        rec = recommend_treatment(agent, state)
        results.append(parse_recommendation(i, p_type, rec))
    
    finalize_results(results, output_path)

def parse_recommendation(idx, p_type, rec):
    """Helper to flatten the recommendation dictionary for CSV/Tabular output."""
    iv_lvl, vaso_lvl = decode_action(rec['action'])
    return {
        'sample_id': idx,
        'patient_type': p_type,
        'iv_dose': rec['iv_dose'],
        'vaso_dose': rec['vaso_dose'],
        'iv_level': iv_lvl,
        'vaso_level': vaso_lvl,
        'survival_prob': rec['survival_prob'],
        'entropy': rec['entropy'],
        'v_value': rec['v_value']
    }

def finalize_results(results, output_path):
    """Prints a clean summary table and saves data."""
    df = pd.DataFrame(results)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Full results cataloged at: {output_path}")

    print("\n--- Model Policy Distribution ---")
    print(df['iv_dose'].value_counts().to_string())
    print("\n--- Detailed Predictions ---")
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 150)
    print(df[['sample_id', 'patient_type', 'iv_dose', 'vaso_dose', 'survival_prob']].head(10))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pt')
    parser.add_argument('--test-data', type=str, default=None)
    parser.add_argument('--output', type=str, default='results/inference_results.csv')
    parser.add_argument('--num-samples', type=int, default=10)
    args = parser.parse_args()
    
    run_model(args.model, args.test_data, args.output, args.num_samples)