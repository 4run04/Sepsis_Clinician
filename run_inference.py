import os
import sys
import torch
import numpy as np
import warnings

# Suppress pandas/bottleneck warnings for cleaner output
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from main import HighlightDDDQNAgent, recommend_treatment, cfg, STATE_FEATURES

def run_model(model_path="best_model.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading '{model_path}' on device: {device}...")
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        sys.exit(1)

    agent = HighlightDDDQNAgent(device=device)
    try:
        agent.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Example Test Patient Recommendation")
    print("=" * 60)
    
    # Synthesize a hypothetical critical patient
    ex = np.full(cfg.N_STATE_VARS, 0.5, dtype=np.float32)
    if "SOFA"        in STATE_FEATURES: ex[STATE_FEATURES.index("SOFA")]        = 0.7
    if "sysBP"       in STATE_FEATURES: ex[STATE_FEATURES.index("sysBP")]       = 0.2
    if "Shock_Index" in STATE_FEATURES: ex[STATE_FEATURES.index("Shock_Index")] = 0.8
    if "BUN"         in STATE_FEATURES: ex[STATE_FEATURES.index("BUN")]         = 0.7
    
    rec = recommend_treatment(agent, ex)
    print(f"\n  Patient state: High SOFA, low BP, elevated BUN, high shock index")
    print(f"  -> IV dose recommended:          {rec['iv_dose']}")
    print(f"  -> Vasopressor recommended:      {rec['vaso_dose']}")
    print(f"  -> Decision confidence (margin): {rec['confidence']:.4f}\n")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "best_model.pt"
    run_model(path)
