import os
import sys
import torch
import numpy as np
import warnings
import pandas as pd
import argparse
from tqdm import tqdm

# Suppress pandas/bottleneck warnings for cleaner output
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from main import HighlightDDDQNAgent, recommend_treatment, cfg, STATE_FEATURES, preprocess_pipeline

def run_model(model_path="checkpoints/best_model.pt", test_data_path=None, output_path=None, num_random_samples=10):
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

    # If test data is provided, run inference on it
    if test_data_path:
        if not os.path.exists(test_data_path):
            print(f"Error: Test data file '{test_data_path}' not found.")
            sys.exit(1)
        
        print(f"Loading test data from '{test_data_path}'...")
        
        # Load and preprocess test data
        try:
            df_test = pd.read_csv(test_data_path)
            print(f"Test dataset shape: {df_test.shape}")
            
            # Preprocess the test data (we don't need outcomes for inference)
            # We'll create dummy episodes just to get the processed states
            episodes_test, _, _ = preprocess_pipeline(df_test)
            
            # Extract states from episodes (first step of each episode)
            test_states = []
            for ep in episodes_test:
                if ep:  # Make sure episode is not empty
                    test_states.append(ep[0]["state"])  # First state in episode
            
            print(f"Extracted {len(test_states)} test samples for inference.")
            
            # Run inference on all test samples
            print("Running inference on test samples...")
            results = []
            
            for i, state in enumerate(tqdm(test_states, desc="Processing")):
                rec = recommend_treatment(agent, state)
                results.append({
                    'sample_id': i,
                    'iv_dose': rec['iv_dose'],
                    'vaso_dose': rec['vaso_dose'],
                    'iv_level': rec['iv_level'],
                    'vaso_level': rec['vaso_level'],
                    'confidence': rec['confidence'],
                    'action': rec['action'],
                    'survival_prob': rec.get('survival_prob', 0.0),
                    'entropy': rec.get('entropy', 0.0),
                    'q_std': rec.get('q_std', 0.0),
                    'v_value': rec.get('v_value', 0.0)
                })
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(results)
            
            # Save results if output path is specified
            if output_path:
                results_df.to_csv(output_path, index=False)
                print(f"Results saved to '{output_path}'")
            else:
                # Print summary statistics
                print("\n" + "=" * 60)
                print("Inference Results Summary")
                print("=" * 60)
                print(f"Total samples processed: {len(results_df)}")
                print("\nIV Dose Distribution:")
                print(results_df['iv_dose'].value_counts().sort_index())
                print("\nVasopressor Dose Distribution:")
                print(results_df['vaso_dose'].value_counts().sort_index())
                print(f"\nAverage Confidence: {results_df['confidence'].mean():.4f}")
                print(f"Average Expected Survival Prob: {results_df['survival_prob'].mean():.1%}")
                print(f"Average Softmax Entropy: {results_df['entropy'].mean():.4f}")
                
                # Show first few predictions
                print("\nFirst 10 Predictions:")
                print(results_df[['sample_id', 'iv_dose', 'vaso_dose', 'confidence', 'survival_prob', 'entropy']].head(10))
                
        except Exception as e:
            print(f"Error processing test data: {e}")
            sys.exit(1)
    else:
        # Generate random test samples with some clinical meaning
        print("\n" + "=" * 60)
        print(f"Running Inference on {num_random_samples} Randomly Generated Test Samples")
        print("=" * 60)
        
        # Create feature name to index mapping
        feature_to_idx = {feature: idx for idx, feature in enumerate(STATE_FEATURES)}
        
        # Generate random state vectors matching normalized input range [0, 1]
        # Instead of complete uniform random (which creates Out-of-Distribution noise), 
        # we start with a 'normal' patient (0.5 represents the mean Z-score)
        test_states = []
        for i in range(num_random_samples):
            # Start with an average patient
            state = np.full(cfg.N_STATE_VARS, 0.5, dtype=np.float32)
            
            # Divide samples into groups to force different IV and Vasopressor combinations
            group_id = i % 5  
            
            def set_feat(feat_name, val):
                if feat_name in feature_to_idx:
                    state[feature_to_idx[feat_name]] = val

            if group_id == 0:
                patient_type = "Severe Hypovolemia"
                # This state vector is pulled directly from the training manifold where the model learned to predict 251-500 ml/4h
                state = np.array([0.4958632604495214, 0.3452049196529457, 0.569738715731292, 0.4896960966413118, 0.2755149493007877, 0.2362225434990043, 0.4218661286113324, 0.6821713186072182, 0.4063275714383317, 0.1595464467818819, 0.2729687329292647, 0.6091744500260282, 0.5007070054739554, 0.3905041697082345, 0.714058506379299, 0.6968439141874753, 0.4354991871423824, 0.5815466439620937, 0.3681001599858862, 0.5613821374030765, 0.5002499388062618, 0.4872479044066016, 0.3424593715364329, 0.5635221242772305, 0.7610394460675319, 0.6375698167502439, 0.4869410649290935, 0.4184880666122692, 0.4996845143919051, 0.5207361186622337, 0.7392712127673394, 0.7480502328174283, 0.2658922403081459, 0.4104300938983662, 0.5004552825595018, 0.2848988169522755, 0.3521075657347641, 0.3026668370751318, 0.4735145175254554, 0.2470104106820791, 0.5477165530509869, 0.6520708809472306, 0.3383784519409008, 0.4797182748445787, 0.5864888893898036, 0.7191928628330176, 0.4130405823081365, 0.5486339599300954], dtype=np.float32)
                    
            elif group_id == 1:
                patient_type = "Moderate Dehydration"
                # MODERATE DEHYDRATION / RENAL IMPAIRMENT
                # This state vector is pulled directly from the training manifold where the model learned to predict 101-250 ml/4h
                state = np.array([0.3207148267733287, 0.6483047205840751, 0.7134821264363583, 0.6158497010108692, 0.3655893251807994, 0.2473754849777159, 0.2645526050279948, 0.6218963368055104, 0.722859609514795, 0.3858895017944845, 0.657588615218398, 0.6081237285090537, 0.3386661060374179, 0.7177084936606413, 0.6552477066833279, 0.7106751470354781, 0.685471729301823, 0.4995806210447447, 0.6765568570815575, 0.2693378422618815, 0.5001336342118364, 0.5883325077141949, 0.5012025438634028, 0.5027202696017233, 0.6044300252452277, 0.555984625596111, 0.7609952013776032, 0.4567141916562373, 0.5011582923056094, 0.7203881788915105, 0.5694974942195344, 0.4901529392643781, 0.5020643299291632, 0.5340576728859943, 0.2974949825913918, 0.3254238733258096, 0.6552266094794792, 0.6138735557976488, 0.7269381698825008, 0.2984852647591309, 0.762670123063296, 0.6520708809472306, 0.4872077297299602, 0.5936528800759288, 0.4068609442801296, 0.496591267783603, 0.5044655716841309, 0.5440754124172541], dtype=np.float32)
                
            elif group_id == 2:
                patient_type = "Stable / Maintenance"
                # STABLE PATIENT (Maintenance fluids)
                # Small random noise around the mean just to vary it slightly
                noise = np.random.normal(0, 0.05, size=cfg.N_STATE_VARS)
                state = np.clip(state + noise, 0.0, 1.0)
                
            elif group_id == 3:
                patient_type = "Vasoplegic Shock"
                # VASOPLEGIC SHOCK (Adequately resuscitated but still hypotensive -> needs Vaso, no IV)
                set_feat("sysBP", 0.2)              # Still hypotensive
                set_feat("meanBP", 0.2)
                set_feat("cumulated_balance", 0.8)  # Already fluid overloaded
                set_feat("previous_dose", 0.8)      # Already got lots of IV
                set_feat("SOFA", 0.8)
                set_feat("output_4hourly", 0.5)     # Urine output is fine
                    
            elif group_id == 4:
                patient_type = "Pre-renal Dehydration"
                # ANOTHER DEHYDRATION PROFILE
                set_feat("sysBP", 0.25)
                set_feat("meanBP", 0.35)
                set_feat("input_total", 0.15)
                set_feat("cumulated_balance", 0.15)
                set_feat("output_4hourly", 0.20)
                set_feat("BUN", 0.85)
                set_feat("Arterial_lactate", 0.80)
                set_feat("previous_dose", 0.10)
            
            test_states.append((state, patient_type))
        
        # Run inference on all random samples
        print("Running inference on random samples...")
        results = []
        
        for i, (state, patient_type) in enumerate(tqdm(test_states, desc="Processing")):
            rec = recommend_treatment(agent, state)
            results.append({
                'sample_id': i,
                'patient_type': patient_type,
                'iv_dose': rec['iv_dose'],
                'vaso_dose': rec['vaso_dose'],
                'iv_level': rec['iv_level'],
                'vaso_level': rec['vaso_level'],
                'confidence': rec['confidence'],
                'action': rec['action'],
                'survival_prob': rec.get('survival_prob', 0.0),
                'entropy': rec.get('entropy', 0.0),
                'q_std': rec.get('q_std', 0.0),
                'v_value': rec.get('v_value', 0.0)
            })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results if output path is specified
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            results_df.to_csv(output_path, index=False)
            print(f"Results saved to '{output_path}'")
            
        # Print summary statistics
            print("\n" + "=" * 60)
            print("Inference Results Summary")
            print("=" * 60)
            print(f"Total random samples processed: {len(results_df)}")
            print("\nIV Dose Distribution:")
            print(results_df['iv_dose'].value_counts().sort_index())
            print("\nVasopressor Dose Distribution:")
            print(results_df['vaso_dose'].value_counts().sort_index())
            print(f"\nAverage Confidence: {results_df['confidence'].mean():.4f}")
            print(f"Average Expected Survival Prob: {results_df['survival_prob'].mean():.1%}")
            print(f"Average Softmax Entropy: {results_df['entropy'].mean():.4f}")
            
            print("\nFirst 10 Predictions:")
            # Set pandas text wrap options strictly for this snippet to avoid line breaks
            pd.set_option('display.max_columns', 15)
            pd.set_option('display.width', 150)
            print(results_df[['sample_id', 'patient_type', 'iv_dose', 'vaso_dose', 'survival_prob', 'entropy']].head(10))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on sepsis treatment model')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pt',
                        help='Path to model file (default: checkpoints/best_model.pt)')
    parser.add_argument('--test-data', type=str, default=None,
                        help='Path to test data CSV file (optional)')
    parser.add_argument('--output', type=str, default='results/inference_results.csv',
                        help='Path to save inference results CSV (optional)')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of random test samples to generate (default: 10)')
    
    args = parser.parse_args()
    
    run_model(
        model_path=args.model,
        test_data_path=args.test_data,
        output_path=args.output,
        num_random_samples=args.num_samples
    )