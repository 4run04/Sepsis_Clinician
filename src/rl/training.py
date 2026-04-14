import os
import numpy as np
import pandas as pd
from config import cfg, SEED
from rl.buffer import ReplayBuffer
from rl.agent import HighlightDDDQNAgent
from eval.evaluation import evaluate_survival_rate, _baseline_survival_rate

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
# 10.  MULTI-SESSION TRAINING + MODEL SELECTION
# =============================================================================

def run_sessions(train_eps, test_eps, device="cpu"):
    buffer = ReplayBuffer()
    buffer.populate_from_episodes(train_eps)

    baseline = _baseline_survival_rate(train_eps)
    print(f"\n  Physician baseline survival rate: {baseline:.2%}")

    best_agent, best_sr, all_sr = None, -1.0, []

    print(f"\nStarting {cfg.N_TRAINING_SESSIONS} training sessions …\n")
    for s in range(1, cfg.N_TRAINING_SESSIONS + 1):
        agent = train_one_session(buffer, session_id=s, device=device)
        sr    = evaluate_survival_rate(agent, test_eps)
        all_sr.append(sr)

        if sr > best_sr:
            best_sr, best_agent = sr, agent
            path = os.path.join(cfg.CHECKPOINT_DIR, "best_model.pt")
            agent.save(path)
            print(f"  ★ New best  session={s:03d}  SR={sr:.4f}  saved → {path}")

    all_sr = np.array(all_sr)
    print(f"\n{'─'*55}")
    print(f"All {cfg.N_TRAINING_SESSIONS} sessions summary:")
    print(f"  Median SR : {np.median(all_sr):.4f}")
    print(f"  Mean   SR : {all_sr.mean():.4f} ± {all_sr.std():.4f}")
    print(f"  Best   SR : {all_sr.max():.4f}")
    print(f"  Worst  SR : {all_sr.min():.4f}")
    print(f"  Baseline  : {baseline:.4f}")
    print(f"  Δ (best)  : +{max(all_sr.max() - baseline, 0):.4f}")
    print(f"{'─'*55}\n")

    pd.DataFrame({"session": range(1, cfg.N_TRAINING_SESSIONS + 1),
                  "survival_rate": all_sr}).to_csv(
        os.path.join(cfg.RESULTS_DIR, "session_results.csv"), index=False)

    return best_agent
