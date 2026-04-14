from typing import List
import numpy as np
from rl.agent import HighlightDDDQNAgent

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

