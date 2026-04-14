import random
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from config import cfg, SEED
from rl.buffer import ReplayBuffer
from rl.network import DuelingDQN

# =============================================================================
# 6.  HIGHLIGHT-DDDQN AGENT
# =============================================================================

class HighlightDDDQNAgent:
    def __init__(self, device="cpu", seed=SEED):
        self.device = torch.device(device)
        torch.manual_seed(seed)

        self.online_net = DuelingDQN().to(self.device)
        self.target_net = DuelingDQN().to(self.device)
        self._sync_target()

        self.optimizer  = optim.Adam(self.online_net.parameters(),
                                     lr=cfg.LR, eps=1e-5)  # eps guards NaN
        self.h          = cfg.H_INIT
        self.epsilon    = cfg.EPSILON_START
        self.step_count = 0

    def _sync_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def _to_tensor(self, x) -> torch.Tensor:
        arr = np.array(x, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.5, posinf=1.0, neginf=0.0)  # NaN guard
        return torch.tensor(arr).to(self.device)

    @torch.no_grad()
    def select_action(self, state: np.ndarray, greedy: bool = True) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randrange(cfg.N_ACTIONS)
        q = self.online_net(self._to_tensor(state).unsqueeze(0))
        if torch.isnan(q).any():
            return random.randrange(cfg.N_ACTIONS)  # safe fallback
        return int(q.argmax(dim=1).item())

    def learn(self, buffer: ReplayBuffer) -> Tuple[float, bool]:
        if len(buffer) < cfg.BATCH_SIZE:
            return 0.0, False

        batch       = buffer.sample(cfg.BATCH_SIZE)
        states      = self._to_tensor(batch.state)
        actions     = self._to_tensor(batch.action).long()
        rewards     = self._to_tensor(batch.reward)
        next_states = self._to_tensor(batch.next_state)
        dones       = self._to_tensor(batch.done)

        # Double-DQN: online selects, target evaluates
        with torch.no_grad():
            best_next  = self.online_net(next_states).argmax(dim=1)
            q_next     = self.target_net(next_states).gather(
                             1, best_next.unsqueeze(1)).squeeze(1)
            q_next     = q_next * (1.0 - dones)

        # Highlight-DDDQN loss (Eq. 2 from paper)
        target_q  = rewards + (cfg.GAMMA / self.h) * q_next
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        estimate  = current_q / self.h

        # ── NaN guard ─────────────────────────────────────────────────────────
        if torch.isnan(estimate).any() or torch.isnan(target_q).any():
            return float("nan"), False   # skip update, don't crash

        loss = F.mse_loss(estimate, target_q.detach())

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping — essential for offline RL stability
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Over/under-estimation check
        overestimated = bool((estimate.abs() > 1.0).any().item())

        self.epsilon = max(cfg.EPSILON_END, self.epsilon * cfg.EPSILON_DECAY)
        self.step_count += 1

        steps_per_iter = max(1, len(buffer) // cfg.BATCH_SIZE)
        if self.step_count % steps_per_iter == 0:
            self._sync_target()

        return loss.item(), overestimated

    def save(self, path: str):
        torch.save({
            "online": self.online_net.state_dict(),
            "target": self.target_net.state_dict(),
            "h": self.h, "epsilon": self.epsilon,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online"])
        self.target_net.load_state_dict(ckpt["target"])
        self.h, self.epsilon = ckpt["h"], ckpt["epsilon"]

