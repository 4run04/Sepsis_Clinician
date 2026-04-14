import torch
import torch.nn as nn
from config import cfg

# =============================================================================
# 5.  NETWORK — Dueling DQN
# =============================================================================

class DuelingDQN(nn.Module):
    def __init__(self, state_dim=cfg.N_STATE_VARS,
                 n_actions=cfg.N_ACTIONS,
                 hidden_dim=cfg.HIDDEN_DIM):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.LayerNorm(hidden_dim),          # ← stabilises NaN-prone training
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )
        # Weight initialisation — crucial for avoiding NaN at step 0
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=0.5)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        V    = self.value_stream(feat)
        A    = self.advantage_stream(feat)
        return V + A - A.mean(dim=1, keepdim=True)

