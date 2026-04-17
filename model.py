import torch
import torch.nn as nn
from typing import Tuple
from config import cfg

class DuelingDQN(nn.Module):
    """
    Dueling Q-Network Architecture.
    Splits the Q-value estimation into two streams:
    1. State Value (V) - How likely is survival purely based on patient status?
    2. Action Advantage (A) - How much does a specific dose change that survival chance?
    """
    def __init__(self, state_dim=cfg.N_STATE_VARS, 
                 n_actions=cfg.N_ACTIONS, 
                 hidden_dim=cfg.HIDDEN_DIM):
        super().__init__()
        
        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        
        # Value Stream (Estimates V(s))
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Advantage Stream (Estimates A(s, a))
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )
        
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        """Orthogonal initialization helps prevent gradient explosion/NaNs in early RL iterations."""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=0.5)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        V = self.value_stream(feat)
        A = self.advantage_stream(feat)
        # Combine using the aggregating function: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        return V + A - A.mean(dim=1, keepdim=True)


def encode_action(iv_level: int, vaso_level: int) -> int:
    """Maps a 2D dose level pair to a single scalar action ID."""
    return iv_level * cfg.VASO_LEVELS + vaso_level


def decode_action(action: int) -> Tuple[int, int]:
    """Maps a scalar action ID back to (IV_level, Vaso_level)."""
    return divmod(action, cfg.VASO_LEVELS)
