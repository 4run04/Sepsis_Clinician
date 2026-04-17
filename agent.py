import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
from typing import List, Tuple
from config import cfg
from model import DuelingDQN

# Transition definition
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class ReplayBuffer:
    """Experience replay memory to break temporal correlations."""
    def __init__(self, capacity: int = cfg.REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def populate_from_episodes(self, episodes: List[List[dict]]):
        """Unpacks generated episodes into transitions for the buffer."""
        for ep in episodes:
            for step in ep:
                self.push(step["state"], step["action"],
                          step["reward"], step["next_state"], step["done"])

    def __len__(self):
        return len(self.buffer)


class HighlightDDDQNAgent:
    """Dueling Double-DQN agent with Highlight-RL tuning for offline sepsis dosing."""
    def __init__(self, device="cpu", seed=42):
        self.device = torch.device(device)
        self.online_net = DuelingDQN().to(self.device)
        self.target_net = DuelingDQN().to(self.device)
        self._sync_target()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=cfg.LR, eps=1e-5)
        self.h = cfg.H_INIT
        self.epsilon = cfg.EPSILON_START
        self.step_count = 0

    def _sync_target(self):
        """Hard update target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def _to_tensor(self, x) -> torch.Tensor:
        arr = np.array(x, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.5)
        return torch.tensor(arr).to(self.device)

    @torch.no_grad()
    def select_action(self, state: np.ndarray, greedy: bool = True) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randrange(cfg.N_ACTIONS)
        q = self.online_net(self._to_tensor(state).unsqueeze(0))
        return int(q.argmax(dim=1).item())

    def learn(self, buffer: ReplayBuffer) -> Tuple[float, bool]:
        """Performs a single gradient descent step."""
        if len(buffer) < cfg.BATCH_SIZE: return 0.0, False

        batch = buffer.sample(cfg.BATCH_SIZE)
        states = self._to_tensor(batch.state)
        actions = self._to_tensor(batch.action).long()
        rewards = self._to_tensor(batch.reward)
        next_states = self._to_tensor(batch.next_state)
        dones = self._to_tensor(batch.done)

        # Double-DQN logic
        with torch.no_grad():
            best_next = self.online_net(next_states).argmax(dim=1)
            q_next = self.target_net(next_states).gather(1, best_next.unsqueeze(1)).squeeze(1)
            q_next = q_next * (1.0 - dones)

        # Target calculation (Highlight modification)
        target_q = rewards + (cfg.GAMMA / self.h) * q_next
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        estimate = current_q / self.h

        loss = F.mse_loss(estimate, target_q.detach())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        # Update stats
        overestimated = bool((estimate.abs() > 1.0).any().item())
        self.epsilon = max(cfg.EPSILON_END, self.epsilon * cfg.EPSILON_DECAY)
        self.step_count += 1
        
        if self.step_count % 100 == 0: self._sync_target()
        return loss.item(), overestimated

    def save(self, path: str):
        torch.save({"online": self.online_net.state_dict(), "h": self.h, "epsilon": self.epsilon}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online"])
        self.target_net.load_state_dict(ckpt["online"])
        self.h, self.epsilon = ckpt.get("h", 1.0), ckpt.get("epsilon", 0.0)


def train_one_session(buffer: ReplayBuffer, session_id: int, device: str = "cpu") -> HighlightDDDQNAgent:
    """Core training loop for a single model session."""
    agent = HighlightDDDQNAgent(device=device)
    steps_per_iter = max(1, len(buffer) // cfg.BATCH_SIZE)

    for iteration in range(cfg.N_ITERATIONS):
        total_loss, n_valid = 0.0, 0
        for _ in range(steps_per_iter):
            loss, overest = agent.learn(buffer)
            if not np.isnan(loss):
                total_loss += loss
                n_valid += 1
            if overest and agent.h < cfg.H_MAX:
                agent.h *= cfg.H_STEP
        
        if iteration % 10 == 0:
            print(f"  [Session {session_id:02d}] Iter {iteration:03d}: Loss={total_loss/max(1, n_valid):.5f}, Epsilon={agent.epsilon:.4f}")
    
    return agent
