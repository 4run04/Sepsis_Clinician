import random
from collections import deque, namedtuple
from typing import List
from config import cfg

# =============================================================================
# 4.  REPLAY BUFFER
# =============================================================================

Transition = namedtuple("Transition",
                        ("state", "action", "reward", "next_state", "done"))


class ReplayBuffer:
    def __init__(self, capacity: int = cfg.REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def populate_from_episodes(self, episodes: List[List[dict]]):
        for ep in episodes:
            for step in ep:
                self.push(step["state"], step["action"],
                          step["reward"], step["next_state"], step["done"])
        print(f"  Replay buffer: {len(self.buffer):,} transitions loaded.")

    def __len__(self):
        return len(self.buffer)

