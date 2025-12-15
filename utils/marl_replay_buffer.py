import random
from collections import deque
import numpy as np


class MultiAgentReplayBuffer:

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        obs_all: np.ndarray,
        acts_all: np.ndarray,
        rews_all: np.ndarray,
        next_obs_all: np.ndarray,
        dones_all: np.ndarray,
    ):
        self.buffer.append(
            (
                obs_all.astype(np.float32),
                acts_all.astype(np.float32),
                rews_all.astype(np.float32),
                next_obs_all.astype(np.float32),
                dones_all.astype(np.float32),
            )
        )

    def sample(self, batch_size: int = 64):
        batch = random.sample(self.buffer, batch_size)
        obs_all, acts_all, rews_all, next_obs_all, dones_all = zip(*batch)
        return (
            np.stack(obs_all, axis=0),  # (B, n_agents, obs_dim)
            np.stack(acts_all, axis=0),  # (B, n_agents, act_dim)
            np.stack(rews_all, axis=0),  # (B, n_agents)
            np.stack(next_obs_all, axis=0),  # (B, n_agents, obs_dim)
            np.stack(dones_all, axis=0),  # (B, n_agents)
        )

    def __len__(self):
        return len(self.buffer)
