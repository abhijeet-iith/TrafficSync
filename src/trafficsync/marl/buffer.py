"""Joint replay buffer (one entry = one decision step of all agents)."""
from __future__ import annotations

import numpy as np

__all__ = ["ReplayBuffer"]


class ReplayBuffer:
    def __init__(self, capacity: int, n_agents: int, obs_dim: int, seed: int = 0) -> None:
        self.cap, self.n, self.d = int(capacity), n_agents, obs_dim
        self.o = np.zeros((capacity, n_agents, obs_dim), np.float32)
        self.o2 = np.zeros_like(self.o)
        self.a = np.zeros((capacity, n_agents), np.int64)
        self.r = np.zeros((capacity, n_agents), np.float32)       # local rewards
        self.R = np.zeros(capacity, np.float32)                   # team reward
        self.done = np.zeros(capacity, np.float32)
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, o, a, r, R, o2, done) -> None:
        i = self.ptr
        self.o[i], self.a[i], self.r[i], self.R[i], self.o2[i], self.done[i] = o, a, r, R, o2, float(done)
        self.ptr = (i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch: int):
        idx = self.rng.integers(0, self.size, size=batch)
        return self.o[idx], self.a[idx], self.r[idx], self.R[idx], self.o2[idx], self.done[idx]
