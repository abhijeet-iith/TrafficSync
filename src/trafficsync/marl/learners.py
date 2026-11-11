"""IQL, VDN and QMIX on a shared Q-network.

All use a target network (hard update every ``target_update`` gradient steps), double-Q
targets, Huber loss and gradient clipping at 10.

* **IQL** (independent Q-learning, parameter sharing): each agent's transition with its
  *local* reward is an independent sample - other agents are part of a non-stationary
  environment.  ``comm`` only changes the observation (``iql+comm``).
* **VDN**: ``Q_tot = sum_i Q_i(o_i, a_i)`` trained on the *team* reward (CTDE).
* **QMIX**: ``Q_tot = f_mix(Q_1..Q_N; s)``, monotonic in every ``Q_i``, state = all observations.
Execution is decentralised: each agent takes ``argmax_a Q_i(o_i, a)``.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from .nets import QMixer, QNet

__all__ = ["Learner", "make_learner"]


class Learner:
    def __init__(self, algo: str, n_agents: int, obs_dim: int, n_actions: int = 2, hidden: int = 64,
                 lr: float | None = None, gamma: float = 0.99, target_update: int = 200, seed: int = 0) -> None:
        torch.manual_seed(seed)
        self.algo, self.N, self.obs_dim, self.nA, self.hidden = algo, n_agents, obs_dim, n_actions, hidden
        self.gamma, self.target_update = gamma, target_update
        self.q = QNet(obs_dim, n_actions, hidden)
        self.q_t = copy.deepcopy(self.q)
        params = list(self.q.parameters())
        self.mixer = self.mixer_t = None
        if algo == "qmix":
            self.mixer = QMixer(n_agents, n_agents * obs_dim)
            self.mixer_t = copy.deepcopy(self.mixer)
            params += list(self.mixer.parameters())
        lr = lr if lr is not None else (1e-3 if algo == "iql" else 5e-4)
        self.opt = torch.optim.Adam(params, lr=lr)
        self.steps = 0

    def act(self, obs: np.ndarray, eps: float, rng: np.random.Generator) -> np.ndarray:
        with torch.no_grad():
            a = self.q(torch.as_tensor(obs)).argmax(-1).numpy()
        explore = rng.random(self.N) < eps
        a[explore] = rng.integers(0, self.nA, size=int(explore.sum()))
        return a

    def update(self, batch) -> float:
        o, a, r, R, o2, d = (torch.as_tensor(x) for x in batch)
        B = o.shape[0]
        q_all = self.q(o)                                             # (B, N, A)
        q_sa = q_all.gather(-1, a.unsqueeze(-1)).squeeze(-1)          # (B, N)
        with torch.no_grad():
            a2 = self.q(o2).argmax(-1, keepdim=True)                   # double-Q: online argmax
            q2 = self.q_t(o2).gather(-1, a2).squeeze(-1)               # (B, N)
        if self.algo == "iql":
            y = r + self.gamma * (1 - d).unsqueeze(-1) * q2
            loss = nn.functional.smooth_l1_loss(q_sa, y)
        elif self.algo == "vdn":
            y = R + self.gamma * (1 - d) * q2.sum(-1)
            loss = nn.functional.smooth_l1_loss(q_sa.sum(-1), y)
        else:
            s, s2 = o.reshape(B, -1), o2.reshape(B, -1)
            with torch.no_grad():
                y = R + self.gamma * (1 - d) * self.mixer_t(q2, s2)
            loss = nn.functional.smooth_l1_loss(self.mixer(q_sa, s), y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_([p for g in self.opt.param_groups for p in g["params"]], 10.0)
        self.opt.step()
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.q_t.load_state_dict(self.q.state_dict())
            if self.mixer is not None:
                self.mixer_t.load_state_dict(self.mixer.state_dict())
        return float(loss.item())

    def checkpoint(self, comm: bool, extra: dict | None = None) -> dict:
        return {"algo": self.algo, "comm": comm, "obs_dim": self.obs_dim, "n_actions": self.nA, "hidden": self.hidden,
                "q": copy.deepcopy(self.q.state_dict()), **(extra or {})}


def make_learner(algo: str, n_agents: int, obs_dim: int, **kw) -> Learner:
    if algo not in ("iql", "vdn", "qmix"):
        raise ValueError(f"unknown algorithm {algo!r}")
    return Learner(algo, n_agents, obs_dim, **kw)
