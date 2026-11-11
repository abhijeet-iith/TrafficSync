"""Networks: shared per-agent Q-network and the QMIX monotonic mixer."""
from __future__ import annotations

import torch
from torch import nn

__all__ = ["QNet", "QMixer"]


class QNet(nn.Module):
    """``Q_i(o_i, .)``: MLP with two hidden layers (shared by all agents; identity is in o_i)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QMixer(nn.Module):
    """``Q_tot = f_mix(Q_1..Q_N; s)`` with ``dQ_tot / dQ_i >= 0`` (hypernetwork weights through abs)."""

    def __init__(self, n_agents: int, state_dim: int, embed: int = 32) -> None:
        super().__init__()
        self.n, self.e = n_agents, embed
        self.hyper_w1 = nn.Sequential(nn.Linear(state_dim, embed), nn.ReLU(), nn.Linear(embed, n_agents * embed))
        self.hyper_b1 = nn.Linear(state_dim, embed)
        self.hyper_w2 = nn.Sequential(nn.Linear(state_dim, embed), nn.ReLU(), nn.Linear(embed, embed))
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, embed), nn.ReLU(), nn.Linear(embed, 1))

    def forward(self, qs: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        B = qs.shape[0]
        w1 = torch.abs(self.hyper_w1(s)).view(B, self.n, self.e)
        h = torch.relu(torch.bmm(qs.view(B, 1, self.n), w1) + self.hyper_b1(s).view(B, 1, self.e))
        w2 = torch.abs(self.hyper_w2(s)).view(B, self.e, 1)
        return (torch.bmm(h, w2) + self.hyper_b2(s).view(B, 1, 1)).view(B)
