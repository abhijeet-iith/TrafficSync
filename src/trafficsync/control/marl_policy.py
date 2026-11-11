"""Observation builder shared by the MARL environment and the learned-policy controller, and the ``marl`` controller that executes a trained policy.

Observation of agent i (fixed length, padded by approach N, E, S, W):

* local (every variant):  ``q_hat_l / C_l`` and ``n_hat_l / C_l`` of the 4 incoming approaches,
  one-hot phase (2), green flag, ``elapsed / g_max``                         -> 12
* communication (``comm: true``): occupancy ``rho_hat_m`` of the 4 outgoing links as reported by
  their owners (0 for exits / missing) and a "fresh report" flag per direction  -> +8
* agent identity one-hot (parameter sharing over heterogeneous positions)     -> +N

Neighbour information arrives only through messages (CoMP's data payloads over the bus), so
the learned agents obey the same information model as every other controller.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..comm.messages import Message
from ..sim.observation import GREEN, Observation
from .base import AgentView, Controller
from .comp import CoMPCore, NeighbourBoard

__all__ = ["ObsBuilder", "obs_dim", "MARLController"]

APPROACHES = ("N", "E", "S", "W")
HEADINGS = ("N", "E", "S", "W")


def obs_dim(n_agents: int, comm: bool) -> int:
    return 12 + (8 if comm else 0) + n_agents


class ObsBuilder:
    """Builds the observation vector of one agent and the messages it shares."""

    def __init__(self, view: AgentView, comm: bool, max_age_s: float = 15.0, period: int = 5) -> None:
        self.view = view
        self.comm = bool(comm)
        self.max_age = float(max_age_s)
        self.period = int(period)
        self.in_by_ap = {view.links[l].approach: l for l in view.incoming}
        self.out_by_h = {view.links[m].heading: m for m in view.outgoing}
        self.board = NeighbourBoard()
        self.core = CoMPCore(view, {"comm_period_s": period}) if comm else None
        self.N = view.n_agents

    def reset(self) -> None:
        self.board = NeighbourBoard()
        if self.core is not None:
            self.core.reset()

    def receive(self, inbox: Sequence[Message]) -> None:
        if self.comm:
            self.board.receive(inbox)

    def vector(self, obs: Observation, t: int) -> np.ndarray:
        v = self.view
        x = np.zeros(obs_dim(self.N, self.comm), dtype=np.float32)
        for k, ap in enumerate(APPROACHES):
            l = self.in_by_ap.get(ap)
            if l is not None:
                e = obs.incoming[l]
                C = float(v.links[l].capacity)
                x[2 * k] = min(1.5, float(e.q_hat) / C)
                x[2 * k + 1] = min(1.5, float(e.n_hat) / C)
        x[8 + int(obs.phase) % 2] = 1.0
        x[10] = 1.0 if obs.signal_state == GREEN else 0.0
        x[11] = min(1.0, float(obs.elapsed) / v.timing.g_max)
        off = 12
        if self.comm:
            for k, h in enumerate(HEADINGS):
                m = self.out_by_h.get(h)
                j = v.downstream_node.get(m) if m is not None else None
                if j is None:
                    continue
                inf = self.board.info.get(j)
                if inf is not None and m in inf.occupancy and t - inf.t_data <= self.max_age:
                    x[off + 2 * k] = min(1.5, float(inf.occupancy[m]))
                    x[off + 2 * k + 1] = 1.0
            off += 8
        x[off + v.agent_id] = 1.0
        return x

    def outbox(self, obs: Observation, t: int) -> list[Message]:
        if not self.comm or t % self.period != 0:
            return []
        nb = set(self.view.neighbors)
        out = []
        for j, p in sorted(self.core.data_payloads(obs, t).items()):
            if j in nb and p.get("queues"):
                out.append(Message(sender=self.view.agent_id, receiver=j, t_sent=t, phase=p["phase"], queues=p["queues"],
                                   occupancy=p["occupancy"], arrival_rate=p["arrival_rate"]))
        return out


_POLICY_CACHE: dict[str, Any] = {}


def _load_policy(path: str):
    if path not in _POLICY_CACHE:
        import torch

        from ..marl.nets import QNet

        ck = torch.load(path, map_location="cpu", weights_only=False)
        net = QNet(ck["obs_dim"], ck["n_actions"], ck["hidden"])
        net.load_state_dict(ck["q"])
        net.eval()
        _POLICY_CACHE[path] = (net, ck)
    return _POLICY_CACHE[path]


class MARLController(Controller):
    """Executes a trained shared Q-network greedily.  ``cfg``: ``policy`` (path to a .pt
    checkpoint from ``marl/train.py``).  Decisions every 5 s; min/max green, yellow and
    all-red are enforced by the signal state machine exactly as during training."""

    name = "marl"

    def reset(self) -> None:
        super().reset()
        path = self.cfg.get("policy")
        if not path:
            raise ValueError("marl controller needs cfg['policy'] (checkpoint path)")
        self.net, ck = _load_policy(str(path))
        self.builder = ObsBuilder(self.view, bool(ck["comm"]), period=int(self.cfg.get("comm_period_s", 5)))

    def receive(self, inbox, t: int) -> None:
        self.builder.receive(inbox)

    def decide(self, obs: Observation, t: int) -> int:
        import torch

        x = torch.as_tensor(self.builder.vector(obs, t))[None]
        with torch.no_grad():
            return int(self.net(x).argmax(-1).item())

    def outbox(self, obs: Observation, t: int) -> list[Message]:
        return self.builder.outbox(obs, t)
