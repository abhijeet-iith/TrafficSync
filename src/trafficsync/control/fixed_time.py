"""C0 - fixed-time Webster plan.

``C0 = (1.5 L_tot + 5) / (1 - Y)``, ``Y = sum_p max_{l in p} lambda_l / s_l``,
``g_p = y_p / Y (C0 - L_tot)`` (effective green), from *design* flows (the scenario's average
entry rates propagated with the turning ratios), never from run-time data.  Lost time per
phase = startup 2 + yellow 3 + all-red 1 = 6 s.  ``C0`` is clipped to [30, 120] s and every
displayed green ``G_p = g_p + startup`` to [g_min, g_max].  Offsets 0.  The plan is a timer
evaluated every second (decision interval 1 s).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..config import SignalTiming
from ..network.graph import Network
from ..sim.observation import Observation
from .base import AgentView, Controller

__all__ = ["WebsterPlan", "webster_plan", "expected_link_flows", "FixedTimeController", "plan_phase"]


@dataclass(frozen=True)
class WebsterPlan:
    cycle: float
    greens: tuple[float, ...]       # displayed greens G_p [s]
    Y: float
    y: tuple[float, ...]
    cycle_webster: float            # unclipped (1.5 L + 5) / (1 - Y)


def expected_link_flows(net: Network, entry_rates: Mapping[int, float], turning: Mapping[str, float],
                        iters: int = 200) -> dict[int, float]:
    """Average flow on every link: entry rates propagated with renormalised turning ratios."""
    ratios: dict[int, list[tuple[int, float]]] = {}
    for i in range(net.N):
        for l in net.incoming_links(i):
            movs = [m for m in net.movements[i] if m.in_link == l]
            tot = sum(float(turning.get(m.turn, 0.0)) for m in movs)
            ratios[l] = [(m.out_link, float(turning.get(m.turn, 0.0)) / tot) for m in movs] if tot > 0 else []
    f = {l: 0.0 for l in net.links}
    for _ in range(iters):
        new = {l: 0.0 for l in net.links}
        for e, r in entry_rates.items():
            new[e] += float(r)
        for l, outs in ratios.items():
            for m, r in outs:
                new[m] += f[l] * r
        if max(abs(new[l] - f[l]) for l in f) < 1e-10:
            f = new
            break
        f = new
    return f


def webster_plan(view: AgentView, flows: Mapping[int, float], timing: SignalTiming | None = None,
                 c_min: float = 30.0, c_max: float = 120.0) -> WebsterPlan:
    tm = timing or view.timing
    P = view.num_phases
    L_tot = P * tm.lost_per_phase
    y = []
    for p in range(P):
        links = view.phase_link_shares(p)
        y.append(max((float(flows.get(l, 0.0)) / view.links[l].sat_flow for l in links), default=0.0))
    Y = float(sum(y))
    cw = (1.5 * L_tot + 5.0) / (1.0 - Y) if Y < 0.95 else c_max
    C = float(np.clip(cw, c_min, c_max))
    g_eff = [(yp / Y if Y > 0 else 1.0 / P) * (C - L_tot) for yp in y]
    G = [float(np.clip(g + tm.startup_lost, tm.g_min, tm.g_max)) for g in g_eff]
    C = sum(G) + P * (tm.yellow + tm.allred)
    return WebsterPlan(C, tuple(G), Y, tuple(y), float(cw))


def plan_phase(t_in_cycle: float, greens, transition: float) -> int:
    """Phase requested at a position of the cycle (window of phase p = G_p + transition)."""
    s = 0.0
    for p, G in enumerate(greens):
        s += G + transition
        if t_in_cycle < s:
            return p
    return len(greens) - 1


class FixedTimeController(Controller):
    """``cfg``: ``design_flows`` (set by the runner), ``c_min``, ``c_max``, ``offset``."""

    name = "fixed"

    def __init__(self, agent_id: int, view: AgentView, cfg: Mapping[str, Any] | None = None) -> None:
        cfg = dict(cfg or {})
        cfg["decision_interval_s"] = 1
        super().__init__(agent_id, view, cfg)

    def reset(self) -> None:
        super().reset()
        flows = self.cfg.get("design_flows") or {}
        self.plan = webster_plan(self.view, flows, self.view.timing, float(self.cfg.get("c_min", 30)),
                                 float(self.cfg.get("c_max", 120)))
        self.offset = float(self.cfg.get("offset", 0.0))

    def decide(self, obs: Observation, t: int) -> int:
        tm = self.view.timing
        c = (t - self.offset) % self.plan.cycle
        return plan_phase(c, self.plan.greens, tm.yellow + tm.allred)
