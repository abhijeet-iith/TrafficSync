"""C2b - consensus cycle + offset coordination (CoCycle) and CoMP+Cycle.

Per agent, every second:

1. **Local design data.**  Per-link flows = bias-corrected EMA of ``lambda_hat`` (time
   constant ``split_window_s`` = 300 s); ``y_p = max_{l in p} f_l / s_l``, ``Y_i = sum_p y_p``,
   local Webster cycle ``C_i* = (1.5 L + 5) / (1 - Y_i)``.
2. **Common cycle by consensus** (``cycle.rule``):
   * ``webster`` (default): agree on ``z = avg(w_i C_i*) / avg(w_i)`` with ``w_i = Y_i`` via two
     consensus channels ``cw`` and ``w`` (the agreement problem solved in ``control.admm``);
   * ``occupancy`` (occupancy formula): ``C = C_min + (C_max - C_min) clip(rho_bar / rho_ref, 0, 1)``
     with ``rho_bar`` from consensus on the load ``rho_i`` (channel ``rho``).
   ``C_net`` moves in 4 s steps with a 3 s hysteresis band and is capped at
   ``2 (g_max + 4)`` so that both greens fit inside ``g_max``.
3. **Local splits** (Webster): ``G_p`` proportional to ``y_p``, inside ``[g_min, g_max]``.
4. **Offsets** (formation consensus on green starts, :mod:`.offset_consensus`): along a road
   edge i -> j served by axis a, j's green of axis a should start ``tau`` after i's; both
   directions are flow-weighted (weights add up to 1); differences wrapped to ``(-C/2, C/2]``;
   agent 0 is the pinned leader; a correction moves the plan by at most 2 s per round.
5. **Actuation** inside ``[g_min, g_max]``: gap-out (served approaches empty while the other
   phase waits) and extension (up to ``max_extension_s`` while the served approaches still
   have a queue).  ``cocycle`` judges with its own queues, ``comp+cycle`` with the full CoMP
   pressure for extension; ``actuation: none`` = plan only.

``consensus.iterations_per_decision = 0`` switches consensus and offsets off (local plan).
Every consensus channel goes through :func:`control.wmsr.make_consensus`, so
``consensus.aggregation: wmsr`` makes CoCycle resilient (E9).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..comm.messages import Message
from ..sim.observation import GREEN, Observation
from .base import AgentView, Controller, axis_phases, local_demand, local_load
from .comp import CoMPCore, NeighbourBoard, merge_payloads
from .fixed_time import plan_phase
from .offset_consensus import OffsetAgent, wrap
from .wmsr import make_consensus

__all__ = ["CoCycleController", "CoMPCycleController", "cycle_from_load", "webster_split"]


def _clip(x, lo, hi):
    """Scalar ``np.clip`` without the numpy call overhead (hot path)."""
    return min(max(float(x), lo), hi)


def cycle_from_load(rho_bar: float, c_min: float = 40.0, c_max: float = 100.0, rho_ref: float = 0.6) -> float:
    """Occupancy-based cycle rule."""
    return _clip(c_min + (c_max - c_min) * np.clip(rho_bar / rho_ref, 0.0, 1.0), c_min, c_max)


def webster_split(y: Sequence[float], C: float, timing) -> list[float]:
    """Displayed greens for cycle ``C`` (2+ phases), proportional to ``y_p``, inside [g_min, g_max]."""
    P = len(y)
    trans = timing.yellow + timing.allred
    avail = C - P * trans
    Y = float(sum(y))
    share = [yp / Y if Y > 0 else 1.0 / P for yp in y]
    G = [_clip(s * avail, timing.g_min, timing.g_max) for s in share]
    # put any remainder where it fits (keeps sum(G) + P*trans == C when feasible)
    for _ in range(3):
        diff = avail - sum(G)
        if abs(diff) < 1e-9:
            break
        room = [(timing.g_max - g) if diff > 0 else (g - timing.g_min) for g in G]
        tot = sum(room)
        if tot <= 1e-9:
            break
        G = [g + diff * r / tot if diff > 0 else g + diff * r / tot for g, r in zip(G, room)]
        G = [_clip(g, timing.g_min, timing.g_max) for g in G]
    return G


class _EMA:
    """Bias-corrected exponential moving average with time constant ``tau`` steps."""

    def __init__(self, tau: float) -> None:
        self.a = 1.0 / max(1.0, float(tau))
        self.m = 0.0
        self.w = 0.0

    def add(self, x: float) -> float:
        self.m = (1 - self.a) * self.m + self.a * x
        self.w = (1 - self.a) * self.w + self.a
        return self.value

    @property
    def value(self) -> float:
        return self.m / self.w if self.w > 0 else 0.0


class CoCycleController(Controller):
    name = "cocycle"
    use_comp = False

    def __init__(self, agent_id: int, view: AgentView, cfg: Mapping[str, Any] | None = None) -> None:
        cfg = dict(cfg or {})
        cfg["decision_interval_s"] = 1          # the plan is a timer; consensus runs on its own clock
        super().__init__(agent_id, view, cfg)

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        super().reset()
        cfg, view = self.cfg, self.view
        tm = view.timing
        self.period = int(cfg.get("comm_period_s", 5))
        policy = str(cfg.get("missing_policy", "last"))
        cyc = dict(cfg.get("cycle", {}) or {})
        self.rule = str(cyc.get("rule", "webster"))
        self.c_min = float(cyc.get("C_min", 40.0))
        self.c_max = min(float(cyc.get("C_max", 100.0)), 2.0 * (tm.g_max + tm.yellow + tm.allred))
        self.rho_ref = float(cyc.get("rho_ref", 0.6))
        self.quant = float(cyc.get("quantum_s", 4.0))
        self.hyst = float(cyc.get("hysteresis_s", 3.0))
        ccfg = dict(cfg.get("consensus", {}) or {})
        self.K = int(ccfg.get("iterations_per_decision", 5))
        self.channels = {ch: make_consensus(view, ccfg, self.period, policy, ch) for ch in ("rho", "cw", "w")}
        self.board = NeighbourBoard()
        self.flow = {l: _EMA(float(cfg.get("split_window_s", 300))) for l in view.incoming}
        self.load_avg = _EMA(float(cfg.get("load_window_s", 300)))
        self.wave_flow = {l: _EMA(float(cfg.get("wave_window_s", 600))) for l in view.incoming}
        self.offsets_on = bool(cfg.get("offsets", True)) and self.K > 0
        eps = 1.0 / (float(view.meta.get("d_max", max(1, len(view.neighbors)))) + 1.0)
        self.off = OffsetAgent(eps, leader=(self.agent_id == int(cfg.get("leader", 0))),
                               max_step=float(cfg.get("max_offset_step_s", 2.0)))
        self.actuation = str(cfg.get("actuation", "local"))
        self.max_ext = float(cfg.get("max_extension_s", 10.0))
        self.ext_used = 0.0
        self.axis = axis_phases(view)
        self.trans = tm.yellow + tm.allred
        self.C = float(cyc.get("C_init", 60.0))
        self.G = webster_split([1.0] * view.num_phases, self.C, tm)
        self.y = [0.0] * view.num_phases
        self.Y = 0.0
        self.c_local = self.C
        self.est = {"rho": 0.0, "cw": 0.0, "w": 0.0}
        self.inputs = {"rho": 0.0, "cw": 0.0, "w": 0.0}
        # road neighbours: out link to j and in link from j
        self.out_to = {view.downstream_node[m]: m for m in view.outgoing if view.downstream_node.get(m) is not None}
        self.in_from = {view.upstream_node[l]: l for l in view.incoming if view.upstream_node.get(l) is not None}
        self.core = CoMPCore(view, cfg) if self.use_comp else None

    # ----------------------------------------------------------- per step
    def receive(self, inbox: Sequence[Message], t: int) -> None:
        self.board.receive(inbox)

    def before_decision(self, obs: Observation, t: int) -> None:
        view, tm = self.view, self.view.timing
        if self.core is not None:
            self.core.observe(obs)
        for l, e in obs.incoming.items():
            self.flow[l].add(max(0.0, float(e.lambda_hat)))
            self.wave_flow[l].add(max(0.0, float(e.lambda_hat)))
        rho = self.load_avg.add(local_load(view, obs))
        y = []
        for p in range(view.num_phases):
            y.append(max((self.flow[l].value / view.links[l].sat_flow for l in view.phase_link_shares(p)), default=0.0))
        self.y = y
        self.Y = Y = min(float(sum(y)), 0.9)
        L = view.num_phases * tm.lost_per_phase
        self.c_local = _clip((1.5 * L + 5.0) / (1.0 - Y), self.c_min, self.c_max)
        w = max(Y, 1e-3)
        self.inputs = {"rho": rho, "cw": w * self.c_local, "w": w}
        for ch, c in self.channels.items():
            self.est[ch] = c.update(t, self.board, self.inputs[ch])
        # common cycle
        if self.K <= 0:
            target = self.c_local
        elif self.rule == "occupancy":
            target = cycle_from_load(self.est["rho"], self.c_min, self.c_max, self.rho_ref)
        else:
            target = self.est["cw"] / self.est["w"] if self.est["w"] > 1e-9 else self.c_local
        target = _clip(target, self.c_min, self.c_max)
        if abs(target - self.C) > self.hyst:
            self.C = _clip(self.quant * round(target / self.quant), self.c_min, self.c_max)
        self.G = webster_split(y, self.C, tm)
        # offsets
        if self.offsets_on:
            fresh = self.board.pop_fresh_offset()
            if fresh:
                errs = {}
                psi_i = self.green_starts()
                for j in fresh:
                    if j not in self.out_to or j not in self.in_from:
                        continue
                    inf = self.board.info[j]
                    psi_j = inf.psi or {}
                    m, l = self.out_to[j], self.in_from[j]
                    axis = "EW" if view.links[m].heading in ("E", "W") else "NS"
                    if axis not in psi_j or axis not in psi_i:
                        continue
                    f_out = float((psi_j.get("fin") or {}).get(str(m), (psi_j.get("fin") or {}).get(m, 0.0)))
                    f_in = self.wave_flow[l].value
                    tau_o, tau_i = view.links[m].tau, view.links[l].tau
                    tot = f_out + f_in
                    d = (f_out * tau_o - f_in * tau_i) / tot if tot > 1e-9 else 0.5 * (tau_o - tau_i)
                    errs[j] = (1.0, wrap((psi_j[axis] - psi_i[axis]) - d, self.C))
                if errs or self.off.leader:
                    self.off.update(errs, self.C)

    def green_starts(self) -> dict[str, float]:
        """Cycle time at which the green of each axis starts (``phi + window start + transition``)."""
        out = {}
        for axis, p in self.axis.items():
            start = sum(self.G[q] + self.trans for q in range(p))
            out[axis] = float(np.mod(self.off.phi + start + self.trans, self.C))
        return out

    def planned_phase(self, t: int) -> int:
        phi = self.off.phi if self.offsets_on else 0.0
        return plan_phase((t - phi) % self.C, self.G, self.trans)

    def _extend(self, obs: Observation, t: int, cur: int, plan: int) -> bool:
        dem = local_demand(self.view, obs)
        return dem[cur] > 0

    def decide(self, obs: Observation, t: int) -> int:
        tm = self.view.timing
        plan = self.planned_phase(t)
        if obs.signal_state != GREEN or self.actuation == "none":
            if obs.signal_state == GREEN and plan == obs.phase:
                self.ext_used = 0.0
            return plan if obs.signal_state == GREEN else (obs.next_phase if obs.next_phase is not None else plan)
        cur, e = int(obs.phase), int(obs.elapsed)
        if e < tm.g_min:
            return cur
        dem = local_demand(self.view, obs)
        if plan == cur:
            self.ext_used = 0.0
            others = [p for p in range(len(dem)) if p != cur and dem[p] > 0]
            if dem[cur] <= 0 and others:                       # gap-out
                return max(others, key=lambda p: (dem[p], -p))
            return cur
        # plan wants to switch: extend while the current approaches still queue
        if e < tm.g_max and self.ext_used < self.max_ext and self._extend(obs, t, cur, plan):
            self.ext_used += 1.0
            return cur
        return plan

    def after_action(self, obs: Observation, action: int, t: int) -> None:
        if self.core is not None:
            self.core.record_departures(obs, action, t)

    def consensus_state(self) -> dict:
        return {"C": self.C, "C_local": self.c_local, "phi": self.off.phi, **{f"x_{k}": v for k, v in self.est.items()}}

    def outbox(self, obs: Observation, t: int) -> list[Message]:
        payload: dict[int, dict[str, Any]] = {}
        nbrs = list(self.view.neighbors)
        anyc = next(iter(self.channels.values()))
        if self.K > 0 and nbrs and anyc.exchange_step(t):
            cons = {ch: float(c.x if c.x is not None else self.inputs[ch]) for ch, c in self.channels.items()}
            fin = {str(l): self.wave_flow[l].value for l in self.view.incoming}
            psi = {**self.green_starts(), "fin": fin}
            merge_payloads(payload, {j: {"load_value": cons["rho"], "cycle_offset": float(self.off.phi),
                                         "extra": {"cons": {"cw": cons["cw"], "w": cons["w"]}, "psi": psi, "C": self.C}}
                                     for j in nbrs})
        if self.core is not None and t % self.period == 0:
            nb = set(nbrs)
            merge_payloads(payload, {j: p for j, p in self.core.data_payloads(obs, t).items() if j in nb})
        return [Message(sender=self.agent_id, receiver=j, t_sent=t, **p) for j, p in sorted(payload.items())]


class CoMPCycleController(CoCycleController):
    """CoCycle whose extension decision uses the CoMP pressure (current vs planned phase)."""

    name = "comp+cycle"
    use_comp = True

    def _extend(self, obs: Observation, t: int, cur: int, plan: int) -> bool:
        P = self.core.pressures(obs, self.board, t, 1.0)
        dem = local_demand(self.view, obs)
        return dem[cur] > 0 and P[cur] > P[plan]
