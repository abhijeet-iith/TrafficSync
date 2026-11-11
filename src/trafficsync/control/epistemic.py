"""Epistemic guards on actions.

Truth vs knowledge: ``Room_m`` = "downstream link m has room for at least delta_room more
vehicles" (``n_m <= C_m - delta_room``).  Agent i knows it only if it holds in every world
compatible with its information::

    worlds W = {0..C_m},   I_i = {n : |n - n_hat_m| <= delta0 + lam_max_m * age}
    K_i Room_m  iff  I_i subseteq V(Room_m)  iff  max I_i <= C_m - delta_room
    M_i Room_m  iff  I_i intersects V(Room_m) iff  min I_i <= C_m - delta_room

``n_hat_m`` comes from the newest message of m's owner (occupancy x C); with no usable
message ``I_i = W`` (nothing known).  Uncertainty grows with message age, so knowledge
shrinks as messages age or get lost (tested).

The guard: a movement (l, m) may be served only if ``K_i Room_m`` (exit links always pass).
The *epistemic feasible set* ``U_i^K`` holds the phases whose movements all pass; if it is
empty the agent falls back to the phase with the least downstream risk (worst case occupancy
over the information set), ties broken by local demand.
"""
from __future__ import annotations

from typing import Any, Mapping

from ..logic.kripke import interval_information_set
from ..sim.observation import Observation
from .base import AgentView, local_demand
from .comp import CoMPController, NeighbourBoard

__all__ = ["EpistemicGuard", "EpistemicCoMPController"]


class EpistemicGuard:
    """Per-agent knowledge of downstream room.  ``cfg``: ``delta0`` (2 veh), ``lam_max``
    (veh/s growth of uncertainty per second of age; default = downstream saturation flow),
    ``delta_room`` (2), ``max_info_age_s`` (60)."""

    def __init__(self, view: AgentView, cfg: Mapping[str, Any] | None = None) -> None:
        cfg = dict(cfg or {})
        self.view = view
        self.delta0 = float(cfg.get("delta0", 2.0))
        self.lam = cfg.get("lam_max")
        self.delta_room = float(cfg.get("delta_room", 2.0))
        self.max_age = float(cfg.get("max_info_age_s", 60.0))
        self.down = {m: view.downstream_node[m] for m in view.outgoing}
        self.blocked = 0          # movements blocked by the guard (decisions)
        self.fallbacks = 0        # decisions where U_i^K was empty
        self.checks = 0

    def lam_max(self, m: int) -> float:
        return float(self.lam) if self.lam is not None else float(self.view.links[m].sat_flow)

    def info_set(self, board: NeighbourBoard, m: int, t: int) -> tuple[int, int]:
        """Integer interval of occupancies of ``m`` compatible with what i knows."""
        C = int(self.view.links[m].capacity)
        j = self.down.get(m)
        if j is None:
            return 0, 0                               # exit link: leaves the network, always room
        inf = board.info.get(j)
        if inf is None or m not in inf.occupancy:
            return 0, C
        age = float(t - inf.t_data)
        if age > self.max_age:
            return 0, C
        return interval_information_set(float(inf.occupancy[m]) * C, age, C, self.delta0, self.lam_max(m))

    def knows_room(self, board: NeighbourBoard, m: int, t: int) -> bool:
        if self.down.get(m) is None:
            return True
        lo, hi = self.info_set(board, m, t)
        return hi <= self.view.links[m].capacity - self.delta_room

    def possible_room(self, board: NeighbourBoard, m: int, t: int) -> bool:
        if self.down.get(m) is None:
            return True
        lo, hi = self.info_set(board, m, t)
        return lo <= self.view.links[m].capacity - self.delta_room

    def feasible_phases(self, board: NeighbourBoard, t: int) -> list[int]:
        """``U_i^K``: phases whose every movement passes the guard."""
        out = []
        for p, movs in enumerate(self.view.phases):
            ok = True
            for mv in movs:
                self.checks += 1
                if not self.knows_room(board, mv.out_link, t):
                    ok = False
                    self.blocked += 1
            if ok:
                out.append(p)
        return out

    def risk(self, board: NeighbourBoard, p: int, t: int) -> float:
        """Worst-case downstream fill of phase p: ``max_m (hi_m - (C_m - delta_room)) / C_m``."""
        r = -1.0
        for mv in self.view.phases[p]:
            m = mv.out_link
            if self.down.get(m) is None:
                continue
            C = float(self.view.links[m].capacity)
            r = max(r, (self.info_set(board, m, t)[1] - (C - self.delta_room)) / C)
        return r

    def choose(self, obs: Observation, wanted: int, board: NeighbourBoard, t: int) -> int:
        """Apply the guard to a desired phase (never interrupts min green / clearance)."""
        feas = self.feasible_phases(board, t)
        if wanted in feas:
            return wanted
        if feas:
            # keep the current phase if it is feasible, else the feasible phase with most demand
            if int(obs.phase) in feas:
                return int(obs.phase)
            dem = local_demand(self.view, obs)
            return max(feas, key=lambda p: (dem[p], -p))
        self.fallbacks += 1
        dem = local_demand(self.view, obs)
        return min(range(self.view.num_phases), key=lambda p: (self.risk(board, p, t), -dem[p], p))


class EpistemicCoMPController(CoMPController):
    """CoMP whose spillback veto is replaced by the epistemic guard (``K_i Room_m``)."""

    name = "comp+epistemic"

    def reset(self) -> None:
        super().reset()
        self.core.use_veto = False
        self.guard = EpistemicGuard(self.view, {**dict(self.cfg.get("epistemic", {}) or {}),
                                                "delta_room": self.cfg.get("delta_room", 2)})

    def decide(self, obs: Observation, t: int) -> int:
        p = super().decide(obs, t)
        if obs.signal_state != "green" or obs.elapsed < self.view.timing.g_min:
            return p                    # min green and clearance intervals are never cut
        q = self.guard.choose(obs, p, self.board, t)
        if q == obs.phase and obs.elapsed >= self.view.timing.g_max:
            return p                    # max green is a hard limit
        return q

    def epistemic_stats(self) -> dict:
        g = self.guard
        return {"guard_blocked": g.blocked, "guard_fallbacks": g.fallbacks, "guard_checks": g.checks}
