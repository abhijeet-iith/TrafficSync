"""C1b - local max-pressure: ``P_i(p) = sum s_l q_hat_l``; the
downstream term is unknown to an isolated agent.  ``theta`` is in pressure units."""
from __future__ import annotations

from typing import Mapping

from ..sim.observation import Observation
from .base import AgentView, Controller, local_demand, phase_scores, select_phase

__all__ = ["LocalMaxPressureController", "local_pressure"]


def local_pressure(view: AgentView, q: Mapping[int, float]) -> list[float]:
    return phase_scores(view, q, weight_sat_flow=True)


class LocalMaxPressureController(Controller):
    name = "maxpressure_local"

    def reset(self) -> None:
        super().reset()
        self.theta = float(self.cfg.get("theta", 1.0))
        self.gap_out = bool(self.cfg.get("gap_out", False))

    def decide(self, obs: Observation, t: int) -> int:
        P = local_pressure(self.view, {l: e.q_hat for l, e in obs.incoming.items()})
        return select_phase(obs, P, self.view.timing, self.theta, self.gap_out, local_demand(self.view, obs))
