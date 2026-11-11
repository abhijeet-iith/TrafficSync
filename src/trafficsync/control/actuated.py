"""C1a - queue-greedy actuated control: local data only, no messages.

Phase score ``sum r_lm q_hat_l``; shared switching rule (min/max green, hysteresis ``theta``,
optional gap-out).
"""
from __future__ import annotations

from ..sim.observation import Observation
from .base import Controller, phase_scores, select_phase

__all__ = ["ActuatedController"]


class ActuatedController(Controller):
    name = "actuated"

    def reset(self) -> None:
        super().reset()
        self.theta = float(self.cfg.get("theta", 2.0))
        self.gap_out = bool(self.cfg.get("gap_out", False))

    def decide(self, obs: Observation, t: int) -> int:
        scores = phase_scores(self.view, {l: e.q_hat for l, e in obs.incoming.items()})
        return select_phase(obs, scores, self.view.timing, self.theta, self.gap_out)
