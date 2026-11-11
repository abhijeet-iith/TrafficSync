"""Agent-facing data types: everything an intersection may see about itself.

``control/`` may import only this module from ``sim/`` (information isolation).
"""
from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["GREEN", "YELLOW", "ALLRED", "LinkEstimate", "Observation"]

GREEN, YELLOW, ALLRED = "green", "yellow", "allred"


@dataclass(frozen=True)
class LinkEstimate:
    """Estimate of one incoming link (probe-based or local detector)."""

    link: int
    q_hat: float          # queue at the stop line [veh]
    n_hat: float          # vehicles on the link [veh]
    rho_hat: float        # n_hat / capacity
    v_hat: float          # mean speed [m/s]
    lambda_hat: float     # arrival rate onto the link [veh/s]
    t_measured: int       # step at which these values were produced
    capacity: int = 0
    sat_flow: float = 0.5


@dataclass(frozen=True)
class Observation:
    agent_id: int
    t: int
    phase: int
    signal_state: str
    elapsed: int                      # seconds in the current signal state
    next_phase: int | None
    num_phases: int
    incoming: dict                    # {link_id: LinkEstimate}
    outgoing: dict = field(default_factory=dict)   # reserved (own outgoing links are not measured)
