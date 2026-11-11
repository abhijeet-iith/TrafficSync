"""Signal controllers."""
from __future__ import annotations

from typing import Any, Mapping

from .actuated import ActuatedController
from .base import AgentView, Controller, make_agent_views, phase_scores, select_phase, target_phase
from .cocycle import CoCycleController, CoMPCycleController, cycle_from_load, webster_split
from .comp import CoMPController, CoMPCore, LoadConsensus, NeighbourBoard
from .fixed_time import FixedTimeController, WebsterPlan, expected_link_flows, webster_plan
from .marl_policy import MARLController
from .maxpressure import LocalMaxPressureController, local_pressure
from .offset_consensus import OffsetAgent, build_constraints, simulate_offset_consensus, wrap
from .dmpc import DMPCController
from .epistemic import EpistemicCoMPController, EpistemicGuard
from .wmsr import FaultInjector, FaultModel, WMSRConsensus, make_consensus

CONTROLLERS: dict[str, type[Controller]] = {
    "fixed": FixedTimeController,
    "actuated": ActuatedController,
    "maxpressure_local": LocalMaxPressureController,
    "comp": CoMPController,
    "cocycle": CoCycleController,
    "comp+cycle": CoMPCycleController,
    "comp+epistemic": EpistemicCoMPController,
    "dmpc": DMPCController,
    "marl": MARLController,          # executes a trained policy (cfg: policy=<checkpoint>)
}


def make_controller(name: str, agent_id: int, view: AgentView, cfg: Mapping[str, Any] | None = None) -> Controller:
    """Instantiate a registered controller by name."""
    if name not in CONTROLLERS:
        raise KeyError(f"unknown controller {name!r}; known: {sorted(CONTROLLERS)}")
    return CONTROLLERS[name](agent_id, view, cfg)


__all__ = [
    "AgentView", "Controller", "make_agent_views", "phase_scores", "select_phase", "target_phase",
    "FixedTimeController", "WebsterPlan", "expected_link_flows", "webster_plan", "ActuatedController",
    "LocalMaxPressureController", "local_pressure", "CONTROLLERS", "make_controller", "CoMPController", "CoMPCore",
    "LoadConsensus", "NeighbourBoard", "CoCycleController", "CoMPCycleController", "cycle_from_load", "webster_split",
    "OffsetAgent", "build_constraints", "simulate_offset_consensus", "wrap", "FaultInjector", "FaultModel",
    "WMSRConsensus", "make_consensus", "DMPCController", "EpistemicCoMPController", "EpistemicGuard",
]
