"""Controllers: information isolation, switching rule, no deadlock, CoCycle agreement, CoMP fallback."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from trafficsync.config import SignalTiming
from trafficsync.control import make_agent_views, select_phase
from trafficsync.network import grid
from trafficsync.runner import run_episode
from trafficsync.sim import build_phases
from trafficsync.sim.observation import GREEN, LinkEstimate, Observation

SRC = Path(__file__).resolve().parents[1] / "src" / "trafficsync"
SHORT = {"duration_s": 900, "warmup_s": 300}
PERFECT = {"mode": "perfect_sensor"}


def test_controllers_never_touch_simulator_internals():
    for f in (SRC / "control").glob("*.py"):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and "sim" in node.module.split("."):
                assert node.module.endswith("sim.observation"), f"{f.name} imports {node.module}"


def test_select_phase_min_max_green():
    net = grid(2, 2)
    v = make_agent_views(net, [build_phases(net, i) for i in range(4)], SignalTiming(),
                         {"through": .7, "left": .15, "right": .15})[0]
    inc = {l: LinkEstimate(l, 0, 0, 0, 13.9, 0, 0) for l in v.incoming}
    o = lambda e: Observation(0, 0, 0, GREEN, e, None, 2, inc)  # noqa: E731
    assert select_phase(o(3), [0, 100], v.timing, 2) == 0
    assert select_phase(o(10), [0, 100], v.timing, 2) == 1
    assert select_phase(o(40), [100, 0], v.timing, 2) == 1


@pytest.mark.parametrize("ctrl", ["fixed", "actuated", "maxpressure_local", "comp", "cocycle", "comp+cycle"])
def test_no_deadlock_grid3x3(ctrl):
    m = run_episode("grid3x3", "D_high", ctrl, 0, sim_cfg={"duration_s": 1500, "warmup_s": 300}, probe_cfg=PERFECT)
    assert not m["gridlock"] and m["throughput_vph"] > 5000


def test_cocycle_agrees_on_one_cycle():
    m = run_episode("grid3x3", "D_med", "cocycle", 0, sim_cfg={"duration_s": 1500, "warmup_s": 300}, probe_cfg=PERFECT)
    assert m["cycle_spread"] == 0.0


def test_comp_falls_back_to_local_with_total_loss():
    kw = dict(sim_cfg={"duration_s": 900, "warmup_s": 300}, probe_cfg=PERFECT)
    from trafficsync.runner import load_controller_params, _deep_merge

    p = _deep_merge(load_controller_params(), {"comp": {"gamma": {"enabled": False}},
                                               "maxpressure_local": {"theta": 8.0, "gap_out": True}})
    a = run_episode("grid2x2", "D_med", "comp", 0, comm_cfg={"drop_prob": 1.0}, controller_params=p, **kw)
    b = run_episode("grid2x2", "D_med", "maxpressure_local", 0, controller_params=p, **kw)
    assert a["mean_delay"] == pytest.approx(b["mean_delay"])


def test_controllers_never_call_true_state():
    for f in (SRC / "control").glob("*.py"):
        assert "true_state" not in f.read_text(), f.name
