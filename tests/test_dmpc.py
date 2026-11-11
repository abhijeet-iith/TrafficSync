"""DMPC: local QP behaviour, profile exchange, STL / epistemic constraints, closed loop."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cvxpy")

from trafficsync.config import SignalTiming  # noqa: E402
from trafficsync.control import make_agent_views  # noqa: E402
from trafficsync.control.dmpc import LocalQP  # noqa: E402
from trafficsync.network import grid  # noqa: E402
from trafficsync.runner import run_episode  # noqa: E402
from trafficsync.sim import build_phases  # noqa: E402

TURN = {"through": 0.7, "left": 0.15, "right": 0.15}


def _qp(stl=False, epi=False, **cfg):
    net = grid(3, 3)
    views = make_agent_views(net, [build_phases(net, i) for i in range(9)], SignalTiming(), TURN)
    v = views[4]
    qp = LocalQP(v, 12, 5, {"w_lin": 1.0, "w_q": 0.05, "w_u": 0.5, "w_switch_l1": 30, "solver_eps": 1e-6, **cfg}, stl, epi)
    L, P, M = len(qp.links), v.num_phases, len(qp.outs)
    qp.q0.value = np.zeros(L)
    qp.a.value = np.zeros((L, 12))
    qp.gprev.value = np.array([1.0, 0.0])
    qp.room.value = np.full((M, 12), 100.0)
    if epi:
        qp.allow.value = np.full(M, 1e6)
    if stl:
        qp.qthr.value = np.full(L, 8.0)
        qp.qhi.value = np.full(L, 24.0)
        qp.need.value = np.zeros(P)
        qp.mask.value = np.zeros((P, 12))
    return v, qp


def _phase_of(v, l):
    return next(p for p in range(v.num_phases) if l in v.phase_link_shares(p))


def test_qp_serves_the_queued_phase():
    v, qp = _qp()
    l = qp.links[0]
    q0 = np.zeros(len(qp.links))
    q0[0] = 20.0
    qp.q0.value = q0
    assert qp.solve()
    p = _phase_of(v, l)
    assert qp.g.value[p, :3].sum() > qp.g.value[1 - p, :3].sum()
    # conservation: q[H] = q0 + sum a - sum d
    q, d = qp.q.value, qp.d.value
    assert np.allclose(q[:, -1], q[:, 0] + qp.a.value.sum(1) - d.sum(1), atol=1e-3)
    assert (qp.g.value.sum(0) <= 1 + 1e-4).all()


def test_room_constraint_limits_flow_into_a_full_link():
    v, qp = _qp(stl=True)
    q0 = np.full(len(qp.links), 15.0)
    qp.q0.value = q0
    room = np.full((len(qp.outs), 12), 100.0)
    room[0, :] = 0.0                                          # downstream link 0 is full for the whole horizon
    qp.room.value = room
    assert qp.solve()
    inflow = qp.R[0] @ qp.d.value
    assert inflow.sum() < 0.05                                # the 1000x penalty makes it (almost) hard


def test_epistemic_constraint_blocks_flow_now():
    v, qp = _qp(epi=True)
    qp.q0.value = np.full(len(qp.links), 15.0)
    allow = np.full(len(qp.outs), 1e6)
    allow[1] = 0.0
    qp.allow.value = allow
    assert qp.solve()
    assert (qp.R[1] @ qp.d.value)[0] < 1e-3


def test_starvation_constraint():
    v, qp = _qp(stl=True)
    q0 = np.zeros(len(qp.links))
    for k, l in enumerate(qp.links):
        if _phase_of(v, l) == 0:
            q0[k] = 30.0
    qp.q0.value = q0                                           # only phase 0 has demand
    need = np.array([0.0, 1.0])
    mask = np.zeros((2, 12))
    mask[1, :4] = 1.0                                          # phase 1 must get a full step within 4 steps
    qp.need.value, qp.mask.value = need, mask
    assert qp.solve()
    assert qp.g.value[1, :4].sum() >= 0.98          # OSQP tolerance


@pytest.mark.parametrize("ctrl", ["dmpc", "dmpc+stl", "dmpc+stl+epistemic"])
def test_closed_loop_short(ctrl):
    m = run_episode("grid2x2", "D_med", ctrl, 0, sim_cfg={"duration_s": 600, "warmup_s": 200},
                    probe_cfg={"mode": "perfect_sensor"}, stl_monitor=True)
    assert not m["gridlock"] and m["solve_failures"] <= 0.01 * m["solves"]
    assert m["msgs_per_s_agent"] > 0


def test_dmpc_profiles_are_exchanged():
    m = run_episode("grid2x2", "D_med", "dmpc", 0, sim_cfg={"duration_s": 300, "warmup_s": 100},
                    probe_cfg={"mode": "perfect_sensor"}, return_sim=True)
    c = m["_controllers"][0]
    assert c.plans and all("dep" in p for _, p in c.plans.values())
    assert any(len(v) == 12 for v in c.dep.values())
