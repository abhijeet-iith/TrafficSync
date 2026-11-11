"""STL robustness semantics and the offline trace monitor."""
from __future__ import annotations


import numpy as np
import pytest

from trafficsync.logic.stl import (Always, And, Eventually, Implies, Not, Or, STLParams, ge, le,
                                   phi_resp, phi_safe, phi_starve, rho, robustness)


def test_example_x6_rho4():
    assert rho(le("x", 10), {"x": np.array([6.0])}) == pytest.approx(4.0)
    assert rho(Not(le("x", 10)), {"x": np.array([6.0])}) == pytest.approx(-4.0)


def test_boolean_and_temporal_operators_hand_computed():
    tr = {"x": np.array([6.0, 9.0, 12.0, 3.0, 8.0])}
    phi = le("x", 10)                      # robustness 4, 1, -2, 7, 2
    assert robustness(phi, tr).tolist() == [4, 1, -2, 7, 2]
    assert rho(Always(phi, 0, 4), tr) == -2
    assert rho(Always(phi, 0, 1), tr) == 1
    assert rho(Eventually(phi, 0, 2), tr) == 4
    assert rho(Eventually(phi, 2, 3), tr) == 7
    assert rho(Always(phi, 3, 10), tr) == 2              # truncated at the trace end
    assert rho(And(phi, ge("x", 5)), tr) == 1            # min(4, 1)
    assert rho(Or(phi, ge("x", 5)), tr) == 4
    assert rho(Implies(ge("x", 7), phi), tr) == 4        # max(-(6-7), 4)
    # G of a window entirely past the end is vacuously true, F is false
    assert robustness(Always(phi, 10, 12), tr)[0] == np.inf
    assert robustness(Eventually(phi, 10, 12), tr)[0] == -np.inf


def test_robustness_sign_matches_boolean_satisfaction():
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = rng.uniform(0, 20, 40)
        tr = {"x": x}
        sat_g = bool(np.all(x[:10] <= 10))
        sat_f = bool(np.any(x[5:15] <= 3))
        r_g = rho(Always(le("x", 10), 0, 9), tr)
        r_f = rho(Eventually(le("x", 3), 5, 14), tr)
        assert (r_g > 0) == sat_g or r_g == 0
        assert (r_f > 0) == sat_f or r_f == 0


def test_traffic_specs():
    p = STLParams(T=99, T_r=10, R_max=120, delta_room=2)
    n = np.full(100, 20.0)
    n[50] = 39.0
    assert rho(phi_safe(7, 40, p), {"n_7": n}) == pytest.approx(-1.0)        # 40 - 2 - 39
    q = np.zeros(100)
    q[20:28] = 30.0                                                           # above q_hi = 24, clears in 8 s
    assert rho(phi_resp(7, 40, p), {"q_7": q}) > 0
    q[20:40] = 30.0                                                           # stays 20 s > T_r = 10
    assert rho(phi_resp(7, 40, p), {"q_7": q}) < 0
    assert rho(phi_starve(0, 1, p), {"tsg_0_1": np.arange(100.0)}) == pytest.approx(120 - 99)


def test_trace_monitor_on_simulation():
    from trafficsync.runner import run_episode

    m = run_episode("grid2x2", "D_med", "actuated", 0, sim_cfg={"duration_s": 900, "warmup_s": 300}, stl_monitor=True)
    assert m["stl_rho_safe"] == pytest.approx(m["stl_rho_safe"])
    assert m["stl_rho_Phi"] <= min(m["stl_rho_safe"], m["stl_rho_resp"], m["stl_rho_starve"]) + 1e-9
    assert m["stl_rho_safe"] > 0 and m["stl_rho_starve"] > 0


# ---------------------------------------------------------------- Kripke


