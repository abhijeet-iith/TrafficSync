"""Simulator: conservation, queue validity, no teleporting, saturation flow, signal safety, determinism, probes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trafficsync.config import SignalTiming
from trafficsync.network import corridor, grid
from trafficsync.runner import run_episode
from trafficsync.sim import Simulator
from trafficsync.sim.observation import GREEN

SRC = Path(__file__).resolve().parents[1] / "src" / "trafficsync"
SHORT = {"duration_s": 900, "warmup_s": 300}
PERFECT = {"mode": "perfect_sensor"}


def _drive(sim, policy=lambda t, i: (t // 30) % 2):
    obs = sim.reset()
    while not sim.done:
        t = sim.t
        obs, _ = sim.step({i: policy(t, i) for i in obs})
        yield sim


@pytest.mark.parametrize("demand", ["D_med", "D_high", "D_incident", "D_hotspot"])
def test_conservation_and_queue_validity(demand):
    sim = Simulator(grid(3, 3), demand, probe_cfg=PERFECT, seed=1, sim_cfg=SHORT)
    for s in _drive(sim):
        st = s.true_state()
        assert st["generated"] == st["n_in_network"] + st["completed"]
        for l, d in st["links"].items():
            assert d["q"] >= 0 and d["q"] <= d["n"]
            if s.network.links[l].kind != "exit":
                assert d["n"] <= d["capacity"]


@pytest.mark.parametrize("lanes", [1, 2])
def test_saturation_flow_within_5_percent(lanes):
    net = corridor(1, lanes=lanes, link_length=2000)
    sim = Simulator(net, {"name": "sat", "rate": 0.0, "entry_rates": [{"entry": {"node": 0, "approach": "W"}, "rate": 3.0}]},
                    probe_cfg=PERFECT, seed=0,
                    sim_cfg={"duration_s": 700, "warmup_s": 0, "signal": {"g_max": 10000}})
    W = [l for l in net.incoming_links(0) if net.links[l].approach == "W"][0]
    out = {}
    for s in _drive(sim, policy=lambda t, i: 0 if t < 400 else 1):     # long red for EW, then EW green
        out[s.t] = sum(1 for v in s.vehicles.values() if v.idx > 0 and v.route[0] == W)
    # 200 s of effective green after start-up (t = 405 .. 605), queue is long throughout
    rate = (out[640] - out[440]) / 200.0
    assert rate == pytest.approx(0.5 * lanes, rel=0.05)


def test_signal_safety_and_timing():
    tm = SignalTiming()
    sim = Simulator(grid(2, 2), "D_med", probe_cfg=PERFECT, seed=0, sim_cfg=SHORT)
    rng = np.random.default_rng(0)
    prev = [None] * 4
    green_len = [0] * 4
    for s in _drive(sim, policy=lambda t, i: int(rng.integers(2))):
        for i, sm in enumerate(s.signals):
            if sm.state == GREEN:
                green_len[i] += 1
                assert green_len[i] <= tm.g_max + 1
            else:
                if prev[i] == GREEN:
                    assert green_len[i] >= tm.g_min
                green_len[i] = 0
            prev[i] = sm.state
            served = {m.approach for m in sm.green_movements()}
            assert not ({"N", "S"} & served and {"E", "W"} & served)


def test_determinism_and_identical_arrivals():
    a = run_episode("grid2x2", "D_high", "actuated", 3, sim_cfg=SHORT)
    b = run_episode("grid2x2", "D_high", "actuated", 3, sim_cfg=SHORT)
    skip = ("wall_s", "compute_ms_per_decision")
    assert repr({k: v for k, v in a.items() if k not in skip}) == repr({k: v for k, v in b.items() if k not in skip})
    s1 = Simulator(grid(2, 2), "D_high", seed=3, sim_cfg=SHORT)
    s2 = Simulator(grid(2, 2), "D_high", seed=3, sim_cfg=SHORT)
    assert s1.arrivals.keys() == s2.arrivals.keys()
    assert all(s1.arrivals[t] == s2.arrivals[t] for t in s1.arrivals)
    f = run_episode("grid2x2", "D_high", "fixed", 3, sim_cfg=SHORT)
    assert f["generated"] == a["generated"]


def test_probe_estimates_approximately_unbiased():
    net = grid(2, 2)
    sim = Simulator(net, "D_high", probe_cfg={"mode": "probe", "penetration": 0.5}, seed=0,
                    sim_cfg={"duration_s": 1800, "warmup_s": 0})
    est_q, true_q = [], []
    for s in _drive(sim):
        if s.t % 5 == 3 and s.t > 300:
            st = s.true_state()["links"]
            for l in net.queue_links:
                est_q.append(s.probe.est[l].n_hat)
                true_q.append(st[l]["n"])
    assert np.mean(est_q) == pytest.approx(np.mean(true_q), rel=0.15)


# --------------------------------------------------------------- consensus


def test_no_vehicle_teleports():
    """Every step a vehicle either stays on its link or moves to the next link of its route."""
    sim = Simulator(grid(3, 3), "D_high", probe_cfg=PERFECT, seed=2, sim_cfg=SHORT)
    prev = {}
    for s in _drive(sim):
        for vid, v in s.vehicles.items():
            if vid in prev:
                assert v.idx - prev[vid] in (0, 1)
            prev[vid] = v.idx
        # a vehicle is in exactly one place: its current link, a boundary buffer, or done
        where = {}
        for l, st in s.ls.items():
            for _, vid in st.travel:
                where[vid] = where.get(vid, 0) + 1
                assert s.vehicles[vid].link == l
            for vid in st.queue:
                where[vid] = where.get(vid, 0) + 1
                assert s.vehicles[vid].link == l
        for e, buf in s.buffers.items():
            for vid in buf:
                where[vid] = where.get(vid, 0) + 1
        assert all(c == 1 for c in where.values())


def test_identical_traces_for_same_seed():
    def trace(seed):
        sim = Simulator(grid(2, 2), "D_med", probe_cfg={"mode": "probe"}, seed=seed, sim_cfg=SHORT)
        list(_drive(sim))
        return sim.logger.q.copy(), sim.logger.n.copy(), [(v.t_gen, tuple(v.route), v.t_done) for v in sim.vehicles.values()]
    a, b, c = trace(4), trace(4), trace(5)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]
    assert a[2] != c[2]


def test_lone_vehicle_on_green_does_not_wait():
    net = corridor(1)
    sim = Simulator(net, {"name": "x", "rate": 0.0, "entry_rates": [{"entry": {"node": 0, "approach": "W"}, "rate": 0.01}]},
                    probe_cfg=PERFECT, seed=1, sim_cfg={"duration_s": 2000, "warmup_s": 0, "signal": {"g_max": 100000}})
    list(_drive(sim, policy=lambda t, i: 1))
    done = [v for v in sim.vehicles.values() if v.t_done is not None]
    assert done and all(v.wait == 0 and v.stops == 0 for v in done)


def test_metrics_cover_metric_table():
    m = run_episode("grid2x2", "D_med", "cocycle", 0, sim_cfg=SHORT, stl_monitor=True)
    for k in ("mean_travel_time", "mean_delay", "mean_waiting", "mean_queue", "max_queue", "queue_per_intersection",
              "throughput_vph", "mean_stops", "spillback_events", "jain_fairness", "max_red_time", "gridlock",
              "consensus_error_mean", "msgs_per_s_agent", "bytes_per_s_agent", "stl_rho_Phi", "compute_ms_per_decision"):
        assert k in m, k
