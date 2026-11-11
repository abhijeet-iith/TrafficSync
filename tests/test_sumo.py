"""SUMO co-simulation (skipped when SUMO / TraCI are not installed)."""
from __future__ import annotations

import pytest

from trafficsync.sumo import sumo_available

pytestmark = pytest.mark.skipif(not sumo_available(), reason="SUMO not installed (pip install eclipse-sumo traci sumolib)")


def test_grid_network_matches_trafficsync_links(tmp_path):
    import sumolib

    from trafficsync.network import load_network
    from trafficsync.sumo.grid import build_sumo_network, link_of

    net = load_network("grid2x2")
    path, bays = build_sumo_network(net, tmp_path)
    snet = sumolib.net.readNet(str(path))
    ids = {link_of(e.getID()) for e in snet.getEdges()}
    assert ids == set(net.links)
    for l, lk in net.links.items():          # total length preserved (main edge + turn bay)
        total = sum(e.getLength() for e in snet.getEdges() if link_of(e.getID()) == l)
        assert abs(total - lk.length) < 1.0
    assert len(snet.getTrafficLights()) == net.N


def test_sumo_episode_runs_trafficsync_controllers_on_same_vehicles(tmp_path):
    from trafficsync.sumo.grid import SumoGridSimulator, run_sumo_episode
    from trafficsync.network import load_network
    from trafficsync.sim import Simulator

    cfg = {"duration_s": 400, "warmup_s": 100}
    for c in ("fixed", "comp", "cocycle"):
        m = run_sumo_episode("grid2x2", "D_med", c, 3, sim_cfg=cfg, probe_cfg={"mode": "perfect_sensor"}, work_dir=tmp_path / c)
        assert m["completed"] > 0 and m["mean_delay"] >= 0
    net = load_network("grid2x2")
    a = SumoGridSimulator(net, "D_med", seed=3, sim_cfg=cfg, work_dir=tmp_path / "x").demand.sample()
    b = Simulator(net, "D_med", seed=3, sim_cfg=cfg).demand.sample()
    assert a == b                              # identical arrivals and routes as the queue simulator


def test_realmap_agents_and_signal_states():
    from trafficsync.sumo.realmap import run_realmap_episode

    m = run_realmap_episode("cologne8", "comp", 0, {"warmup_s": 60}, end_offset_s=300)
    assert m["agents"] == 8 and m["msgs"] > 0 and m["completed"] > 0


def test_utm_inverse_and_web_viewer(tmp_path):
    from trafficsync.sumo.realmap import run_realmap_episode
    from trafficsync.sumo.webviz import WebRecorder, utm_to_lonlat, write_html

    lon, lat = utm_to_lonlat(500000.0, 5500000.0, 32)            # central meridian of zone 32
    assert abs(lon - 9.0) < 1e-9 and abs(lat - 49.6527) < 1e-3
    rec = WebRecorder(rel_t0=0, rel_len=60)
    m = run_realmap_episode("cologne8", "comp", 0, {"warmup_s": 0}, recorder=rec, end_offset_s=60, work_dir=tmp_path)
    assert len(rec.frames) >= 59 and rec.scene["geo"] is not None and len(rec.scene["agents"]) == 8
    p = write_html(tmp_path / "v.html", "t", "s", rec.scene, [rec.pack("a", "A", m), rec.pack("b", "B", m)])
    html = p.read_text()
    assert "/*DATA*/" not in html and "L.map" in html and len(html) > 150_000


def test_cocycle_agents_agree_on_a_cycle_in_a_real_city():
    from trafficsync.sumo.realmap import run_realmap_episode

    m = run_realmap_episode("cologne8", "cocycle", 0, {"warmup_s": 60}, end_offset_s=600)
    assert m["msgs"] > 0 and m["completed"] > 0
    assert 40 <= m["mean_cycle_s"] <= 120 and m["cycle_spread_s"] <= 8      # consensus: (almost) one common cycle
