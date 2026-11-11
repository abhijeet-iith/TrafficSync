"""TrafficSync networks and controllers inside SUMO.

``run_sumo_episode`` mirrors :func:`trafficsync.runner.run_episode`, with the queue simulator replaced by
SUMO (car-following, lane changing, intersection internals):

1. the TrafficSync :class:`~trafficsync.network.graph.Network` is written as SUMO nodes / edges
   (one SUMO edge ``L<id>`` per link, same length, lanes and speed) and compiled with ``netconvert``;
2. the TrafficSync demand model samples the vehicles (same seed -> identical arrival times and routes as the
   queue simulator) and writes them as a SUMO route file;
3. each step the **unchanged** TrafficSync controllers receive observations built from SUMO lane data
   (halting vehicles = queue, vehicles on the edge = occupancy, entries = arrival rate; local detectors or the
   probe-vehicle model), exchange messages through the TrafficSync bus, and request phases;
4. the TrafficSync signal state machine (min/max green, yellow, all-red) turns requests into SUMO signal
   states (``G`` through/right, ``g`` permissive left, ``y``, ``r``) via ``setRedYellowGreenState``;
5. metrics come from SUMO's trip information (time loss + insertion delay = delay).
"""
from __future__ import annotations

import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..comm.bus import CommBus, CommConfig
from ..config import ProbeConfig, SimConfig
from ..control import expected_link_flows, make_agent_views, make_controller
from ..network.graph import Network
from ..sim.demand import Demand, DemandScenario
from ..sim.intersection import GREEN, YELLOW, SignalStateMachine, build_phases
from ..sim.observation import Observation
from ..sim.probe import ProbeModel
from .common import start_traci, tool_binary

__all__ = ["build_sumo_network", "write_routes", "SumoGridSimulator", "run_sumo_episode"]

_DIR = {"E": (1.0, 0.0), "W": (-1.0, 0.0), "N": (0.0, 1.0), "S": (0.0, -1.0)}


def edge_id(l: int) -> str:
    return f"L{l}"


def bay_id(l: int) -> str:
    return f"L{l}b"


def link_of(sumo_edge_or_lane: str) -> int:
    """TrafficSync link id of a SUMO edge / lane id (``L12``, ``L12b``, ``L12b_1``)."""
    return int(sumo_edge_or_lane.split("_")[0][1:].rstrip("b"))


# --------------------------------------------------------------------------- network
def build_sumo_network(net: Network, out_dir: str | Path, turn_bay_m: float = 60.0) -> tuple[Path, set[int]]:
    """Write nodes / edges / connections and compile ``<name>.net.xml`` with netconvert.

    Every link that ends at an intersection gets, over its last ``turn_bay_m`` metres, an extra lane used
    only by left-turners (edge ``L<id>b``), so a left-turner waiting for a gap in oncoming traffic does not
    block through traffic -- the same assumption as the queue simulator.  ``turn_bay_m = 0``: one shared lane.
    Returns the network file and the set of links that have a bay."""
    import subprocess

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    nodes = ET.Element("nodes")
    for i, (x, y) in enumerate(net.pos):
        ET.SubElement(nodes, "node", id=f"J{i}", x=f"{x:.2f}", y=f"{y:.2f}", type="traffic_light", tl=f"J{i}")
    boundary: dict[tuple[float, float], str] = {}

    def bnode(x: float, y: float) -> str:
        key = (round(x, 1), round(y, 1))
        if key not in boundary:
            boundary[key] = f"B{len(boundary)}"
            ET.SubElement(nodes, "node", id=boundary[key], x=f"{x:.2f}", y=f"{y:.2f}", type="dead_end")
        return boundary[key]

    edges, cons = ET.Element("edges"), ET.Element("connections")
    bays: set[int] = set()
    for l, lk in sorted(net.links.items()):
        dx, dy = _DIR[lk.heading]
        if lk.from_node is None:          # entry: starts outside, ends at to_node
            x, y = net.pos[lk.to_node]
            frm, to = bnode(x - dx * lk.length, y - dy * lk.length), f"J{lk.to_node}"
        elif lk.to_node is None:          # exit
            x, y = net.pos[lk.from_node]
            frm, to = f"J{lk.from_node}", bnode(x + dx * lk.length, y + dy * lk.length)
        else:
            frm, to = f"J{lk.from_node}", f"J{lk.to_node}"
        common = {"speed": f"{lk.free_flow_speed:.2f}"}
        if lk.to_node is not None and turn_bay_m > 0:
            bays.add(l)
            tx, ty = net.pos[lk.to_node]
            mid = f"M{l}"
            ET.SubElement(nodes, "node", id=mid, x=f"{tx - dx * turn_bay_m:.2f}", y=f"{ty - dy * turn_bay_m:.2f}", type="priority")
            ET.SubElement(edges, "edge", id=edge_id(l), attrib={"from": frm, "to": mid, "numLanes": str(lk.lanes),
                                                                "length": f"{lk.length - turn_bay_m:.1f}", **common})
            ET.SubElement(edges, "edge", id=bay_id(l), attrib={"from": mid, "to": to, "numLanes": str(lk.lanes + 1),
                                                               "length": f"{turn_bay_m:.1f}", **common})
            for ln in range(lk.lanes + 1):
                ET.SubElement(cons, "connection", attrib={"from": edge_id(l), "to": bay_id(l),
                                                          "fromLane": str(min(ln, lk.lanes - 1)), "toLane": str(ln)})
        else:
            ET.SubElement(edges, "edge", id=edge_id(l), attrib={"from": frm, "to": to, "numLanes": str(lk.lanes),
                                                                "length": f"{lk.length:.1f}", **common})
    for i in range(net.N):
        for m in net.movements[i]:
            src = bay_id(m.in_link) if m.in_link in bays else edge_id(m.in_link)
            nl = net.links[m.in_link].lanes
            from_lanes = [nl] if (m.in_link in bays and m.turn == "left") else range(nl)
            for fl in from_lanes:
                ET.SubElement(cons, "connection", attrib={"from": src, "to": edge_id(m.out_link), "fromLane": str(fl),
                                                          "toLane": "0"})
    nf, ef, cf, nt = (out / f"{net.name}.{k}.xml" for k in ("nod", "edg", "con", "net"))
    ET.ElementTree(nodes).write(nf)
    ET.ElementTree(edges).write(ef)
    ET.ElementTree(cons).write(cf)
    r = subprocess.run([tool_binary("netconvert"), "-n", str(nf), "-e", str(ef), "-x", str(cf), "-o", str(nt),
                        "--no-turnarounds", "true", "--tls.default-type", "static", "--no-warnings", "true",
                        "--offset.disable-normalization", "true"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"netconvert failed: {r.stderr[-2000:]}")
    return nt, bays


def write_routes(net: Network, arrivals: Mapping[int, list], path: str | Path, vtype: Mapping[str, Any] | None = None,
                 bays: set[int] | None = None) -> dict:
    """Write the TrafficSync arrivals ``{t: [(entry, route, is_probe), ...]}`` as SUMO vehicles.

    Returns ``{vehicle_id: (t_gen, route, is_probe)}``."""
    root = ET.Element("routes")
    vt = {"id": "car", "length": "5.0", "minGap": "2.5", "accel": "2.6", "decel": "4.5", "sigma": "0.5", "maxSpeed": "13.9"}
    vt.update({k: str(v) for k, v in (vtype or {}).items()})
    ET.SubElement(root, "vType", attrib=vt)
    info, k = {}, 0
    for t in sorted(arrivals):
        for _, route, is_probe in arrivals[t]:
            vid = f"v{k}"
            v = ET.SubElement(root, "vehicle", id=vid, type="car", depart=f"{t:.1f}", departLane="best",
                              departSpeed="max", departPos="base")
            ET.SubElement(v, "route", edges=" ".join(e for l in route for e in
                                                    ((edge_id(l), bay_id(l)) if l in (bays or ()) else (edge_id(l),))))
            info[vid] = (int(t), list(route), bool(is_probe))
            k += 1
    ET.ElementTree(root).write(path)
    return info


# --------------------------------------------------------------------------- simulator
class _Logger:
    """The two hooks the episode loop needs (compute time and communication counts)."""

    def __init__(self) -> None:
        self.compute: list[float] = []
        self.msgs = 0
        self.bytes = 0

    def record_compute_time(self, s: float) -> None:
        self.compute.append(s)

    def record_comm(self, t: int, n_msgs: int, n_bytes: int) -> None:
        self.msgs += n_msgs
        self.bytes += n_bytes


class SumoGridSimulator:
    """Drop-in replacement for :class:`trafficsync.sim.Simulator` that runs SUMO underneath."""

    def __init__(self, net: Network, demand_cfg: Any, probe_cfg: ProbeConfig | Mapping | None = None, seed: int = 0,
                 sim_cfg: SimConfig | Mapping | None = None, gui: bool = False, work_dir: str | Path | None = None,
                 sumo_seed: int | None = None, turn_bay_m: float = 60.0) -> None:
        self.network = net
        self.cfg = sim_cfg if isinstance(sim_cfg, SimConfig) else SimConfig.from_dict(sim_cfg)
        self.probe_cfg = probe_cfg if isinstance(probe_cfg, ProbeConfig) else ProbeConfig.from_dict(probe_cfg)
        self.seed, self.gui = int(seed), gui
        self.scenario = DemandScenario.get(demand_cfg)
        self.duration = int(self.cfg.duration_s)
        self.agents = list(range(net.N))
        self.phases = [build_phases(net, i, self.cfg.phase_scheme, self.cfg.drive_side) for i in range(net.N)]
        self.demand = Demand(net, self.scenario, self.duration, self.cfg.turning, self.seed, self.probe_cfg.penetration)
        self.qlinks = list(net.queue_links)
        self.work = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="ts_sumo_"))
        self.netfile, self.bays = build_sumo_network(net, self.work, turn_bay_m)
        self.sumo_seed = self.seed if sumo_seed is None else int(sumo_seed)
        self.conn = None

    # ------------------------------------------------------------------ API
    def reset(self) -> dict[int, Observation]:
        net = self.network
        self.close()
        self.t = 0
        self.signals = [SignalStateMachine(i, self.phases[i], self.cfg.signal, self.cfg.initial_phase) for i in self.agents]
        self.vinfo = write_routes(net, self.demand.sample(), self.work / "routes.rou.xml", bays=self.bays)
        self.tripfile = self.work / f"trips_{time.time_ns()}.xml"
        args = ["-n", str(self.netfile), "-r", str(self.work / "routes.rou.xml"), "--begin", "0",
                "--end", str(self.duration), "--step-length", "1", "--seed", str(self.sumo_seed),
                "--tripinfo-output", str(self.tripfile), "--no-step-log", "true", "--no-warnings", "true",
                "--time-to-teleport", "300", "--collision.action", "warn"]
        self.conn = start_traci(args, self.gui)
        c = self.conn
        import traci.constants as tc

        self._tc = tc
        self._edges = {l: [edge_id(l)] + ([bay_id(l)] if l in self.bays else []) for l in net.links}
        for es in self._edges.values():
            for e in es:
                c.edge.subscribe(e, [tc.LAST_STEP_VEHICLE_HALTING_NUMBER, tc.LAST_STEP_VEHICLE_NUMBER,
                                     tc.LAST_STEP_VEHICLE_ID_LIST])
        # signal index -> (in_link, out_link) for every traffic light
        self._links = {}
        for i in self.agents:
            idx = []
            for conns in c.trafficlight.getControlledLinks(f"J{i}"):
                if not conns:
                    idx.append(None)
                    continue
                a, b, _ = conns[0]
                idx.append((link_of(a), link_of(b)))
            self._links[i] = idx
        self._turn = {(m.in_link, m.out_link): m.turn for ph in self.phases for p in ph for m in p}
        self._served = [[{(m.in_link, m.out_link) for m in p} for p in self.phases[i]] for i in self.agents]
        self._on_edge = {l: set() for l in net.links}
        self.probe = ProbeModel(self.probe_cfg, net.links, self.qlinks, self.seed)
        self.logger = _Logger()
        self.q_hist: list[np.ndarray] = []
        self.spill = 0
        self.max_red = 0
        self._apply_signals()
        self._read()
        return self._observations()

    @property
    def done(self) -> bool:
        return self.t >= self.duration

    def step(self, actions: Mapping[int, int | None] | None = None):
        actions = actions or {}
        for i, sm in enumerate(self.signals):
            sm.update(actions.get(i))
        self._apply_signals()
        self.conn.simulationStep()
        for sm in self.signals:
            sm.advance()
            if self.t >= self.cfg.warmup_s:
                self.max_red = max(self.max_red, max(sm.tsg))
        self.t += 1
        self._read()
        return self._observations(), {"t": self.t - 1}

    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:  # noqa: BLE001 - SUMO may already have exited
                pass
            self.conn = None

    # -------------------------------------------------------------- internals
    def _state_string(self, i: int) -> str:
        sm = self.signals[i]
        out = []
        for mv in self._links[i]:
            if mv is None:
                out.append("r")
            elif sm.state == GREEN and mv in self._served[i][sm.phase]:
                out.append("g" if self._turn.get(mv) == "left" else "G")
            elif sm.state == YELLOW and mv in self._served[i][sm.phase]:
                out.append("y")
            else:
                out.append("r")
        return "".join(out)

    def _apply_signals(self) -> None:
        for i in self.agents:
            self.conn.trafficlight.setRedYellowGreenState(f"J{i}", self._state_string(i))

    def _read(self) -> None:
        sub = self.conn.edge.getAllSubscriptionResults()
        tc = self._tc
        state = {}
        probe_mode = self.probe_cfg.mode == "probe"
        for l in self.network.links:
            rs = [sub.get(e, {}) for e in self._edges[l]]
            ids = set().union(*(r.get(tc.LAST_STEP_VEHICLE_ID_LIST, ()) for r in rs))
            for vid in ids - self._on_edge[l]:
                self.probe.on_enter(l, self.vinfo[vid][2])
            self._on_edge[l] = ids
            if l in self.probe.est:
                q = sum(int(r.get(tc.LAST_STEP_VEHICLE_HALTING_NUMBER, 0)) for r in rs)
                n = sum(int(r.get(tc.LAST_STEP_VEHICLE_NUMBER, 0)) for r in rs)
                pq = pt = 0
                if probe_mode:
                    for vid in ids:
                        if self.vinfo[vid][2]:
                            if self.conn.vehicle.getSpeed(vid) < 0.1:
                                pq += 1
                            else:
                                pt += 1
                state[l] = (q, n, pq, pt)
        self.probe.update(self.t, state)
        if self.t > self.cfg.warmup_s:
            self.q_hist.append(np.array([state[l][0] for l in self.qlinks], float))
            caps = {l: self.network.links[l].capacity for l in self.qlinks}
            self.spill += sum(1 for l in self.network.internal_links if state.get(l, (0, 0))[1] >= caps.get(l, 1e9))

    def _observations(self) -> dict[int, Observation]:
        est = self.probe.est
        return {i: Observation(i, self.t, sm.phase, sm.state, sm.t_in_state, sm.next_phase, sm.P,
                               {l: est[l] for l in self.network.incoming_links(i)}, {})
                for i, sm in enumerate(self.signals)}

    def metrics(self) -> dict:
        self.close()
        warm = self.cfg.warmup_s
        rows = []
        for tr in ET.parse(self.tripfile).getroot().iter("tripinfo"):
            t_gen = self.vinfo[tr.get("id")][0]
            if t_gen >= warm:
                rows.append((float(tr.get("timeLoss")) + float(tr.get("departDelay")),
                             float(tr.get("waitingTime")) + float(tr.get("departDelay")),
                             int(tr.get("waitingCount")), float(tr.get("duration")) + float(tr.get("departDelay"))))
        arr = np.array(rows) if rows else np.zeros((0, 4))
        gen = sum(1 for v in self.vinfo.values() if v[0] >= warm)
        Q = np.array(self.q_hist) if self.q_hist else np.zeros((1, len(self.qlinks)))
        hours = (self.duration - warm) / 3600.0
        return {"mean_delay": float(arr[:, 0].mean()) if len(arr) else float("nan"),
                "mean_stopped_delay": float(arr[:, 1].mean()) if len(arr) else float("nan"),
                "mean_stops": float(arr[:, 2].mean()) if len(arr) else float("nan"),
                "mean_travel_time": float(arr[:, 3].mean()) if len(arr) else float("nan"),
                "mean_queue": float(Q.mean()), "max_queue": float(Q.max()),
                "throughput_vph": len(arr) / hours, "completed": len(arr), "generated_after_warmup": gen,
                "unfinished": gen - len(arr), "spillback_link_seconds": int(self.spill), "max_red_time": int(self.max_red),
                "msgs_per_s_agent": self.logger.msgs / max(1, self.duration) / len(self.agents),
                "compute_ms_per_decision": 1000 * float(np.mean(self.logger.compute)) if self.logger.compute else 0.0}


# --------------------------------------------------------------------------- episode
def run_sumo_episode(network: Network | str, demand: Any, controller: str, seed: int, *,
                     sim_cfg: Mapping | None = None, probe_cfg: Mapping | None = None, comm_cfg: Mapping | None = None,
                     controller_params: Mapping | None = None, gui: bool = False, work_dir: str | Path | None = None,
                     recorder: Any = None, turn_bay_m: float = 60.0) -> dict:
    """:func:`trafficsync.runner.run_episode` with SUMO as the traffic simulator."""
    from ..runner import build_network, load_controller_params, resolve_controller

    t_wall = time.perf_counter()
    sim_cfg = sim_cfg if isinstance(sim_cfg, SimConfig) else SimConfig.from_dict(sim_cfg)
    comm_cfg = comm_cfg if isinstance(comm_cfg, CommConfig) else CommConfig.from_dict(comm_cfg)
    net = build_network(network, comm_cfg) if isinstance(network, str) else network
    params = dict(controller_params) if controller_params is not None else load_controller_params()
    sim = SumoGridSimulator(net, demand, probe_cfg=probe_cfg, seed=seed, sim_cfg=sim_cfg, gui=gui, work_dir=work_dir,
                            turn_bay_m=turn_bay_m)
    turning = dict(sim.scenario.turning or sim_cfg.turning)
    cls, ccfg = resolve_controller(controller, params)
    ccfg.setdefault("comm_period_s", comm_cfg.period_s)
    ccfg.setdefault("missing_policy", comm_cfg.missing_policy)
    views = make_agent_views(net, sim.phases, sim_cfg.signal, turning,
                             decision_interval=int(ccfg.get("decision_interval_s", sim_cfg.decision_interval_s)))
    if cls == "fixed":
        entry_rates = {e: float(sim.demand.rate_vector(e).mean()) for e in net.entry_links}
        ccfg["design_flows"] = expected_link_flows(net, entry_rates, turning)
    ccfg.pop("faults", None)
    ctrls = {i: make_controller(cls, i, views[i], ccfg) for i in sim.agents}
    bus = CommBus(net, comm_cfg, seed=seed)
    obs = sim.reset()
    if recorder is not None:
        recorder.start(sim)
    try:
        while not sim.done:
            t = sim.t
            inboxes = bus.deliver(t)
            actions = {}
            for i, c in ctrls.items():
                t0 = time.perf_counter()
                actions[i] = c.act(obs[i], inboxes.get(i, []), t)
                if c.last_was_decision:
                    sim.logger.record_compute_time(time.perf_counter() - t0)
            for i, c in ctrls.items():
                bus.send_many(c.outbox(obs[i], t))
            n_msgs, n_bytes = bus.pop_step_counts()
            if n_msgs:
                sim.logger.record_comm(t, n_msgs, n_bytes)
            obs, _ = sim.step(actions)
            if recorder is not None:
                recorder.step(sim)
    finally:
        m = sim.metrics()
    m.update(network=net.name, demand=str(demand), controller=controller, seed=seed, simulator="sumo",
             wall_s=time.perf_counter() - t_wall)
    return m
