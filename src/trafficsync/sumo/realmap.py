"""Multi-agent signal control on real city maps in SUMO.

Every traffic light of a SUMO network becomes an agent.  The phase model is read from the map's own signal
program: its green phases (no ``y``) are the agent's phases, and switching from green ``p`` to green ``q`` goes
through a 3 s yellow state built from the two (``G``/``g`` in ``p`` and not green in ``q`` -> ``y``).

Agents observe only lane detectors on their own approaches (halting vehicles per incoming / exit lane) and the
messages they receive.  Controllers:

``fixed``       the map's own fixed-time program (what the city runs; the baseline)
``actuated``    TrafficSync switching rule on own queues: hold during min green, switch at max green, otherwise
                switch when another phase's queue beats the current one by ``theta``; gap-out on empty greens
``maxpressure`` local max-pressure with exit-lane detectors: ``sum (q_in - q_exit)`` over the phase's movements
``cocycle``     CoCycle: every agent estimates its Webster cycle from measured lane flows; a common cycle is
                agreed by dynamic average consensus over messages with neighbouring agents; green splits are
                local Webster splits of the common cycle; offsets follow formation-control consensus so that
                a downstream green starts one travel time after its upstream neighbour (agent 0 = leader);
                greens gap out on empty approaches and extend (at most ``max_extension_s`` per cycle)
``comp``        CoMP: max-pressure whose downstream term also uses the per-lane queue *reported by the downstream
                agent* for the approach the movement leads to (through the message bus: latency, drop
                probability, period); the larger of reported and local exit-lane queue is used, and movements
                into a downstream approach reported as nearly full are vetoed

The bundled maps (``sumo_scenarios/``) are the RESCO benchmark scenarios built from OpenStreetMap with real
travel demand: ``cologne8`` (8 signals, TAPASCologne, 7:00-8:00) and ``ingolstadt7`` (7 signals, InTAS,
16:00-17:00).  Any other ``.sumocfg`` works the same way, e.g. one made from OpenStreetMap with SUMO's
``osmWebWizard.py``.
"""
from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .common import start_traci

__all__ = ["RealMapConfig", "TLSAgent", "run_realmap_episode", "scenario_path", "SCENARIOS"]

SCENARIOS = {name: f"sumo_scenarios/{name}/{name}.sumocfg" for name in
             ("cologne3", "cologne8", "ingolstadt7", "ingolstadt21", "bologna_acosta", "bologna_pasubio")}


def tuned_params(scenario: str) -> dict:
    """Per-city parameters from ``configs/sumo_tuned.yaml`` (written by ``experiments/sumo_tune.py``)."""
    from ..config import project_root

    p = project_root() / "configs" / "sumo_tuned.yaml"
    if not p.exists():
        return {}
    import yaml

    return dict((yaml.safe_load(p.read_text()) or {}).get(Path(str(scenario)).stem, {}))


def scenario_path(name_or_path: str) -> Path:
    from ..config import project_root

    p = Path(SCENARIOS.get(name_or_path, name_or_path))
    if not p.suffix and (project_root() / "sumo_scenarios" / p / f"{p}.sumocfg").exists():   # imported city
        p = Path("sumo_scenarios") / p / f"{p}.sumocfg"
    return p if p.is_absolute() else project_root() / p


@dataclass
class RealMapConfig:
    g_min: int = 5
    g_max: int = 40
    yellow: int = 3
    decision_interval: int = 5
    theta: float = 3.0            # switching margin [veh per approach lane]; tuned on SUMO seed 100 over 5 maps:
    theta_per_lane: bool = True   # absolute 2 / 8 veh and 1 / 2 / 3 veh per lane tried (README, design decisions)
    gap_out: bool = True
    comm_period: int = 5
    comm_latency: int = 1
    comm_drop: float = 0.0
    veto_occupancy: float = 0.9   # comp: no release into an approach this full (reported)
    warmup_s: int = 600           # metrics from begin + warmup
    # cocycle
    c_min: float = 40.0
    c_max: float = 120.0
    sat_flow: float = 0.5         # veh/s per lane
    flow_window_s: float = 300.0  # EMA time constant of the measured lane flows
    max_extension_s: int = 40
    offsets: bool = True
    offset_hold_s: int = 10       # longest hold before the first phase to correct the offset
    cycle_quantum: float = 4.0


# --------------------------------------------------------------------------- agent
@dataclass
class TLSAgent:
    tls: str
    greens: list[str]                       # green state strings
    links: list[tuple[str, str] | None]     # signal index -> (in lane, out lane)
    phase: int = 0
    state: str = "green"                    # green | yellow
    t_in_state: int = 0
    target: int = 0
    in_lanes: list[str] = field(default_factory=list)
    out_lanes: list[str] = field(default_factory=list)
    moves: list[list[tuple[str, str]]] = field(default_factory=list)   # per phase
    downstream: dict = field(default_factory=dict)   # out lane -> (agent, its approach edge) reached downstream
    neighbours: set = field(default_factory=set)
    inbox: dict = field(default_factory=dict)        # agent -> (t_sent, {edge: (halting, occupancy)})

    def __post_init__(self) -> None:
        self.moves = [[mv for k, mv in enumerate(self.links) if mv and g[k] in "Gg"] for g in self.greens]
        self.in_lanes = sorted({mv[0] for mv in self.links if mv})
        self.lanes_per_phase = float(np.mean([len({m[0] for m in mv}) for mv in self.moves])) if self.moves else 1.0
        self.out_lanes = sorted({mv[1] for mv in self.links if mv})

    def yellow_state(self, a: int, b: int) -> str:
        ga, gb = self.greens[a], self.greens[b]
        return "".join("y" if ca in "Gg" and cb not in "Gg" else (ca if ca in "Gg" else "r") for ca, cb in zip(ga, gb))


def _greens(conn, tls: str) -> list[str]:
    logics = conn.trafficlight.getAllProgramLogics(tls)
    active = conn.trafficlight.getProgram(tls)
    logic = next((lg for lg in logics if lg.programID == active), logics[0])
    states = []
    for ph in logic.phases:
        s = ph.state
        if "y" in s.lower().replace("g", "") or not any(ch in "Gg" for ch in s):
            continue
        if s not in states:
            states.append(s)
    # drop transitional greens whose green set is contained in another green phase (e.g. a short
    # through-only clearance phase between two main phases): agents choose among the main phases and
    # transitions are generated as yellow states
    sets = [{k for k, ch in enumerate(s) if ch in "Gg"} for s in states]
    keep = [s for i, s in enumerate(states) if not any(i != j and sets[i] < sets[j] for j in range(len(states)))]
    return keep if len(keep) >= 2 else states


def _downstream_map(netfile: str, agents: dict[str, TLSAgent], max_hops: int = 6) -> None:
    """For each exit lane of an agent: the first downstream agent and its approach edge (road search)."""
    import sumolib

    net = sumolib.net.readNet(netfile)
    approach = {}
    for a in agents.values():
        for ln in a.in_lanes:
            approach[ln.rsplit("_", 1)[0]] = a.tls
    for a in agents.values():
        for ln in a.out_lanes:
            e0 = ln.rsplit("_", 1)[0]
            frontier, seen = deque([(net.getEdge(e0), 0)]), set()
            while frontier:
                e, h = frontier.popleft()
                if e.getID() in seen or h > max_hops:
                    continue
                seen.add(e.getID())
                if e.getID() in approach and approach[e.getID()] != a.tls:
                    a.downstream[ln] = (approach[e.getID()], e.getID())
                    a.neighbours.add(approach[e.getID()])
                    break
                for nxt in e.getOutgoing():
                    frontier.append((nxt, h + 1))


# --------------------------------------------------------------------------- CoCycle
class _CoCycle:
    """Per-agent CoCycle state on a real map (consensus cycle, formation offsets, local splits, actuation)."""

    def __init__(self, a: TLSAgent, cfg: RealMapConfig, travel: dict[str, float]) -> None:
        self.a, self.cfg = a, cfg
        self.flow = {ln: 0.0 for ln in a.in_lanes}         # EMA arrivals per lane [veh/s]
        self.ids = {ln: set() for ln in a.in_lanes}
        self.travel = travel                                # neighbour -> travel time from this agent [s]
        self.u_prev = np.zeros(2)
        self.x = np.zeros(2)                                # consensus state on (w C*, w)
        self.C = 60.0
        self.phi = 0.0
        self.ext_used = 0
        self.cycle_start = None

    def local(self) -> tuple[float, float, list[float]]:
        a, cfg = self.a, self.cfg
        y = []
        for moves in a.moves:
            lanes = {m[0] for m in moves}
            y.append(max((self.flow[ln] / cfg.sat_flow for ln in lanes), default=0.0))
        Y = min(0.9, sum(y))
        L = len(a.greens) * (cfg.yellow + 1)
        c = (1.5 * L + 5) / max(0.1, 1 - Y)
        return float(np.clip(c, cfg.c_min, cfg.c_max)), max(Y, 1e-3), y

    def observe(self, ids_by_lane: Mapping[str, Sequence[str]]) -> None:
        k = 1.0 / self.cfg.flow_window_s
        for ln in self.a.in_lanes:
            now = set(ids_by_lane.get(ln, ()))
            new = len(now - self.ids[ln])
            self.ids[ln] = now
            self.flow[ln] += k * (new - self.flow[ln])

    def message(self) -> dict:
        return {"x": self.x.tolist(), "phi": self.phi}

    def consensus(self, inbox: Mapping[str, tuple], n_max_deg: int, leader: bool) -> None:
        c_star, w, _ = self.local()
        u = np.array([w * c_star, w])
        if not self.x.any():
            self.x = u.copy()
        eps = 1.0 / (n_max_deg + 1)
        nb = [p for snd, (ts, p) in inbox.items() if "x" in p]
        self.x = self.x + eps * sum((np.array(p["x"]) - self.x for p in nb), np.zeros(2)) + (u - self.u_prev)
        self.u_prev = u
        z = self.x[0] / max(self.x[1], 1e-6)
        q = self.cfg.cycle_quantum * round(z / self.cfg.cycle_quantum)
        if abs(q - self.C) >= 3:
            self.C = float(np.clip(q, self.cfg.c_min, self.cfg.c_max))
        if self.cfg.offsets and not leader:
            corr = 0.0
            for snd, (ts, p) in inbox.items():
                if "phi" in p and snd in self.travel:
                    d = (p["phi"] + self.travel[snd]) - self.phi           # want phi_i = phi_j + tau_ji
                    corr += (d + self.C / 2) % self.C - self.C / 2
            self.phi = (self.phi + eps * max(-2.0, min(2.0, corr))) % self.C

    def splits(self) -> list[float]:
        cfg = self.cfg
        _, Y, y = self.local()
        L = len(self.a.greens) * (cfg.yellow + 1)
        return [max(cfg.g_min, (self.C - L) * yp / Y) for yp in y]

    def decide(self, t: int, hal: Mapping[str, float]) -> int | None:
        """Next phase to switch to, or None to hold."""
        a, cfg = self.a, self.cfg
        if a.phase == 0 and self.cycle_start is None:
            self.cycle_start = t
        g = self.splits()[a.phase]
        e = a.t_in_state
        own = sum(hal[ln] for ln in {m[0] for m in a.moves[a.phase]})
        others = sum(hal[ln] for p, mv in enumerate(a.moves) if p != a.phase for ln in {m[0] for m in mv})
        n = len(a.greens)
        demand = [sum(hal[ln] + self.flow[ln] * 10 for ln in {m[0] for m in a.moves[p]}) for p in range(n)]
        nxt = next(((a.phase + k) % n for k in range(1, n) if demand[(a.phase + k) % n] > 0.5), (a.phase + 1) % n)
        if e < cfg.g_min:
            return None
        if e >= cfg.g_max:
            return nxt
        if own == 0 and others > 0:                         # gap-out
            return nxt
        if e < g:
            return None
        if own > 0 and self.ext_used < cfg.max_extension_s:  # extension
            self.ext_used += 1
            return None
        if nxt == 0 and self.cfg.offsets:                   # align the next cycle start with the offset
            lag = ((t - self.phi) % self.C)
            if self.C - lag <= cfg.offset_hold_s and e < cfg.g_max:
                return None
        return nxt

    def started_phase(self, p: int, t: int) -> None:
        if p == 0:
            self.ext_used, self.cycle_start = 0, t


# --------------------------------------------------------------------------- episode
def run_realmap_episode(scenario: str, controller: str = "fixed", seed: int = 0, cfg: RealMapConfig | Mapping | None = None,
                        gui: bool = False, recorder: Any = None, work_dir: str | Path | None = None,
                        end_offset_s: int | None = None) -> dict:
    """Run one hour of a real-map scenario with one agent per traffic light; returns metrics."""
    import tempfile

    if not isinstance(cfg, RealMapConfig):
        cfg = RealMapConfig(**{**tuned_params(scenario), **dict(cfg or {})})
    sumocfg = scenario_path(scenario)
    root = ET.parse(sumocfg).getroot()
    begin = int(float(root.find("time/begin").get("value")))
    end = int(float(root.find("time/end").get("value")))
    if end_offset_s is not None:
        end = begin + int(end_offset_s)
    netfile = str(sumocfg.parent / root.find("input/net-file").get("value"))
    work = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="ts_map_"))
    work.mkdir(parents=True, exist_ok=True)
    trip = work / f"trips_{controller}_{seed}_{time.time_ns()}.xml"
    t_wall = time.perf_counter()
    conn = start_traci(["-c", str(sumocfg), "--seed", str(seed), "--end", str(end), "--tripinfo-output", str(trip),
                        "--no-step-log", "true", "--no-warnings", "true", "--time-to-teleport", "300",
                        "--step-length", "1"], gui)
    import traci.constants as tc

    agents: dict[str, TLSAgent] = {}
    for tls in conn.trafficlight.getIDList():
        links = [(c[0][0], c[0][1]) if c else None for c in conn.trafficlight.getControlledLinks(tls)]
        greens = _greens(conn, tls)
        if len(greens) >= 2:
            agents[tls] = TLSAgent(tls, greens, links)
    _downstream_map(netfile, agents)
    lanes = sorted({ln for a in agents.values() for ln in a.in_lanes + a.out_lanes})
    for ln in lanes:
        conn.lane.subscribe(ln, [tc.LAST_STEP_VEHICLE_HALTING_NUMBER, tc.LAST_STEP_OCCUPANCY, tc.LAST_STEP_VEHICLE_ID_LIST])
    cc: dict[str, _CoCycle] = {}
    if controller == "cocycle":
        import sumolib

        snet = sumolib.net.readNet(netfile)
        pos = {k: np.array(_agent_xy(snet, a), float) for k, a in agents.items()}
        sym = {k: set(a.neighbours) for k, a in agents.items()}
        for k, a in agents.items():
            for j in a.neighbours:
                sym[j].add(k)
        for k, a in agents.items():
            a.neighbours = sym[k]
            # travel time from neighbour j to k (straight-line distance at 50 km/h)
            cc[k] = _CoCycle(a, cfg, {j: float(np.hypot(*(pos[k] - pos[j]))) / 13.9 for j in sym[k]})
        max_deg = max((len(v) for v in sym.values()), default=1)
        leader = next(iter(agents), None)
    rng = np.random.default_rng(seed)
    pipe: deque = deque()       # (t_deliver, receiver, sender, t_sent, payload)
    n_msgs = n_dropped = 0
    if controller != "fixed":
        for a in agents.values():
            conn.trafficlight.setRedYellowGreenState(a.tls, a.greens[0])
    q_hist, switches = [], 0
    t = begin
    if recorder is not None:
        recorder.start(conn, netfile, agents)
    while t < end:
        conn.simulationStep()
        t += 1
        sub = conn.lane.getAllSubscriptionResults()
        hal = {ln: float(sub.get(ln, {}).get(tc.LAST_STEP_VEHICLE_HALTING_NUMBER, 0)) for ln in lanes}
        occ = {ln: float(sub.get(ln, {}).get(tc.LAST_STEP_OCCUPANCY, 0.0)) for ln in lanes}
        if cc:
            ids = {ln: sub.get(ln, {}).get(tc.LAST_STEP_VEHICLE_ID_LIST, ()) for ln in lanes}
            for st in cc.values():
                st.observe(ids)
        if t >= begin + cfg.warmup_s:
            q_hist.append(np.mean([hal[ln] for a in agents.values() for ln in a.in_lanes]))
        # messages: deliver, then send (each agent reports its approach edges)
        while pipe and pipe[0][0] <= t:
            _, rcv, snd, ts, payload = pipe.popleft()
            agents[rcv].inbox[snd] = (ts, payload)
        if controller == "comp" and (t - begin) % cfg.comm_period == 0:
            for a in agents.values():
                payload: dict[str, list] = {}
                for ln in a.in_lanes:
                    e = ln.rsplit("_", 1)[0]
                    p = payload.setdefault(e, [0.0, 0.0, 0])
                    p[0] += hal[ln]
                    p[1] = max(p[1], occ[ln])
                    p[2] += 1
                for b in agents.values():
                    if a.tls in b.neighbours:
                        n_msgs += 1
                        if rng.random() < cfg.comm_drop:
                            n_dropped += 1
                        else:
                            pipe.append((t + cfg.comm_latency, b.tls, a.tls, t, payload))
        if controller == "cocycle" and (t - begin) % cfg.comm_period == 0:
            for k, st in cc.items():
                st.consensus(agents[k].inbox, max_deg, k == leader)
            for k, st in cc.items():
                payload = st.message()
                for j in agents[k].neighbours:
                    n_msgs += 1
                    if rng.random() < cfg.comm_drop:
                        n_dropped += 1
                    else:
                        pipe.append((t + cfg.comm_latency, j, k, t, payload))
        if controller == "cocycle":
            for k, a in agents.items():
                a.t_in_state += 1
                if a.state == "yellow":
                    if a.t_in_state >= cfg.yellow:
                        a.state, a.phase, a.t_in_state = "green", a.target, 0
                        cc[k].started_phase(a.phase, t)
                        conn.trafficlight.setRedYellowGreenState(a.tls, a.greens[a.phase])
                    continue
                nxt = cc[k].decide(t, hal)
                if nxt is not None and nxt != a.phase:
                    a.target, a.state, a.t_in_state = nxt, "yellow", 0
                    switches += 1
                    conn.trafficlight.setRedYellowGreenState(a.tls, a.yellow_state(a.phase, nxt))
        elif controller != "fixed":
            for a in agents.values():
                a.t_in_state += 1
                if a.state == "yellow":
                    if a.t_in_state >= cfg.yellow:
                        a.state, a.phase, a.t_in_state = "green", a.target, 0
                        conn.trafficlight.setRedYellowGreenState(a.tls, a.greens[a.phase])
                    continue
                if (t - begin) % cfg.decision_interval:
                    continue
                score = _scores(a, controller, hal, occ, t, cfg)
                cur = a.phase
                others = [p for p in range(len(a.greens)) if p != cur]
                best = max(others, key=lambda p: score[p])
                switch = False
                if a.t_in_state >= cfg.g_max:
                    switch = True
                elif a.t_in_state >= cfg.g_min:
                    own = sum(hal[m[0]] for m in a.moves[cur])
                    th = cfg.theta * (a.lanes_per_phase if cfg.theta_per_lane else 1.0)
                    if score[best] - score[cur] > th:
                        switch = True
                    elif cfg.gap_out and own == 0 and score[best] > 0:
                        switch = True
                if switch:
                    a.target, a.state, a.t_in_state = best, "yellow", 0
                    switches += 1
                    conn.trafficlight.setRedYellowGreenState(a.tls, a.yellow_state(cur, best))
        if recorder is not None:
            recorder.step(conn, t)
    conn.close()
    rows = []
    for tr in ET.parse(trip).getroot().iter("tripinfo"):
        if float(tr.get("depart")) - float(tr.get("departDelay")) >= begin + cfg.warmup_s:
            rows.append((float(tr.get("timeLoss")) + float(tr.get("departDelay")),
                         float(tr.get("waitingTime")), int(tr.get("waitingCount")), float(tr.get("duration"))))
    arr = np.array(rows) if rows else np.zeros((0, 4))
    hours = (end - begin - cfg.warmup_s) / 3600
    extra = {"mean_cycle_s": float(np.mean([st.C for st in cc.values()])),
             "cycle_spread_s": float(np.ptp([st.C for st in cc.values()]))} if cc else {}
    return {**extra, "scenario": scenario, "controller": controller, "seed": seed, "agents": len(agents),
            "mean_delay": float(arr[:, 0].mean()) if len(arr) else float("nan"),
            "mean_waiting": float(arr[:, 1].mean()) if len(arr) else float("nan"),
            "mean_stops": float(arr[:, 2].mean()) if len(arr) else float("nan"),
            "mean_travel_time": float(arr[:, 3].mean()) if len(arr) else float("nan"),
            "completed": len(arr), "throughput_vph": len(arr) / hours if hours > 0 else float("nan"),
            "mean_queue_per_lane": float(np.mean(q_hist)) if q_hist else float("nan"),
            "switches_per_agent_h": switches / max(1, len(agents)) / max(hours, 1e-9),
            "msgs": n_msgs, "msgs_dropped": n_dropped, "wall_s": time.perf_counter() - t_wall}


def _agent_xy(snet, a: TLSAgent) -> tuple[float, float]:
    if snet.hasNode(a.tls):
        return snet.getNode(a.tls).getCoord()
    pts = [snet.getLane(ln).getShape()[-1] for ln in a.in_lanes]
    return tuple(np.mean(pts, axis=0))


def _scores(a: TLSAgent, controller: str, hal: Mapping[str, float], occ: Mapping[str, float], t: int,
            cfg: RealMapConfig) -> list[float]:
    out = []
    for moves in a.moves:
        s = 0.0
        for ln_in, ln_out in moves:
            q_in = hal[ln_in]
            if controller == "actuated":
                s += q_in
            elif controller == "maxpressure":
                s += q_in - hal.get(ln_out, 0.0)
            elif controller == "comp":
                down = a.downstream.get(ln_out)
                q_dn, veto = hal.get(ln_out, 0.0), False      # fallback: own exit-lane detector
                if down is not None and down[0] in a.inbox:
                    ts, payload = a.inbox[down[0]]
                    if down[1] in payload:
                        hq, o, nl = payload[down[1]]
                        q_dn = max(q_dn, hq / max(1, nl))    # per-lane queue reported downstream
                        veto = o >= 100 * cfg.veto_occupancy
                s += 0.0 if veto else q_in - q_dn
            else:
                raise ValueError(f"unknown controller {controller!r} (fixed, actuated, maxpressure, comp, cocycle)")
        # count each incoming lane once per phase for the actuated score
        if controller == "actuated":
            s = sum(hal[ln] for ln in {m[0] for m in moves})
        out.append(s)
    return out
