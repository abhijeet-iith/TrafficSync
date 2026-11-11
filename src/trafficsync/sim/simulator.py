"""Queue-based, vehicle-level simulator.

One step (1 s), in this order:

1. signals apply the controllers' requests (min/max green, yellow, all-red);
2. travelling vehicles that reach the stop line join the FIFO queue of their link
   (on an exit link they leave the network: travel time recorded);
3. new vehicles enter their entry link (a full entry link keeps them in a boundary
   point queue - counted as in-network, their wait counts as delay);
4. downstream space ``C_m - n_m`` is measured once (order-independent spillback);
5. discharge: an effectively green link earns ``s_l * incident_factor`` credit per second;
   the head vehicle moves when credit >= 1 and its next link has space, otherwise the
   lane is blocked (spillback event).  An empty stop line on green banks at most one
   vehicle of credit (``sim/link.py``), so a lone arrival passes without waiting while a
   standing queue discharges at exactly ``s_l``;
6. every queued / buffered vehicle waits one more second (stops counted on the first
   second of each wait); metrics and probe reports are recorded.

``true_state()`` is for metrics, tests and visualisation only - never for agents.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Mapping

import numpy as np

from ..config import ProbeConfig, SimConfig
from ..network.graph import Network
from .demand import Demand, DemandScenario
from .intersection import SignalStateMachine, build_phases
from .link import CREDIT_BANK, LinkState
from .vehicle import Vehicle
from .metrics_logger import MetricsLogger
from .observation import Observation
from .probe import ProbeModel

__all__ = ["Simulator", "Vehicle", "LinkState"]


class Simulator:
    def __init__(self, network: Network, demand_cfg: Any = "D_med", probe_cfg: ProbeConfig | Mapping | None = None,
                 seed: int = 0, sim_cfg: SimConfig | Mapping | None = None) -> None:
        self.network = net = network
        self.cfg = sim_cfg if isinstance(sim_cfg, SimConfig) else SimConfig.from_dict(sim_cfg)
        self.probe_cfg = probe_cfg if isinstance(probe_cfg, ProbeConfig) else ProbeConfig.from_dict(probe_cfg)
        self.seed = int(seed)
        self.scenario = DemandScenario.get(demand_cfg)
        self.duration = int(self.cfg.duration_s)
        self.agents = list(range(net.N))
        self.phases = [build_phases(net, i, self.cfg.phase_scheme, self.cfg.drive_side) for i in range(net.N)]
        self.demand = Demand(net, self.scenario, self.duration, self.cfg.turning, self.seed, self.probe_cfg.penetration)
        self.qlinks = list(net.queue_links)
        self.qcol = {l: k for k, l in enumerate(self.qlinks)}
        self.tau = {l: net.links[l].tau for l in net.links}
        self.cap = {l: net.links[l].capacity for l in net.links}
        self.reset()

    # ------------------------------------------------------------------ API
    def reset(self) -> dict[int, Observation]:
        net = self.network
        self.t = 0
        self.ls = {l: LinkState() for l in net.links}
        self.buffers = {e: deque() for e in net.entry_links}
        self.signals = [SignalStateMachine(i, self.phases[i], self.cfg.signal, self.cfg.initial_phase) for i in self.agents]
        self.vehicles: dict[int, Vehicle] = {}
        self.arrivals = self.demand.sample()
        self.n_generated = 0
        self.n_completed = 0
        self.probe = ProbeModel(self.probe_cfg, net.links, self.qlinks, self.seed)
        self.logger = MetricsLogger(self.duration, self.qlinks, net.N, self.cfg.warmup_s)
        self._phase_links = [[sorted({m.in_link for m in ph}) for ph in self.phases[i]] for i in self.agents]
        self._probe_update()
        return self._observations()

    @property
    def done(self) -> bool:
        return self.t >= self.duration

    def step(self, actions: Mapping[int, int | None] | None = None) -> tuple[dict[int, Observation], dict]:
        if self.done:
            raise RuntimeError("episode finished; call reset()")
        t = self.t
        actions = actions or {}
        net = self.network
        for i, sm in enumerate(self.signals):
            sm.update(actions.get(i))
        # 2. stop-line arrivals / exits
        done_now = 0
        for l, st in self.ls.items():
            tr = st.travel
            if not tr:
                continue
            exit_link = net.links[l].to_node is None
            while tr and tr[0][0] <= t:
                _, vid = tr.popleft()
                if exit_link:
                    v = self.vehicles[vid]
                    v.t_done = t
                    self.n_completed += 1
                    done_now += 1
                else:
                    st.queue.append(vid)
        # 3. new vehicles
        gen_now = 0
        for e, route, is_probe in self.arrivals.get(t, ()):
            vid = len(self.vehicles)
            ff = sum(self.tau[l] for l in route)
            self.vehicles[vid] = Vehicle(vid, route, t, is_probe, ff)
            self.buffers[e].append(vid)
            gen_now += 1
        self.n_generated += gen_now
        blocked = 0
        for e, buf in self.buffers.items():
            st = self.ls[e]
            while buf and st.n < self.cap[e]:
                vid = buf.popleft()
                self.vehicles[vid].waiting = False
                st.travel.append((t + self.tau[e], vid))
                self.vehicles[vid].t_link_enter = t
                self.probe.on_enter(e, self.vehicles[vid].is_probe)
            if buf:
                blocked += 1
        # 4. space snapshot
        space = {m: self.cap[m] - self.ls[m].n for m in net.internal_links}
        # 5. discharge
        spill = 0
        for i, sm in enumerate(self.signals):
            eff = sm.effective_green()
            for l in net.incoming_links(i):
                st = self.ls[l]
                outs = sm.served_out(l) if eff else ()
                if not outs:
                    st.credit = 0.0
                    continue
                gain = net.links[l].sat_flow * self.demand.capacity_factor(l, t)
                if not st.queue:            # empty stop line on green: bank at most one vehicle
                    st.credit = min(CREDIT_BANK, st.credit + gain)
                    continue
                st.credit += gain
                while st.credit >= 1.0 - 1e-12 and st.queue:
                    v = self.vehicles[st.queue[0]]
                    m = v.route[v.idx + 1]
                    if m not in outs:
                        break
                    if m in space and space[m] <= 0:
                        spill += 1
                        st.credit = min(st.credit, 1.0)
                        break
                    st.queue.popleft()
                    st.credit -= 1.0
                    self.logger.record_discharge_wait(t, l, v.link_wait)
                    v.idx += 1
                    v.waiting = False
                    v.link_wait = 0
                    self.ls[m].travel.append((t + self.tau[m], v.id))
                    v.t_link_enter = t
                    if m in space:
                        space[m] -= 1
                    self.probe.on_enter(m, v.is_probe)
        # 6. waiting, logging, probes
        V = self.vehicles
        for st in self.ls.values():
            for vid in st.queue:
                v = V[vid]
                v.wait += 1
                v.link_wait += 1
                if not v.waiting:
                    v.waiting = True
                    v.stops += 1
        for buf in self.buffers.values():
            for vid in buf:
                v = V[vid]
                v.wait += 1
                if not v.waiting:
                    v.waiting = True
                    v.stops += 1
        q = np.array([len(self.ls[l].queue) for l in self.qlinks], dtype=np.float32)
        n = np.array([self.ls[l].n for l in self.qlinks], dtype=np.float32)
        self.logger.record_step(t, q, n, gen_now, done_now, spill, blocked)
        for sm in self.signals:
            sm.advance()
        lg = self.logger
        if lg.tsg is None:
            lg.tsg = np.zeros((self.duration, len(self.signals), max(s.P for s in self.signals)), dtype=np.int16)
        for i, sm in enumerate(self.signals):
            lg.tsg[t, i, : sm.P] = sm.tsg
            if t >= self.cfg.warmup_s:
                lg.max_red = max(lg.max_red, max(sm.tsg))
        self.t += 1
        self._probe_update()
        info = {"t": t, "generated": gen_now, "completed": done_now, "spillback": spill}
        return self._observations(), info

    # -------------------------------------------------------------- internals
    def _probe_update(self) -> None:
        V = self.vehicles
        state = {}
        for l in self.qlinks:
            st = self.ls[l]
            pq = sum(1 for vid in st.queue if V[vid].is_probe) if self.probe_cfg.mode == "probe" else 0
            pt = sum(1 for _, vid in st.travel if V[vid].is_probe) if self.probe_cfg.mode == "probe" else 0
            state[l] = (len(st.queue), st.n, pq, pt)
        self.probe.update(self.t, state)

    def _observations(self) -> dict[int, Observation]:
        est = self.probe.est
        out = {}
        for i, sm in enumerate(self.signals):
            inc = {l: est[l] for l in self.network.incoming_links(i)}
            out[i] = Observation(i, self.t, sm.phase, sm.state, sm.t_in_state, sm.next_phase, sm.P, inc, {})
        return out

    def in_network(self) -> int:
        return sum(st.n for l, st in self.ls.items()) + sum(len(b) for b in self.buffers.values())

    def true_state(self) -> dict:
        """Ground truth (metrics / tests / visualisation only)."""
        links = {l: {"q": len(st.queue), "n": st.n, "capacity": self.cap[l]} for l, st in self.ls.items()}
        sig = {i: {"phase": sm.phase, "state": sm.state, "t_in_state": sm.t_in_state} for i, sm in enumerate(self.signals)}
        return {"t": self.t, "links": links, "signals": sig, "n_in_network": self.in_network(),
                "generated": self.n_generated, "completed": self.n_completed,
                "buffers": {e: len(b) for e, b in self.buffers.items()}}

    def node_loads(self) -> np.ndarray:
        """True ``rho_i = mean_{l in I_i} n_l / C_l`` (analysis, e.g. E11)."""
        net = self.network
        return np.array([np.mean([self.ls[l].n / self.cap[l] for l in net.incoming_links(i)]) for i in self.agents])

    def stl_trace(self) -> dict:
        """Signals for the STL monitor: ``n_<l>``, ``q_<l>`` (queue links), ``tsg_<i>_<p>``."""
        lg = self.logger
        tr = {}
        for k, l in enumerate(self.qlinks):
            tr[f"n_{l}"] = lg.n[:, k]
            tr[f"q_{l}"] = lg.q[:, k]
        for i, sm in enumerate(self.signals):
            for p in range(sm.P):
                tr[f"tsg_{i}_{p}"] = lg.tsg[:, i, p]
        return tr

    def metrics(self) -> dict:
        caps = np.array([self.cap[l] if self.network.links[l].kind == "internal" else 0 for l in self.qlinks], float)
        return self.logger.metrics(self.vehicles, self._phase_links, caps, float(self.cfg.stl.get("delta_room", 2)),
                                   self.cfg.gridlock_window_s, self.cfg.gridlock_ratio, self.cfg.travel_time_rule)
