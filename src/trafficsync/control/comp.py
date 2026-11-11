"""C2a - neighbour-informed max-pressure, CoMP.

Phase pressure of agent ``i``::

    P_i(p) = gamma_i * sum_{(l,m) in M_p} s_l * r_lm * ( q_hat_l + kappa * a_hat_l - q_tilde_m )

* ``q_tilde_m`` - queue on the downstream link ``m``, reported by the neighbour that
  owns ``m`` and extrapolated over the message age ``alpha``:
  ``q_tilde_m = min(C_m, q_hat_m + lambda_m * alpha)`` (same for the occupancy ``n_tilde_m``).
* ``a_hat_l``  - vehicles the upstream neighbour announced as departed onto ``l`` that
  reach the stop line within ``H_a = 15 s`` (platoon anticipation).  Agents estimate
  their own departures from their signal state and queue estimates
  (:class:`DepartureLog`); nobody reads the simulator.
* spillback veto - a movement contributes nothing when ``n_tilde_m >= C_m - delta_room``.
* ``gamma_i = clip(1 + beta_c (rho_bar_i - rho_i), gamma_min, gamma_max)`` with
  ``rho_bar_i`` from load consensus.  ``gamma_i`` multiplies every phase of agent ``i``,
  so it changes no argmax; what it does change is the switching margin: an
  intersection more loaded than the network average (``gamma < 1``) needs a larger
  pressure difference to switch, i.e. keeps greens longer and loses less time.

Timing rules are the shared ones (:func:`~trafficsync.control.base.select_phase`),
identical to C1 for fairness.  Gap-out tests the local queue of the served approaches,
never the pressure: a negative pressure (longer queue downstream) does not mean there is
nothing to serve.  Information older than ``max_info_age_s`` is dropped, so with total
message loss CoMP falls back to local max-pressure.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..comm.messages import Message
from ..sim.observation import GREEN, Observation
from .base import AgentView, Controller, local_demand, local_load, select_phase

__all__ = [
    "NeighbourInfo",
    "NeighbourBoard",
    "DepartureLog",
    "CoMPCore",
    "LoadConsensus",
    "CoMPController",
    "merge_payloads",
]

_NEVER = -(10 ** 9)


# ----------------------------------------------------------------------------
# What an agent has heard from its neighbours
# ----------------------------------------------------------------------------
@dataclass
class NeighbourInfo:
    """Newest values received from one neighbour (each field keeps its own send time)."""

    t_data: int = _NEVER
    queues: dict = field(default_factory=dict)
    occupancy: dict = field(default_factory=dict)
    data_rate: dict = field(default_factory=dict)
    departures: list = field(default_factory=list)
    phase: int = -1
    t_rate: int = _NEVER
    arrival_rate: dict = field(default_factory=dict)
    t_load: int = _NEVER
    load: float | None = None
    cons: dict = field(default_factory=dict)      # named consensus channels: {name: value}
    t_cons: dict = field(default_factory=dict)
    t_offset: int = _NEVER
    phi: float | None = None
    psi: dict = field(default_factory=dict)
    cycle: float | None = None


class NeighbourBoard:
    """Stores the newest message content per neighbour; tracks what is new this step."""

    def __init__(self) -> None:
        self.info: dict[int, NeighbourInfo] = {}
        self.fresh: dict[str, set[int]] = {}
        self.fresh_offset: set[int] = set()

    def _put(self, inf: NeighbourInfo, sender: int, name: str, value: float, t_sent: int) -> None:
        if t_sent >= inf.t_cons.get(name, _NEVER):
            inf.cons[name] = float(value)
            inf.t_cons[name] = t_sent
            self.fresh.setdefault(name, set()).add(sender)

    def receive(self, msgs: Sequence[Message]) -> None:
        for m in msgs:
            inf = self.info.get(m.sender)
            if inf is None:
                inf = self.info[m.sender] = NeighbourInfo()
            if (m.queues or m.departures) and m.t_sent >= inf.t_data:
                inf.t_data = m.t_sent
                inf.queues = dict(m.queues)
                inf.occupancy = dict(m.occupancy)
                inf.data_rate = dict(m.arrival_rate)
                inf.departures = list(m.departures)
                inf.phase = m.phase
            if m.arrival_rate and m.t_sent >= inf.t_rate:
                inf.t_rate = m.t_sent
                inf.arrival_rate = dict(m.arrival_rate)
            if m.load_value is not None:
                if m.t_sent >= inf.t_load:
                    inf.t_load = m.t_sent
                    inf.load = float(m.load_value)
                self._put(inf, m.sender, "rho", m.load_value, m.t_sent)
            for name, val in (m.extra.get("cons") or {}).items():
                self._put(inf, m.sender, name, val, m.t_sent)
            if m.cycle_offset is not None and m.t_sent >= inf.t_offset:
                inf.t_offset = m.t_sent
                inf.phi = float(m.cycle_offset)
                inf.psi = dict(m.extra.get("psi", {}))
                inf.cycle = m.extra.get("C")
                self.fresh_offset.add(m.sender)

    def pop_fresh(self, channel: str = "rho") -> set[int]:
        return self.fresh.pop(channel, set())

    def pop_fresh_load(self) -> set[int]:
        return self.pop_fresh("rho")

    def pop_fresh_offset(self) -> set[int]:
        out, self.fresh_offset = self.fresh_offset, set()
        return out


def merge_payloads(dst: dict[int, dict[str, Any]], src: Mapping[int, Mapping[str, Any]]) -> None:
    """Merge per-receiver message fields (one message per neighbour and step)."""
    for j, p in src.items():
        d = dst.setdefault(j, {})
        for k, v in p.items():
            if k == "extra":
                d.setdefault("extra", {}).update(v)
            elif k == "arrival_rate":
                d.setdefault("arrival_rate", {}).update(v)
            else:
                d[k] = v


# ----------------------------------------------------------------------------
# Own departure estimates (for the upstream announcement)
# ----------------------------------------------------------------------------
class DepartureLog:
    """Estimated departures onto each outgoing internal link, kept for ``tau`` steps."""

    def __init__(self, view: AgentView) -> None:
        self.links = [m for m in view.outgoing if view.downstream_node.get(m) is not None]
        self.tau = {m: int(view.links[m].tau) for m in self.links}
        self.log: dict[int, deque[tuple[int, float]]] = {m: deque() for m in self.links}
        self.total = 0.0     # all departures estimated so far (diagnostics)

    def add(self, t: int, deps: Mapping[int, float]) -> None:
        for m, c in deps.items():
            if c > 1e-9:
                self.log[m].append((int(t), float(c)))
                self.total += float(c)
        for m, dq in self.log.items():
            while dq and dq[0][0] < t - self.tau[m] - 1:
                dq.popleft()

    def export(self, m: int, t: int) -> list[list[float]]:
        """Departures still travelling on ``m`` at time ``t``: ``[[t_d, count], ...]``."""
        return [[td, c] for td, c in self.log[m] if td >= t - self.tau[m]]


# ----------------------------------------------------------------------------
# Pressure computation shared by CoMP and CoMP+Cycle
# ----------------------------------------------------------------------------
class CoMPCore:
    """Neighbour-informed pressure, departure estimates and the data messages of CoMP."""

    def __init__(self, view: AgentView, cfg: Mapping[str, Any]) -> None:
        self.view = view
        self.kappa = float(cfg.get("kappa", 0.5))
        self.H_a = int(cfg.get("anticipation_horizon_s", 15))
        self.delta_room = float(cfg.get("delta_room", 2))
        self.use_down = bool(cfg.get("downstream", True))
        self.use_ant = bool(cfg.get("anticipation", True)) and self.kappa != 0.0
        self.use_veto = bool(cfg.get("veto", True))
        self.max_age = float(cfg.get("max_info_age_s", 30))
        self.policy = str(cfg.get("missing_policy", "last"))
        self.period = int(cfg.get("comm_period_s", 5))
        self.timing = view.timing
        self.phase_movs = [[(m.in_link, m.out_link, float(view.turning_ratio[(m.in_link, m.out_link)]))
                            for m in ph] for ph in view.phases]
        self.served_links = [sorted({l for l, _, _ in movs}) for movs in self.phase_movs]
        self.up_of = {l: view.upstream_node[l] for l in view.incoming if view.upstream_node.get(l) is not None}
        self.down_of = {m: view.downstream_node[m] for m in view.outgoing
                        if view.downstream_node.get(m) is not None}
        self.in_from = {}   # neighbour -> my incoming links from it
        for l, u in self.up_of.items():
            self.in_from.setdefault(u, []).append(l)
        self.out_to = {}    # neighbour -> my outgoing links to it
        for m, j in self.down_of.items():
            self.out_to.setdefault(j, []).append(m)
        self.road_neighbors = sorted(set(self.in_from) | set(self.out_to))
        self.reset()

    def reset(self) -> None:
        self.q_track: dict[int, float] = {}
        self.t_meas: dict[int, int] = {}
        self.deplog = DepartureLog(self.view)

    # -- local tracking ------------------------------------------------------
    def observe(self, obs: Observation) -> None:
        """Track each incoming queue between estimate updates (adds estimated arrivals)."""
        for l, e in obs.incoming.items():
            if self.t_meas.get(l) != e.t_measured:
                self.t_meas[l] = e.t_measured
                self.q_track[l] = float(e.q_hat)
            else:
                self.q_track[l] = self.q_track.get(l, float(e.q_hat)) + max(0.0, float(e.lambda_hat))

    def record_departures(self, obs: Observation, action: int, t: int) -> None:
        """Estimate this step's discharge onto every outgoing link (signal state + queues)."""
        tm = self.timing
        e = int(obs.elapsed)
        discharging = (obs.signal_state == GREEN and e >= tm.startup_lost and e < tm.g_max
                       and not (action != obs.phase and e >= tm.g_min))
        deps: dict[int, float] = {}
        if discharging:
            per_link = {}
            for l in self.served_links[obs.phase]:
                est = obs.incoming[l]
                s = self.view.links[l].sat_flow
                q = self.q_track.get(l, float(est.q_hat))
                d = s if q >= 1.0 else min(s, max(0.0, float(est.lambda_hat)))
                self.q_track[l] = max(0.0, q - d)
                per_link[l] = d
            for l, m, r in self.phase_movs[obs.phase]:
                if m in self.down_of:
                    deps[m] = deps.get(m, 0.0) + r * per_link[l]
        self.deplog.add(t, deps)

    # -- neighbour information -----------------------------------------------
    def _usable(self, age: float) -> float:
        """Weight of information of a given age (0 = unusable)."""
        if age > self.max_age:
            return 0.0
        if self.policy == "ignore" and age > 2 * self.period:
            return 0.0
        if self.policy == "age_discounted":
            return max(0.0, 1.0 - age / self.max_age)
        return 1.0

    def downstream(self, board: NeighbourBoard, m: int, t: int) -> tuple[float, float, float] | None:
        """``(q_tilde_m, n_tilde_m, weight)`` from the owner of ``m`` or ``None``."""
        j = self.down_of.get(m)
        if j is None:
            return None
        inf = board.info.get(j)
        if inf is None or m not in inf.queues:
            return None
        age = float(t - inf.t_data)
        w = self._usable(age)
        if w <= 0.0:
            return None
        C = float(self.view.links[m].capacity)
        lam = max(0.0, float(inf.data_rate.get(m, 0.0)))
        q_t = min(C, float(inf.queues[m]) + lam * age)
        n_t = min(C, float(inf.occupancy.get(m, 0.0)) * C + lam * age)
        return q_t, n_t, w

    def anticipated(self, board: NeighbourBoard, l: int, t: int) -> float:
        """Announced upstream departures reaching the stop line of ``l`` in ``(t, t + H_a]``."""
        u = self.up_of.get(l)
        if u is None or not self.use_ant:
            return 0.0
        inf = board.info.get(u)
        if inf is None or self._usable(float(t - inf.t_data)) <= 0.0:
            return 0.0
        tau = self.view.links[l].tau
        return float(sum(c for td, c in inf.departures if t < td + tau <= t + self.H_a))

    def pressures(self, obs: Observation, board: NeighbourBoard, t: int, gamma: float = 1.0) -> list[float]:
        out = []
        for movs in self.phase_movs:
            P = 0.0
            for l, m, r in movs:
                e = obs.incoming[l]
                up = float(e.q_hat)
                if self.use_ant:
                    up += self.kappa * self.anticipated(board, l, t)
                down = 0.0
                if self.use_down or self.use_veto:
                    ds = self.downstream(board, m, t)
                    if ds is not None:
                        q_t, n_t, w = ds
                        if self.use_veto and n_t >= float(self.view.links[m].capacity) - self.delta_room:
                            continue
                        if self.use_down:
                            down = w * q_t
                P += self.view.links[l].sat_flow * r * (up - down)
            out.append(gamma * P)
        return out

    def data_payloads(self, obs: Observation, t: int) -> dict[int, dict[str, Any]]:
        """CoMP data for every road neighbour: queues/occupancy/rate of the links it feeds me,
        and my announced departures towards it."""
        out: dict[int, dict[str, Any]] = {}
        for j in self.road_neighbors:
            p: dict[str, Any] = {"phase": int(obs.phase)}
            ls = self.in_from.get(j, [])
            if ls:
                inc = obs.incoming
                p["queues"] = {l: inc[l].q_hat for l in ls}
                p["occupancy"] = {l: inc[l].rho_hat for l in ls}
                p["arrival_rate"] = {l: inc[l].lambda_hat for l in ls}
            deps = []
            for m in self.out_to.get(j, []):
                deps.extend(self.deplog.export(m, t))
            p["departures"] = deps
            out[j] = p
        return out


# ----------------------------------------------------------------------------
# Average consensus on the load index (gamma in CoMP, cycle length in CoCycle)
# ----------------------------------------------------------------------------
class LoadConsensus:
    """Distributed (dynamic) average consensus on ``rho_i`` with one exchange per round.

    ``K_c`` rounds per message period ``T_c``: rounds happen every ``max(1, T_c // K_c)``
    steps in the first part of each period (with ``T_c = 5``, ``K_c = 5``: every step).
    ``dynamic`` (default): ``x_i <- sum_j w_ij x_j + (rho_i[k] - rho_i[k-1])``, which keeps
    ``sum x = sum rho`` when no message is lost.  ``static``: the restarted
    consensus, ``x_i = rho_i`` at the start of every period, read just before the restart.
    ``K_c = 0`` disables consensus (estimate = own load).
    """

    def __init__(self, view: AgentView, cfg: Mapping[str, Any] | None, period: int,
                 missing_policy: str = "last", channel: str = "rho") -> None:
        cfg = dict(cfg or {})
        self.channel = channel
        self.nbrs = list(view.neighbors)
        kind = str(cfg.get("weights", "laplacian"))
        if kind == "metropolis":
            di = int(view.meta.get("degree", len(self.nbrs)))
            nd = view.meta.get("neighbor_degree", {})
            self.w = {j: 1.0 / (1.0 + max(di, int(nd.get(j, di)))) for j in self.nbrs}
        elif kind == "laplacian":
            eps = cfg.get("epsilon", "auto")
            e = 1.0 / (float(view.meta.get("d_max", len(self.nbrs))) + 1.0) if eps in (None, "auto") else float(eps)
            self.w = {j: e for j in self.nbrs}
        else:
            raise ValueError(f"unknown consensus weights {kind!r}")
        self.w_self = 1.0 - sum(self.w.values())
        self.K = int(cfg.get("iterations_per_decision", 5))
        self.mode = str(cfg.get("mode", "dynamic"))
        if self.mode not in ("dynamic", "static"):
            raise ValueError("consensus mode must be dynamic or static")
        self.period = max(1, int(period))
        self.interval = max(1, self.period // max(self.K, 1))
        self.policy = missing_policy
        self.x: float | None = None
        self.estimate = 0.0
        self.rho_prev = 0.0
        self.last: dict[int, float] = {}

    def exchange_step(self, t: int) -> bool:
        if self.K <= 0 or not self.nbrs:
            return False
        r = t % self.period
        return r % self.interval == 0 and r < self.K * self.interval

    def update(self, t: int, board: NeighbourBoard, rho: float) -> float:
        fresh = board.pop_fresh(self.channel)
        if self.K <= 0:
            self.estimate = rho
            return rho
        if self.x is None:
            self.x, self.rho_prev, self.estimate = rho, rho, rho
            return rho
        for j in fresh:
            v = board.info[j].cons.get(self.channel)
            if v is not None:
                self.last[j] = v
        if fresh:
            acc = self.w_self * self.x
            for j, w in self.w.items():
                if j in fresh or (self.policy == "last" and j in self.last):
                    acc += w * self.last[j]
                else:
                    acc += w * self.x
            self.x = acc
        if self.mode == "dynamic":
            self.x += rho - self.rho_prev
            self.estimate = self.x
        elif t % self.period == 0:
            self.estimate = self.x      # value agreed during the last period ...
            self.x = rho                # ... then restart from the current load
        self.rho_prev = rho
        return self.estimate


# ----------------------------------------------------------------------------
# The controller
# ----------------------------------------------------------------------------
class CoMPController(Controller):
    """C2a.  ``cfg``: ``theta`` [pressure units], ``gap_out``, ``kappa``, ``anticipation_horizon_s``,
    ``delta_room``, ``downstream``, ``anticipation``, ``veto``, ``max_info_age_s``,
    ``gamma: {enabled, beta, min, max}``, ``consensus: {weights, epsilon, iterations_per_decision, mode,
    aggregation (average | wmsr), F}``, ``priority: {agents, gamma}`` (E11 / H6),
    plus ``comm_period_s`` / ``missing_policy`` from the communication model."""

    name = "comp"

    def reset(self) -> None:
        super().reset()
        cfg = self.cfg
        self.theta = float(cfg.get("theta", 0.0))
        self.gap_out = bool(cfg.get("gap_out", False))
        self.period = int(cfg.get("comm_period_s", 5))
        self.core = CoMPCore(self.view, cfg)
        self.board = NeighbourBoard()
        g = dict(cfg.get("gamma", {}) or {})
        self.use_gamma = bool(g.get("enabled", True))
        self.beta = float(g.get("beta", 1.0))
        self.g_lo, self.g_hi = float(g.get("min", 0.5)), float(g.get("max", 1.5))
        ccfg = dict(cfg.get("consensus", {}) or {})
        from .wmsr import make_consensus   # aggregation: average (LoadConsensus) | wmsr (E9)

        self.cons = (make_consensus(self.view, ccfg, self.period, str(cfg.get("missing_policy", "last")))
                     if self.use_gamma else None)
        # H6 / E11 'vaccination': a fixed extra priority weight for listed (central) agents
        pr = dict(cfg.get("priority", {}) or {})
        self.priority = float(pr.get("gamma", 1.0)) if self.agent_id in set(pr.get("agents", []) or []) else 1.0
        self.gamma = 1.0
        self.rho = 0.0

    def receive(self, inbox: Sequence[Message], t: int) -> None:
        self.board.receive(inbox)

    def before_decision(self, obs: Observation, t: int) -> None:
        self.core.observe(obs)
        if self.cons is not None:
            self.rho = local_load(self.view, obs)
            est = self.cons.update(t, self.board, self.rho)
            self.gamma = min(self.g_hi, max(self.g_lo, 1.0 + self.beta * (est - self.rho)))

    def pressures(self, obs: Observation, t: int) -> list[float]:
        return self.core.pressures(obs, self.board, t, self.gamma * self.priority)

    def decide(self, obs: Observation, t: int) -> int:
        demand = local_demand(self.view, obs) if self.gap_out else None
        return select_phase(obs, self.pressures(obs, t), self.view.timing, self.theta, self.gap_out, demand)

    def after_action(self, obs: Observation, action: int, t: int) -> None:
        self.core.record_departures(obs, action, t)

    def outbox(self, obs: Observation, t: int) -> list[Message]:
        payload: dict[int, dict[str, Any]] = {}
        if self.cons is not None and self.cons.exchange_step(t):
            merge_payloads(payload, {j: {"load_value": float(self.cons.x if self.cons.x is not None else self.rho)}
                                     for j in self.view.neighbors})
        if t % self.period == 0:
            nb = set(self.view.neighbors)
            merge_payloads(payload, {j: p for j, p in self.core.data_payloads(obs, t).items() if j in nb})
        return [Message(sender=self.agent_id, receiver=j, t_sent=t, **p) for j, p in sorted(payload.items())]
