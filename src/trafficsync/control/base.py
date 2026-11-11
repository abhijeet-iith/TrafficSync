"""Controller interface and the static information an agent may use.

Every controller implements::

    act(obs, inbox, t) -> desired phase
    outbox(obs, t)     -> list[Message] for its communication neighbours
    reset()

``act`` is called every simulation step.  :class:`Controller` keeps the last
decision and calls :meth:`Controller.decide` only every ``decision_interval_s``
seconds (default 5 s); the inbox is passed to :meth:`receive` every step so that
coordinated controllers never miss a message.

Information isolation: this package may import only ``trafficsync.sim.observation``
from the simulator (``tests/test_controllers.py`` checks it).  Static knowledge
(phases, link geometry, turning ratios, timing limits) comes in an
:class:`AgentView` built once per episode.
"""
from __future__ import annotations

import numpy as np

import abc
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..comm.messages import Message
from ..config import SignalTiming
from ..network.graph import Link, Movement, Network
from ..sim.observation import GREEN, Observation

__all__ = ["AgentView", "make_agent_views", "Controller", "target_phase", "select_phase", "phase_scores", "local_demand",
           "axis_phases", "local_load"]


@dataclass(frozen=True)
class AgentView:
    """Static, locally known information of intersection ``agent_id``."""

    agent_id: int
    phases: tuple[tuple[Movement, ...], ...]
    incoming: tuple[int, ...]
    outgoing: tuple[int, ...]
    links: Mapping[int, Link]                        # static data of incoming + outgoing links
    neighbors: tuple[int, ...]                        # communication neighbours N_i
    upstream_node: Mapping[int, int | None]           # incoming link -> upstream intersection
    downstream_node: Mapping[int, int | None]         # outgoing link -> downstream intersection
    turning_ratio: Mapping[tuple[int, int], float]    # r_lm for every movement (l, m)
    timing: SignalTiming
    decision_interval: int
    n_agents: int
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def num_phases(self) -> int:
        return len(self.phases)

    def phase_link_shares(self, p: int) -> dict[int, float]:
        """``{l: sum_m r_lm}`` over the movements of phase ``p`` (1.0 for 2-phase)."""
        out: dict[int, float] = {}
        for m in self.phases[p]:
            out[m.in_link] = out.get(m.in_link, 0.0) + self.turning_ratio[(m.in_link, m.out_link)]
        return out


def make_agent_views(
    network: Network,
    phases: Sequence[Sequence[tuple[Movement, ...]]],
    timing: SignalTiming,
    turning: Mapping[str, float],
    decision_interval: int = 5,
) -> dict[int, AgentView]:
    """Build one :class:`AgentView` per intersection.

    ``phases`` comes from the simulator configuration (``Simulator.phases``) and
    ``turning`` are the design turning probabilities (renormalised over the turns
    that exist at each approach).
    """
    views: dict[int, AgentView] = {}
    degree = {i: len(network.neighbors(i)) for i in range(network.N)}
    # static sweep schedule for DMPC: hop distance from intersection 0 on the road graph
    hops = {0: 0}
    frontier = [0]
    while frontier:
        nxt = []
        for u in frontier:
            for v in np.flatnonzero(np.asarray(network.road_A)[u] > 0):
                if int(v) not in hops:
                    hops[int(v)] = hops[u] + 1
                    nxt.append(int(v))
        frontier = nxt
    d_max = max(degree.values(), default=0)
    for i in range(network.N):
        inc = tuple(network.incoming_links(i))
        out = tuple(network.outgoing_links(i))
        ratios: dict[tuple[int, int], float] = {}
        for l in inc:
            movs = [m for m in network.movements[i] if m.in_link == l]
            tot = sum(float(turning.get(m.turn, 0.0)) for m in movs)
            for m in movs:
                ratios[(l, m.out_link)] = float(turning.get(m.turn, 0.0)) / tot if tot > 0 else 0.0
        views[i] = AgentView(
            agent_id=i,
            phases=tuple(tuple(p) for p in phases[i]),
            incoming=inc,
            outgoing=out,
            links={l: network.links[l] for l in inc + out},
            neighbors=tuple(network.neighbors(i)),
            upstream_node={l: network.links[l].from_node for l in inc},
            downstream_node={l: network.links[l].to_node for l in out},
            turning_ratio=ratios,
            timing=timing,
            decision_interval=int(decision_interval),
            n_agents=network.N,
            meta={
                "network": network.name,
                "degree": degree[i],                                   # communication degree d_i
                "d_max": d_max,                                         # global constant (sets eps = 1/(d_max+1))
                "neighbor_degree": {j: degree[j] for j in network.neighbors(i)},  # for Metropolis weights
                "sweep_stage": hops.get(i, 0),                          # DMPC sequential sweep order
            },
        )
    return views


def axis_phases(view: AgentView) -> dict[str, int]:
    """``{"NS": p, "EW": p}``: phase serving the through movement of each axis."""
    out: dict[str, int] = {}
    for axis, approaches in (("NS", ("N", "S")), ("EW", ("E", "W"))):
        for p, movs in enumerate(view.phases):
            if any(m.approach in approaches and m.turn == "through" for m in movs):
                out[axis] = p
                break
    return out


def local_load(view: AgentView, obs: Observation) -> float:
    """``rho_i = mean_{l in I_i} n_hat_l / C_l``."""
    vals = [e.rho_hat for l, e in obs.incoming.items()]
    return float(sum(vals) / len(vals)) if vals else 0.0


def local_demand(view: AgentView, obs: Observation) -> list[float]:
    """Per-phase local queue to serve, ``sum r_lm * q_hat_l`` (the gap-out test)."""
    return phase_scores(view, {l: e.q_hat for l, e in obs.incoming.items()}, weight_sat_flow=False)


def target_phase(obs: Observation) -> int:
    """Phase the signal is serving or moving to (hold this during yellow/all-red)."""
    if obs.signal_state != GREEN and obs.next_phase is not None:
        return int(obs.next_phase)
    return int(obs.phase)


def phase_scores(view: AgentView, link_value: Mapping[int, float], weight_sat_flow: bool = False) -> list[float]:
    """``score(p) = sum_{(l,m) in M_p} r_lm * w_l * value_l`` with ``w_l = s_l`` or 1.

    For the 2-phase scheme ``sum_m r_lm = 1`` per approach, so this is the sum of
    the served approaches' values (the movement sum without counting
    an approach once per movement).
    """
    scores = []
    for p in range(view.num_phases):
        s = 0.0
        for l, share in view.phase_link_shares(p).items():
            w = view.links[l].sat_flow if weight_sat_flow else 1.0
            s += share * w * float(link_value.get(l, 0.0))
        scores.append(s)
    return scores


def select_phase(
    obs: Observation,
    scores: Sequence[float],
    timing: SignalTiming,
    theta: float = 0.0,
    gap_out: bool = False,
    demand: Sequence[float] | None = None,
) -> int:
    """Shared switching rule (C1a; reused by every adaptive controller for fairness).

    * during yellow/all-red keep the target phase;
    * ``e < g_min``: keep the current phase;
    * ``e >= g_max``: switch to the best *other* phase;
    * ``gap_out`` (optional, standard actuated practice): the current phase has nothing
      to serve while another phase has demand -> switch to the best phase that has demand;
    * otherwise switch to ``argmax`` only if it beats the current phase by more than ``theta``.
    Ties go to the lowest phase index.

    ``demand`` is the per-phase *local* queue to serve (``phase_scores`` of ``q_hat``).  It
    decides gap-out; ``scores`` decide which phase to take.  When ``scores`` are themselves
    queues (actuated, local max-pressure) it can be omitted.  Controllers whose scores can be
    zero or negative while vehicles are waiting (CoMP's pressure with a downstream term) must
    pass it, otherwise "pressure <= 0" would end greens that still have a queue.
    """
    if obs.signal_state != GREEN:
        return target_phase(obs)
    cur, e = int(obs.phase), int(obs.elapsed)
    if e < timing.g_min or len(scores) <= 1:
        return cur
    if e >= timing.g_max:
        others = [p for p in range(len(scores)) if p != cur]
        return max(others, key=lambda p: (scores[p], -p))
    best = max(range(len(scores)), key=lambda p: (scores[p], -p))
    if gap_out:
        d = scores if demand is None else demand
        waiting = [p for p in range(len(scores)) if p != cur and d[p] > 0]
        if d[cur] <= 0 and waiting:
            return max(waiting, key=lambda p: (scores[p], -p))
    if best != cur and scores[best] > scores[cur] + theta:
        return best
    return cur


class Controller(abc.ABC):
    """Base class: decision-interval bookkeeping, inbox handling and defaults."""

    name = "base"

    def __init__(self, agent_id: int, view: AgentView, cfg: Mapping[str, Any] | None = None) -> None:
        self.agent_id = int(agent_id)
        self.view = view
        self.cfg: dict[str, Any] = dict(cfg or {})
        self.decision_interval = int(self.cfg.get("decision_interval_s", view.decision_interval))
        if self.decision_interval < 1:
            raise ValueError("decision_interval_s must be >= 1")
        self.reset()

    def reset(self) -> None:
        self._last: int | None = None
        self.last_was_decision = False
        self.n_decisions = 0

    def is_decision_step(self, t: int) -> bool:
        return self._last is None or t % self.decision_interval == 0

    def act(self, obs: Observation, inbox: Sequence[Message] | None, t: int) -> int:
        """Desired phase at step ``t``."""
        self.receive(inbox or (), t)
        self.before_decision(obs, t)
        self.last_was_decision = self.is_decision_step(t)
        if self.last_was_decision:
            p = int(self.decide(obs, t))
            if not 0 <= p < self.view.num_phases:
                raise ValueError(f"{self.name}: phase {p} out of range")
            self._last = p
            self.n_decisions += 1
        action = int(self._last)  # type: ignore[arg-type]
        self.after_action(obs, action, t)
        return action

    def receive(self, inbox: Sequence[Message], t: int) -> None:
        """Store incoming messages (local controllers ignore them)."""

    def before_decision(self, obs: Observation, t: int) -> None:
        """Per-step hook after ``receive`` (consensus updates, local tracking)."""

    def after_action(self, obs: Observation, action: int, t: int) -> None:
        """Per-step hook once this step's request is known (e.g. departure estimates)."""

    @abc.abstractmethod
    def decide(self, obs: Observation, t: int) -> int:
        """New desired phase; called every ``decision_interval`` seconds."""

    def outbox(self, obs: Observation, t: int) -> list[Message]:
        """Messages to neighbours (local controllers send nothing)."""
        return []
