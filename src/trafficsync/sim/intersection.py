"""Phases, movement conflicts and the signal state machine.

State machine (per intersection, evaluated at the start of every 1 s step with the
phase requested by the controller):

* GREEN of phase ``p`` for ``t`` completed seconds:

  - request ``r != p`` and ``t >= g_min``  -> start the change to ``r``;
  - ``t >= g_max``                        -> forced change to ``r`` if ``r != p``,
    otherwise to the next phase in cyclic order;
  - otherwise stay (a request made before ``g_min`` is honoured once ``g_min`` is
    reached, as long as the controller keeps requesting it).

* A change is GREEN -> YELLOW (``yellow`` s) -> ALL-RED (``allred`` s) -> GREEN of the
  target phase.  Requests are ignored during YELLOW/ALL-RED.
* Discharge is allowed only in GREEN after the startup lost time
  (``t >= startup_lost``), so an effective green lasts ``G - startup_lost`` seconds.
"""
from __future__ import annotations

from typing import Sequence

from ..config import SignalTiming
from ..network.graph import OPPOSITE, Movement, Network
from .observation import ALLRED, GREEN, YELLOW

__all__ = [
    "GREEN",
    "YELLOW",
    "ALLRED",
    "crossing_turn",
    "build_phases",
    "movements_conflict",
    "SignalStateMachine",
]

_EMPTY: frozenset[int] = frozenset()


def crossing_turn(drive_side: str) -> str:
    """Turn that crosses opposing traffic: left for right-hand traffic, right otherwise."""
    return "left" if drive_side == "right" else "right"


def build_phases(
    network: Network, node: int, scheme: str = "2phase", drive_side: str = "right"
) -> list[tuple[Movement, ...]]:
    """Phases of ``node`` as tuples of movements.

    * ``2phase`` (core): 0 = all N/S-approach movements, 1 = all E/W-approach movements
      (crossing turns are permissive).
    * ``4phase`` (extended): 0 = N/S through + nearside turn, 1 = N/S protected crossing
      turn, 2 = E/W through + nearside turn, 3 = E/W protected crossing turn.
    """
    movs = network.movements[node]
    ns = [m for m in movs if m.approach in ("N", "S")]
    ew = [m for m in movs if m.approach in ("E", "W")]
    if scheme == "2phase":
        groups = [ns, ew]
    elif scheme == "4phase":
        x = crossing_turn(drive_side)
        groups = [
            [m for m in ns if m.turn != x],
            [m for m in ns if m.turn == x],
            [m for m in ew if m.turn != x],
            [m for m in ew if m.turn == x],
        ]
    else:
        raise ValueError(f"unknown phase scheme {scheme!r}")
    return [tuple(g) for g in groups]


def movements_conflict(m1: Movement, m2: Movement, protected: bool, drive_side: str = "right") -> bool:
    """Hard conflict between two movements of the same intersection.

    Perpendicular approaches always conflict.  Opposing approaches conflict only
    when exactly one of the two movements is the crossing turn and turns are
    protected (``4phase``); in ``2phase`` that pair is a permitted permissive turn.
    """
    if m1.node != m2.node or m1.approach == m2.approach:
        return False
    if OPPOSITE[m1.approach] == m2.approach:
        if not protected:
            return False
        x = crossing_turn(drive_side)
        return (m1.turn == x) != (m2.turn == x)
    return True


class SignalStateMachine:
    """Signal state of one intersection: ``(phase, state, t_in_state)``."""

    def __init__(
        self,
        node: int,
        phases: Sequence[tuple[Movement, ...]],
        timing: SignalTiming,
        initial_phase: int = 0,
    ) -> None:
        if not phases:
            raise ValueError("an intersection needs at least one phase")
        self.node = node
        self.phases = list(phases)
        self.P = len(self.phases)
        self.timing = timing
        self.initial_phase = int(initial_phase) % self.P
        self.served: list[dict[int, frozenset[int]]] = []
        for ph in self.phases:
            d: dict[int, set[int]] = {}
            for m in ph:
                d.setdefault(m.in_link, set()).add(m.out_link)
            self.served.append({k: frozenset(v) for k, v in d.items()})
        self.reset()

    def reset(self) -> None:
        self.phase = self.initial_phase
        self.state = GREEN
        self.t_in_state = 0
        self.next_phase: int | None = None
        self.tsg = [0] * self.P  # seconds since each phase was last green
        self.n_switches = 0

    # ----------------------------------------------------------------- logic
    def update(self, requested: int | None = None) -> None:
        """Apply the controller request at the start of a step."""
        tm = self.timing
        if requested is not None:
            requested = int(requested)
            if not 0 <= requested < self.P:
                raise ValueError(f"node {self.node}: phase {requested} not in 0..{self.P - 1}")
        if self.state == GREEN:
            wants = requested is not None and requested != self.phase
            if (wants and self.t_in_state >= tm.g_min) or self.t_in_state >= tm.g_max:
                target = requested if wants else (self.phase + 1) % self.P
                if target == self.phase:  # single-phase intersection
                    return
                self.next_phase = target
                self.n_switches += 1
                if tm.yellow > 0:
                    self._enter(YELLOW)
                elif tm.allred > 0:
                    self._enter(ALLRED)
                else:
                    self._enter(GREEN)
        elif self.state == YELLOW:
            if self.t_in_state >= tm.yellow:
                self._enter(ALLRED if tm.allred > 0 else GREEN)
        elif self.state == ALLRED:
            if self.t_in_state >= tm.allred:
                self._enter(GREEN)

    def _enter(self, state: str) -> None:
        self.state = state
        self.t_in_state = 0
        if state == GREEN:
            self.phase = int(self.next_phase)  # type: ignore[arg-type]
            self.next_phase = None

    def effective_green(self) -> bool:
        return self.state == GREEN and self.t_in_state >= self.timing.startup_lost

    def served_out(self, in_link: int) -> frozenset[int]:
        """Outgoing links that ``in_link`` may discharge to right now (if green)."""
        if self.state != GREEN:
            return _EMPTY
        return self.served[self.phase].get(in_link, _EMPTY)

    def green_movements(self) -> tuple[Movement, ...]:
        return self.phases[self.phase] if self.state == GREEN else ()

    def advance(self) -> None:
        """End of step: one more second in the current state."""
        for p in range(self.P):
            self.tsg[p] = 0 if (self.state == GREEN and p == self.phase) else self.tsg[p] + 1
        self.t_in_state += 1
