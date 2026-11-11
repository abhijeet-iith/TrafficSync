"""Message format exchanged between neighbouring intersections.

Only information a neighbour could realistically share is included; no agent
ever reads another agent's internal state or the simulator's true state.

Two optional fields are added to the base format without changing it:
``iteration`` (index of an inner consensus round) and ``extra`` (room for later
modules, e.g. W-MSR). ``size_bytes`` gives the approximate wire size used for the
communication-overhead metric, assuming 32-bit ids and numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["Message", "HEADER_BYTES", "WORD_BYTES", "payload_bytes"]

WORD_BYTES = 4      # one id or one 32-bit number
HEADER_BYTES = 20   # sender, receiver, t_sent, iteration, phase, time_to_switch (4 B each, rounded)


def payload_bytes(obj: Any) -> int:
    """Approximate encoded size: 4 B per number/key, recursing into containers."""
    if obj is None:
        return 0
    if isinstance(obj, (bool, int, float, np.generic)):
        return WORD_BYTES
    if isinstance(obj, str):
        return len(obj.encode("utf-8"))
    if isinstance(obj, dict):
        n = 0
        for v in obj.values():
            n += WORD_BYTES + (WORD_BYTES if isinstance(v, (int, float)) else payload_bytes(v))
        return n
    if isinstance(obj, (list, tuple)):
        return sum(payload_bytes(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return WORD_BYTES * int(obj.size)
    raise TypeError(f"cannot size message field of type {type(obj).__name__}")


@dataclass(frozen=True)
class Message:
    """One message from ``sender`` to ``receiver`` created at step ``t_sent``."""

    sender: int
    receiver: int
    t_sent: int
    queues: dict = field(default_factory=dict)        # {link_id: estimated queue} (links shared with receiver)
    occupancy: dict = field(default_factory=dict)     # {link_id: n_hat / capacity}
    arrival_rate: dict = field(default_factory=dict)  # {link_id: lambda_hat}
    departures: list = field(default_factory=list)    # announced departures toward receiver, next H_a steps
    phase: int = -1                                   # current phase of sender
    time_to_switch: float = 0.0                       # earliest next phase change (s)
    load_value: float | None = None                   # x_i for consensus (or W-MSR)
    cycle_offset: float | None = None                 # phi_i for offset consensus
    plan: dict | None = None                          # DMPC: predicted trajectories
    iteration: int = 0                                # inner consensus round (added)
    extra: dict = field(default_factory=dict)         # reserved for later modules (added)

    def size_bytes(self) -> int:
        return HEADER_BYTES + sum(
            payload_bytes(v)
            for v in (self.queues, self.occupancy, self.arrival_rate, self.departures,
                      self.load_value, self.cycle_offset, self.plan, self.extra)
        )

    def age(self, t: int) -> int:
        """Steps since the message was created."""
        return int(t) - int(self.t_sent)
