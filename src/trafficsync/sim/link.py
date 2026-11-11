"""Queue-based link model.

A link holds travelling vehicles (``travel``: FIFO of ``(t_stopline, vehicle_id)``, they
reach the stop line ``tau_l`` steps after entering) and a FIFO stop-line queue.  ``n`` =
travelling + queued <= storage capacity ``C_l``.

Discharge credit: on effective green a link earns ``s_l * incident_factor`` vehicles of
credit per second; the head vehicle leaves when the credit reaches 1 and its next link has
room.  While the stop line is empty on green the credit is banked up to **one** vehicle, so a
vehicle that arrives at an empty green stop line passes without waiting, while a standing
queue still discharges at exactly ``s_l`` (headway 1 / s_l).  Off green the credit is 0.
"""
from __future__ import annotations

from collections import deque

__all__ = ["LinkState", "CREDIT_BANK"]

CREDIT_BANK = 1.0     # at most one vehicle of credit can be banked while the stop line is empty


class LinkState:
    __slots__ = ("travel", "queue", "credit")

    def __init__(self) -> None:
        self.travel: deque = deque()
        self.queue: deque = deque()
        self.credit = 0.0

    @property
    def n(self) -> int:
        return len(self.travel) + len(self.queue)
