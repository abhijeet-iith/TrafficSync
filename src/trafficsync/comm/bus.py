"""Communication bus: delivers, delays or drops messages and counts overhead.

Timing. ``send`` at step ``t`` schedules delivery at ``t + latency_steps``;
``deliver(t)`` returns every pending message with delivery time ``<= t``.  In the
simulation main loop ``deliver(t)`` runs before agents act and ``send`` after, so a
message is readable at the earliest one step after it was sent (latency 0 and 1
behave the same there).  Inner rounds of a consensus exchange call
``send``/``deliver`` with the round index as clock, where latency 0 really is
synchronous.

Drops are i.i.d. with probability ``drop_prob`` and drawn from a stream seeded
by the experiment seed.  Inside ``[outage_start_s, outage_end_s)`` every message
is dropped (experiment E7's total outage).  Transmitted messages and bytes are
counted whether or not they are dropped.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

import numpy as np

from ..network.graph import Network
from .messages import Message

__all__ = ["CommConfig", "CommBus"]


@dataclass(frozen=True)
class CommConfig:
    """Communication model."""

    period_s: int = 5
    latency_steps: int = 1
    drop_prob: float = 0.0
    enforce_topology: bool = True
    missing_policy: str = "last"          # last | age_discounted | ignore  (applied by receivers)
    outage_start_s: int | None = None
    outage_end_s: int | None = None
    topology: Any = None                  # optional override of the network's comm graph (runner)

    def __post_init__(self) -> None:
        if self.period_s < 1 or self.latency_steps < 0:
            raise ValueError("period_s >= 1 and latency_steps >= 0 required")
        if not 0.0 <= self.drop_prob <= 1.0:
            raise ValueError("drop_prob must be in [0, 1]")
        if self.missing_policy not in ("last", "age_discounted", "ignore"):
            raise ValueError("missing_policy must be last | age_discounted | ignore")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "CommConfig":
        data = dict(data or {})
        if "comm" in data and len(data) == 1:
            data = dict(data["comm"])
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown CommConfig keys: {sorted(unknown)}")
        return cls(**data)


class CommBus:
    """Message transport between neighbouring agents."""

    def __init__(self, network: Network, cfg: CommConfig | Mapping[str, Any] | None = None, seed: int = 0) -> None:
        self.network = network
        self.cfg = cfg if isinstance(cfg, CommConfig) else CommConfig.from_dict(cfg)
        self.seed = int(seed)
        self._nbrs = [set(network.neighbors(i)) for i in range(network.N)]
        self.reset()

    def reset(self) -> None:
        self._rng = np.random.default_rng(np.random.SeedSequence([self.seed, 0xC0FFEE]))
        self._heap: list[tuple[int, int, Message]] = []
        self._seq = 0
        self.n_sent = 0
        self.n_dropped = 0
        self.n_delivered = 0
        self.bytes_sent = 0
        self.sent_by_agent = np.zeros(self.network.N, dtype=np.int64)
        self._step_msgs = 0
        self._step_bytes = 0
        # optional per-message log (sender, receiver, t_sent, dropped) for visualisation;
        # off by default, switched on by ``enable_trace``
        self.trace: list[tuple[int, int, int, bool]] | None = None

    def enable_trace(self) -> list[tuple[int, int, int, bool]]:
        """Start logging every transmission as ``(sender, receiver, t_sent, dropped)``."""
        self.trace = []
        return self.trace

    # ---------------------------------------------------------------- sending
    def in_outage(self, t: int) -> bool:
        a, b = self.cfg.outage_start_s, self.cfg.outage_end_s
        return a is not None and b is not None and a <= t < b

    def send(self, msg: Message) -> bool:
        """Transmit ``msg``; returns False if it was dropped."""
        if self.cfg.enforce_topology and msg.receiver not in self._nbrs[msg.sender]:
            raise ValueError(f"agent {msg.sender} cannot message {msg.receiver}: not a communication neighbour")
        size = msg.size_bytes()
        self.n_sent += 1
        self.bytes_sent += size
        self.sent_by_agent[msg.sender] += 1
        self._step_msgs += 1
        self._step_bytes += size
        dropped = self.in_outage(msg.t_sent)
        if not dropped and self.cfg.drop_prob > 0:
            dropped = bool(self._rng.random() < self.cfg.drop_prob)
        if self.trace is not None:
            self.trace.append((msg.sender, msg.receiver, msg.t_sent, dropped))
        if dropped:
            self.n_dropped += 1
            return False
        heapq.heappush(self._heap, (msg.t_sent + self.cfg.latency_steps, self._seq, msg))
        self._seq += 1
        return True

    def send_many(self, msgs: Iterable[Message]) -> int:
        return sum(self.send(m) for m in msgs)

    def broadcast(self, sender: int, t: int, **payload: Any) -> list[Message]:
        """Build (not send) one message per communication neighbour of ``sender``."""
        return [Message(sender=sender, receiver=j, t_sent=t, **payload) for j in sorted(self._nbrs[sender])]

    # --------------------------------------------------------------- delivery
    def deliver(self, t: int) -> dict[int, list[Message]]:
        """All messages due by ``t``, grouped by receiver, in (due time, send order)."""
        out: dict[int, list[Message]] = {}
        heap = self._heap
        while heap and heap[0][0] <= t:
            _, _, msg = heapq.heappop(heap)
            out.setdefault(msg.receiver, []).append(msg)
            self.n_delivered += 1
        return out

    @property
    def pending(self) -> int:
        return len(self._heap)

    def pop_step_counts(self) -> tuple[int, int]:
        """(messages, bytes) transmitted since the previous call (for the metrics logger)."""
        out = (self._step_msgs, self._step_bytes)
        self._step_msgs = self._step_bytes = 0
        return out

    def stats(self) -> dict[str, Any]:
        return {
            "sent": self.n_sent,
            "dropped": self.n_dropped,
            "delivered": self.n_delivered,
            "pending": self.pending,
            "bytes_sent": self.bytes_sent,
            "drop_rate": (self.n_dropped / self.n_sent) if self.n_sent else 0.0,
        }

    def with_config(self, **changes: Any) -> "CommBus":
        """Fresh bus on the same network with some config fields changed."""
        return CommBus(self.network, replace(self.cfg, **changes), self.seed)
