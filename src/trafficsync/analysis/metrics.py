"""Metric helpers that are not per-episode simulator outputs.

* :func:`consensus_error` - ``||x - mean(rho) 1||_2`` of the agents' load-consensus states
  (the "consensus error" metric), NaN for controllers without load consensus;
* :func:`per_intersection_queues` - mean and max stop-line queue per intersection;
* :func:`percent_improvement` - ``100 (m_base - m_ctrl) / m_base`` (re-exported from stats).
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .stats import percent_improvement

__all__ = ["consensus_error", "per_intersection_queues", "percent_improvement"]


def consensus_error(controllers: Mapping[int, Any]) -> float:
    xs, rs = [], []
    for c in controllers.values():
        c = getattr(c, "inner", c)                       # unwrap faulty-reporter wrappers
        cons = getattr(c, "cons", None)
        if cons is None and hasattr(c, "channels"):      # CoCycle: load channel
            cons = c.channels.get("rho")
        if cons is None or getattr(cons, "K", 0) <= 0 or cons.x is None:
            return math.nan
        xs.append(float(cons.x))
        rs.append(float(getattr(c, "rho", getattr(c, "inputs", {}).get("rho", 0.0))))
    if not xs:
        return math.nan
    return float(np.linalg.norm(np.asarray(xs) - np.mean(rs)))


def per_intersection_queues(sim, t0: int | None = None) -> dict[str, list[float]]:
    """Mean and max queue over the incoming stop lines of every intersection (t >= t0)."""
    lg = sim.logger
    t0 = int(sim.cfg.warmup_s if t0 is None else t0)
    q = lg.q[t0:]
    mean, mx = [], []
    for i in range(sim.network.N):
        cols = [sim.qcol[l] for l in sim.network.incoming_links(i)]
        mean.append(float(q[:, cols].mean()) if q.size else math.nan)
        mx.append(float(q[:, cols].max()) if q.size else math.nan)
    return {"mean": mean, "max": mx}
