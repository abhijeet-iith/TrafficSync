"""TrafficSync: multi-agent traffic-signal control on a graph of intersections.

AI3403 Multi-Agent Systems project, Group MadmaX.

Packages:

* :mod:`trafficsync.network` - road/communication graphs, generators, spectral tools, (r,s)-robustness.
* :mod:`trafficsync.sim` - queue-based simulator, demand scenarios, probe-vehicle model, metrics logger.
* :mod:`trafficsync.comm` - messages and the communication bus (latency, drops, outages).
* :mod:`trafficsync.control` - fixed-time, actuated, max-pressure, CoMP, CoCycle, W-MSR, ADMM/DGD,
  epistemic guard, DMPC and the MARL policy wrapper.
* :mod:`trafficsync.marl` - multi-agent environment and IQL / VDN / QMIX learners.
* :mod:`trafficsync.logic` - Signal Temporal Logic monitor and Kripke models.
* :mod:`trafficsync.analysis` - statistics, metrics, epidemic analysis.
* :mod:`trafficsync.viz` - plots, animation, dashboard, recordings.
* :mod:`trafficsync.sumo` - SUMO co-simulation, real-city agents and browser replays.
"""
from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
