"""SUMO co-simulation for TrafficSync (optional; needs ``pip install eclipse-sumo traci sumolib``).

Two ways to run the project in the microscopic simulator SUMO:

* :mod:`trafficsync.sumo.grid` - the TrafficSync networks (corridor / grid) are converted to SUMO networks,
  the *same* vehicles (same seed, same routes) are inserted, and the **unchanged** TrafficSync controllers
  (fixed, actuated, max-pressure, CoMP, CoCycle, ...) drive the SUMO traffic lights through TraCI, with the
  same message bus.  Used to validate the queue simulator against car-following physics.
* :mod:`trafficsync.sumo.realmap` - real city maps with real demand (bundled: Cologne and Ingolstadt
  scenarios from the RESCO benchmark, built from OpenStreetMap and the TAPASCologne / InTAS traffic studies),
  or any network imported from OpenStreetMap.  Every traffic light becomes an agent with a generic phase model;
  controllers: the map's own fixed-time program, actuated, local max-pressure and CoMP with messages.
* :mod:`trafficsync.sumo.render` - draws the map, vehicles and signal states to PNG / GIF (``sumo-gui``
  is used instead when ``gui=True``).
"""
from __future__ import annotations

from .common import sumo_available, sumo_binary

__all__ = ["sumo_available", "sumo_binary"]
