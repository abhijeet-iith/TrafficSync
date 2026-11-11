"""Vehicle state: pre-computed route, position on the route, waiting time.

Delay of a vehicle = its accumulated waiting time (vehicles travel at free flow between stop
lines), which equals travel time minus the free-flow travel time of its route.
"""
from __future__ import annotations

__all__ = ["Vehicle"]


class Vehicle:
    __slots__ = ("id", "route", "idx", "t_gen", "t_done", "wait", "stops", "waiting", "is_probe", "link_wait",
                 "freeflow", "t_link_enter")

    def __init__(self, vid: int, route: list[int], t_gen: int, is_probe: bool, freeflow: int) -> None:
        self.id, self.route, self.idx, self.t_gen = vid, route, 0, t_gen
        self.t_done: int | None = None
        self.wait = 0                 # seconds waited in total (stop lines + boundary buffer)
        self.stops = 0                # number of separate waiting episodes (speed < 1 m/s)
        self.waiting = False
        self.is_probe = is_probe
        self.link_wait = 0            # seconds waited on the current link
        self.freeflow = freeflow      # free-flow travel time of the route [s]
        self.t_link_enter = t_gen     # time the vehicle entered its current link

    @property
    def link(self) -> int:
        return self.route[self.idx]
