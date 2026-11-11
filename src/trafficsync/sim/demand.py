"""Demand scenarios and pre-sampled arrivals.

All arrivals, routes and probe flags of an episode are sampled at reset from streams
seeded by ``(seed, entry link)``, so every controller faces *identical* vehicles for the
same seed (paired comparisons).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ..config import config_dir, load_yaml
from ..network.graph import Network

__all__ = ["DemandScenario", "Demand", "load_scenarios", "sample_route"]


def load_scenarios() -> dict[str, dict]:
    return load_yaml(config_dir() / "demand_scenarios.yaml")["scenarios"]


@dataclass
class DemandScenario:
    name: str
    rate: float = 0.12
    heading_rates: dict = field(default_factory=dict)
    profile: list | None = None
    entry_rates: list = field(default_factory=list)
    incidents: list = field(default_factory=list)
    turning: dict | None = None
    description: str = ""

    @classmethod
    def get(cls, spec: Any) -> "DemandScenario":
        if isinstance(spec, DemandScenario):
            return spec
        if isinstance(spec, (int, float)):
            return cls(name=f"uniform_{float(spec):g}", rate=float(spec))
        if isinstance(spec, str):
            sc = load_scenarios()
            if spec not in sc:
                raise KeyError(f"unknown demand scenario {spec!r}; known: {sorted(sc)}")
            return cls(name=spec, **{k: v for k, v in sc[spec].items()})
        d = dict(spec)
        d.setdefault("name", "custom")
        return cls(**d)


def _center_node(net: Network) -> int:
    pos = np.array(net.pos)
    c = pos.mean(axis=0)
    return int(np.argmin(((pos - c) ** 2).sum(axis=1)))


def _resolve_link(net: Network, spec: Any, kind: str) -> int:
    if isinstance(spec, int):
        return spec
    if spec == "auto" or spec is None:
        if kind == "incident":           # internal eastbound link into the most central intersection
            c = _center_node(net)
            cands = [l for l in net.incoming_links(c) if net.links[l].kind == "internal" and net.links[l].heading == "E"]
            cands = cands or [l for l in net.incoming_links(c) if net.links[l].kind == "internal"]
            return cands[0]
        # hotspot: eastbound entry, middle row, west edge
        rows = net.meta.get("rows", 1)
        node = (rows // 2) * net.meta.get("cols", net.N)
        return [l for l in net.incoming_links(node) if net.links[l].kind == "entry" and net.links[l].heading == "E"][0]
    d = dict(spec)
    node = int(d["node"])
    ap = d.get("approach", "W")
    return [l for l in net.incoming_links(node) if net.links[l].approach == ap][0]


def sample_route(net: Network, entry: int, turning: Mapping[str, float], rng: np.random.Generator) -> list[int]:
    """Route = list of link ids from ``entry`` to an exit link (no node revisited)."""
    route = [entry]
    visited: set[int] = set()
    l = entry
    while True:
        node = net.links[l].to_node
        if node is None:
            return route
        visited.add(node)
        movs = [m for m in net.movements[node] if m.in_link == l
                and (net.links[m.out_link].to_node is None or net.links[m.out_link].to_node not in visited)]
        if not movs:   # dead end: take any exit of this node
            movs = [m for m in net.movements[node] if m.in_link == l and net.links[m.out_link].to_node is None]
        if not movs:   # boxed in by visited nodes (rare): allow a revisit
            movs = [m for m in net.movements[node] if m.in_link == l]
        p = np.array([float(turning.get(m.turn, 0.0)) for m in movs])
        p = p / p.sum() if p.sum() > 0 else np.full(len(movs), 1 / len(movs))
        l = movs[int(rng.choice(len(movs), p=p))].out_link
        route.append(l)


class Demand:
    """Rates per entry and second, and the pre-sampled vehicles of one episode."""

    def __init__(self, net: Network, scenario: DemandScenario, duration: int, turning: Mapping[str, float],
                 seed: int, penetration: float) -> None:
        self.net, self.sc, self.T = net, scenario, int(duration)
        self.turning = dict(scenario.turning or turning)
        self.seed = int(seed)
        self.penetration = float(penetration)
        self._rates = {e: self._rate_vector(e) for e in net.entry_links}
        self.incidents = [(_resolve_link(net, d.get("link", "auto"), "incident"), int(d["start_s"]), int(d["end_s"]),
                           float(d.get("capacity_factor", 0.3))) for d in (scenario.incidents or [])]

    def _rate_vector(self, e: int) -> np.ndarray:
        T = self.T
        sc = self.sc
        lk = self.net.links[e]
        if sc.profile:
            pts = np.array(sc.profile, float)
            base = np.interp(np.arange(T), pts[:, 0], pts[:, 1])
        else:
            base = np.full(T, float(sc.rate))
        if lk.heading in (sc.heading_rates or {}):
            base = np.full(T, float(sc.heading_rates[lk.heading]))
        for d in sc.entry_rates or []:
            if _resolve_link(self.net, d.get("entry", "auto"), "hotspot") == e:
                base = np.full(T, float(d["rate"]))
        return base

    def rate_vector(self, e: int) -> np.ndarray:
        return self._rates[e]

    def capacity_factor(self, l: int, t: int) -> float:
        f = 1.0
        for link, a, b, cf in self.incidents:
            if link == l and a <= t < b:
                f *= cf
        return f

    def sample(self) -> dict[int, list[tuple[int, list[int], bool]]]:
        """``{t: [(entry, route, is_probe), ...]}`` for the whole episode."""
        out: dict[int, list] = {}
        for e in self.net.entry_links:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, e, 11]))
            prng = np.random.default_rng(np.random.SeedSequence([self.seed, e, 23]))
            counts = rng.poisson(self._rates[e])
            for t in np.flatnonzero(counts):
                for _ in range(int(counts[t])):
                    route = sample_route(self.net, e, self.turning, rng)
                    out.setdefault(int(t), []).append((e, route, bool(prng.random() < self.penetration)))
        return out
