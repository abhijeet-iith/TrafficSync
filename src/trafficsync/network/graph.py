"""Road graph and communication graph.

Road graph: directed links between signalised intersections plus *entry* links (from a
boundary source) and *exit* links (to a boundary sink).  Every link has a heading (direction
of travel E/W/N/S); at its downstream node it arrives on the *approach* opposite to its
heading (an eastbound link arrives on the W approach).

Communication graph: undirected adjacency ``A`` over intersections (default: intersections
that share a road link), degree matrix ``D``, Laplacian ``L = D - A``, oriented incidence
``B`` (``L = B B^T`` unweighted, ``L = B W B^T`` weighted).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

__all__ = ["Link", "Movement", "Network", "OPPOSITE", "LEFT_OF", "RIGHT_OF", "turn_between", "VEHICLE_LENGTH"]

OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
RIGHT_OF = {"E": "S", "S": "W", "W": "N", "N": "E"}      # heading after a right turn
LEFT_OF = {v: k for k, v in RIGHT_OF.items()}
VEHICLE_LENGTH = 7.5


def turn_between(h_in: str, h_out: str) -> str | None:
    """Turn from heading ``h_in`` to ``h_out`` (``None`` for a U-turn)."""
    if h_out == h_in:
        return "through"
    if h_out == RIGHT_OF[h_in]:
        return "right"
    if h_out == LEFT_OF[h_in]:
        return "left"
    return None


@dataclass(frozen=True)
class Link:
    """A directed road segment (static data)."""

    id: int
    from_node: int | None          # None = entry link (boundary source)
    to_node: int | None            # None = exit link (boundary sink)
    heading: str
    length: float
    lanes: int = 1
    free_flow_speed: float = 13.9
    vehicle_length: float = VEHICLE_LENGTH
    sat_headway: float = 2.0

    @property
    def kind(self) -> str:
        if self.from_node is None:
            return "entry"
        if self.to_node is None:
            return "exit"
        return "internal"

    @property
    def tau(self) -> int:
        """Free-flow travel time in steps, ``ceil(l / v_f)``."""
        return int(math.ceil(self.length / self.free_flow_speed - 1e-9))

    @property
    def capacity(self) -> int:
        """Storage capacity ``floor(n_l * l / l_veh)``."""
        return int(math.floor(self.lanes * self.length / self.vehicle_length + 1e-9))

    @property
    def sat_flow(self) -> float:
        """Saturation flow ``s_l = n_l / h`` (veh/s)."""
        return self.lanes / self.sat_headway

    @property
    def approach(self) -> str:
        """Approach at the downstream node."""
        return OPPOSITE[self.heading]


@dataclass(frozen=True)
class Movement:
    node: int
    in_link: int
    out_link: int
    approach: str
    turn: str


@dataclass
class Network:
    """Road network + communication graph."""

    name: str
    N: int
    pos: list[tuple[float, float]]
    links: dict[int, Link]
    road_A: np.ndarray                       # undirected road adjacency between intersections
    A: np.ndarray = field(default=None)      # communication adjacency (weighted, symmetric)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.A is None:
            self.A = self.road_A.copy()
        self._index()

    # ----------------------------------------------------------------- indexes
    def _index(self) -> None:
        self._in = {i: [] for i in range(self.N)}
        self._out = {i: [] for i in range(self.N)}
        for l in sorted(self.links):
            lk = self.links[l]
            if lk.to_node is not None:
                self._in[lk.to_node].append(l)
            if lk.from_node is not None:
                self._out[lk.from_node].append(l)
        self.movements: dict[int, list[Movement]] = {}
        for i in range(self.N):
            movs = []
            for l in self._in[i]:
                hin = self.links[l].heading
                for m in self._out[i]:
                    t = turn_between(hin, self.links[m].heading)
                    if t is not None:
                        movs.append(Movement(i, l, m, self.links[l].approach, t))
            self.movements[i] = movs
        self.entry_links = [l for l in sorted(self.links) if self.links[l].kind == "entry"]
        self.exit_links = [l for l in sorted(self.links) if self.links[l].kind == "exit"]
        self.internal_links = [l for l in sorted(self.links) if self.links[l].kind == "internal"]
        self.queue_links = self.entry_links + self.internal_links          # links with a stop line
        self.set_comm(self.A)

    def set_comm(self, A: np.ndarray) -> None:
        A = np.asarray(A, dtype=float)
        if A.shape != (self.N, self.N) or not np.allclose(A, A.T):
            raise ValueError("communication adjacency must be symmetric N x N")
        self.A = A.copy()
        np.fill_diagonal(self.A, 0.0)
        self._nbrs = [sorted(np.flatnonzero(self.A[i] > 0).tolist()) for i in range(self.N)]

    # ------------------------------------------------------------- road queries
    def incoming_links(self, i: int) -> list[int]:
        return list(self._in[i])

    def outgoing_links(self, i: int) -> list[int]:
        return list(self._out[i])

    def link_between(self, i: int, j: int) -> int | None:
        for l in self._out[i]:
            if self.links[l].to_node == j:
                return l
        return None

    # ------------------------------------------------------ communication graph
    def neighbors(self, i: int) -> list[int]:
        return list(self._nbrs[i])

    @property
    def D(self) -> np.ndarray:
        return np.diag(self.A.sum(axis=1))

    @property
    def L(self) -> np.ndarray:
        return self.D - self.A

    def edges(self) -> list[tuple[int, int]]:
        return [(i, j) for i in range(self.N) for j in range(i + 1, self.N) if self.A[i, j] > 0]

    @property
    def B(self) -> np.ndarray:
        """Oriented incidence matrix ``N x M`` (+1 at the lower index of each edge)."""
        E = self.edges()
        B = np.zeros((self.N, len(E)))
        for k, (i, j) in enumerate(E):
            B[i, k], B[j, k] = 1.0, -1.0
        return B

    @property
    def edge_weights(self) -> np.ndarray:
        return np.array([self.A[i, j] for i, j in self.edges()])

    def spectral_summary(self) -> dict:
        from .spectral import spectral_summary

        out = spectral_summary(self.A)
        if "er_p" in self.meta:                          # Erdos-Renyi: expected mean degree (N-1) p
            out["expected_mean_degree"] = (self.N - 1) * float(self.meta["er_p"])
        return out

    def copy_with_comm(self, A: np.ndarray, label: str | None = None) -> "Network":
        net = Network(self.name, self.N, list(self.pos), dict(self.links), self.road_A.copy(), np.asarray(A, float),
                      dict(self.meta))
        if label:
            net.meta["comm"] = label
        return net

    def road_edges(self) -> Iterable[tuple[int, int]]:
        return [(i, j) for i in range(self.N) for j in range(i + 1, self.N) if self.road_A[i, j] > 0]
