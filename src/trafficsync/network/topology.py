"""Network generators and communication topologies.

Roads are always a corridor or a Manhattan grid; ring / complete / Erdos-Renyi / k-hop /
custom / link removal change only *who talks to whom* (communication graph).
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..config import config_dir, load_yaml
from .graph import Link, Network

__all__ = ["corridor", "grid", "load_network", "apply_communication", "comm_adjacency", "erdos_renyi",
           "ring", "complete", "k_hop", "path"]

_DIRS = {"E": (0, 1), "W": (0, -1), "N": (-1, 0), "S": (1, 0)}     # (d_row, d_col)


def _build(name: str, rows: int, cols: int, length: float, entry_length: float, lanes: int, vf: float,
           lveh: float, headway: float, side_streets_only_ns: bool = False) -> Network:
    """Generic builder: intersections on a rows x cols lattice, internal links both ways
    between lattice neighbours, one entry and one exit link on every boundary side."""
    N = rows * cols
    pos = [(c * length, -r * length) for r in range(rows) for c in range(cols)]
    links: dict[int, Link] = {}
    nid = lambda r, c: r * cols + c  # noqa: E731
    lid = 0

    def add(fr, to, h, ln):
        nonlocal lid
        links[lid] = Link(lid, fr, to, h, ln, lanes, vf, lveh, headway)
        lid += 1

    road_A = np.zeros((N, N))
    for r in range(rows):
        for c in range(cols):
            i = nid(r, c)
            for h, (dr, dc) in _DIRS.items():
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    add(i, nid(rr, cc), h, length)
                    road_A[i, nid(rr, cc)] = 1.0
    for r in range(rows):
        for c in range(cols):
            i = nid(r, c)
            for h, (dr, dc) in _DIRS.items():
                rr, cc = r + dr, c + dc
                if not (0 <= rr < rows and 0 <= cc < cols):
                    add(i, None, h, entry_length)                 # exit towards the boundary
                    from .graph import OPPOSITE

                    add(None, i, OPPOSITE[h], entry_length)       # entry from that boundary
    return Network(name, N, pos, links, road_A, meta={"rows": rows, "cols": cols, "link_length": length})


def corridor(n: int = 3, link_length: float = 400.0, entry_length: float | None = None, lanes: int = 1,
             free_flow_speed: float = 13.9, vehicle_length: float = 7.5, sat_headway: float = 2.0,
             name: str | None = None) -> Network:
    """``n`` intersections on an E-W arterial; side streets (N/S entries and exits) at each."""
    return _build(name or f"corridor{n}", 1, n, link_length, entry_length or link_length, lanes, free_flow_speed,
                  vehicle_length, sat_headway)


def grid(rows: int = 3, cols: int = 3, link_length: float = 300.0, entry_length: float | None = None, lanes: int = 1,
         free_flow_speed: float = 13.9, vehicle_length: float = 7.5, sat_headway: float = 2.0,
         name: str | None = None) -> Network:
    return _build(name or f"grid{rows}x{cols}", rows, cols, link_length, entry_length or link_length, lanes,
                  free_flow_speed, vehicle_length, sat_headway)


# ------------------------------------------------------------------ comm graphs
def path(N: int) -> np.ndarray:
    A = np.zeros((N, N))
    for i in range(N - 1):
        A[i, i + 1] = A[i + 1, i] = 1
    return A


def ring(N: int) -> np.ndarray:
    A = path(N)
    if N > 2:
        A[0, N - 1] = A[N - 1, 0] = 1
    return A


def complete(N: int) -> np.ndarray:
    return np.ones((N, N)) - np.eye(N)


def erdos_renyi(N: int, p: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    U = rng.random((N, N))
    A = (np.triu(U, 1) < p) & np.triu(np.ones((N, N), bool), 1)
    A = A | A.T
    return A.astype(float)


def k_hop(A: np.ndarray, k: int) -> np.ndarray:
    N = A.shape[0]
    R = np.eye(N)
    P = np.eye(N)
    for _ in range(k):
        P = ((P @ (np.asarray(A) + np.eye(N))) > 0).astype(float)
        R = np.maximum(R, P)
    out = (R > 0).astype(float)
    np.fill_diagonal(out, 0)
    return out


def comm_adjacency(net: Network, cfg: Mapping[str, Any] | None) -> np.ndarray:
    """Communication adjacency from a ``communication:`` config section."""
    cfg = dict(cfg or {})
    topo = cfg.get("topology", "road")
    N = net.N
    if topo == "road":
        A = net.road_A.copy()
    elif topo == "path":
        A = path(N)
    elif topo == "ring":
        A = ring(N)
    elif topo == "complete":
        A = complete(N)
    elif topo == "erdos_renyi":
        A = erdos_renyi(N, float(cfg["p"]), int(cfg.get("seed", 0)))
    elif topo == "k_hop":
        A = k_hop(net.road_A, int(cfg.get("k", 2)))
    elif topo == "custom":
        A = np.zeros((N, N))
        for e in cfg.get("edges", []) or []:
            A[e[0], e[1]] = A[e[1], e[0]] = 1.0
        for i, j, w in cfg.get("weights", []) or []:
            A[i, j] = A[j, i] = float(w)
    else:
        raise ValueError(f"unknown communication topology {topo!r}")
    for e in cfg.get("add_links", []) or []:                      # extra (non-road) V2I links, e.g. E9
        A[e[0], e[1]] = A[e[1], e[0]] = 1.0
    for e in cfg.get("remove_links", []) or []:
        A[e[0], e[1]] = A[e[1], e[0]] = 0.0
    frac = float(cfg.get("remove_fraction", 0.0) or 0.0)
    if frac > 0:
        rng = np.random.default_rng(int(cfg.get("remove_seed", 0)))
        E = [(i, j) for i in range(N) for j in range(i + 1, N) if A[i, j] > 0]
        order = rng.permutation(len(E))
        k = int(round(frac * len(E)))
        from .spectral import is_connected

        removed = 0
        for idx in order:
            if removed >= k:
                break
            i, j = E[idx]
            w = A[i, j]
            A[i, j] = A[j, i] = 0
            if cfg.get("keep_connected") and not is_connected(A):
                A[i, j] = A[j, i] = w
                continue
            removed += 1
    np.fill_diagonal(A, 0)
    return A


def apply_communication(net: Network, cfg: Mapping[str, Any] | None) -> Network:
    if not cfg:
        return net
    out = net.copy_with_comm(comm_adjacency(net, cfg), label=str(dict(cfg).get("topology", "road")))
    if dict(cfg).get("topology") == "erdos_renyi":
        out.meta["er_p"] = float(cfg["p"])
    return out


def load_network(name: str) -> Network:
    """Build a network from ``configs/network_<name>.yaml`` (or a path)."""
    from pathlib import Path

    p = Path(name)
    if not p.exists():
        p = config_dir() / f"network_{name}.yaml"
    raw = load_yaml(p)
    n = raw["network"]
    kw = dict(link_length=float(n.get("link_length_m", 300.0)), entry_length=float(n.get("entry_length_m", n.get("link_length_m", 300.0))),
              lanes=int(n.get("lanes", 1)), free_flow_speed=float(n.get("free_flow_speed_mps", 13.9)),
              vehicle_length=float(n.get("vehicle_length_m", 7.5)), sat_headway=float(n.get("sat_headway_s", 2.0)),
              name=str(n.get("id", name)))
    if n["type"] == "corridor":
        net = corridor(int(n["n"]), **kw)
    elif n["type"] == "grid":
        net = grid(int(n["rows"]), int(n["cols"]), **kw)
    else:
        raise ValueError(f"unknown network type {n['type']!r}")
    comm = raw.get("communication")
    if comm and comm.get("topology", "road") != "road" or (comm and (comm.get("remove_links") or comm.get("remove_fraction"))):
        net = apply_communication(net, comm)
    return net
