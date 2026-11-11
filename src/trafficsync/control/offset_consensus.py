"""Signal offsets as formation control.

``phi_i[k+1] = phi_i + eps [ sum_j a_ij wrap((phi_j - phi_i) - (r_j - r_i)) + b_i wrap(phi_0 + r_i - phi_i) ]``

differences wrapped to ``(-C/2, C/2]`` (offsets are circular); ``b_i = 1`` for the pinned
leader.  In CoCycle the "positions" ``r`` are travel times: along a road edge i -> j the
green of j's serving phase should start ``tau`` after i's (green wave); both directions of
an edge are combined with flow weights that add up to 1.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

__all__ = ["wrap", "build_constraints", "simulate_offset_consensus", "OffsetAgent", "formation_error"]


def _clip(x, lo, hi):
    """Scalar ``np.clip`` without the numpy call overhead (hot path)."""
    return min(max(float(x), lo), hi)


def wrap(x, C: float):
    """Wrap to ``(-C/2, C/2]``."""
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        h = C / 2.0
        y = (float(x) + h) % C - h
        return h if abs(y + h) <= 1e-8 + 1e-5 * h else y
    y = np.mod(np.asarray(x, float) + C / 2.0, C) - C / 2.0
    y = np.where(np.isclose(y, -C / 2.0), C / 2.0, y)
    return float(y) if np.ndim(y) == 0 else y


def build_constraints(A: np.ndarray, r: Sequence[float]) -> np.ndarray:
    """Desired differences ``d_ij = r_j - r_i`` on every edge (antisymmetric matrix)."""
    r = np.asarray(r, float)
    return (np.asarray(A) > 0) * (r[None, :] - r[:, None])


def formation_error(phi: np.ndarray, d: np.ndarray, A: np.ndarray, C: float) -> float:
    """Max |wrap((phi_j - phi_i) - d_ij)| over edges."""
    ii, jj = np.nonzero(np.asarray(A) > 0)
    if ii.size == 0:
        return 0.0
    return float(np.max(np.abs(wrap(phi[jj] - phi[ii] - d[ii, jj], C))))


def simulate_offset_consensus(A: np.ndarray, d: np.ndarray, phi0: Sequence[float], C: float,
                              eps: float | None = None, leader: int | None = 0, phi_leader: float = 0.0,
                              iters: int = 500) -> np.ndarray:
    """Synchronous formation consensus on a circle; returns the history ``(iters+1, N)``."""
    A = np.asarray(A, float)
    N = A.shape[0]
    e = 1.0 / (A.sum(1).max() + 1.0 + (1.0 if leader is not None else 0.0)) if eps is None else float(eps)
    phi = np.mod(np.asarray(phi0, float), C)
    b = np.zeros(N)
    if leader is not None:
        b[leader] = 1.0
    H = [phi.copy()]
    for _ in range(iters):
        diff = wrap(phi[None, :] - phi[:, None] - d, C)
        u = (A * diff).sum(1) + b * wrap(phi_leader - phi, C)
        phi = np.mod(phi + e * u, C)
        H.append(phi.copy())
    return np.array(H)


class OffsetAgent:
    """Per-agent offset state for CoCycle (one update per exchange round)."""

    def __init__(self, eps: float, leader: bool = False, phi0: float = 0.0, max_step: float = 2.0) -> None:
        self.eps, self.leader, self.phi0, self.max_step = float(eps), bool(leader), float(phi0), float(max_step)
        self.phi = 0.0

    def update(self, errors: Mapping[int, tuple[float, float]], C: float) -> float:
        """``errors[j] = (weight, wrapped error)``; returns the applied correction."""
        u = sum(w * e for w, e in errors.values())
        if self.leader:
            u += wrap(self.phi0 - self.phi, C)
        step = _clip(self.eps * u, -self.max_step, self.max_step)
        self.phi = float(np.mod(self.phi + step, C))
        return step
