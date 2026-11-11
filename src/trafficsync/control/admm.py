"""Cycle-length agreement as consensus optimisation, solved by ADMM.

Problem::

    min_{C_1..C_N, z}  sum_i f_i(C_i)   s.t.  C_i = z,     f_i(C) = 1/2 w_i (C - C_i*)^2

``C_i* = (1.5 L_tot + 5) / (1 - Y_i)`` is agent i's local Webster cycle and ``w_i`` its
load weight.  Closed-form optimum ``z* = sum w_i C_i* / sum w_i`` (validation ground truth).

Solvers (all return an :class:`OptResult` with per-iteration histories):

* :func:`admm_consensus` - scaled-form consensus ADMM with a central ``z``::

      C_i+ = (w_i C_i* + rho (z - u_i)) / (w_i + rho)
      z+   = mean_i (C_i+ + u_i)
      u_i+ = u_i + C_i+ - z+

  primal residual ``r = (C_i - z)_i``, dual residual ``s = rho (z - z_prev)`` (both as
  vectors over agents, ``||s|| = rho sqrt(N) |dz|``), stop when both norms < ``tol``.
* :func:`admm_neighbour_z` - the course "ADMM hack": the central average is replaced by
  a local estimate ``z_i`` from ``K`` rounds of average consensus on ``C + u``
  (``K = 1``: neighbourhood average).  It is *inexact*: its fixed points only satisfy
  ``C_i = z_i`` locally, so it stalls with the residual near 0 but the agents still apart;
  the gap shrinks as ``K`` grows (measured in E10).
* :func:`admm_decentralized` - exact decentralised consensus ADMM (edge-based, Shi et al.
  2014 / Mateos et al. 2010)::

      C_i+ = (w_i C_i* - p_i + rho sum_{j in N_i} (C_i + C_j)) / (w_i + 2 rho d_i)
      p_i+ = p_i + rho sum_{j in N_i} (C_i+ - C_j+)

  one neighbour exchange per iteration, converges to ``z*`` on any connected graph.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = [
    "CycleAgreementProblem",
    "OptResult",
    "webster_cycle",
    "admm_consensus",
    "admm_neighbour_z",
    "admm_decentralized",
    "iterations_to",
]


def webster_cycle(Y: float, L_tot: float, c_min: float = 30.0, c_max: float = 120.0) -> float:
    """Webster's optimal cycle ``(1.5 L + 5) / (1 - Y)``, clipped (``Y >= 0.95`` -> ``c_max``)."""
    if Y >= 0.95:
        return float(c_max)
    return float(np.clip((1.5 * L_tot + 5.0) / (1.0 - Y), c_min, c_max))


@dataclass(frozen=True)
class CycleAgreementProblem:
    """``f_i(C) = 1/2 w_i (C - C_i*)^2``."""

    c_star: np.ndarray
    w: np.ndarray

    def __post_init__(self) -> None:
        c = np.asarray(self.c_star, dtype=float)
        w = np.asarray(self.w, dtype=float)
        if c.shape != w.shape or c.ndim != 1:
            raise ValueError("c_star and w must be 1-D arrays of equal length")
        if (w <= 0).any():
            raise ValueError("load weights must be positive")
        object.__setattr__(self, "c_star", c)
        object.__setattr__(self, "w", w)

    @classmethod
    def from_lists(cls, c_star: Sequence[float], w: Sequence[float]) -> "CycleAgreementProblem":
        return cls(np.asarray(c_star, float), np.asarray(w, float))

    @property
    def N(self) -> int:
        return int(self.c_star.size)

    @property
    def z_star(self) -> float:
        return float(np.dot(self.w, self.c_star) / self.w.sum())

    def grad(self, theta: np.ndarray) -> np.ndarray:
        return self.w * (np.asarray(theta, float) - self.c_star)

    def objective(self, theta: np.ndarray) -> float:
        return float(0.5 * np.sum(self.w * (np.asarray(theta, float) - self.c_star) ** 2))

    @property
    def L_smooth(self) -> float:
        return float(self.w.max())

    @property
    def mu_strong(self) -> float:
        return float(self.w.min())


@dataclass
class OptResult:
    method: str
    x: np.ndarray                                 # final local estimates (one per agent)
    z: float | None                               # final consensus variable (if any)
    z_star: float
    err: list[float] = field(default_factory=list)       # max_i |x_i - z*| per iteration
    primal: list[float] = field(default_factory=list)    # ||r^k||_2 (ADMM)
    dual: list[float] = field(default_factory=list)      # ||s^k||_2 (ADMM)
    disagreement: list[float] = field(default_factory=list)  # max_i x_i - min_i x_i
    iterations: int = 0
    converged: bool = False
    messages: int = 0                             # scalar values sent between agents

    @property
    def final_error(self) -> float:
        return float(self.err[-1]) if self.err else float("nan")


def admm_consensus(prob: CycleAgreementProblem, rho: float = 1.0, tol: float = 1e-3,
                   max_iter: int = 1000, z0: float | None = None, run_to: int | None = None) -> OptResult:
    """Scaled consensus ADMM with a central ``z`` (stops on both residual norms < ``tol``).

    ``run_to`` keeps iterating to that iteration count after convergence (for plots).
    """
    N = prob.N
    z = float(np.mean(prob.c_star)) if z0 is None else float(z0)
    u = np.zeros(N)
    C = np.full(N, z)
    res = OptResult("ADMM (central z)", C, z, prob.z_star)
    for k in range(1, max(max_iter, run_to or 0) + 1):
        C = (prob.w * prob.c_star + rho * (z - u)) / (prob.w + rho)
        z_prev = z
        z = float(np.mean(C + u))
        u = u + C - z
        r = float(np.linalg.norm(C - z))
        s = float(rho * np.sqrt(N) * abs(z - z_prev))
        res.primal.append(r)
        res.dual.append(s)
        res.err.append(float(np.max(np.abs(C - prob.z_star))))
        res.disagreement.append(float(C.max() - C.min()))
        res.messages += 2 * N          # each agent sends C_i+u_i up and receives z
        if not res.converged and r < tol and s < tol:
            res.converged, res.iterations = True, k
            if run_to is None:
                break
        if k >= max_iter and run_to is None:
            break
    if not res.converged:
        res.iterations = len(res.err)
    res.x, res.z = C, z
    return res


def _nbr_lists(A: np.ndarray) -> list[np.ndarray]:
    A = np.asarray(A) > 0
    return [np.flatnonzero(A[i]) for i in range(A.shape[0])]


def admm_neighbour_z(prob: CycleAgreementProblem, A: np.ndarray, rho: float = 1.0,
                     iters: int = 500, rounds: int = 1, W: np.ndarray | None = None) -> OptResult:
    """Course "ADMM hack": the central ``z`` is replaced by a local estimate ``z_i``
    obtained with ``rounds`` steps of average consensus (``W``, Metropolis by default) on
    ``v = C + u`` - ``rounds = 1`` is the plain neighbourhood average.

    Any fixed point only needs ``C_i = z_i`` *locally*, i.e. ``C = W^K (C + u)``; that does
    not force agreement, so the hack stalls with residual ~0 but agents still apart.  The
    gap shrinks as ``W^K -> 11^T/N`` (E10 plots error vs ``rounds``).
    """
    from .dgd import metropolis_weights

    Wm = metropolis_weights(A) if W is None else np.asarray(W, float)
    WK = np.linalg.matrix_power(Wm, int(rounds))
    nnz = int((np.abs(Wm) > 0).sum() - Wm.shape[0])
    N = prob.N
    C = prob.c_star.copy()
    u = np.zeros(N)
    z = C.copy()
    res = OptResult(f"ADMM (local z, {rounds} round{'s' if rounds != 1 else ''})", C, None, prob.z_star)
    for k in range(1, iters + 1):
        C = (prob.w * prob.c_star + rho * (z - u)) / (prob.w + rho)
        z_new = WK @ (C + u)
        u = u + C - z_new
        res.primal.append(float(np.linalg.norm(C - z_new)))
        res.dual.append(float(rho * np.linalg.norm(z_new - z)))
        z = z_new
        res.err.append(float(np.max(np.abs(C - prob.z_star))))
        res.disagreement.append(float(C.max() - C.min()))
        res.messages += nnz * int(rounds)
    res.x, res.iterations = C, iters
    res.converged = res.err[-1] < 1e-6
    return res


def admm_decentralized(prob: CycleAgreementProblem, A: np.ndarray, rho: float = 1.0,
                       iters: int = 500, tol: float | None = None) -> OptResult:
    """Exact decentralised consensus ADMM (one neighbour exchange per iteration)."""
    Ab = (np.asarray(A) > 0).astype(float)
    N = prob.N
    d = Ab.sum(axis=1)
    if (d == 0).any():
        raise ValueError("decentralised ADMM needs every agent to have a neighbour")
    L = np.diag(d) - Ab
    ii, jj = np.nonzero(np.triu(Ab))
    C = prob.c_star.copy()
    p = np.zeros(N)
    wc = prob.w * prob.c_star
    den = prob.w + 2 * rho * d
    n_msgs = int(d.sum())
    res = OptResult("ADMM (decentralised)", C, None, prob.z_star)
    for k in range(1, iters + 1):
        C_new = (wc - p + rho * (d * C + Ab @ C)) / den
        p = p + rho * (L @ C_new)
        res.dual.append(float(rho * np.linalg.norm(C_new - C)))
        C = C_new
        res.primal.append(float(np.linalg.norm(C[ii] - C[jj])))
        res.err.append(float(np.max(np.abs(C - prob.z_star))))
        res.disagreement.append(float(C.max() - C.min()))
        res.messages += n_msgs
        if tol is not None and res.primal[-1] < tol and res.dual[-1] < tol and not res.converged:
            res.converged, res.iterations = True, k
            break
    res.x = C
    if not res.converged:
        res.iterations = len(res.err)
        res.converged = res.err[-1] < 1e-6
    return res


def iterations_to(err: Sequence[float], level: float) -> int | None:
    """First iteration (1-based) from which the error stays below ``level``."""
    e = np.asarray(err, float)
    above = np.flatnonzero(e >= level)
    if above.size == 0:
        return 1
    k = int(above[-1]) + 2
    return k if k <= e.size else None
