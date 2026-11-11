"""Distributed gradient descent and gradient tracking.

Both solve ``min sum_i f_i(theta)`` over a connected communication graph with a
doubly-stochastic mixing matrix ``W`` (Metropolis weights by default, so no global
knowledge is needed)::

    DGD:                theta+ = W theta - alpha_k grad f(theta)
    gradient tracking:  theta+ = W theta - alpha s
                        s+     = W s + grad f(theta+) - grad f(theta),   s^0 = grad f(theta^0)

With a constant step, DGD stops at a point whose distance from ``z*`` is O(alpha / (1 -
sigma_2(W))) - the *constant-step bias* E10 shows; with ``alpha_k = a / (k + 1)^p`` it
converges slowly to ``z*``.  Gradient tracking converges exactly and linearly for a small
enough constant step (each ``s_i`` tracks the network-average gradient).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .admm import CycleAgreementProblem, OptResult

__all__ = ["metropolis_weights", "laplacian_weights", "dgd", "gradient_tracking", "dgd_bias_bound", "mixing_rate"]


def metropolis_weights(A: np.ndarray) -> np.ndarray:
    """``w_ij = 1 / (1 + max(d_i, d_j))`` for edges, ``w_ii = 1 - sum_j w_ij``."""
    A = (np.asarray(A) > 0).astype(float)
    d = A.sum(axis=1)
    N = A.shape[0]
    W = np.zeros((N, N))
    ii, jj = np.nonzero(A)
    W[ii, jj] = 1.0 / (1.0 + np.maximum(d[ii], d[jj]))
    W[np.arange(N), np.arange(N)] = 1.0 - W.sum(axis=1)
    return W


def laplacian_weights(A: np.ndarray, eps: float | None = None) -> np.ndarray:
    """``W = I - eps L`` with ``eps = 1 / (d_max + 1)`` by default."""
    A = (np.asarray(A) > 0).astype(float)
    L = np.diag(A.sum(axis=1)) - A
    e = 1.0 / (A.sum(axis=1).max() + 1.0) if eps is None else float(eps)
    return np.eye(A.shape[0]) - e * L


def mixing_rate(W: np.ndarray) -> float:
    """Second-largest singular value of ``W`` (``sigma_2 < 1`` on a connected graph)."""
    s = np.linalg.svd(W - np.ones_like(W) / W.shape[0], compute_uv=False)
    return float(s[0])


def dgd_bias_bound(prob: CycleAgreementProblem, W: np.ndarray, alpha: float) -> float:
    """Order-of-magnitude bound ``alpha * max|grad f_i(z*)| / (1 - sigma_2(W))`` on the
    constant-step DGD error (used only to annotate the E10 plot)."""
    g = np.abs(prob.grad(np.full(prob.N, prob.z_star))).max()
    return float(alpha * g / max(1e-12, 1.0 - mixing_rate(W)))


def _messages(W: np.ndarray) -> int:
    return int(((np.abs(W) > 0).sum() - W.shape[0]))


def dgd(prob: CycleAgreementProblem, W: np.ndarray, alpha: float | Callable[[int], float] = 0.1,
        iters: int = 1000, theta0: np.ndarray | None = None) -> OptResult:
    """Distributed gradient descent; ``alpha`` constant or a function of the iteration ``k``."""
    theta = prob.c_star.copy() if theta0 is None else np.asarray(theta0, float).copy()
    step = alpha if callable(alpha) else (lambda k, a=float(alpha): a)
    label = "DGD (diminishing)" if callable(alpha) else f"DGD (alpha={float(alpha):g})"
    res = OptResult(label, theta, None, prob.z_star)
    nm = _messages(W)
    for k in range(iters):
        with np.errstate(over="ignore", invalid="ignore"):
            theta = W @ theta - step(k) * prob.grad(theta)
        res.err.append(float(np.max(np.abs(theta - prob.z_star))))
        res.disagreement.append(float(theta.max() - theta.min()))
        res.messages += nm
    res.x, res.iterations = theta, iters
    res.converged = res.err[-1] < 1e-6
    return res


def gradient_tracking(prob: CycleAgreementProblem, W: np.ndarray, alpha: float = 0.1,
                      iters: int = 1000, theta0: np.ndarray | None = None) -> OptResult:
    """Gradient tracking (DIGing form); two scalars per neighbour per iteration."""
    theta = prob.c_star.copy() if theta0 is None else np.asarray(theta0, float).copy()
    g = prob.grad(theta)
    s = g.copy()
    res = OptResult(f"Gradient tracking (alpha={alpha:g})", theta, None, prob.z_star)
    nm = 2 * _messages(W)
    for k in range(iters):
        with np.errstate(over="ignore", invalid="ignore"):     # a too-large step diverges; reported, not raised
            theta_new = W @ theta - alpha * s
            g_new = prob.grad(theta_new)
            s = W @ s + g_new - g
        theta, g = theta_new, g_new
        res.err.append(float(np.max(np.abs(theta - prob.z_star))))
        res.disagreement.append(float(theta.max() - theta.min()))
        res.messages += nm
    res.x, res.iterations = theta, iters
    res.converged = res.err[-1] < 1e-6
    return res
