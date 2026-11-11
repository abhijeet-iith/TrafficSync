"""Average consensus, matrix form.

* ``x[k+1] = (I - eps L) x[k]`` with ``0 < eps < 2 / lambda_max(L)`` (default ``1/(d_max+1)``);
* Metropolis weights ``w_ij = 1 / (1 + max(d_i, d_j))`` (no global knowledge);
* dynamic average consensus ``x[k+1] = W x[k] + (u[k+1] - u[k])`` for time-varying inputs.

The distributed (message-passing) versions used by controllers are
``control.comp.LoadConsensus`` and ``control.wmsr.WMSRConsensus``.
"""
from __future__ import annotations

import numpy as np

__all__ = ["laplacian_weights", "metropolis_weights", "epsilon_bound", "default_epsilon", "static_consensus",
           "dynamic_consensus", "consensus_error"]


def _L(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, float)
    return np.diag(A.sum(1)) - A


def epsilon_bound(A: np.ndarray) -> float:
    lm = float(np.linalg.eigvalsh(_L(A)).max())
    return 2.0 / lm if lm > 0 else np.inf


def default_epsilon(A: np.ndarray) -> float:
    return 1.0 / (float((np.asarray(A) > 0).sum(1).max()) + 1.0)


def laplacian_weights(A: np.ndarray, eps: float | None = None) -> np.ndarray:
    e = default_epsilon(A) if eps is None else float(eps)
    return np.eye(np.asarray(A).shape[0]) - e * _L(A)


def metropolis_weights(A: np.ndarray) -> np.ndarray:
    A = (np.asarray(A) > 0).astype(float)
    d = A.sum(1)
    N = A.shape[0]
    W = np.zeros((N, N))
    ii, jj = np.nonzero(A)
    W[ii, jj] = 1.0 / (1.0 + np.maximum(d[ii], d[jj]))
    W[np.arange(N), np.arange(N)] = 1.0 - W.sum(1)
    return W


def static_consensus(x0, W: np.ndarray, iters: int) -> np.ndarray:
    """History ``(iters+1, N)``."""
    x = np.asarray(x0, float).copy()
    H = [x.copy()]
    for _ in range(iters):
        x = W @ x
        H.append(x.copy())
    return np.array(H)


def dynamic_consensus(U: np.ndarray, W: np.ndarray, K: int = 1, order: str = "add_then_mix") -> np.ndarray:
    """Track the average of time-varying inputs ``U[k, i]`` with ``K`` mixing rounds per step."""
    U = np.asarray(U, float)
    x = U[0].copy()
    out = [x.copy()]
    for k in range(1, U.shape[0]):
        if order == "add_then_mix":
            x = x + U[k] - U[k - 1]
            for _ in range(K):
                x = W @ x
        else:
            for _ in range(K):
                x = W @ x
            x = x + U[k] - U[k - 1]
        out.append(x.copy())
    return np.array(out)


def consensus_error(x: np.ndarray, target: float | None = None) -> float:
    x = np.asarray(x, float)
    t = x.mean() if target is None else target
    return float(np.linalg.norm(x - t))
