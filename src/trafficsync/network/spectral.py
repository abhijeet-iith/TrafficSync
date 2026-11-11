"""Spectral quantities of a (communication) graph."""
from __future__ import annotations

import numpy as np

__all__ = ["laplacian", "spectral_summary", "is_connected", "diameter"]


def laplacian(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, float)
    return np.diag(A.sum(axis=1)) - A


def _hops(A: np.ndarray) -> np.ndarray:
    N = A.shape[0]
    adj = np.asarray(A) > 0
    D = np.full((N, N), np.inf)
    for s in range(N):
        D[s, s] = 0
        frontier = [s]
        d = 0
        while frontier:
            d += 1
            nxt = []
            for u in frontier:
                for v in np.flatnonzero(adj[u]):
                    if D[s, v] == np.inf:
                        D[s, v] = d
                        nxt.append(v)
            frontier = nxt
    return D


def is_connected(A: np.ndarray) -> bool:
    return bool(np.isfinite(_hops(A)).all())


def diameter(A: np.ndarray) -> float:
    return float(_hops(A).max())


def spectral_summary(A: np.ndarray, er_p: float | None = None) -> dict:
    A = np.asarray(A, float)
    N = A.shape[0]
    L = laplacian(A)
    ev = np.sort(np.linalg.eigvalsh(L))
    ea = np.sort(np.linalg.eigvalsh(A))
    m = int((A > 0).sum() // 2)
    return {
        "N": N,
        "edges": m,
        "lambda2": float(ev[1]) if N > 1 else 0.0,
        "lambda_max_L": float(ev[-1]),
        "lambda_max_A": float(ea[-1]),
        "diameter": diameter(A),
        "mean_degree": 2.0 * m / N if N else 0.0,
        "d_max": float((A > 0).sum(axis=1).max()) if N else 0.0,
        "connected": is_connected(A),
        "eps_bound": 2.0 / float(ev[-1]) if ev[-1] > 0 else float("inf"),
        **({"expected_mean_degree": (N - 1) * float(er_p)} if er_p is not None else {}),
    }
