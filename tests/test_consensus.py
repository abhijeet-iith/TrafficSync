"""Consensus and offsets: path-graph example, sum conservation, epsilon bound, rate vs lambda_2, wrap-around, formation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trafficsync.control.consensus import (dynamic_consensus, epsilon_bound, laplacian_weights, metropolis_weights,
                                           static_consensus)
from trafficsync.control.offset_consensus import build_constraints, formation_error, simulate_offset_consensus, wrap
from trafficsync.network import corridor, erdos_renyi, grid, ring
from trafficsync.network.spectral import is_connected

SRC = Path(__file__).resolve().parents[1] / "src" / "trafficsync"
SHORT = {"duration_s": 900, "warmup_s": 300}
PERFECT = {"mode": "perfect_sensor"}


def test_path_graph_example():
    A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)
    H = static_consensus([2, 5, 8], laplacian_weights(A), 200)
    assert np.allclose(H[-1], [5, 5, 5])
    assert np.allclose(H.sum(1), 15)


def test_epsilon_bound_divergence():
    A = ring(6)
    b = epsilon_bound(A)
    x0 = np.arange(6.0)
    ok = static_consensus(x0, laplacian_weights(A, 0.9 * b), 400)[-1]
    bad = static_consensus(x0, laplacian_weights(A, 1.1 * b), 400)[-1]
    assert np.ptp(ok) < 1e-6 and np.ptp(bad) > 1e3


def test_dynamic_consensus_tracks_average():
    rng = np.random.default_rng(0)
    U = np.cumsum(rng.normal(0, 0.01, (300, 9)), axis=0) + rng.uniform(0, 1, 9)
    X = dynamic_consensus(U, metropolis_weights(grid(3, 3).A), K=5)
    assert np.allclose(X.sum(1), U.sum(1))
    assert np.abs(X[-1] - U[-1].mean()).max() < 0.05


# --------------------------------------------------------------- offsets


def test_wrap():
    assert wrap(35, 60) == pytest.approx(-25)
    assert wrap(-30, 60) == pytest.approx(30)
    assert wrap(30, 60) == pytest.approx(30)
    assert wrap(119, 60) == pytest.approx(-1)


def test_offsets_converge_to_travel_time_differences():
    A = corridor(4).A
    r = np.array([0, 22, 44, 66.0])
    d = build_constraints(A, r)
    H = simulate_offset_consensus(A, d, phi0=[10, 50, 5, 30], C=80, iters=600)
    assert formation_error(H[-1], d, A, 80) < 1e-6
    assert H[-1][0] == pytest.approx(0.0, abs=1e-6)


# -------------------------------------------------------- controllers / isolation


def test_convergence_speed_grows_with_lambda2():
    from scipy.stats import spearmanr

    lam2, its = [], []
    for k, p in enumerate([0.15, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0] * 2):
        A = erdos_renyi(16, p, 50 + k)
        if not is_connected(A):
            continue
        L = np.diag(A.sum(1)) - A
        lam2.append(np.sort(np.linalg.eigvalsh(L))[1])
        x0 = np.random.default_rng(k).uniform(0, 1, 16)
        H = static_consensus(x0, laplacian_weights(A), 3000)
        err = np.linalg.norm(H - x0.mean(), axis=1)
        its.append(int(np.argmax(err < 1e-6 * err[0])))
    assert len(lam2) >= 8 and spearmanr(lam2, its).correlation < -0.8


def test_metropolis_static_conserves_sum_and_converges():
    A = grid(3, 3).A
    x0 = np.arange(9.0)
    H = static_consensus(x0, metropolis_weights(A), 500)
    assert np.allclose(H.sum(1), x0.sum()) and np.ptp(H[-1]) < 1e-8
