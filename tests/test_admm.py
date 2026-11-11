"""E10 building blocks: ADMM / decentralised ADMM / DGD / gradient tracking on the
cycle-agreement problem."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from scipy.stats import spearmanr

from trafficsync.control.admm import (CycleAgreementProblem, admm_consensus, admm_decentralized,
                                      admm_neighbour_z, iterations_to, webster_cycle)
from trafficsync.control.dgd import dgd, gradient_tracking, laplacian_weights, metropolis_weights, mixing_rate


def _problem(seed: int, N: int | None = None) -> CycleAgreementProblem:
    rng = np.random.default_rng(seed)
    N = int(rng.integers(3, 25)) if N is None else N
    return CycleAgreementProblem(rng.uniform(35.0, 120.0, N), rng.uniform(0.05, 2.0, N))


def _grid(r: int, c: int) -> np.ndarray:
    return nx.to_numpy_array(nx.convert_node_labels_to_integers(nx.grid_2d_graph(r, c)))


def test_closed_form_optimum_is_weighted_mean():
    p = CycleAgreementProblem.from_lists([60.0, 90.0], [1.0, 3.0])
    assert p.z_star == pytest.approx(82.5)
    assert np.allclose(p.grad([82.5, 82.5]).sum(), 0.0)


@pytest.mark.parametrize("seed", range(20))
def test_admm_matches_closed_form_within_1e4(seed):
    p = _problem(seed)
    r = admm_consensus(p, rho=1.0, tol=1e-5)
    assert r.converged
    assert abs(r.z - p.z_star) < 1e-4
    assert np.max(np.abs(r.x - p.z_star)) < 1e-4


def test_admm_default_tolerance_is_about_1e3_accurate():
    """With the default stop (both residual norms < 1e-3) z is within ~1e-3, not 1e-4
    in general - this is why the validation above uses tol = 1e-5."""
    errs = [abs(admm_consensus(_problem(s), tol=1e-3).z - _problem(s).z_star) for s in range(100)]
    assert max(errs) < 1e-3


@pytest.mark.parametrize("seed", range(10))
def test_admm_residuals_decrease(seed):
    r = admm_consensus(_problem(seed), tol=1e-9, max_iter=500)
    primal, dual = np.array(r.primal), np.array(r.dual)
    assert np.all(np.diff(primal) <= 1e-12)                       # primal residual monotone
    half = len(dual) // 2
    assert dual[half:].max() < dual[:3].min()                      # dual residual decays overall
    assert primal[-1] < 1e-9 and dual[-1] < 1e-9


@pytest.mark.parametrize("graph", ["path", "grid", "ring", "star"])
def test_decentralised_admm_is_exact(graph):
    A = {"path": nx.to_numpy_array(nx.path_graph(9)), "grid": _grid(3, 3),
         "ring": nx.to_numpy_array(nx.cycle_graph(9)), "star": nx.to_numpy_array(nx.star_graph(8))}[graph]
    p = _problem(3, N=9)
    r = admm_decentralized(p, A, rho=1.0, iters=3000)
    assert r.final_error < 1e-6


def test_gradient_tracking_exact_and_dgd_biased():
    A = _grid(3, 3)
    W = metropolis_weights(A)
    p = _problem(1, N=9)
    gt = gradient_tracking(p, W, alpha=0.2, iters=3000)
    assert gt.final_error < 1e-8
    d = dgd(p, W, alpha=0.2, iters=3000)
    assert d.final_error > 1e-2                                      # constant-step bias
    assert d.disagreement[-1] > 1e-3
    d_small = dgd(p, W, alpha=0.02, iters=20000)
    assert d_small.final_error < d.final_error                       # bias shrinks with alpha
    dim = dgd(p, W, alpha=lambda k: 0.5 / (k + 1) ** 0.6, iters=3000)
    assert dim.final_error < d.final_error


def test_metropolis_doubly_stochastic():
    for A in (_grid(3, 3), nx.to_numpy_array(nx.star_graph(5))):
        W = metropolis_weights(A)
        assert np.allclose(W.sum(axis=0), 1) and np.allclose(W.sum(axis=1), 1)
        assert (W >= -1e-15).all() and mixing_rate(W) < 1
        Wl = laplacian_weights(A)
        assert np.allclose(Wl.sum(axis=1), 1)


def test_local_z_admm_is_inexact_and_improves_with_rounds():
    A = _grid(3, 3)
    p = _problem(0, N=9)
    e1 = admm_neighbour_z(p, A, iters=3000, rounds=1).final_error
    e10 = admm_neighbour_z(p, A, iters=3000, rounds=10).final_error
    e60 = admm_neighbour_z(p, A, iters=3000, rounds=60).final_error
    assert e1 > 1.0 and e60 < e10 < e1 and e60 < 1e-3


def _er_family(N=16):
    for k, prob_edge in enumerate([0.15, 0.25, 0.35, 0.5, 0.65, 0.8, 1.0]):
        G = nx.erdos_renyi_graph(N, prob_edge, seed=k)
        if nx.is_connected(G):
            A = nx.to_numpy_array(G)
            L = np.diag(A.sum(1)) - A
            yield np.sort(np.linalg.eigvalsh(L))[1], A


def test_rate_vs_lambda2():
    """Gradient tracking and decentralised ADMM with rho tuned per graph get faster as
    lambda_2(L) grows; with a *fixed* rho = 1 decentralised ADMM gets slower on denser
    graphs (the best rho shrinks roughly like 1/degree) - reported in E10."""
    p = _problem(7, N=16)
    lam2, gt, tuned, fixed = [], [], [], []
    for l2, A in _er_family():
        W = metropolis_weights(A)
        lam2.append(l2)
        gt.append(min(iterations_to(gradient_tracking(p, W, a, 2500).err, 1e-6) or 2500 for a in (0.1, 0.2)))
        tuned.append(min(iterations_to(admm_decentralized(p, A, r, 2500).err, 1e-6) or 2500
                         for r in (0.05, 0.1, 0.2, 0.5)))
        fixed.append(iterations_to(admm_decentralized(p, A, 1.0, 2500).err, 1e-6) or 2500)
    assert len(lam2) >= 5
    assert spearmanr(lam2, gt).correlation < -0.5
    assert spearmanr(lam2, tuned).correlation < -0.5
    assert spearmanr(lam2, fixed).correlation > 0.5


def test_webster_cycle_clip():
    assert webster_cycle(0.5, 12) == pytest.approx((1.5 * 12 + 5) / 0.5)
    assert webster_cycle(0.97, 12) == 120.0
    assert webster_cycle(0.0, 12, c_min=30) == 30.0
