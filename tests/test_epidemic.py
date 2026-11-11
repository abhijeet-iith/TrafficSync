"""E11 building blocks: congestion events, rate estimation, spectral threshold."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from trafficsync.analysis.epidemic import (congestion_states, estimate_rates, mean_field, observed_spreading,
                                           prioritised_agents, spectral_epidemic, spreading_contribution)

GRID3 = nx.to_numpy_array(nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3)))


def _sis(A, beta, beta0, gamma, T, seed):
    rng = np.random.default_rng(seed)
    N = A.shape[0]
    S = np.zeros((T, N), bool)
    for t in range(1, T):
        p = 1 - np.exp(-(beta0 + beta * (A @ S[t - 1])))
        u = rng.random(N)
        S[t] = np.where(S[t - 1], u >= 1 - np.exp(-gamma), u < p)
    return S


def test_congestion_definition():
    loads = np.zeros((600, 3))
    loads[100:200, 0] = 0.8          # 100 s -> congested
    loads[205:300, 0] = 0.8          # 5 s gap -> merged into one event
    loads[50:100, 1] = 0.9           # 50 s -> too short
    loads[540:, 2] = 0.75            # still congested at the end -> censored
    S, ev = congestion_states(loads, threshold=0.7, min_on_s=60, merge_gap_s=30)
    assert [(e.node, e.onset, e.end, e.censored) for e in ev] == [(0, 100, 300, False)]
    loads[500:, 2] = 0.75
    S, ev = congestion_states(loads)
    assert ev[-1].node == 2 and ev[-1].censored and S[599, 2]
    assert not S[:, 1].any()


def test_spectral_quantities():
    assert spectral_epidemic(nx.to_numpy_array(nx.path_graph(3)))["lambda_max_A"] == pytest.approx(np.sqrt(2))
    sp = spectral_epidemic(GRID3)
    assert sp["lambda_max_A"] == pytest.approx(2 * np.sqrt(2))
    assert sp["ranking"][0] == 4 and sp["v_max"].sum() == pytest.approx(1)
    assert prioritised_agents(GRID3, 1) == [4]


def test_rates_recovered_from_synthetic_sis():
    beta, beta0, gamma = 0.004, 0.0005, 1 / 120
    runs = [_sis(GRID3, beta, beta0, gamma, 3600, s) for s in range(12)]
    est = estimate_rates(runs, GRID3)
    assert est.beta == pytest.approx(beta, rel=0.25)
    assert est.beta0 == pytest.approx(beta0, rel=0.25)
    assert est.gamma == pytest.approx(gamma, rel=0.15)
    assert est.threshold_ratio == pytest.approx(beta * 2 * np.sqrt(2) / gamma, rel=0.35)


def test_no_contagion_gives_beta_near_zero():
    runs = [_sis(GRID3, 0.0, 0.001, 1 / 60, 3600, s) for s in range(8)]
    est = estimate_rates(runs, GRID3)
    assert est.beta < 3e-4 and not est.spreading_predicted
    assert est.beta0 == pytest.approx(0.001, rel=0.25)


def test_mean_field_threshold():
    lam = 2 * np.sqrt(2)
    gamma = 0.01
    seed = np.zeros(9)
    seed[4] = 0.01
    below = mean_field(GRID3, beta=0.8 * gamma / lam, gamma=gamma, iota0=seed, T=3000)
    above = mean_field(GRID3, beta=2.0 * gamma / lam, gamma=gamma, iota0=seed, T=3000)
    assert below["fraction"][-1] < below["fraction"][0]
    assert above["fraction"][-1] > 10 * above["fraction"][0]
    sir = mean_field(GRID3, beta=2.0 * gamma / lam, gamma=gamma, iota0=seed, T=3000, model="SIR")
    assert sir["fraction"][-1] < above["fraction"][-1]


def test_observed_and_contribution_consistent():
    S = _sis(GRID3, 0.004, 0.0005, 1 / 120, 3600, 0)
    obs = observed_spreading(S, GRID3)
    contrib = spreading_contribution([S], GRID3)
    assert contrib.sum() == pytest.approx(obs["n_onsets_exposed"])
    assert 0 <= obs["mean_fraction"] <= obs["peak_fraction"] <= 1
