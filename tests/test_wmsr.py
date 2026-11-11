"""W-MSR and (r, s)-robustness."""
from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from trafficsync.comm.messages import Message
from trafficsync.control.wmsr import (FaultInjector, FaultModel, WMSRConsensus, choose_faulty, is_f_local,
                                      simulate_resilient, wmsr_filter, wmsr_update)
from trafficsync.network.robustness import (augment_to_robustness, is_r_robust, is_rs_robust, khop_adjacency,
                                            max_r_robust, robustness_profile, sampled_rs_robust)


def _A(G) -> np.ndarray:
    return nx.to_numpy_array(nx.convert_node_labels_to_integers(G))


GRID3 = _A(nx.grid_2d_graph(3, 3))


# ---------------------------------------------------------------- robustness
@pytest.mark.parametrize("N", [3, 4, 5, 6, 7, 8, 11, 12])
def test_complete_graph_is_ceil_half_robust(N):
    A = _A(nx.complete_graph(N))
    assert max_r_robust(A) == (N + 1) // 2


@pytest.mark.parametrize("G", [nx.path_graph(5), nx.cycle_graph(8), nx.grid_2d_graph(3, 3), nx.grid_2d_graph(2, 2)])
def test_sparse_road_graphs_are_only_1_robust(G):
    A = _A(G)
    assert is_r_robust(A, 1) and not is_r_robust(A, 2)
    assert not is_rs_robust(A, 2, 2)


def test_disconnected_graph_not_1_robust():
    A = _A(nx.disjoint_union(nx.path_graph(3), nx.path_graph(3)))
    assert not is_r_robust(A, 1)


def test_witness_violates_definition():
    ok, (S1, S2) = is_rs_robust(GRID3, 2, 2, return_witness=True)
    assert not ok and S1 and S2 and not set(S1) & set(S2)

    def X(S):
        return [i for i in S if sum(GRID3[i, j] for j in range(9) if j not in S) >= 2]

    x1, x2 = X(S1), X(S2)
    assert len(x1) < len(S1) and len(x2) < len(S2) and len(x1) + len(x2) < 2


@pytest.mark.parametrize("seed", range(6))
def test_sampled_check_never_contradicts_exact(seed):
    G = nx.erdos_renyi_graph(9, 0.5, seed=seed)
    A = _A(G)
    for r, s in ((1, 1), (2, 1), (2, 2), (3, 1)):
        exact = bool(is_rs_robust(A, r, s))
        ok, wit, _ = sampled_rs_robust(A, r, s, n_samples=4000, seed=seed)
        if exact:
            assert ok and wit is None          # a robust graph has no counterexample
        if not ok:
            assert not exact


def test_augmentation_reaches_targets_on_grid3x3():
    for target in ("F_total", "F_local"):
        for method in ("greedy", "khop"):
            res = augment_to_robustness(GRID3, F=1, target=target, method=method)
            assert res.satisfied and res.n_added > 0
            assert (res.A >= GRID3).all()                        # only adds links
    prof = robustness_profile(khop_adjacency(GRID3, 2))
    assert prof["F_total_ok"] and prof["F_local_sufficient"]


# ---------------------------------------------------------------- W-MSR rule
def test_filter_removes_only_strict_extremes():
    assert sorted(wmsr_filter(5.0, [1, 2, 9, 10], F=1)) == [2.0, 5.0, 9.0]
    assert sorted(wmsr_filter(5.0, [5, 5, 10], F=1)) == [5.0, 5.0, 5.0]   # equal values kept
    assert sorted(wmsr_filter(5.0, [10, 11], F=1)) == [5.0, 10.0]          # only F removed
    assert sorted(wmsr_filter(5.0, [1, 10], F=0)) == [1.0, 5.0, 10.0]
    assert wmsr_update(5.0, [1, 2, 9, 10], F=1) == pytest.approx(16 / 3)


def test_f_local_choice():
    bad = choose_faulty(GRID3, F=1)
    assert bad == [4] and is_f_local(GRID3, bad, 1)
    assert not is_f_local(GRID3, [1, 3], 1)                     # node 0 would see two


# ------------------------------------------- benign interval invariance
@pytest.mark.parametrize("kind", ["constant", "noise", "drift", "byzantine"])
@pytest.mark.parametrize("seed", range(5))
def test_benign_interval_invariance(kind, seed):
    rng = np.random.default_rng(seed)
    for A in (GRID3, _A(nx.path_graph(5)), khop_adjacency(GRID3, 2), _A(nx.erdos_renyi_graph(10, 0.4, seed=seed))):
        bad = choose_faulty(A, F=1, prefer="random", seed=seed)
        x0 = rng.uniform(0.1, 0.6, A.shape[0])
        fm = FaultModel(kind=kind, value=5.0, sigma=2.0, rate=0.05, low=-5, high=5, seed=seed)
        run = simulate_resilient(A, x0, bad, fm, F=1, rule="wmsr", iters=150)
        assert run.interval_invariant, (kind, seed, run.max_excursion)


def test_plain_averaging_is_dragged_out_of_the_interval():
    x0 = np.linspace(0.2, 0.5, 9)
    run = simulate_resilient(GRID3, x0, [4], FaultModel("constant", value=1.0), rule="average", iters=300)
    assert not run.interval_invariant and run.final_bias > 0.4


@pytest.mark.parametrize("kind", ["constant", "noise", "drift"])
def test_resilient_agreement_on_robust_graph(kind):
    A = khop_adjacency(GRID3, 2)                      # (2,2)- and 3-robust
    assert is_rs_robust(A, 2, 2)
    x0 = np.random.default_rng(0).uniform(0.1, 0.6, 9)
    run = simulate_resilient(A, x0, [4], FaultModel(kind, value=1.0, sigma=0.5, rate=0.01), rule="wmsr", iters=400)
    assert run.interval_invariant and run.final_spread < 1e-6


def test_non_robust_graph_may_not_agree():
    """Path graph with the middle node faulty: the ends are cut off (necessity of robustness)."""
    run = simulate_resilient(_A(nx.path_graph(3)), [0.2, 0.9, 0.6], [1], FaultModel("constant", 0.9), rule="wmsr")
    assert run.final_spread == pytest.approx(0.4)


# ---------------------------------------------------- controller integration
def _board(values: dict[int, float]):
    info = {j: SimpleNamespace(cons={"rho": v}) for j, v in values.items()}
    return SimpleNamespace(info=info, pop_fresh=lambda ch: set(values))


def test_wmsr_consensus_interface():
    view = SimpleNamespace(neighbors=(1, 2, 3))
    c = WMSRConsensus(view, {"F": 1, "iterations_per_decision": 5, "mode": "static"}, period=5)
    assert c.exchange_step(0) and c.exchange_step(4) and c.exchange_step(5)
    c0 = WMSRConsensus(view, {"iterations_per_decision": 0}, period=5)
    assert not c0.exchange_step(0)
    c.update(0, _board({}), 0.3)                                   # initialise
    est = c.update(1, _board({1: 0.2, 2: 0.4, 3: 5.0}), 0.3)       # 5.0 (largest) and 0.2 (smallest) dropped
    assert c.x == pytest.approx(np.mean([0.3, 0.4]))
    assert c.n_filtered == 2
    assert est == pytest.approx(0.3)                               # static: estimate updated at period start


def test_wmsr_consensus_f0_is_plain_average():
    view = SimpleNamespace(neighbors=(1, 2))
    c = WMSRConsensus(view, {"F": 0, "mode": "dynamic"}, period=5)
    c.update(0, _board({}), 0.3)
    c.update(1, _board({1: 0.0, 2: 0.9}), 0.3)
    assert c.x == pytest.approx(0.4)


def test_fault_injector_rewrites_only_consensus_values():
    m = Message(sender=4, receiver=1, t_sent=10, queues={7: 3.0}, load_value=0.3, extra={"cons": {"cw": 50.0}, "C": 60})
    inj = FaultInjector({"kind": "constant", "value": 1.0}, channels={"cw": {"kind": "constant", "value": 3.0, "relative": True}})
    (out,) = inj.rewrite([m])
    assert out.load_value == 1.0 and out.extra["cons"]["cw"] == 150.0
    assert out.queues == {7: 3.0} and out.extra["C"] == 60 and out.t_sent == 10
    assert m.load_value == 0.3                                      # original untouched
