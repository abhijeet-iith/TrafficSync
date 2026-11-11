"""Kripke models, information sets and the epistemic guard."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from trafficsync.control.epistemic import EpistemicGuard
from trafficsync.logic.kripke import KripkeModel, hidden_coin, interval_information_set, partition_from_observation


def test_hidden_coin():
    M = hidden_coin()
    assert M.is_s5("a") and M.is_s5("b")
    assert M.holds(("K", "a", "heads"), "H")                     # a looked: knows heads
    assert not M.holds(("K", "b", "heads"), "H")                 # b did not look
    assert M.holds(("M", "b", "heads"), "H") and M.holds(("M", "b", ("not", "heads")), "H")
    assert M.holds(("K", "b", ("or", "heads", ("not", "heads"))), "T")     # tautology known
    assert M.holds(("K", "b", ("or", ("K", "a", "heads"), ("K", "a", ("not", "heads")))), "H")   # b knows a knows
    assert not M.holds(("C", ["a", "b"], "heads"), "H")
    assert M.truth_set(("K", "a", "heads")) == {"H"}


def test_knowledge_is_set_containment():
    W = list(range(6))
    M = KripkeModel(W, {"i": partition_from_observation(W, lambda w: w // 2)}, {"p": {0, 1, 2}})
    for w in W:
        assert M.holds(("K", "i", "p"), w) == (M.info_set("i", w) <= {0, 1, 2})
    assert M.truth_set(("K", "i", "p")) == {0, 1}


def test_information_set_widens_with_age():
    widths = [np.diff(interval_information_set(20, age, 40, 2.0, 0.5))[0] for age in range(0, 30, 5)]
    assert all(b >= a for a, b in zip(widths, widths[1:])) and widths[-1] > widths[0]
    assert interval_information_set(20, 1000, 40, 2.0, 0.5) == (0, 40)


def _view_with_one_downstream():
    links = {5: SimpleNamespace(capacity=40, sat_flow=0.5), 6: SimpleNamespace(capacity=40, sat_flow=0.5)}
    mv = SimpleNamespace(out_link=5)
    return SimpleNamespace(links=links, outgoing=(5, 6), downstream_node={5: 1, 6: None},
                           phases=((mv,), (SimpleNamespace(out_link=6),)), num_phases=2)


def test_knowledge_shrinks_with_message_age():
    view = _view_with_one_downstream()
    g = EpistemicGuard(view, {"delta0": 2.0, "lam_max": 0.5, "delta_room": 2.0, "max_info_age_s": 120})
    board = SimpleNamespace(info={1: SimpleNamespace(occupancy={5: 25 / 40}, t_data=100)})
    known = [g.knows_room(board, 5, 100 + age) for age in range(0, 60)]
    # n_hat = 25, room needs n <= 38: max I = floor(25 + 2 + 0.5 age) <= 38  ->  age <= 23
    assert all(known[:24]) and not any(known[24:])
    assert all(g.possible_room(board, 5, 100 + age) for age in range(60))      # still possible
    assert g.knows_room(board, 6, 500)                                         # exit link: always room
    assert not g.knows_room(SimpleNamespace(info={}), 5, 0)                     # no message: no knowledge


def test_feasible_set_and_fallback():
    view = _view_with_one_downstream()
    g = EpistemicGuard(view, {"delta0": 2.0, "lam_max": 0.5})
    full = SimpleNamespace(info={1: SimpleNamespace(occupancy={5: 39 / 40}, t_data=0)})
    assert g.feasible_phases(full, 0) == [1]
    obs = SimpleNamespace(phase=0, incoming={})
    view.phase_link_shares = lambda p: {}
    assert g.choose(obs, 0, full, 0) == 1                     # phase 0 would release into a full link


