"""Kripke models and epistemic logic S5.

A model ``M = (W, ~_1..~_n, V)``: worlds, one indistinguishability (equivalence) relation per
agent, and a valuation ``V(p) subset W``.  The information set of agent ``i`` in world ``w`` is
``I_i(w) = {v : w ~_i v}`` and::

    M, w |= K_i phi   iff   I_i(w) subseteq [[phi]]           (set containment)
    M_i phi = not K_i not phi   (phi is possible for i)

Formulas are nested tuples: ``"p"``, ``("not", f)``, ``("and", f, g)``, ``("or", f, g)``,
``("K", i, f)``, ``("M", i, f)``, ``("E", group, f)`` (everybody knows), ``("C", group, f)``
(common knowledge: reachability over the union of the group's relations).

:func:`hidden_coin` builds the course's toy model; :func:`interval_information_set` is the
traffic version used by ``control/epistemic.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Iterable

import numpy as np

__all__ = ["KripkeModel", "hidden_coin", "interval_information_set", "partition_from_observation"]


@dataclass
class KripkeModel:
    worlds: list
    relations: dict                                  # agent -> partition (list of sets) or set of (w, v) pairs
    valuation: dict                                  # proposition -> set of worlds

    def __post_init__(self) -> None:
        """``relations[agent]`` is either a *partition* (list of sets of worlds) or an explicit
        *set* of ``(w, v)`` pairs."""
        rel = {}
        for a, r in self.relations.items():
            if isinstance(r, list):
                rel[a] = {(w, v) for cell in r for w in cell for v in cell}
            else:
                rel[a] = set(r)
        self.relations = rel

    # ------------------------------------------------------------------ frames
    def info_set(self, agent: Hashable, w: Any) -> set:
        return {v for (u, v) in self.relations[agent] if u == w}

    def is_s5(self, agent: Hashable) -> bool:
        """Reflexive, symmetric and transitive (equivalence relation)."""
        R = self.relations[agent]
        refl = all((w, w) in R for w in self.worlds)
        sym = all((v, u) in R for (u, v) in R)
        trans = all((u, x) in R for (u, v) in R for (v2, x) in R if v == v2)
        return refl and sym and trans

    # --------------------------------------------------------------- semantics
    def truth_set(self, f: Any) -> set:
        W = set(self.worlds)
        if isinstance(f, str):
            return set(self.valuation.get(f, set()))
        op = f[0]
        if op == "not":
            return W - self.truth_set(f[1])
        if op == "and":
            return self.truth_set(f[1]) & self.truth_set(f[2])
        if op == "or":
            return self.truth_set(f[1]) | self.truth_set(f[2])
        if op == "implies":
            return (W - self.truth_set(f[1])) | self.truth_set(f[2])
        if op == "K":
            t = self.truth_set(f[2])
            return {w for w in W if self.info_set(f[1], w) <= t}
        if op == "M":
            t = self.truth_set(f[2])
            return {w for w in W if self.info_set(f[1], w) & t}
        if op == "E":
            out = W
            for a in f[1]:
                out = out & self.truth_set(("K", a, f[2]))
            return out
        if op == "C":
            t = self.truth_set(f[2])
            union = set().union(*(self.relations[a] for a in f[1]))
            out = set()
            for w in W:
                seen, stack = set(), [v for (u, v) in union if u == w]
                while stack:
                    v = stack.pop()
                    if v in seen:
                        continue
                    seen.add(v)
                    stack += [x for (u, x) in union if u == v]
                if seen <= t:
                    out.add(w)
            return out
        raise ValueError(f"unknown operator {op!r}")

    def holds(self, f: Any, w: Any) -> bool:
        return w in self.truth_set(f)


def partition_from_observation(worlds: Iterable, observe) -> list[set]:
    """Partition worlds by what an agent observes (``observe(w)``): S5 by construction."""
    cells: dict = {}
    for w in worlds:
        cells.setdefault(observe(w), set()).add(w)
    return list(cells.values())


def hidden_coin() -> KripkeModel:
    """Course toy model: a coin lies heads (H) or tails (T); agent ``a`` looked at it, agent
    ``b`` did not.  Worlds H, T; ``heads`` true in H."""
    W = ["H", "T"]
    rel = {"a": partition_from_observation(W, lambda w: w),      # a distinguishes the worlds
           "b": partition_from_observation(W, lambda w: 0)}      # b cannot
    return KripkeModel(W, rel, {"heads": {"H"}})


def interval_information_set(n_hat: float, age: float, capacity: int, delta0: float, lam_max: float) -> tuple[int, int]:
    """Worlds ``n in {0..C}`` compatible with a report ``n_hat`` of age ``age``:
    ``|n - n_hat| <= delta0 + lam_max * age`` (closed integer interval, clipped to [0, C])."""
    r = float(delta0) + float(lam_max) * max(0.0, float(age))
    lo = int(np.ceil(max(0.0, n_hat - r) - 1e-9))
    hi = int(np.floor(min(float(capacity), n_hat + r) + 1e-9))
    return lo, max(lo, hi)
