"""(r, s)-robustness of a communication graph.

Definitions (LeBlanc et al., course notes).  For an undirected graph and a node set ``S``:

* ``X_S^r = {i in S : |N_i \\ S| >= r}`` - the nodes of ``S`` with at least ``r`` neighbours
  outside ``S`` (``S`` is *r-reachable* if ``X_S^r`` is non-empty).
* The graph is **(r, s)-robust** if for every pair of non-empty, disjoint ``S1, S2``
  at least one of the following holds:
  (1) ``X_{S1}^r = S1``;  (2) ``X_{S2}^r = S2``;  (3) ``|X_{S1}^r| + |X_{S2}^r| >= s``.
* **r-robust** = (r, 1)-robust (at least one of every disjoint pair is r-reachable).

Results used in E9 for ``F`` faulty (malicious) neighbours:

* F-total model: W-MSR reaches resilient consensus **iff** the graph is (F+1, F+1)-robust.
* F-local model: (2F+1)-robust is **sufficient** (needs min degree >= 2F+1, and since K_N is
  only ceil(N/2)-robust, N >= 5 for F = 1: corridor3 and grid2x2 can never meet it).

Exact check (``N <= 12``): every pair is enumerated with bitmasks (``3^N`` pairs; 0.5 M for
N = 12, about a second per check).  Above that, :func:`sampled_rs_robust` searches random
and greedy-grown subset pairs; it can *refute* robustness (returns a counterexample) but a
pass is only "no counterexample found in M samples", never a proof.

:func:`augment_to_robustness` adds communication links (k-hop first, then greedy single
edges) until the target holds, and reports how many links were needed.  Communication links
need not coincide with roads (V2I / cloud relay).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Iterable

import numpy as np

__all__ = [
    "adjacency_of",
    "reachable_counts",
    "is_rs_robust",
    "is_r_robust",
    "max_r_robust",
    "max_s_for_r",
    "sampled_rs_robust",
    "robustness_profile",
    "khop_adjacency",
    "augment_to_robustness",
    "AugmentResult",
    "EXACT_MAX_N",
]

EXACT_MAX_N = 12


def adjacency_of(G: Any) -> np.ndarray:
    """Symmetric 0/1 adjacency from a ``Network`` (``.A``), a networkx graph or an array."""
    if hasattr(G, "A") and isinstance(getattr(G, "A"), np.ndarray):
        A = np.asarray(G.A, dtype=float)
    elif hasattr(G, "nodes") and hasattr(G, "edges"):
        import networkx as nx

        A = nx.to_numpy_array(G, nodelist=sorted(G.nodes()), dtype=float)
    else:
        A = np.asarray(G, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be square")
    A = ((A + A.T) > 0).astype(float)
    np.fill_diagonal(A, 0.0)
    return A


def _nbr_masks(A: np.ndarray) -> list[int]:
    N = A.shape[0]
    return [sum(1 << j for j in range(N) if A[i, j] > 0) for i in range(N)]


def reachable_counts(A: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
    """For every subset mask ``S`` (0 .. 2^N-1): ``(|X_S^r|, |S|)`` as two int arrays."""
    A = adjacency_of(A)
    N = A.shape[0]
    if N > 20:
        raise ValueError("bitmask enumeration is limited to N <= 20")
    M = 1 << N
    masks = np.arange(M, dtype=np.int64)
    member = ((masks[:, None] >> np.arange(N)[None, :]) & 1).astype(bool)      # (M, N)
    outside = ~member
    # neighbours of i outside S = sum_j A_ij [j not in S]
    n_out = outside.astype(np.int32) @ A.T.astype(np.int32)                      # (M, N)
    in_X = member & (n_out >= r)
    return in_X.sum(axis=1).astype(np.int32), member.sum(axis=1).astype(np.int32)


def is_rs_robust(G: Any, r: int, s: int, return_witness: bool = False):
    """Exact (r, s)-robustness by enumerating all disjoint non-empty pairs (``N <= 12``).

    With ``return_witness`` returns ``(ok, (S1, S2))`` where the pair violates the condition
    (node lists) or ``None``.
    """
    A = adjacency_of(G)
    N = A.shape[0]
    if N > EXACT_MAX_N:
        raise ValueError(f"exact check is limited to N <= {EXACT_MAX_N}; use sampled_rs_robust")
    if r < 0 or s < 1:
        raise ValueError("need r >= 0 and s >= 1")
    if r == 0:
        return (True, None) if return_witness else True
    X, size = reachable_counts(A, r)
    full = (1 << N) - 1
    fullX = X == size                       # condition (1)/(2) for that set
    for S1 in range(1, full + 1):
        if fullX[S1]:
            continue
        rest = full & ~S1
        x1 = int(X[S1])
        sub = rest
        while sub:
            if not fullX[sub] and x1 + int(X[sub]) < s:
                if return_witness:
                    nodes = lambda m: [i for i in range(N) if m >> i & 1]  # noqa: E731
                    return False, (nodes(S1), nodes(sub))
                return False
            sub = (sub - 1) & rest
    return (True, None) if return_witness else True


def is_r_robust(G: Any, r: int) -> bool:
    return bool(is_rs_robust(G, r, 1))


def max_r_robust(G: Any) -> int:
    """Largest ``r`` with the graph r-robust (``<= ceil(N/2)``, ``<= min degree``)."""
    A = adjacency_of(G)
    N = A.shape[0]
    if N <= 1:
        return 0
    r_hi = min(int(A.sum(axis=1).min()), (N + 1) // 2)
    best = 0
    for r in range(1, r_hi + 1):
        if is_r_robust(A, r):
            best = r
        else:
            break
    return best


def max_s_for_r(G: Any, r: int) -> int:
    """Largest ``s`` (0 if not even r-robust) such that the graph is (r, s)-robust."""
    A = adjacency_of(G)
    N = A.shape[0]
    best = 0
    for s in range(1, N + 1):
        if is_rs_robust(A, r, s):
            best = s
        else:
            break
    return best


def _violates(A: np.ndarray, S1: np.ndarray, S2: np.ndarray, r: int, s: int) -> bool:
    out1 = (A[S1][:, ~S1].sum(axis=1) >= r) if S1.any() else np.array([], bool)
    out2 = (A[S2][:, ~S2].sum(axis=1) >= r) if S2.any() else np.array([], bool)
    full1, full2 = bool(out1.all()), bool(out2.all())
    return (not full1) and (not full2) and int(out1.sum()) + int(out2.sum()) < s


def sampled_rs_robust(G: Any, r: int, s: int, n_samples: int = 20000, seed: int = 0):
    """Randomised search for a pair violating (r, s)-robustness (any ``N``).

    Samples (i) random disjoint pairs and (ii) pairs of connected sets grown greedily from
    random seeds (the usual worst cases: dense clusters with few outside links).
    Returns ``(no_counterexample_found, witness_or_None, n_checked)``.
    """
    A = adjacency_of(G)
    N = A.shape[0]
    rng = np.random.default_rng(seed)
    nb = [np.flatnonzero(A[i]) for i in range(N)]

    def grow(start: int, size: int, banned: np.ndarray) -> np.ndarray:
        S = np.zeros(N, bool)
        S[start] = True
        frontier = set(nb[start]) - set(np.flatnonzero(banned))
        while S.sum() < size and frontier:
            j = rng.choice(sorted(frontier))
            S[j] = True
            frontier |= set(nb[j])
            frontier -= set(np.flatnonzero(S | banned))
        return S

    for k in range(n_samples):
        if k % 2 == 0:
            lab = rng.integers(0, 3, size=N)          # 0 none, 1 -> S1, 2 -> S2
            S1, S2 = lab == 1, lab == 2
        else:
            a = int(rng.integers(N))
            S1 = grow(a, int(rng.integers(1, max(2, N // 2) + 1)), np.zeros(N, bool))
            rest = np.flatnonzero(~S1)
            if rest.size == 0:
                continue
            b = int(rng.choice(rest))
            S2 = grow(b, int(rng.integers(1, max(2, rest.size) + 1)), S1)
        if not S1.any() or not S2.any():
            continue
        if _violates(A, S1, S2, r, s):
            return False, (np.flatnonzero(S1).tolist(), np.flatnonzero(S2).tolist()), k + 1
    return True, None, n_samples


def robustness_profile(G: Any, F: int = 1, n_samples: int = 20000, seed: int = 0) -> dict:
    """Summary for E9: min degree, max r, and the two resilience conditions for ``F``.

    ``exact`` tells whether the answers are proofs (N <= 12) or sampled (no counterexample).
    """
    A = adjacency_of(G)
    N = A.shape[0]
    exact = N <= EXACT_MAX_N
    out: dict[str, Any] = {"N": N, "edges": int(A.sum() // 2), "min_degree": int(A.sum(axis=1).min()) if N else 0,
                           "exact": exact, "F": F}
    if exact:
        out["max_r"] = max_r_robust(A)
        out[f"rs_{F+1}_{F+1}"] = bool(is_rs_robust(A, F + 1, F + 1))
        out[f"r_{2*F+1}"] = bool(is_r_robust(A, 2 * F + 1)) if out["min_degree"] >= 2 * F + 1 else False
        out["max_s_for_r_F+1"] = max_s_for_r(A, F + 1)
    else:
        ok1, w1, _ = sampled_rs_robust(A, F + 1, F + 1, n_samples, seed)
        ok2 = False
        if out["min_degree"] >= 2 * F + 1:
            ok2, _, _ = sampled_rs_robust(A, 2 * F + 1, 1, n_samples, seed + 1)
        out[f"rs_{F+1}_{F+1}"] = bool(ok1)
        out[f"r_{2*F+1}"] = bool(ok2)
        out["max_r"] = None
        out["max_s_for_r_F+1"] = None
    out["F_total_ok"] = out[f"rs_{F+1}_{F+1}"]          # necessary and sufficient (F-total)
    out["F_local_sufficient"] = out[f"r_{2*F+1}"]       # sufficient (F-local)
    return out


def khop_adjacency(A: np.ndarray, k: int) -> np.ndarray:
    """Connect every pair within ``k`` hops of ``A`` (k = 1 returns ``A``)."""
    A = adjacency_of(A)
    N = A.shape[0]
    R = np.eye(N)
    P = np.eye(N)
    for _ in range(k):
        P = (P @ (A + np.eye(N)) > 0).astype(float)
        R = np.maximum(R, P)
    out = (R > 0).astype(float)
    np.fill_diagonal(out, 0.0)
    return out


@dataclass
class AugmentResult:
    A: np.ndarray
    added: list[tuple[int, int]] = field(default_factory=list)
    method: str = ""
    k_hop: int = 1
    satisfied: bool = False
    original_edges: int = 0

    @property
    def n_added(self) -> int:
        return len(self.added)


def _check(A: np.ndarray, target: str, F: int, n_samples: int, verify: int = 0) -> bool:
    """Exact for N <= 12; sampled above (a pass with ``n_samples`` is re-checked with
    ``verify`` further samples from other seeds, so a weak pass is not accepted)."""
    if target == "F_total":
        r, s = F + 1, F + 1
    elif target == "F_local":
        if A.sum(axis=1).min() < 2 * F + 1:
            return False
        r, s = 2 * F + 1, 1
    else:
        raise ValueError("target must be F_total or F_local")
    if A.shape[0] <= EXACT_MAX_N:
        return bool(is_rs_robust(A, r, s))
    if not sampled_rs_robust(A, r, s, n_samples, seed=0)[0]:
        return False
    k = 1
    while verify > 0:
        m = min(verify, n_samples)
        if not sampled_rs_robust(A, r, s, m, seed=1000 + k)[0]:
            return False
        verify -= m
        k += 1
    return True


def augment_to_robustness(G: Any, F: int = 1, target: str = "F_total", method: str = "greedy",
                          pos: dict | None = None, n_samples: int = 5000, max_k: int = 6,
                          verify_samples: int = 60000) -> AugmentResult:
    """Add communication links until the robustness target holds.

    ``method="khop"``: smallest k such that the k-hop graph satisfies the target.
    ``method="greedy"``: add one edge at a time; candidates are non-edges whose ends have the
    smallest degree, ties broken by shortest hop distance (short-range V2I links first) then
    by Euclidean distance if ``pos`` is given.  Greedy is not guaranteed minimal, but it is
    checked against the k-hop count in E9.
    """
    A0 = adjacency_of(G)
    N = A0.shape[0]
    res = AugmentResult(A=A0.copy(), method=method, original_edges=int(A0.sum() // 2))
    if _check(A0, target, F, n_samples, verify_samples):
        res.satisfied = True
        return res
    if method == "khop":
        for k in range(2, max_k + 1):
            Ak = khop_adjacency(A0, k)
            if _check(Ak, target, F, n_samples, verify_samples):
                res.A, res.k_hop, res.satisfied = Ak, k, True
                res.added = [(i, j) for i, j in combinations(range(N), 2) if Ak[i, j] > 0 and A0[i, j] == 0]
                return res
        return res
    if method != "greedy":
        raise ValueError("method must be greedy or khop")
    import networkx as nx

    A = A0.copy()
    for _ in range(N * (N - 1) // 2):
        Gx = nx.from_numpy_array(A)
        hop = dict(nx.all_pairs_shortest_path_length(Gx))
        deg = A.sum(axis=1)
        cands = []
        for i, j in combinations(range(N), 2):
            if A[i, j] > 0:
                continue
            h = hop.get(i, {}).get(j, N + 1)
            d = 0.0
            if pos is not None:
                d = float(np.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1]))
            cands.append((min(deg[i], deg[j]), deg[i] + deg[j], h, d, i, j))
        if not cands:
            break
        # try the best few candidates; keep the first that satisfies, else the best one
        cands.sort()
        best = cands[0]
        chosen = None
        for c in cands[: min(len(cands), 8)]:
            B = A.copy()
            B[c[4], c[5]] = B[c[5], c[4]] = 1.0
            if _check(B, target, F, n_samples, verify_samples):
                chosen = c
                break
        c = chosen or best
        A[c[4], c[5]] = A[c[5], c[4]] = 1.0
        res.added.append((c[4], c[5]))
        if chosen is not None:
            res.A, res.satisfied = A, True
            return res
    res.A = A
    res.satisfied = _check(A, target, F, n_samples, verify_samples)
    return res


def _pairs(nodes: Iterable[int]) -> list[tuple[int, int]]:  # pragma: no cover - helper for notebooks
    return list(combinations(sorted(nodes), 2))
