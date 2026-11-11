"""Congestion spreading as an epidemic process.

"Intersection i is congested" (infected) when its load ``rho_i`` stays above ``threshold``
(0.7) for more than ``min_on_s`` (60 s); it recovers when the load falls back below the
threshold (gaps shorter than ``merge_gap_s`` are bridged so one episode is not split by a
single green).  Loads are ``rho_i = mean_{l in I_i} n_l / C_l`` from the
recorded ground truth - analysis, never control.

Model (network SIR / SIS, mean field)::

    d iota_i/dt = beta_0 s_i + beta s_i sum_j a_ij iota_j - gamma iota_i

``beta_0`` is a background (demand-driven) infection rate that a pure SIR model does
not have; without it every congestion episode that starts with no congested neighbour
would be forced onto ``beta``.  Linearised early growth: ``d iota/dt ~ (beta A - gamma I)
iota``, so spreading is suppressed when **beta lambda_max(A) / gamma < 1**.

Estimation (maximum likelihood, pooled over runs):
* ``gamma``  = recoveries / total congested time (exponential durations, right-censored);
* ``beta, beta_0``: Poisson hazard ``beta_0 + beta E_i(t)`` for a non-congested node with
  exposure ``E_i(t) = sum_j a_ij iota_j(t)``; log-likelihood
  ``sum_onsets log(beta_0 + beta E) - sum_{i,t: s_i(t)=1} (beta_0 + beta E_i(t)) dt``.
Recovery to "susceptible" (SIS) is what the logs show - congestion recurs - so estimation
treats every uncongested node as susceptible; the SIR curve is reported for comparison.
Both models share the threshold ``beta lambda_max / gamma``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = [
    "CongestionEvent",
    "congestion_states",
    "congestion_events",
    "spectral_epidemic",
    "RateEstimate",
    "estimate_rates",
    "mean_field",
    "observed_spreading",
    "spreading_contribution",
    "prioritised_agents",
]


@dataclass(frozen=True)
class CongestionEvent:
    node: int
    onset: int          # first second of the qualifying run
    end: int            # first second after recovery (== T if censored)
    censored: bool      # still congested at the end of the record

    @property
    def duration(self) -> int:
        return self.end - self.onset


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of True as ``[start, end)``."""
    m = np.r_[False, mask.astype(bool), False]
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def congestion_states(loads: np.ndarray, threshold: float = 0.7, min_on_s: int = 60,
                      merge_gap_s: int = 30, dt: float = 1.0) -> tuple[np.ndarray, list[CongestionEvent]]:
    """Infected indicator ``(T, N)`` and the list of congestion events.

    ``loads[t, i]`` sampled every ``dt`` seconds.  A run above ``threshold`` qualifies if it
    lasts more than ``min_on_s``; runs separated by less than ``merge_gap_s`` are merged
    before that test.
    """
    X = np.asarray(loads, dtype=float)
    T, N = X.shape
    on_n = int(np.ceil(min_on_s / dt))
    gap_n = int(np.ceil(merge_gap_s / dt))
    states = np.zeros((T, N), dtype=bool)
    events: list[CongestionEvent] = []
    for i in range(N):
        runs = _runs(X[:, i] > threshold)
        merged: list[list[int]] = []
        for a, b in runs:
            if merged and a - merged[-1][1] < gap_n:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        for a, b in merged:
            if b - a > on_n:
                states[a:b, i] = True
                events.append(CongestionEvent(i, a, b, censored=(b >= T)))
    events.sort(key=lambda e: (e.onset, e.node))
    return states, events


def congestion_events(states: np.ndarray) -> list[CongestionEvent]:
    """Events from an infected indicator (no thresholding)."""
    T, N = states.shape
    out = [CongestionEvent(i, a, b, b >= T) for i in range(N) for a, b in _runs(states[:, i])]
    return sorted(out, key=lambda e: (e.onset, e.node))


def spectral_epidemic(A: np.ndarray) -> dict:
    """``lambda_max(A)`` and the (positive, unit-sum) dominant eigenvector = eigenvector
    centrality of the road graph."""
    A = np.asarray(A, dtype=float)
    A = (A + A.T) / 2
    vals, vecs = np.linalg.eigh(A)
    v = vecs[:, -1]
    v = np.abs(v) / np.abs(v).sum()
    return {"lambda_max_A": float(vals[-1]), "v_max": v, "ranking": np.argsort(-v, kind="stable").tolist()}


@dataclass
class RateEstimate:
    beta: float
    beta0: float
    gamma: float
    lambda_max: float
    n_onsets: int
    n_onsets_exposed: int
    n_recoveries: int
    infected_time_s: float
    susceptible_time_s: float
    exposure_time_s: float          # sum over susceptible node-seconds of E_i(t)
    loglik: float = float("nan")
    notes: list[str] = field(default_factory=list)

    @property
    def threshold_ratio(self) -> float:
        """``beta lambda_max(A) / gamma`` (> 1: spreading predicted)."""
        if self.gamma <= 0:
            return float("inf") if self.beta > 0 else float("nan")
        return self.beta * self.lambda_max / self.gamma

    @property
    def spreading_predicted(self) -> bool:
        return self.threshold_ratio > 1.0

    @property
    def mean_duration_s(self) -> float:
        return 1.0 / self.gamma if self.gamma > 0 else float("inf")

    def as_dict(self) -> dict:
        return {"beta": self.beta, "beta0": self.beta0, "gamma": self.gamma, "lambda_max_A": self.lambda_max,
                "R_spectral": self.threshold_ratio, "spreading_predicted": self.spreading_predicted,
                "mean_duration_s": self.mean_duration_s, "n_onsets": self.n_onsets,
                "n_onsets_exposed": self.n_onsets_exposed, "n_recoveries": self.n_recoveries,
                "infected_time_s": self.infected_time_s, "susceptible_time_s": self.susceptible_time_s}


def _exposure(states: np.ndarray, A: np.ndarray) -> np.ndarray:
    return states.astype(float) @ np.asarray(A, float).T        # E[t, i] = sum_j a_ij iota_j(t)


def estimate_rates(state_list: Sequence[np.ndarray], A: np.ndarray, dt: float = 1.0,
                   background: bool = True, t_from: int = 0) -> RateEstimate:
    """Pooled MLE of ``(beta, beta_0, gamma)`` from one or more infected indicators.

    Only ``t >= t_from`` (e.g. after warm-up) is used; an episode already running at
    ``t_from`` counts towards infected time and recoveries but not as an onset.
    """
    from scipy.optimize import minimize

    A = np.asarray(A, float)
    lam = spectral_epidemic(A)["lambda_max_A"]
    E_on: list[float] = []
    sum_s = 0.0
    sum_sE = 0.0
    inf_time = 0.0
    n_rec = 0
    for S in state_list:
        S = np.asarray(S, bool)
        E = _exposure(S, A)
        T = S.shape[0]
        for t in range(max(1, t_from), T):
            new = S[t] & ~S[t - 1]
            E_on.extend(E[t - 1, new].tolist())          # exposure just before the onset
            sus = ~S[t - 1]
            sum_s += float(sus.sum()) * dt
            sum_sE += float(E[t - 1, sus].sum()) * dt
            n_rec += int((S[t - 1] & ~S[t]).sum())
        inf_time += float(S[t_from:].sum()) * dt
    E_on_arr = np.asarray(E_on, float)
    n_on = int(E_on_arr.size)
    n_exp = int((E_on_arr > 0).sum())
    gamma = n_rec / inf_time if inf_time > 0 else 0.0
    notes: list[str] = []
    if n_on == 0:
        return RateEstimate(0.0, 0.0, gamma, lam, 0, 0, n_rec, inf_time, sum_s, sum_sE, 0.0,
                            ["no congestion onsets: beta not identifiable (set to 0)"])
    if not background:
        beta = n_on / sum_sE if sum_sE > 0 else 0.0
        if n_on > n_exp:
            notes.append(f"{n_on - n_exp} onsets had no congested neighbour; pure SIR attributes them to beta")
        ll = float(np.sum(np.log(np.maximum(beta * E_on_arr, 1e-300))) - beta * sum_sE)
        return RateEstimate(beta, 0.0, gamma, lam, n_on, n_exp, n_rec, inf_time, sum_s, sum_sE, ll, notes)

    def nll(p: np.ndarray) -> float:
        b0, b = np.exp(p)
        h = b0 + b * E_on_arr
        return -(float(np.sum(np.log(h))) - b0 * sum_s - b * sum_sE)

    b0_init = max(1e-9, (n_on - n_exp) / max(sum_s, 1e-9))
    b_init = max(1e-9, n_exp / max(sum_sE, 1e-9))
    best = None
    for x0 in (np.log([b0_init, b_init]), np.log([n_on / max(sum_s, 1e-9), 1e-6]), np.log([1e-9, n_on / max(sum_sE, 1e-9)])):
        r = minimize(nll, x0, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 5000})
        if best is None or r.fun < best.fun:
            best = r
    b0, b = np.exp(best.x)
    if b < 1e-8:
        notes.append("beta ~ 0: onsets are not more frequent next to congested neighbours")
    if sum_sE == 0:
        notes.append("no exposure observed (no congested node ever had a congested neighbour)")
    return RateEstimate(float(b), float(b0), gamma, lam, n_on, n_exp, n_rec, inf_time, sum_s, sum_sE,
                        -float(best.fun), notes)


def mean_field(A: np.ndarray, beta: float, gamma: float, iota0: np.ndarray | float, T: float,
               beta0: float = 0.0, model: str = "SIS", dt: float = 1.0) -> dict:
    """Integrate the mean-field network model (forward Euler, ``dt`` seconds).

    Returns ``{"t", "iota" (T x N), "fraction" (mean over nodes)}``; for SIR also ``"r"``.
    """
    A = np.asarray(A, float)
    N = A.shape[0]
    n = int(np.ceil(T / dt)) + 1
    io = np.full(N, float(iota0)) if np.isscalar(iota0) else np.asarray(iota0, float).copy()
    rr = np.zeros(N)
    I = np.empty((n, N))
    R = np.empty((n, N))
    for k in range(n):
        I[k], R[k] = io, rr
        s = 1.0 - io - (rr if model == "SIR" else 0.0)
        inf = (beta0 + beta * (A @ io)) * s
        rec = gamma * io
        io = np.clip(io + dt * (inf - rec), 0.0, 1.0)
        if model == "SIR":
            rr = np.clip(rr + dt * rec, 0.0, 1.0)
    out = {"t": np.arange(n) * dt, "iota": I, "fraction": I.mean(axis=1)}
    if model == "SIR":
        out["r"] = R
    return out


def observed_spreading(states: np.ndarray, A: np.ndarray, t_from: int = 0) -> dict:
    """What actually happened in one run (``t >= t_from``)."""
    S = np.asarray(states, bool)[t_from:]
    A = np.asarray(A, float)
    T, N = S.shape
    if T == 0:
        return {}
    E = _exposure(S, A)
    on_t, on_i = np.nonzero(S[1:] & ~S[:-1])
    exposed = E[on_t, on_i] > 0
    frac = S.mean(axis=1)
    return {
        "mean_fraction": float(frac.mean()),
        "peak_fraction": float(frac.max()),
        "ever_fraction": float(S.any(axis=0).mean()),
        "n_onsets": int(on_t.size),
        "n_onsets_exposed": int(exposed.sum()),
        "share_onsets_exposed": float(exposed.mean()) if on_t.size else float("nan"),
        "congested_node_s": float(S.sum()),
        "final_fraction": float(frac[-1]),
    }


def spreading_contribution(state_list: Sequence[np.ndarray], A: np.ndarray, t_from: int = 0) -> np.ndarray:
    """Per node: neighbour onsets that happened while the node was congested, each onset
    split equally among its congested neighbours (an attribution, not a causal estimate)."""
    A = np.asarray(A, float)
    N = A.shape[0]
    out = np.zeros(N)
    for S in state_list:
        S = np.asarray(S, bool)[t_from:]
        for t in range(1, S.shape[0]):
            new = np.flatnonzero(S[t] & ~S[t - 1])
            for j in new:
                src = np.flatnonzero((A[j] > 0) & S[t - 1])
                if src.size:
                    out[src] += 1.0 / src.size
    return out


def prioritised_agents(A: np.ndarray, k: int) -> list[int]:
    """The ``k`` intersections with the largest eigenvector centrality (H6 'vaccination')."""
    return spectral_epidemic(A)["ranking"][: int(k)]
