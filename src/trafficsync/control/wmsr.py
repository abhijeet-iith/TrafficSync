"""Resilient aggregation with W-MSR.

W-MSR step of a benign agent ``i`` with own value ``x_i`` and neighbour values ``x_j``:

1. ``U = {j : x_j > x_i}`` sorted descending - remove the first ``min(F, |U|)``;
2. ``D = {j : x_j < x_i}`` sorted ascending  - remove the first ``min(F, |D|)``;
3. ``x_i+ = sum_{j in J_i \\ R_i} w_ij x_j`` over the kept set (own value always kept),
   with ``w_ij > 0`` summing to 1 (uniform by default).

Values equal to ``x_i`` are never removed.  A benign agent therefore only ever averages
values inside ``[min, max]`` of benign values (whenever it has at most ``F`` faulty
neighbours), which gives the **invariance of the benign interval** tested in
``tests/test_wmsr.py``.

Fault models (``F``-local, ``F = 1`` in E9): a faulty node keeps broadcasting the *same*
value to all neighbours (malicious, not Byzantine) chosen by :class:`FaultModel`:

* ``constant`` - a fixed extreme value (default 1.0 = "fully congested" load);
* ``noise``    - its true value plus Gaussian noise ``N(0, sigma^2)`` every round;
* ``drift``    - its true value plus ``rate * k`` (a slowly drifting value);
* ``byzantine``- a different random value to each neighbour (stress test only;
  used to show the limits of the malicious-model guarantee).

Two layers:

* :func:`wmsr_update`, :func:`simulate_resilient` - synchronous network-level iteration
  used by E9 part 1 and the tests;
* :class:`WMSRConsensus` - drop-in replacement for ``control.comp.LoadConsensus`` (same
  ``exchange_step / update / x / estimate`` interface) used by CoMP / CoCycle when the
  consensus config has ``aggregation: wmsr``; :class:`FaultInjector` turns a controller into
  a faulty reporter by rewriting the consensus values it sends.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

__all__ = [
    "wmsr_filter",
    "wmsr_update",
    "average_update",
    "FaultModel",
    "choose_faulty",
    "is_f_local",
    "simulate_resilient",
    "ResilientRun",
    "WMSRConsensus",
    "FaultInjector",
    "make_consensus",
]


# ----------------------------------------------------------------------------
# Core rule
# ----------------------------------------------------------------------------
def wmsr_filter(own: float, received: Sequence[float], F: int) -> list[float]:
    """Values kept by W-MSR: own value + neighbours not among the ``F`` most extreme
    strictly-larger or strictly-smaller ones."""
    vals = [float(v) for v in received]
    up = sorted((v for v in vals if v > own), reverse=True)
    dn = sorted(v for v in vals if v < own)
    eq = [v for v in vals if v == own]
    kept = up[min(F, len(up)):] + dn[min(F, len(dn)):] + eq
    return [float(own)] + kept


def wmsr_update(own: float, received: Sequence[float], F: int,
                weights: Callable[[int], float] | None = None) -> float:
    """One W-MSR step for one agent (uniform weights over the kept set by default)."""
    kept = wmsr_filter(own, received, F)
    if weights is None:
        return float(np.mean(kept))
    w = np.array([weights(len(kept))] * len(kept))
    return float(np.dot(w / w.sum(), kept))


def average_update(own: float, received: Sequence[float]) -> float:
    """Plain (unfiltered) neighbourhood average - the vulnerable baseline."""
    return float(np.mean([own, *received]))


# ----------------------------------------------------------------------------
# Fault models
# ----------------------------------------------------------------------------
@dataclass
class FaultModel:
    """What a faulty node reports in round ``k`` given its true value."""

    kind: str = "constant"            # constant | noise | drift | byzantine
    value: float = 1.0                # constant: reported value (or multiplier, see `relative`)
    relative: bool = False            # constant: value * true_value instead of value
    sigma: float = 0.3                # noise: std of the added noise
    rate: float = 0.002               # drift: added per round
    low: float = 0.0                  # byzantine: range of per-neighbour random values
    high: float = 1.0
    seed: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.kind not in ("constant", "noise", "drift", "byzantine"):
            raise ValueError(f"unknown fault kind {self.kind!r}")
        self._rng = np.random.default_rng(self.seed)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "FaultModel":
        d = dict(d or {})
        known = {k: d[k] for k in ("kind", "value", "relative", "sigma", "rate", "low", "high", "seed") if k in d}
        return cls(**known)

    def report(self, k: int, true_value: float, n_receivers: int = 1) -> np.ndarray:
        """Values sent in round ``k`` (one per receiver; identical unless byzantine)."""
        if self.kind == "constant":
            v = self.value * true_value if self.relative else self.value
            return np.full(n_receivers, float(v))
        if self.kind == "noise":
            return np.full(n_receivers, float(true_value + self._rng.normal(0.0, self.sigma)))
        if self.kind == "drift":
            return np.full(n_receivers, float(true_value + self.rate * k))
        return self._rng.uniform(self.low, self.high, size=n_receivers)


def is_f_local(A: np.ndarray, faulty: Sequence[int], F: int) -> bool:
    """Every benign node has at most ``F`` faulty neighbours."""
    A = np.asarray(A) > 0
    bad = np.zeros(A.shape[0], bool)
    bad[list(faulty)] = True
    return bool(((A[:, bad].sum(axis=1) <= F) | bad).all())


def choose_faulty(A: np.ndarray, F: int = 1, n_faulty: int | None = None, seed: int = 0,
                  prefer: str = "central") -> list[int]:
    """Pick faulty nodes respecting the F-local model.

    ``prefer="central"``: highest degree first (worst case for plain averaging);
    ``"random"``: random order.  ``n_faulty`` defaults to ``F``.
    """
    A = np.asarray(A) > 0
    N = A.shape[0]
    rng = np.random.default_rng(seed)
    order = list(np.argsort(-A.sum(axis=1), kind="stable")) if prefer == "central" else list(rng.permutation(N))
    chosen: list[int] = []
    for i in order:
        if len(chosen) >= (F if n_faulty is None else n_faulty):
            break
        if is_f_local(A, chosen + [int(i)], F):
            chosen.append(int(i))
    return chosen


# ----------------------------------------------------------------------------
# Network-level simulation (E9 part 1, tests)
# ----------------------------------------------------------------------------
@dataclass
class ResilientRun:
    X: np.ndarray                 # (K+1, N) states; faulty columns hold what they reported
    benign: np.ndarray            # bool mask
    interval0: tuple[float, float]
    target: float                 # average of benign initial values

    @property
    def final_spread(self) -> float:
        xb = self.X[-1, self.benign]
        return float(xb.max() - xb.min())

    @property
    def final_bias(self) -> float:
        """Mean benign final value minus the benign initial average."""
        return float(self.X[-1, self.benign].mean() - self.target)

    @property
    def max_excursion(self) -> float:
        """How far benign values ever left the initial benign interval (0 = invariant)."""
        lo, hi = self.interval0
        xb = self.X[:, self.benign]
        return float(max(0.0, lo - xb.min(), xb.max() - hi))

    @property
    def interval_invariant(self) -> bool:
        return self.max_excursion <= 1e-12

    def spread_history(self) -> np.ndarray:
        xb = self.X[:, self.benign]
        return xb.max(axis=1) - xb.min(axis=1)


def simulate_resilient(A: np.ndarray, x0: Sequence[float], faulty: Sequence[int] = (),
                       fault: FaultModel | None = None, F: int = 1, rule: str = "wmsr",
                       iters: int = 200) -> ResilientRun:
    """Synchronous iteration of all benign agents with ``rule`` in {``wmsr``, ``average``}.

    Faulty agents do not update; each round they broadcast ``fault.report(k, x0_i)``.
    """
    A = np.asarray(A) > 0
    N = A.shape[0]
    x = np.asarray(x0, dtype=float).copy()
    bad = np.zeros(N, bool)
    bad[list(faulty)] = True
    fault = fault or FaultModel()
    nbrs = [np.flatnonzero(A[i]) for i in range(N)]
    benign0 = x[~bad]
    X = np.empty((iters + 1, N))
    true_bad = x.copy()
    # what each faulty node sends to each of its neighbours in round k
    for k in range(iters + 1):
        sent: dict[int, dict[int, float]] = {}
        for f in np.flatnonzero(bad):
            vals = fault.report(k, float(true_bad[f]), len(nbrs[f]))
            sent[int(f)] = {int(j): float(v) for j, v in zip(nbrs[f], vals)}
            x[f] = float(np.mean(vals))
        X[k] = x
        if k == iters:
            break
        new = x.copy()
        for i in np.flatnonzero(~bad):
            rec = [sent[int(j)][int(i)] if bad[j] else x[j] for j in nbrs[i]]
            new[i] = wmsr_update(x[i], rec, F) if rule == "wmsr" else average_update(x[i], rec)
        x = new
    return ResilientRun(X=X, benign=~bad, interval0=(float(benign0.min()), float(benign0.max())),
                        target=float(benign0.mean()))


# ----------------------------------------------------------------------------
# Controller integration
# ----------------------------------------------------------------------------
class WMSRConsensus:
    """W-MSR version of ``control.comp.LoadConsensus`` (same interface and timing).

    ``cfg`` = the controller's ``consensus`` section with ``aggregation: wmsr`` and ``F``.
    Per exchange round, the newest value of every neighbour (fresh this round, or the last
    one under the ``last`` missing policy) is filtered by W-MSR and averaged uniformly
    with the own state.  ``mode: dynamic`` then adds the own input increment
    ``rho_i[k] - rho_i[k-1]``.  W-MSR does not conserve the sum of states, so dynamic
    W-MSR tracks a value inside the range of benign loads rather than their exact mean;
    ``mode: static`` (restart from the own load each period) has no drift (E9 compares both).
    """

    def __init__(self, view: Any, cfg: Mapping[str, Any] | None, period: int,
                 missing_policy: str = "last", channel: str = "rho") -> None:
        cfg = dict(cfg or {})
        self.channel = channel
        self.nbrs = list(view.neighbors)
        self.F = int(cfg.get("F", 1))
        self.K = int(cfg.get("iterations_per_decision", 5))
        self.mode = str(cfg.get("mode", "dynamic"))
        if self.mode not in ("dynamic", "static"):
            raise ValueError("consensus mode must be dynamic or static")
        self.period = max(1, int(period))
        self.interval = max(1, self.period // max(self.K, 1))
        self.policy = missing_policy
        self.x: float | None = None
        self.estimate = 0.0
        self.rho_prev = 0.0
        self.last: dict[int, float] = {}
        self.n_filtered = 0          # neighbour values discarded (diagnostics)
        self.n_rounds = 0

    def exchange_step(self, t: int) -> bool:
        if self.K <= 0 or not self.nbrs:
            return False
        r = t % self.period
        return r % self.interval == 0 and r < self.K * self.interval

    def update(self, t: int, board: Any, rho: float) -> float:
        fresh = board.pop_fresh(self.channel)
        if self.K <= 0:
            self.estimate = rho
            return rho
        if self.x is None:
            self.x, self.rho_prev, self.estimate = rho, rho, rho
            return rho
        for j in fresh:
            v = board.info[j].cons.get(self.channel)
            if v is not None:
                self.last[j] = float(v)
        if fresh:
            rec = [self.last[j] for j in self.nbrs
                   if j in fresh or (self.policy == "last" and j in self.last)]
            kept = wmsr_filter(self.x, rec, self.F)
            self.n_filtered += len(rec) + 1 - len(kept)
            self.n_rounds += 1
            self.x = float(np.mean(kept))
        if self.mode == "dynamic":
            self.x += rho - self.rho_prev
            self.estimate = self.x
        elif t % self.period == 0:
            self.estimate = self.x
            self.x = rho
        self.rho_prev = rho
        return self.estimate


def make_consensus(view: Any, cfg: Mapping[str, Any] | None, period: int,
                   missing_policy: str = "last", channel: str = "rho"):
    """Factory used by CoMP / CoCycle: ``aggregation: average`` (default) -> LoadConsensus,
    ``aggregation: wmsr`` -> :class:`WMSRConsensus`."""
    agg = str(dict(cfg or {}).get("aggregation", "average"))
    if agg == "wmsr":
        return WMSRConsensus(view, cfg, period, missing_policy, channel)
    if agg != "average":
        raise ValueError(f"unknown consensus aggregation {agg!r}")
    from .comp import LoadConsensus

    return LoadConsensus(view, cfg, period, missing_policy, channel)


class FaultInjector:
    """Makes one agent a faulty *reporter*: its signals run normally, but every consensus
    value it sends (``load_value`` and the named channels in ``extra["cons"]``) is replaced
    by the fault model's report.  Nothing else in the message is touched.

    ``channels``: optional per-channel overrides, e.g. ``{"rho": {"kind": "constant",
    "value": 1.0}, "cw": {"kind": "constant", "value": 5.0, "relative": true}}``; channels
    not listed use the default model.
    """

    def __init__(self, default: Mapping[str, Any] | FaultModel | None = None,
                 channels: Mapping[str, Mapping[str, Any]] | None = None, seed: int = 0) -> None:
        base = default if isinstance(default, FaultModel) else FaultModel.from_dict({"seed": seed, **dict(default or {})})
        self.default = base
        self.models = {name: FaultModel.from_dict({"seed": seed + 17 * (n + 1), **dict(c)})
                       for n, (name, c) in enumerate(sorted((channels or {}).items()))}
        self.rounds: dict[str, int] = {}

    def _model(self, ch: str) -> FaultModel:
        return self.models.get(ch, self.default)

    def rewrite(self, msgs: Sequence[Any]) -> list[Any]:
        """Return the messages with falsified consensus values (dataclasses.replace)."""
        from dataclasses import replace

        if not msgs:
            return list(msgs)
        out = []
        cache: dict[str, np.ndarray] = {}
        n = len(msgs)
        for idx, m in enumerate(msgs):
            changes: dict[str, Any] = {}
            if m.load_value is not None:
                v = self._value("rho", float(m.load_value), idx, n, cache)
                changes["load_value"] = v
            cons = (m.extra or {}).get("cons")
            if cons:
                new_cons = {ch: self._value(ch, float(val), idx, n, cache) for ch, val in cons.items()}
                changes["extra"] = {**m.extra, "cons": new_cons}
            out.append(replace(m, **changes) if changes else m)
        for ch in cache:
            self.rounds[ch] = self.rounds.get(ch, 0) + 1
        return out

    def _value(self, ch: str, true_value: float, idx: int, n: int, cache: dict) -> float:
        if ch not in cache:
            cache[ch] = self._model(ch).report(self.rounds.get(ch, 0), true_value, n)
        return float(cache[ch][idx])
