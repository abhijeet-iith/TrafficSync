"""Signal Temporal Logic: robustness semantics and an offline trace monitor.

Discrete time (1 s samples).  Quantitative semantics (course notes)::

    rho(mu, t)               = mu(x(t))                  e.g. mu = C - delta - n  for  n <= C - delta
    rho(not phi)             = -rho(phi)
    rho(phi and psi)         = min,     rho(phi or psi) = max,     phi => psi  =  (not phi) or psi
    rho(G_[a,b] phi, t)      = min_{tau in [t+a, t+b]} rho(phi, tau)
    rho(F_[a,b] phi, t)      = max_{tau in [t+a, t+b]} rho(phi, tau)

Finite traces: a window is cut at the end of the trace (only available samples count); a
window with no sample gives +inf for G (vacuous) and -inf for F (unsatisfiable).

:func:`robustness` returns the whole signal ``rho(phi, t)`` for every t (vectorised sliding
min/max), :func:`rho` the value at one time.

Traffic specifications (:func:`phi_safe`, :func:`phi_clear`, :func:`phi_resp`,
:func:`phi_starve`, :func:`Phi_i`, :func:`Phi`) are evaluated on recorded traces by
:class:`TraceMonitor` (signals ``n_<l>``, ``q_<l>``, ``tsg_<i>_<p>``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

__all__ = ["Formula", "Pred", "Not", "And", "Or", "Implies", "Always", "Eventually", "robustness", "rho",
           "le", "ge", "phi_safe", "phi_clear", "phi_resp", "phi_starve", "Phi_i", "Phi", "TraceMonitor", "STLParams"]

Trace = Mapping[str, np.ndarray]


class Formula:
    def __and__(self, other: "Formula") -> "Formula":
        return And(self, other)

    def __or__(self, other: "Formula") -> "Formula":
        return Or(self, other)

    def __invert__(self) -> "Formula":
        return Not(self)


@dataclass(frozen=True)
class Pred(Formula):
    """Atomic predicate ``mu(x) >= 0``; ``fn(trace) -> array`` of ``mu`` values."""

    fn: Callable[[Trace], np.ndarray]
    name: str = "mu"

    def __repr__(self) -> str:
        return self.name


def le(signal: str, c: float) -> Pred:
    """``signal <= c``  (robustness ``c - x``)."""
    return Pred(lambda tr, s=signal, c=c: float(c) - np.asarray(tr[s], float), f"{signal}<={c:g}")


def ge(signal: str, c: float) -> Pred:
    """``signal >= c``  (robustness ``x - c``)."""
    return Pred(lambda tr, s=signal, c=c: np.asarray(tr[s], float) - float(c), f"{signal}>={c:g}")


@dataclass(frozen=True)
class Not(Formula):
    phi: Formula


@dataclass(frozen=True)
class And(Formula):
    phi: Formula
    psi: Formula


@dataclass(frozen=True)
class Or(Formula):
    phi: Formula
    psi: Formula


def Implies(phi: Formula, psi: Formula) -> Formula:  # noqa: N802 - logic notation
    return Or(Not(phi), psi)


@dataclass(frozen=True)
class Always(Formula):
    phi: Formula
    a: int = 0
    b: int | None = None     # None = to the end of the trace


@dataclass(frozen=True)
class Eventually(Formula):
    phi: Formula
    a: int = 0
    b: int | None = None


def _window(r: np.ndarray, a: int, b: int | None, op: str) -> np.ndarray:
    """``out[t] = op(r[t+a : t+b+1])`` with truncation at the end of the trace."""
    T = r.size
    fill = np.inf if op == "min" else -np.inf
    b_eff = T - 1 if b is None else int(b)
    w = b_eff - int(a) + 1
    if w <= 0:
        raise ValueError("empty interval [a, b]")
    pad = np.full(T + b_eff + 1, fill)
    pad[:T] = r
    if b is None or b_eff >= T - 1:   # window reaches the end of the trace: suffix min/max from t + a
        acc = (np.minimum if op == "min" else np.maximum).accumulate(r[::-1])[::-1]
        out = np.full(T, fill)
        idx = np.arange(T) + int(a)
        ok = idx < T
        out[ok] = acc[idx[ok]]
        return out
    from numpy.lib.stride_tricks import sliding_window_view

    win = sliding_window_view(pad[int(a):int(a) + T + w - 1], w)[:T]
    return win.min(axis=1) if op == "min" else win.max(axis=1)


def robustness(phi: Formula, trace: Trace) -> np.ndarray:
    """``rho(phi, t)`` for every sample ``t`` of ``trace``."""
    if isinstance(phi, Pred):
        return np.asarray(phi.fn(trace), float)
    if isinstance(phi, Not):
        return -robustness(phi.phi, trace)
    if isinstance(phi, And):
        return np.minimum(robustness(phi.phi, trace), robustness(phi.psi, trace))
    if isinstance(phi, Or):
        return np.maximum(robustness(phi.phi, trace), robustness(phi.psi, trace))
    if isinstance(phi, Always):
        return _window(robustness(phi.phi, trace), phi.a, phi.b, "min")
    if isinstance(phi, Eventually):
        return _window(robustness(phi.phi, trace), phi.a, phi.b, "max")
    raise TypeError(f"unknown formula {phi!r}")


def rho(phi: Formula, trace: Trace, t: int = 0) -> float:
    return float(robustness(phi, trace)[t])


# ----------------------------------------------------------------------------
# Traffic specifications
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class STLParams:
    T: int = 3600
    T_c: int = 300
    T_r: int = 180
    q_thr_frac: float = 0.2
    q_hi_frac: float = 0.6
    R_max: int = 120
    delta_room: float = 2.0

    @classmethod
    def from_dict(cls, d: Mapping | None) -> "STLParams":
        d = dict(d or {})
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def phi_safe(l: int, C: float, p: STLParams) -> Formula:
    """``G_[0,T] (n_l <= C_l - delta_room)``  (no spillback)."""
    return Always(le(f"n_{l}", C - p.delta_room), 0, p.T)


def phi_clear(l: int, C: float, p: STLParams) -> Formula:
    """``F_[0,T_c] (q_l <= q_thr)``  (queue clears within T_c)."""
    return Eventually(le(f"q_{l}", p.q_thr_frac * C), 0, p.T_c)


def phi_resp(l: int, C: float, p: STLParams) -> Formula:
    """``G_[0,T] (q_l > q_hi  =>  F_[0,T_r] q_l < q_hi)``  (response)."""
    hi = p.q_hi_frac * C
    return Always(Implies(ge(f"q_{l}", hi), Eventually(le(f"q_{l}", hi), 0, p.T_r)), 0, p.T)


def phi_starve(i: int, ph: int, p: STLParams) -> Formula:
    """``G_[0,T] (time since last green of phase p <= R_max)``."""
    return Always(le(f"tsg_{i}_{ph}", p.R_max), 0, p.T)


def Phi_i(i: int, links: Sequence[int], caps: Mapping[int, float], n_phases: int, p: STLParams) -> Formula:  # noqa: N802
    """``AND_l (safe_l AND resp_l) AND AND_p starve_{i,p}`` (``links`` = internal incoming links of i)."""
    parts: list[Formula] = []
    for l in links:
        parts += [phi_safe(l, caps[l], p), phi_resp(l, caps[l], p)]
    parts += [phi_starve(i, ph, p) for ph in range(n_phases)]
    out = parts[0]
    for f in parts[1:]:
        out = And(out, f)
    return out


def Phi(per_agent: Sequence[Formula]) -> Formula:  # noqa: N802
    out = per_agent[0]
    for f in per_agent[1:]:
        out = And(out, f)
    return out


class TraceMonitor:
    """Offline monitor of the traffic specifications on a recorded episode.

    ``trace``: ``n_<l>``, ``q_<l>`` for every queue link, ``tsg_<i>_<p>`` for every phase; the
    evaluation starts at ``t0`` (warm-up excluded).  Spillback safety and response are
    checked on *internal* links (entry links have a boundary buffer, not a stop line upstream).
    """

    def __init__(self, network, n_phases: Sequence[int], params: STLParams | None = None) -> None:
        self.net = network
        self.n_phases = list(n_phases)
        self.p = params or STLParams()

    def evaluate(self, trace: Trace, t0: int = 0) -> dict:
        tr = {k: np.asarray(v, float)[t0:] for k, v in trace.items()}
        T = len(next(iter(tr.values())))
        p = STLParams(**{**self.p.__dict__, "T": T - 1})
        net = self.net
        out: dict = {"safe": [], "resp": [], "starve": [], "clear": [], "Phi_i": []}
        for i in range(net.N):
            links = [l for l in net.incoming_links(i) if net.links[l].kind == "internal"]
            caps = {l: float(net.links[l].capacity) for l in net.incoming_links(i)}
            for l in links:
                out["safe"].append(rho(phi_safe(l, caps[l], p), tr))
                out["resp"].append(rho(phi_resp(l, caps[l], p), tr))
            for l in net.incoming_links(i):
                out["clear"].append(rho(phi_clear(l, caps[l], p), tr))
            for ph in range(self.n_phases[i]):
                out["starve"].append(rho(phi_starve(i, ph, p), tr))
            out["Phi_i"].append(rho(Phi_i(i, links, caps, self.n_phases[i], p), tr))
        res = {
            "rho_Phi": float(min(out["Phi_i"])),
            "rho_safe": float(min(out["safe"])) if out["safe"] else float("nan"),
            "rho_resp": float(min(out["resp"])) if out["resp"] else float("nan"),
            "rho_starve": float(min(out["starve"])),
            "rho_clear": float(min(out["clear"])),
            "Phi_satisfied_agents": int(sum(r >= 0 for r in out["Phi_i"])),
            "safe_violated_links": int(sum(r < 0 for r in out["safe"])),
            "resp_violated_links": int(sum(r < 0 for r in out["resp"])),
            "starve_violated_phases": int(sum(r < 0 for r in out["starve"])),
        }
        return res
