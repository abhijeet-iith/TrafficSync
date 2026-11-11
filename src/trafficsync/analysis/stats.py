"""Statistics for paired controller comparisons.

t-interval and bootstrap 95% CIs, paired Wilcoxon signed-rank test, percent improvement
with a paired-bootstrap CI, Holm adjustment.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = ["mean_ci", "bootstrap_ci", "paired_wilcoxon", "percent_improvement", "holm", "summarize"]


def mean_ci(x: Sequence[float], level: float = 0.95) -> dict[str, float]:
    a = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    n = a.size
    if n == 0:
        return {"n": 0, "mean": math.nan, "half_width": math.nan, "lo": math.nan, "hi": math.nan, "std": math.nan}
    m = float(a.mean())
    if n == 1:
        return {"n": 1, "mean": m, "half_width": math.nan, "lo": math.nan, "hi": math.nan, "std": 0.0}
    from scipy.stats import t

    s = float(a.std(ddof=1))
    h = float(t.ppf(0.5 + level / 2, n - 1) * s / math.sqrt(n))
    return {"n": n, "mean": m, "half_width": h, "lo": m - h, "hi": m + h, "std": s}


def bootstrap_ci(x: Sequence[float], stat=np.mean, n_boot: int = 5000, level: float = 0.95, seed: int = 0) -> tuple[float, float]:
    a = np.asarray(x, float)
    if a.size < 2:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    b = np.apply_along_axis(stat, 1, a[idx])
    return float(np.quantile(b, (1 - level) / 2)), float(np.quantile(b, 1 - (1 - level) / 2))


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    out = {"n_better": int((d < 0).sum()), "n_worse": int((d > 0).sum()), "n_tied": int((d == 0).sum())}
    if np.allclose(d, 0):
        return {**out, "statistic": 0.0, "p_value": 1.0, "effect_r": 0.0}
    from scipy.stats import wilcoxon

    r = wilcoxon(a, b, zero_method="wilcox")
    nz = int((d != 0).sum())
    # matched-pairs rank-biserial correlation as effect size
    ranks = np.argsort(np.argsort(np.abs(d[d != 0]))) + 1
    rb = float((ranks[d[d != 0] > 0].sum() - ranks[d[d != 0] < 0].sum()) / ranks.sum()) if nz else 0.0
    return {**out, "statistic": float(r.statistic), "p_value": float(r.pvalue), "effect_r": rb}


def percent_improvement(ctrl: Sequence[float], base: Sequence[float], n_boot: int = 5000, seed: int = 0) -> dict:
    """``100 (m_base - m_ctrl) / m_base`` with a paired-bootstrap 95% CI."""
    c, b = np.asarray(ctrl, float), np.asarray(base, float)
    imp = 100.0 * (b.mean() - c.mean()) / b.mean()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, c.size, size=(n_boot, c.size))
    bs = 100.0 * (b[idx].mean(1) - c[idx].mean(1)) / b[idx].mean(1)
    return {"improvement_pct": float(imp), "ci_lo": float(np.quantile(bs, 0.025)), "ci_hi": float(np.quantile(bs, 0.975))}


def holm(pvals: Sequence[float]) -> list[float]:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = p.size
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def summarize(rows: Sequence[Mapping[str, Any]], keys: Sequence[str], metrics: Sequence[str]) -> list[dict]:
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for g, rs in groups.items():
        for m in metrics:
            vals = [float(r[m]) for r in rs if m in r and r[m] is not None and r[m] != ""]
            out.append({**dict(zip(keys, g)), "metric": m, **mean_ci(vals)})
    return out
