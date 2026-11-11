"""Publication plots: 300 dpi, one colour per controller everywhere,
units on every axis, 95% confidence intervals as error bars or bands.

Figure list (``experiments/make_figures.py`` builds them all from ``results/``):

=====  ==========================================================  ============================
F01    bar chart with CI for each metric and scenario               :func:`bar_ci`
F02    delay vs demand                                              :func:`lines_ci`
F03    gain vs network size                                         :func:`gain_vs_size`
F04    performance vs lambda_2(L) (and mean degree)                  :func:`perf_vs_lambda2`
F05    performance vs drop probability                              :func:`lines_ci`
F06    consensus error vs iteration (log scale)                     :func:`consensus_error`
F07    DGD / gradient tracking / ADMM convergence                   :func:`optimizer_convergence`
F08    W-MSR vs plain averaging under attack                        :func:`wmsr_vs_average`
F09    queue time series with the incident marked                   :func:`incident_queue`
F10    offset time-space diagram (green-wave bands)                 :func:`time_space`
F11    epidemic threshold                                           :func:`epidemic_threshold`
=====  ==========================================================  ============================
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

__all__ = ["CONTROLLER_COLORS", "CONTROLLER_LABELS", "color_for", "label_for", "style_axes", "save",
           "bar_ci", "lines_ci", "gain_vs_size", "perf_vs_lambda2", "consensus_error", "optimizer_convergence",
           "wmsr_vs_average", "incident_queue", "time_space", "epidemic_threshold",
           "INK", "INK2", "MUTED", "AXIS", "GRID", "SURFACE", "DPI"]

DPI = 300
INK, INK2, MUTED, AXIS, GRID, SURFACE = "#1f2328", "#3d444d", "#8b949e", "#57606a", "#d8dee4", "#ffffff"
CONTROLLER_COLORS = {
    "fixed": "#8c959f", "actuated": "#1f2328", "maxpressure_local": "#57606a", "comp": "#8250df",
    "cocycle": "#1a7f37", "comp+cycle": "#2da44e", "comp+epistemic": "#a475f9", "dmpc": "#bc4c00",
    "dmpc+stl": "#0969da", "dmpc+stl+epistemic": "#54aeff", "iql": "#e16f24", "iql+comm": "#953800",
    "vdn": "#218bff", "qmix": "#6639ba", "marl": "#e16f24",
}
CONTROLLER_LABELS = {
    "fixed": "Fixed-time (Webster)", "actuated": "Actuated (queue-greedy)", "maxpressure_local": "Local max-pressure",
    "comp": "CoMP", "cocycle": "CoCycle", "comp+cycle": "CoMP+Cycle", "comp+epistemic": "CoMP + epistemic guard",
    "dmpc": "DMPC", "dmpc+stl": "DMPC+STL", "dmpc+stl+epistemic": "DMPC+STL+epistemic", "iql": "IQL",
    "iql+comm": "IQL+comm", "vdn": "VDN", "qmix": "QMIX",
}
_EXTRA = ["#cf222e", "#4d2d00", "#0550ae", "#116329", "#8a4600"]


def color_for(name: str) -> str:
    key = str(name).split("|")[0]
    if key in CONTROLLER_COLORS:
        return CONTROLLER_COLORS[key]
    return _EXTRA[abs(hash(key)) % len(_EXTRA)]


def label_for(name: str) -> str:
    return CONTROLLER_LABELS.get(str(name), str(name))


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlesize": 9.5, "legend.frameon": False})
    return plt


def style_axes(ax, grid_axis: str = "both") -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


# ---------------------------------------------------------------- F01
def bar_ci(groups: Sequence[str], controllers: Sequence[str], mean: Mapping, ci: Mapping, ylabel: str, title: str,
           path: str | Path, log: bool = False) -> Path:
    """Grouped bars ``mean[(group, controller)]`` with 95% CI half-widths ``ci[...]``."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(max(6.0, 1.3 * len(groups) + 2), 3.8))
    w = 0.8 / max(1, len(controllers))
    for k, c in enumerate(controllers):
        x = np.arange(len(groups)) + (k - (len(controllers) - 1) / 2) * w
        ax.bar(x, [mean.get((g, c), np.nan) for g in groups], w, yerr=[ci.get((g, c), 0) for g in groups], capsize=2,
               color=color_for(c), label=label_for(c), error_kw={"lw": 0.8})
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.set_title(title, loc="left")
    style_axes(ax, "y")
    ax.legend(fontsize=7, ncol=min(3, len(controllers)))
    return save(fig, path)


# ---------------------------------------------------------------- F02 / F05
def lines_ci(x: Sequence[float], series: Mapping[str, tuple], xlabel: str, ylabel: str, title: str,
             path: str | Path, ref: Mapping[str, float] | None = None, logx: bool = False) -> Path:
    """One line with a 95% CI band per controller: ``series[c] = (means, halfwidths)``."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    x = np.asarray(x, float)
    for c, (m, h) in series.items():
        m, h = np.asarray(m, float), np.asarray(h, float)
        ax.plot(x, m, "o-", ms=3.5, lw=1.5, color=color_for(c), label=label_for(c))
        ax.fill_between(x, m - h, m + h, color=color_for(c), alpha=0.15, lw=0)
    for c, v in (ref or {}).items():
        ax.axhline(v, color=color_for(c), ls="--", lw=1, label=f"{label_for(c)} (reference)")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    style_axes(ax)
    ax.legend(fontsize=7)
    return save(fig, path)


# ---------------------------------------------------------------- F03
def gain_vs_size(N: Sequence[int], series: Mapping[str, tuple], title: str, path: str | Path) -> Path:
    """``series[label] = (change %, lo, hi)`` vs number of intersections."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    for k, (lab, (m, lo, hi)) in enumerate(series.items()):
        m, lo, hi = (np.asarray(v, float) for v in (m, lo, hi))
        ax.errorbar(np.asarray(N) + 0.15 * k, m, yerr=[m - lo, hi - m], marker="os^v"[k % 4], capsize=3, lw=1.4, label=lab,
                    color=_EXTRA[k % len(_EXTRA)] if k else color_for("cocycle"))
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xlabel("number of intersections N [–]")
    ax.set_ylabel("delay change vs actuated [%] (negative = better)")
    ax.set_title(title, loc="left")
    style_axes(ax)
    ax.legend(fontsize=7)
    return save(fig, path)


# ---------------------------------------------------------------- F04
def perf_vs_lambda2(lam2: Sequence[float], degree: Sequence[float], delay: Sequence[float], delay_ci: Sequence[float],
                    ref: float, title: str, path: str | Path) -> Path:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    for ax, x, lab in ((axes[0], lam2, r"algebraic connectivity $\lambda_2(L)$ [–]"), (axes[1], degree, "mean degree [–]")):
        ax.errorbar(x, delay, yerr=delay_ci, fmt="o", ms=4, capsize=2, color=color_for("cocycle"), label=label_for("cocycle"))
        ax.axhline(ref, color=color_for("actuated"), ls="--", lw=1, label=f"{label_for('actuated')} (no communication)")
        ax.set_xlabel(lab)
        style_axes(ax)
    axes[0].set_ylabel("mean delay [s/veh]")
    axes[0].legend(fontsize=7)
    fig.suptitle(title, x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout()
    return save(fig, path)


# ---------------------------------------------------------------- F06
def consensus_error(curves: Mapping[str, np.ndarray], title: str, path: str | Path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    for lab, e in curves.items():
        e = np.asarray(e, float)
        ax.semilogy(np.arange(e.size), np.maximum(e / e[0], 1e-16), lw=1.4, label=lab)
    ax.set_xlabel("iteration k [–]")
    ax.set_ylabel(r"$\|x[k]-\bar x\mathbf{1}\|_2\,/\,\|x[0]-\bar x\mathbf{1}\|_2$ [–]")
    ax.set_title(title, loc="left")
    style_axes(ax)
    ax.legend(fontsize=7)
    return save(fig, path)


# ---------------------------------------------------------------- F07
def optimizer_convergence(errors: Mapping[str, Sequence[float]], title: str, path: str | Path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for lab, e in errors.items():
        e = np.maximum(np.asarray(e, float), 1e-16)
        ax.semilogy(np.arange(1, e.size + 1), e, lw=1.4, label=lab)
    ax.set_xlabel("iteration [–]")
    ax.set_ylabel(r"$\max_i |C_i - z^\star|$ [s]")
    ax.set_ylim(1e-14, None)
    ax.set_title(title, loc="left")
    style_axes(ax)
    ax.legend(fontsize=7)
    return save(fig, path)


# ---------------------------------------------------------------- F08
def wmsr_vs_average(runs: Sequence[tuple], faulty: Sequence[int], title: str, path: str | Path) -> Path:
    """``runs = [(panel title, ResilientRun), ...]``."""
    plt = _plt()
    fig, axes = plt.subplots(1, len(runs), figsize=(4.0 * len(runs), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (t, run) in zip(axes, runs):
        for i in range(run.X.shape[1]):
            if i in faulty:
                ax.plot(run.X[:, i], color="#cf222e", lw=2, ls="--", label="faulty agent")
            else:
                ax.plot(run.X[:, i], color=INK2, lw=0.9, alpha=0.85, label="benign agents" if i == 0 else None)
        ax.axhspan(*run.interval0, color="#2da44e", alpha=0.12, label="initial benign interval")
        ax.axhline(run.target, color="#1a7f37", ls=":", lw=1, label="benign average")
        ax.set_title(t, fontsize=8.5, loc="left")
        ax.set_xlabel("consensus iteration [–]")
        style_axes(ax)
    axes[0].set_ylabel("load estimate $x_i$ [–]")
    axes[0].legend(fontsize=6.5, loc="center right")
    fig.suptitle(title, x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout()
    return save(fig, path)


# ---------------------------------------------------------------- F09
def incident_queue(t: np.ndarray, series: Mapping[str, np.ndarray], incident: tuple[int, int], title: str,
                   path: str | Path, t0: int = 600) -> Path:
    """``series[c]`` = (seeds x T) network mean queue (60 s average) per controller."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    for c, S in series.items():
        S = np.asarray(S, float)
        m = S.mean(0)
        ax.plot(t, m, color=color_for(c), lw=1.4, label=f"{label_for(c)} (n={S.shape[0]})")
        if S.shape[0] > 1:
            h = 1.96 * S.std(0, ddof=1) / np.sqrt(S.shape[0])
            ax.fill_between(t, m - h, m + h, color=color_for(c), alpha=0.15, lw=0)
    ax.axvspan(*incident, color="#cf222e", alpha=0.08, label="incident (30% capacity)")
    ax.set_xlim(t0, t[-1])
    ax.set_xlabel("time [s]")
    ax.set_ylabel("network mean queue, 60 s average [veh/link]")
    ax.set_title(title, loc="left")
    style_axes(ax)
    ax.legend(fontsize=7)
    return save(fig, path)


# ---------------------------------------------------------------- F10
def time_space(datas: Sequence, path: str | Path, title: str = "") -> Path:
    """Time-space diagram per controller: vehicle trajectories along an arterial and the
    green / yellow / red indication of every stop line (green-wave bands appear as aligned
    green bars shifted by the travel time)."""
    plt = _plt()
    fig, axes = plt.subplots(1, len(datas), figsize=(5.2 * len(datas), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    sig_col = {0: "#2da44e", 1: "#d4a72c", 2: "#cf222e"}
    for ax, d in zip(axes, datas):
        for vid, tr in d.trajectories.items():
            if len(tr) > 1:
                ax.plot(tr[:, 0], tr[:, 1], color=INK2, lw=0.35, alpha=0.55)
        for j, x in enumerate(d.node_pos):
            s = d.signal[:, j]
            for code, col in sig_col.items():
                on = s == code
                if on.any():
                    starts = np.flatnonzero(on & ~np.r_[False, on[:-1]])
                    ends = np.flatnonzero(on & ~np.r_[on[1:], False])
                    for a, b in zip(starts, ends):
                        ax.plot([d.t[a] - 1, d.t[b]], [x, x], color=col, lw=4, solid_capstyle="butt")
        ax.set_xlabel("time [s]")
        ax.set_title(f"{label_for(d.meta.get('controller', ''))} — mean delay {d.meta.get('mean_delay', float('nan')):.1f} s",
                     fontsize=8.5, loc="left")
        style_axes(ax)
    axes[0].set_ylabel("distance along the eastbound arterial [m]")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout()
    return save(fig, path)


# ---------------------------------------------------------------- F11
def epidemic_threshold(names: Sequence[str], R: Mapping[str, Sequence[float]], observed: Mapping[str, Sequence[float]],
                       predicted: Mapping[str, Sequence[float]], title: str, path: str | Path) -> Path:
    """Left: ``beta lambda_max / gamma`` per scenario and controller with the threshold 1;
    right: observed vs mean-field predicted congested fraction."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(names))
    w = 0.8 / max(1, len(R))
    for k, (c, r) in enumerate(R.items()):
        axes[0].bar(x + (k - (len(R) - 1) / 2) * w, r, w, color=color_for(c), label=label_for(c))
    axes[0].axhline(1.0, color="#cf222e", ls="--", lw=1.2, label=r"threshold $\beta\lambda_{max}/\gamma=1$")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, fontsize=8)
    axes[0].set_ylabel(r"$\hat\beta\,\lambda_{max}(A)\,/\,\hat\gamma$ [–]")
    axes[0].legend(fontsize=7)
    style_axes(axes[0], "y")
    lim = 0.05
    for c in observed:
        o, p = np.asarray(observed[c]), np.asarray(predicted[c])
        axes[1].scatter(p, o, color=color_for(c), label=label_for(c), s=18)
        for n, a, b in zip(names, p, o):
            axes[1].annotate(n, (a, b), fontsize=6)
        lim = max(lim, float(np.nanmax(np.r_[o, p])) * 1.1)
    axes[1].plot([0, lim], [0, lim], color=MUTED, ls=":", lw=1)
    axes[1].set_xlim(0, lim)
    axes[1].set_ylim(0, lim)
    axes[1].set_xlabel("mean-field SIS prediction, congested fraction [–]")
    axes[1].set_ylabel("observed congested fraction [–]")
    axes[1].legend(fontsize=7)
    style_axes(axes[1])
    fig.suptitle(title, x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout()
    return save(fig, path)
