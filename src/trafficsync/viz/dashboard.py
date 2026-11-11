"""Dashboard panel next to the animation.

Three real-time lines per controller: network mean queue [veh/link], throughput
[veh/h] over a sliding window, and the consensus error ``||x - mean(rho) 1||_2`` of the
agents' load-consensus states.  The lines grow up to the current animation time; a
vertical cursor marks it.  Controllers keep their project-wide colours
(:data:`trafficsync.viz.plots.CONTROLLER_COLORS`); with more than one controller a legend
names them, with one the panel title does.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .plots import AXIS, GRID, INK, INK2, MUTED, SURFACE, color_for, label_for, style_axes
from .recording import Recording

__all__ = ["DASHBOARD_METRICS", "dashboard_series", "Dashboard"]

# key, axis label, smoothing window [s]
DASHBOARD_METRICS = [
    ("mean_queue", "Network mean queue [veh/link]", 60),
    ("throughput", "Throughput [veh/h]", 300),
    ("consensus_error", "Consensus error ‖x − ρ̄1‖ [–]", 60),
]


def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """Trailing moving average that ignores NaNs (NaN where the whole window is NaN)."""
    x = np.asarray(x, dtype=float)
    if w <= 1:
        return x
    ok = np.isfinite(x)
    xs = np.where(ok, x, 0.0)
    c = np.cumsum(np.r_[0.0, xs])
    n = np.cumsum(np.r_[0, ok.astype(int)])
    idx = np.arange(1, x.size + 1)
    lo = np.maximum(idx - w, 0)
    cnt = n[idx] - n[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (c[idx] - c[lo]) / cnt
    out[cnt == 0] = np.nan
    return out


def dashboard_series(rec: Recording) -> dict[str, np.ndarray]:
    """Per-step dashboard lines of one recording (time axis = step end, seconds)."""
    s = rec.series
    out = {"t": np.arange(1, s["mean_queue"].size + 1, dtype=float)}
    for key, _label, w in DASHBOARD_METRICS:
        if key == "throughput":
            done = s["completed"].astype(float)
            c = np.cumsum(np.r_[0.0, done])
            idx = np.arange(1, done.size + 1)
            lo = np.maximum(idx - w, 0)
            out[key] = (c[idx] - c[lo]) * 3600.0 / np.maximum(idx - lo, 1)
        else:
            out[key] = _moving_average(s[key], w)
    return out


class Dashboard:
    """Three metric axes that follow the animation clock."""

    def __init__(self, axes: Sequence, recordings: Sequence[Recording], t_range: tuple[float, float]) -> None:
        if len(axes) != len(DASHBOARD_METRICS):
            raise ValueError(f"need {len(DASHBOARD_METRICS)} axes")
        self.axes = list(axes)
        self.recs = list(recordings)
        self.series = [dashboard_series(r) for r in self.recs]
        self.t0, self.t1 = t_range
        self.lines: list[list] = []
        self.cursors = []
        multi = len(self.recs) > 1
        warm = float(self.recs[0].meta.get("warmup_s", 0))
        for ax, (key, label, _w) in zip(self.axes, DASHBOARD_METRICS):
            style_axes(ax, grid_axis="y")
            ax.set_xlim(self.t0, self.t1)
            ax.set_ylabel(label, fontsize=8)
            vals = [s[key][(s["t"] >= self.t0) & (s["t"] <= self.t1)] for s in self.series]
            finite = np.concatenate([v[np.isfinite(v)] for v in vals]) if vals else np.array([])
            hi = float(finite.max()) if finite.size else 1.0
            ax.set_ylim(0.0, hi * 1.12 if hi > 0 else 1.0)
            if warm > self.t0:
                ax.axvspan(self.t0, min(warm, self.t1), color=GRID, alpha=0.45, lw=0, zorder=0)
            row = []
            for rec in self.recs:
                (ln,) = ax.plot([], [], color=color_for(rec.meta.get("controller", "")), lw=1.6,
                                label=label_for(rec.label), solid_capstyle="round")
                row.append(ln)
            self.lines.append(row)
            self.cursors.append(ax.axvline(self.t0, color=AXIS, lw=0.8, zorder=1))
            if key == "consensus_error":
                none = [label_for(r.label) for r, s in zip(self.recs, self.series) if not np.isfinite(s[key]).any()]
                if none:
                    ax.text(0.99, 0.95, "no consensus: " + ", ".join(none), transform=ax.transAxes, ha="right",
                            va="top", fontsize=6.5, color=MUTED)
        self.axes[-1].set_xlabel("Simulation time [s]", fontsize=8)
        for ax in self.axes[:-1]:
            ax.tick_params(labelbottom=False)
        if warm > self.t0:
            self.axes[0].text(self.t0 + 0.01 * (self.t1 - self.t0), self.axes[0].get_ylim()[1] * 0.97, "warm-up",
                              fontsize=6.5, color=MUTED, va="top")
        if multi:
            self.axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=len(self.recs), frameon=False,
                                fontsize=7, handlelength=1.4, labelcolor=INK2)
        else:
            self.axes[0].set_title(label_for(self.recs[0].label), fontsize=8.5, color=INK, loc="left")
        for ax in self.axes:
            ax.set_facecolor(SURFACE)

    def update(self, t: float) -> list:
        artists = []
        for row, ax, (key, _l, _w) in zip(self.lines, self.axes, DASHBOARD_METRICS):
            for ln, s in zip(row, self.series):
                m = (s["t"] >= self.t0) & (s["t"] <= t)
                ln.set_data(s["t"][m], s[key][m])
                artists.append(ln)
        for c in self.cursors:
            c.set_xdata([t, t])
            artists.append(c)
        return artists
