"""Animated network view.

Each panel draws one controller's episode (a :class:`~trafficsync.viz.recording.Recording`):

* intersections at their grid positions (circle with the signal state letter G / Y / R and
  the seconds spent in it);
* every directed road link as a thick segment, offset to its side of the road, coloured by
  occupancy ``n_l / C_l`` (green -> yellow -> red) with the width growing with occupancy;
  a dark queue bar at the stop line (length ~ ``q_l / C_l``);
* a signal head at every stop line: green / yellow / red for the approach (red when its
  phase is not served);
* communication: an arc between two agents flashes black when messages were delivered in
  the last frame interval and dashed grey when messages were dropped.

Side-by-side mode = several recordings with the same seed (identical arrivals) drawn in
synchronised panels, with the dashboard (network mean queue, throughput, consensus error)
underneath.  :func:`animate` writes ``.mp4`` (ffmpeg) or ``.gif`` (Pillow);
:func:`render_still` writes one frame as PNG.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .dashboard import DASHBOARD_METRICS, Dashboard
from .plots import INK, INK2, MUTED, SURFACE, label_for
from .recording import Recording

__all__ = ["NetworkPanel", "animate", "render_still"]

_SIG = {0: "#2da44e", 1: "#d4a72c", 2: "#cf222e"}
_LETTER = {0: "G", 1: "Y", 2: "R"}
_DIR = {"E": (1.0, 0.0), "W": (-1.0, 0.0), "N": (0.0, 1.0), "S": (0.0, -1.0)}


def _cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("occ", ["#2da44e", "#d4a72c", "#cf222e"])


class NetworkPanel:
    """One controller panel; ``update(frame)`` returns the changed artists."""

    def __init__(self, ax, rec: Recording) -> None:
        from matplotlib.collections import LineCollection
        from matplotlib.patches import Circle, FancyArrowPatch

        from ..network import load_network

        self.ax, self.rec = ax, rec
        net = load_network(rec.meta["network"])
        self.net = net
        L = float(net.meta.get("link_length", 300.0))
        pos = {i: np.array(p, float) for i, p in enumerate(net.pos)}
        self.cmap = _cmap()
        segs, heads, bars = [], [], []
        off = 0.06 * L
        stub = 0.45 * L
        for l in rec.link_ids:
            lk = net.links[int(l)]
            d = np.array(_DIR[lk.heading])
            n = np.array([d[1], -d[0]]) * off                 # right-hand side of the road
            if lk.to_node is not None:
                b = pos[lk.to_node] - d * 0.09 * L
                a = pos[lk.from_node] + d * 0.09 * L if lk.from_node is not None else pos[lk.to_node] - d * stub
            else:
                continue
            segs.append([a + n, b + n])
            heads.append(b + n + d * 0.03 * L)
            bars.append((b + n, -d))
        # exit links (drawn faintly, not part of the recording)
        ex = []
        for l in net.exit_links:
            lk = net.links[l]
            d = np.array(_DIR[lk.heading])
            n = np.array([d[1], -d[0]]) * off
            a = pos[lk.from_node] + d * 0.09 * L
            ex.append([a + n, a + n + d * (stub - 0.09 * L)])
        ax.add_collection(LineCollection(ex, colors="#d0d7de", linewidths=1.2, zorder=1))
        self.links = LineCollection(segs, linewidths=2.0, zorder=2, capstyle="butt")
        ax.add_collection(self.links)
        self.bar_base = bars
        self.qbars = LineCollection([[b, b] for b, _ in bars], colors=INK, linewidths=3.2, zorder=3, capstyle="butt")
        ax.add_collection(self.qbars)
        hp = np.array(heads)
        self.heads = ax.scatter(hp[:, 0], hp[:, 1], s=14, marker="s", zorder=5, edgecolors="white", linewidths=0.4)
        self.qscale = 0.75 * L
        self.cap = np.array([net.links[int(l)].capacity for l in rec.link_ids], float)
        # nodes
        self.node_txt = []
        for i in range(net.N):
            ax.add_patch(Circle(
                pos[i], 0.085 * L, facecolor="white", edgecolor=INK2, lw=0.9, zorder=6))
            self.node_txt.append(ax.text(*pos[i], "", ha="center", va="center", fontsize=6, family="monospace", zorder=7,
                                         color=INK))
        # comm arcs

        self.arcs = []
        for s_, r_ in rec.comm_pairs:
            arc = FancyArrowPatch(pos[int(s_)], pos[int(r_)], connectionstyle="arc3,rad=0.22", arrowstyle="-|>",
                                  mutation_scale=6, lw=0.7, color=INK, alpha=0.0, zorder=4,
                                  shrinkA=0.1 * L, shrinkB=0.1 * L)
            ax.add_patch(arc)
            self.arcs.append(arc)
        xs = np.array([p[0] for p in pos.values()])
        ys = np.array([p[1] for p in pos.values()])
        ax.set_xlim(xs.min() - 0.6 * L, xs.max() + 0.6 * L)
        ax.set_ylim(ys.min() - 0.6 * L, ys.max() + 0.6 * L)
        ax.set_aspect("equal")
        ax.axis("off")
        self.title = ax.set_title("", fontsize=8.5, loc="left", color=INK)
        self.clock = ax.text(0.01, 0.01, "", transform=ax.transAxes, fontsize=7, color=MUTED, va="bottom")
        self.served_idx = rec.served

    def update(self, f: int) -> list:
        r = self.rec
        occ = np.clip(r.occ[f], 0, 1)
        self.links.set_color(self.cmap(occ))
        self.links.set_linewidth(1.6 + 3.2 * occ)
        q = r.queue[f] / self.cap
        self.qbars.set_segments([[b, b + d * self.qscale * min(1.0, float(qq))] for (b, d), qq in zip(self.bar_base, q)])
        cols = []
        for k, l in enumerate(r.link_ids):
            node = self.net.links[int(l)].to_node
            st = int(r.sig_state[f, node])
            cols.append(_SIG[st] if r.served[f, k] else _SIG[2])
        self.heads.set_color(cols)
        for i, txt in enumerate(self.node_txt):
            txt.set_text(f"{_LETTER[int(r.sig_state[f, i])]}{int(r.sig_tis[f, i]):3d}")
        for k, arc in enumerate(self.arcs):
            if r.msg_ok[f, k] > 0:
                arc.set_alpha(0.85)
                arc.set_color(INK)
                arc.set_linestyle("-")
            elif r.msg_drop[f, k] > 0:
                arc.set_alpha(0.8)
                arc.set_color("#afb8c1")
                arc.set_linestyle("--")
            else:
                arc.set_alpha(0.0)
        delay = r.meta.get("mean_delay", float("nan"))
        self.title.set_text(f"{label_for(r.label)}  ·  episode mean delay {delay:.1f} s/veh")
        self.clock.set_text(f"t = {int(r.t[f])} s   mean queue {float(r.queue[f].mean()):.1f} veh/link")
        return [self.links, self.qbars, self.heads, *self.node_txt, *self.arcs, self.title, self.clock]


def _figure(recs: Sequence[Recording], dashboard: bool = True):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n = len(recs)
    fig = plt.figure(figsize=(5.2 * n if n > 1 else 9.0, 7.2 if dashboard else 5.0), facecolor=SURFACE)
    if dashboard:
        gs = GridSpec(2, 1, height_ratios=[3.0, 1.9], hspace=0.12, figure=fig)
        top = gs[0].subgridspec(1, n, wspace=0.04)
        bot = gs[1].subgridspec(1, len(DASHBOARD_METRICS), wspace=0.32)
    else:
        top = GridSpec(1, n, figure=fig, wspace=0.04)
    panels = [NetworkPanel(fig.add_subplot(top[0, k]), r) for k, r in enumerate(recs)]
    dash = None
    if dashboard:
        axes = [fig.add_subplot(bot[0, k]) for k in range(len(DASHBOARD_METRICS))]
        t0, t1 = float(recs[0].t[0]), float(recs[0].t[-1])
        dash = Dashboard(axes, recs, (t0, t1))
        for ax in axes:                                   # dashboard axes side by side: label each one
            ax.tick_params(labelbottom=True)
            ax.set_xlabel("Simulation time [s]", fontsize=7.5)
    title = " vs ".join(label_for(r.label) for r in recs)
    m = recs[0].meta
    fig.text(0.01, 0.985, f"TrafficSync — {title}", fontsize=11, color=INK, va="top", weight="bold")
    fig.text(0.01, 0.955, f"{m.get('network')} · {m.get('demand')} · seed {m.get('seed')} · same arrivals in every panel"
             + (f" · message drop {m['drop_prob']:.0%}" if "drop_prob" in m else ""), fontsize=8, color=MUTED, va="top")
    return fig, panels, dash


def _frame_list(recs: Sequence[Recording], every: int) -> list[int]:
    n = min(r.n_frames for r in recs)
    return list(range(0, n, max(1, int(every))))


def render_still(recs: Sequence[Recording], t: float, path: str | Path, dpi: int = 150, dashboard: bool = True) -> Path:
    import matplotlib.pyplot as plt

    fig, panels, dash = _figure(recs, dashboard)
    f = recs[0].frame_index(t)
    for p in panels:
        p.update(f)
    if dash is not None:
        dash.update(float(recs[0].t[f]))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return path


def animate(recs: Sequence[Recording], path: str | Path, every: int = 1, fps: int = 12, dpi: int = 100,
            dashboard: bool = True) -> Path:
    """Write an ``.mp4`` (ffmpeg, H.264) or ``.gif`` (Pillow) of synchronised panels."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    fig, panels, dash = _figure(recs, dashboard)
    frames = _frame_list(recs, every)

    def draw(f):
        arts = []
        for p in panels:
            arts += p.update(f)
        if dash is not None:
            arts += dash.update(float(recs[0].t[f]))
        return arts

    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gif":
        writer = PillowWriter(fps=fps)
    else:
        writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20"])
    anim.save(str(path), writer=writer, dpi=dpi, savefig_kwargs={"facecolor": SURFACE})
    plt.close(fig)
    return path
