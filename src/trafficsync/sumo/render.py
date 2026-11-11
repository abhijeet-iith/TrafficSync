"""Draw a SUMO run: road map, vehicles (coloured by speed) and every signal head (green / yellow / red).

``MapRecorder`` is passed as ``recorder=`` to :func:`~trafficsync.sumo.realmap.run_realmap_episode` (or wrapped by
:class:`GridRecorder` for :func:`~trafficsync.sumo.grid.run_sumo_episode`).  It captures a frame every ``every``
seconds inside ``[t0, t1)`` and writes a PNG still and an animated GIF / MP4 with matplotlib - no display needed.
For interactive viewing run the same episode with ``gui=True`` (opens ``sumo-gui``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = ["MapRecorder", "GridRecorder", "render_frames"]

_SIG = {"G": "#1a7f37", "g": "#2da44e", "y": "#d4a72c", "r": "#cf222e", "R": "#cf222e", "s": "#cf222e", "u": "#d4a72c",
        "o": "#999999", "O": "#999999"}


class MapRecorder:
    def __init__(self, every: int = 5, t0: float | None = None, t1: float | None = None, title: str = "",
                 rel_t0: int | None = None, rel_len: int | None = None) -> None:
        self.every, self.t0, self.t1, self.title = every, t0, t1, title
        self.rel_t0, self.rel_len = rel_t0, rel_len
        self.frames: list[dict] = []

    # ------------------------------------------------------------ capture
    def start(self, conn, netfile: str, agents=None) -> None:
        import sumolib

        net = sumolib.net.readNet(netfile)
        self.lanes = [np.asarray(ln.getShape(), float) for e in net.getEdges() if not e.isSpecial() for ln in e.getLanes()]
        self.tls_heads: dict[str, list[tuple[float, float]]] = {}
        for tls in conn.trafficlight.getIDList():
            pts = []
            for c in conn.trafficlight.getControlledLinks(tls):
                if c:
                    shp = np.asarray(conn.lane.getShape(c[0][0]), float)
                    pts.append(tuple(shp[-1]))
                else:
                    pts.append(None)
            self.tls_heads[tls] = pts
        xy = np.vstack(self.lanes)
        self.bbox = (xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max())
        if self.rel_t0 is not None:
            b = conn.simulation.getTime()
            self.t0 = b + self.rel_t0
            self.t1 = self.t0 + (self.rel_len or 10 ** 9)

    def step(self, conn, t: float) -> None:
        if self.t0 is not None and t < self.t0:
            return
        if self.t1 is not None and t >= self.t1:
            return
        if int(t) % self.every:
            return
        vids = conn.vehicle.getIDList()
        pos = np.array([conn.vehicle.getPosition(v) for v in vids], float).reshape(-1, 2)
        spd = np.array([conn.vehicle.getSpeed(v) for v in vids], float)
        sig = {tls: conn.trafficlight.getRedYellowGreenState(tls) for tls in self.tls_heads}
        self.frames.append({"t": t, "pos": pos, "spd": spd, "sig": sig})

    # ------------------------------------------------------------ output
    def save(self, path_stem: str | Path, fps: int = 8, dpi: int = 110, still_index: int | None = None,
             formats: Sequence[str] = (".gif",)) -> list[Path]:
        return render_frames(self, path_stem, fps=fps, dpi=dpi, still_index=still_index, formats=formats)


class GridRecorder(MapRecorder):
    """Adapter for :func:`trafficsync.sumo.grid.run_sumo_episode` (its recorder hooks get the simulator)."""

    def start(self, sim, *_):  # type: ignore[override]
        super().start(sim.conn, str(sim.netfile))

    def step(self, sim, *_):  # type: ignore[override]
        super().step(sim.conn, sim.t)


def render_frames(rec: MapRecorder, path_stem: str | Path, fps: int = 8, dpi: int = 110, still_index: int | None = None,
                  formats: Sequence[str] = (".gif",)) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.collections import LineCollection

    if not rec.frames:
        raise ValueError("no frames captured")
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    x0, x1, y0, y1 = rec.bbox
    w, h = x1 - x0, y1 - y0
    fig, ax = plt.subplots(figsize=(8.0, max(3.5, 8.0 * h / max(w, 1.0))))
    ax.add_collection(LineCollection(rec.lanes, colors="#c9ced6", linewidths=1.6, zorder=1))
    ax.set_xlim(x0 - 0.03 * w, x1 + 0.03 * w)
    ax.set_ylim(y0 - 0.03 * h, y1 + 0.03 * h)
    ax.set_aspect("equal")
    ax.axis("off")
    heads = [(tls, k, p) for tls, pts in rec.tls_heads.items() for k, p in enumerate(pts) if p is not None]
    hp = np.array([p for _, _, p in heads], float).reshape(-1, 2)
    sig_sc = ax.scatter(hp[:, 0], hp[:, 1], s=26, marker="s", c="#cf222e", zorder=3, edgecolors="#24292f", linewidths=0.4)
    veh_sc = ax.scatter([], [], s=10, c=[], cmap="viridis", vmin=0, vmax=14, zorder=4, edgecolors="none")
    txt = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", fontsize=9, color="#24292f")
    ax.set_title(rec.title + "\nsquares = signal heads (green / yellow / red), dots = vehicles", loc="left", fontsize=9)
    cb = fig.colorbar(veh_sc, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("vehicle speed [m/s]", fontsize=8)

    def draw(k: int):
        f = rec.frames[k]
        veh_sc.set_offsets(f["pos"] if len(f["pos"]) else np.zeros((0, 2)))
        veh_sc.set_array(f["spd"])
        sig_sc.set_color([_SIG.get(f["sig"][tls][i] if i < len(f["sig"][tls]) else "r", "#999999") for tls, i, _ in heads])
        txt.set_text(f"t = {f['t']:.0f} s   vehicles: {len(f['pos'])}")
        return veh_sc, sig_sc, txt

    outs = []
    draw(still_index if still_index is not None else len(rec.frames) // 2)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=dpi * 1.5)
    outs.append(stem.with_suffix(".png"))
    anim = animation.FuncAnimation(fig, draw, frames=len(rec.frames), blit=False)
    for ext in formats:
        p = stem.with_suffix(ext)
        writer = animation.PillowWriter(fps=fps) if ext == ".gif" else animation.FFMpegWriter(fps=fps, bitrate=2400)
        anim.save(str(p), writer=writer, dpi=dpi)
        outs.append(p)
    plt.close(fig)
    return outs
