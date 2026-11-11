"""Demo videos into ``results/demo/``.

* ``side_by_side_D_med.{mp4,gif}``  fixed vs actuated vs CoCycle, grid 3x3, D_med, seed 0
* ``side_by_side_D_high.mp4``       the same at D_high (congestion colours, CoCycle losing)
* ``cocycle_drop30.{mp4,gif}``      one CoCycle view + dashboard with 30% message loss
                                    (dropped messages flash grey)

All panels use the tuned E2 settings with local detectors and the 600-2400 s window
(one frame per 5 s, 12 fps).  Recordings are saved as ``.npz`` next to the videos, so
``--replot`` re-renders without simulating.

    python experiments/make_demo.py            # ~12 min on 2 cores
    python experiments/make_demo.py --quick    # gifs only, every 2nd frame
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trafficsync.viz.animate import animate, render_still  # noqa: E402
from trafficsync.viz.recording import Recording, experiment_settings, record_side_by_side  # noqa: E402

OUT = ROOT / "results" / "demo"
DEMOS = [
    # name, demand, controllers, drop probability, outputs
    ("side_by_side_D_med", "D_med", ["fixed", "actuated", "cocycle"], None, [".mp4", ".gif"]),
    ("side_by_side_D_high", "D_high", ["fixed", "actuated", "cocycle"], None, [".mp4"]),
    ("cocycle_drop30", "D_med", ["cocycle"], 0.3, [".mp4", ".gif"]),
]


def recordings(name: str, demand: str, ctrls: list[str], drop: float | None, replot: bool) -> list[Recording]:
    files = [OUT / f"{name}_{c.replace('+', 'plus')}.npz" for c in ctrls]
    if replot and all(f.exists() for f in files):
        return [Recording.load(f) for f in files]
    kw = experiment_settings("E2")
    kw["probe_cfg"] = {**kw["probe_cfg"], "mode": "perfect_sensor"}
    if drop is not None:
        kw["comm_cfg"] = {**kw["comm_cfg"], "drop_prob": drop}
    recs = record_side_by_side("grid3x3", demand, ctrls, 0, stride=5, t_start=600, t_end=2400, **kw)
    for r, f in zip(recs, files):
        if drop is not None:
            r.meta["label"] = r.meta["controller"]
            r.meta["drop_prob"] = drop
        r.save(f)
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replot", action="store_true", help="reuse saved recordings")
    ap.add_argument("--quick", action="store_true", help="gifs only, every second frame")
    ap.add_argument("--only", default=None, help="comma-separated demo names")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    only = set(a.only.split(",")) if a.only else None
    for name, demand, ctrls, drop, exts in DEMOS:
        if only and name not in only:
            continue
        t0 = time.perf_counter()
        recs = recordings(name, demand, ctrls, drop, a.replot)
        render_still(recs, 1500, OUT / f"{name}_t1500.png", dpi=150)
        for ext in exts:
            if a.quick and ext != ".gif":
                continue
            gif = ext == ".gif"
            animate(recs, OUT / f"{name}{ext}", every=2 if (gif or a.quick) else 1, fps=6 if gif else 12,
                    dpi=60 if gif else 100)
        print(f"{name}: {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
