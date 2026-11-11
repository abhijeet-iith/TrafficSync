"""E12 add-on: are VDN / QMIX worse only because of their learning rate / target period?

Retrains VDN and QMIX with lr = 1e-4 and target update every 500 steps (3 training seeds each,
same protocol and budget as e12_marl.py) and tests them on D_med, D_high, D_tidal (seeds 0-9).
Outputs: results/E12_marl/sensitivity/{checkpoints, summary.md}
Run:  python experiments/e12_ctde_sensitivity.py [--workers 2]   (~15 min)
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trafficsync.analysis.stats import mean_ci  # noqa: E402

OUT = ROOT / "results" / "E12_marl" / "sensitivity"
DEMANDS = ["D_med", "D_high", "D_tidal"]


def _train(args):
    v, s = args
    from trafficsync.marl.train import TrainConfig, train

    r = train(v, s, OUT / "checkpoints", TrainConfig(episodes=120, min_episodes=40, lr=1e-4, target_update=500),
              log=lambda m: print(m, flush=True))
    r.pop("log")
    return r


def _eval(args):
    path, v, s, d, seed = args
    from trafficsync.marl.train import evaluate_policy

    return v, s, d, seed, evaluate_policy(path, "grid3x3", d, seed)["mean_delay"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as ex:
        tr = list(ex.map(_train, [(v, s) for s in range(3) for v in ("vdn", "qmix")]))
        res = list(ex.map(_eval, [(r["checkpoint"], r["variant"], r["seed"], d, k) for r in tr for d in DEMANDS for k in range(10)]))
    main_rows = list(csv.DictReader(open(ROOT / "results" / "E12_marl" / "summary.csv")))
    L = ["# E12 sensitivity - VDN / QMIX with lr 1e-4 and target update 500", "",
         "| variant | best validation delay per training seed [s] |", "|---|---|"]
    for v in ("vdn", "qmix"):
        L.append(f"| {v} | " + ", ".join(f"{r['best_val_delay']:.1f}" for r in tr if r["variant"] == v) + " |")
    L += ["", "| scenario | variant | delay, lr 1e-4 [s] | spread over training seeds | delay, main run (lr 5e-4) [s] | best handcrafted [s] |",
          "|---|---|---|---|---|---|"]
    for d in DEMANDS:
        for v in ("vdn", "qmix"):
            per_seed = [np.mean([x[4] for x in res if x[0] == v and x[2] == d and x[3] == k]) for k in range(10)]
            per_pol = [np.mean([x[4] for x in res if x[0] == v and x[2] == d and x[1] == s]) for s in range(3)]
            ci = mean_ci(per_seed)
            main = next(r for r in main_rows if r["demand"] == d and r["controller"] == v)
            best = next(r for r in main_rows if r["demand"] == d and r["controller"] == r["best_handcrafted"])
            L.append(f"| {d} | {v} | {ci['mean']:.1f} ± {ci['half_width']:.1f} | {np.std(per_pol, ddof=1):.1f} | "
                     f"{float(main['mean_delay']):.1f} | {float(best['mean_delay']):.1f} ({best['controller']}) |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
