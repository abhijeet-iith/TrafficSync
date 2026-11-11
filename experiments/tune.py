"""Re-tune the threshold parameters on tuning seeds 100-102 (never used in evaluation).

Score = geometric mean over (network, demand, seed) of delay / fixed-time delay (lower is better).
Writes ``results/TUNING/tuning.csv`` and prints the best setting per controller; the chosen values
are copied by hand into ``configs/controllers_tuned.yaml``.

    python experiments/tune.py --workers 2
"""
from __future__ import annotations

import argparse
import copy
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trafficsync.runner import RunSpec, load_experiment, results_dir, run_many, write_csv  # noqa: E402

NETS, DEMS, SEEDS = ["grid2x2", "grid3x3"], ["D_med", "D_high", "D_tidal"], [100, 101, 102]

GRID = {
    "actuated": [{"theta": th, "gap_out": g} for th, g in itertools.product([2, 4, 8, 12, 16, 24, 32], [True, False])],
    "comp": [{"theta": th, "kappa": k, "gap_out": True} for th, k in itertools.product([4, 8, 12, 16], [0.0, 0.5])],
    "cocycle": [{"cycle": {"C_min": cm, "rho_ref": rr}, "max_extension_s": me}
                for cm, rr, me in [(40, 0.6, 10), (30, 0.6, 10), (30, 0.8, 10), (30, 0.8, 0), (40, 0.8, 20)]],
    # second round after the first showed the actuation window (max_extension_s) mattered most
    "cocycle_ext": [{"max_extension_s": me, "cycle": {"C_min": cm}} for me, cm in [(20, 40), (30, 40), (40, 40), (20, 30)]],
    "comp+cycle": [{"max_extension_s": me} for me in (10, 40)],
}
BASE_OF = {"cocycle_ext": "cocycle"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--controllers", default="actuated,comp,cocycle")
    a = ap.parse_args()
    raw = {"experiment": {"id": "TUNING", "duration_s": 2400, "controllers": ["fixed"], "seeds": SEEDS}}
    base = load_experiment(raw)
    specs, keys = [], []
    def add(ctrl, params, tag):
        cp = copy.deepcopy(base["controller_params"])
        cp[f"{ctrl}@{tag}"] = {"base": BASE_OF.get(ctrl, ctrl), **params}
        for n, d, s in itertools.product(NETS, DEMS, SEEDS):
            specs.append(RunSpec("TUNING", n, d, f"{ctrl}@{tag}" if ctrl != "fixed" else "fixed", s, "perfect_sensor",
                                 base["sim"], {**base["probe"], "mode": "perfect_sensor"}, base["comm"], cp))
            keys.append((ctrl, tag, n, d, s))
    add("fixed", {}, "ref")
    for c in a.controllers.split(","):
        for k, p in enumerate(GRID[c]):
            add(c, p, str(k))
    rows = run_many(specs, a.workers, "tuning")
    tagname = a.controllers.replace(",", "_")
    ref = {(k[2], k[3], k[4]): r["mean_delay"] for k, r in zip(keys, rows) if k[0] == "fixed"}
    out, score = [], {}
    for k, r in zip(keys, rows):
        ratio = r["mean_delay"] / ref[(k[2], k[3], k[4])]
        out.append({"controller": k[0], "setting": k[1], "params": str(GRID.get(k[0], [{}])[int(k[1])] if k[0] != "fixed" else {}),
                    "network": k[2], "demand": k[3], "seed": k[4], "mean_delay": r["mean_delay"], "ratio_vs_fixed": ratio})
        score.setdefault((k[0], k[1]), []).append(np.log(ratio))
    write_csv(results_dir() / "TUNING" / f"tuning_{tagname}.csv", out)
    lines = ["controller,setting,params,geomean_ratio_vs_fixed"]
    for (c, t), v in sorted(score.items(), key=lambda kv: (kv[0][0], np.mean(kv[1]))):
        p = GRID[c][int(t)] if c != "fixed" else {}
        lines.append(f"{c},{t},\"{p}\",{np.exp(np.mean(v)):.3f}")
    (results_dir() / "TUNING" / f"tuning_summary_{tagname}.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
