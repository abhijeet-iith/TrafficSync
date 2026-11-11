"""Per-city tuning of the switching margin and green limits on SUMO seed 100 (never used for evaluation).

Like a real deployment, each city's agents get their own setting.  Candidates: margin 2 vehicles (absolute) or
1 / 2 / 3 / 6 vehicles per approach lane, green 5-40 s or longer (10-90, 15-120, 25-150 s) for heavily loaded,
multi-lane cities.  Objective: lowest mean delay of max-pressure agents among margins that
complete at least 98 % of the trips the city's own program completes (so that a margin cannot "win" by
leaving vehicles stuck outside the network).  Writes configs/sumo_tuned.yaml and results/SUMO/tuning.csv.

    python experiments/sumo_tune.py --workers 2
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CITIES = ["cologne3", "cologne8", "ingolstadt7", "ingolstadt21", "bologna_acosta", "bologna_pasubio"]
CANDS = {"2 veh": {"theta": 2.0, "theta_per_lane": False}, "1 veh/lane": {"theta": 1.0, "theta_per_lane": True},
         "2 veh/lane": {"theta": 2.0, "theta_per_lane": True}, "3 veh/lane": {"theta": 3.0, "theta_per_lane": True},
         "6 veh/lane, green 10-90 s": {"theta": 6.0, "theta_per_lane": True, "g_min": 10, "g_max": 90},
         "6 veh/lane, green 15-120 s": {"theta": 6.0, "theta_per_lane": True, "g_min": 15, "g_max": 120},
         "6 veh/lane, green 25-150 s": {"theta": 6.0, "theta_per_lane": True, "g_min": 25, "g_max": 150}}


def _job(a):
    from trafficsync.sumo.realmap import run_realmap_episode

    city, ctrl, name = a
    m = run_realmap_episode(city, ctrl, 100, CANDS.get(name, {}), work_dir=ROOT / "results" / "SUMO" / "work_tune")
    return {"city": city, "controller": ctrl, "candidate": name, "mean_delay": m["mean_delay"], "completed": m["completed"]}


def main() -> None:
    import shutil

    import yaml

    from trafficsync.runner import write_csv

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    jobs = [(c, "fixed", "-") for c in CITIES] + [(c, "maxpressure", n) for c in CITIES for n in CANDS]
    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(pool.map(_job, jobs))
    write_csv(ROOT / "results" / "SUMO" / "tuning.csv", rows)
    shutil.rmtree(ROOT / "results" / "SUMO" / "work_tune", ignore_errors=True)
    out = {}
    for c in CITIES:
        ref = next(r for r in rows if r["city"] == c and r["controller"] == "fixed")
        ok = [r for r in rows if r["city"] == c and r["controller"] == "maxpressure" and r["completed"] >= 0.98 * ref["completed"]]
        pool_ = ok or [r for r in rows if r["city"] == c and r["controller"] == "maxpressure"]
        best = min(pool_, key=lambda r: r["mean_delay"]) if ok else max(pool_, key=lambda r: r["completed"])
        out[c] = dict(CANDS[best["candidate"]])
        print(f"{c:16s} city {ref['mean_delay']:6.1f} s / {ref['completed']} trips | " + " | ".join(
            f"{r['candidate']}: {r['mean_delay']:.1f} s / {r['completed']}" for r in rows if r["city"] == c and r["controller"] == "maxpressure")
              + f"  -> {best['candidate']}")
    (ROOT / "configs" / "sumo_tuned.yaml").write_text(
        "# Per-city switching margin for the real-map agents (experiments/sumo_tune.py, SUMO seed 100)\n" + yaml.safe_dump(out))


if __name__ == "__main__":
    main()
