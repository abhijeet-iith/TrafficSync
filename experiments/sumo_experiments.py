"""SUMO experiments: the project's controllers on a microscopic simulator and on real city maps.

S1  validation: grid 3x3, D_low / D_med / D_high, fixed / actuated / CoMP / CoCycle, seeds 0-4, run in the
    TrafficSync queue simulator AND in SUMO with the same vehicles and the same controller code
S2  real cities, real demand, 1 h (bundled: Cologne 3 / 8 signals and Ingolstadt 7 / 21 signals from RESCO,
    Bologna Acosta 7 / Pasubio 8 signals from DLR sumo-scenarios): the city's own signal program vs actuated,
    local max-pressure, CoMP and CoCycle agents, SUMO seeds 0-9
S3  CoMP on Cologne under message loss 0 / 50 / 100 %
S4  animations: Cologne fixed vs CoMP, Ingolstadt CoMP, SUMO grid 3x3 CoCycle (PNG + GIF)

Writes results/SUMO/{summary.md, *.csv, figures/}.   python experiments/sumo_experiments.py --workers 2
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trafficsync.analysis.stats import mean_ci, percent_improvement  # noqa: E402
from trafficsync.runner import write_csv  # noqa: E402

OUT = ROOT / "results" / "SUMO"
GRID_CTRL = ["fixed", "actuated", "comp", "cocycle"]
GRID_DEM = ["D_low", "D_med", "D_high"]
MAPS = ["cologne3", "cologne8", "ingolstadt7", "ingolstadt21", "bologna_acosta", "bologna_pasubio"]
MAP_CTRL = ["fixed", "actuated", "maxpressure", "comp", "cocycle"]
MAP_LABEL = {"fixed": "city program", "actuated": "actuated", "maxpressure": "max-pressure", "comp": "CoMP", "cocycle": "CoCycle"}
PERFECT = {"mode": "perfect_sensor"}


def _grid_job(a):
    sim, d, c, s = a
    print(f"  grid {sim:11s} {d:6s} {c:9s} seed {s}", flush=True)
    if sim == "sumo":
        from trafficsync.sumo.grid import run_sumo_episode

        m = run_sumo_episode("grid3x3", d, c, s, probe_cfg=PERFECT, work_dir=OUT / "work" / f"g_{d}_{c}_{s}")
    else:
        from trafficsync.runner import run_episode

        m = run_episode("grid3x3", d, c, s, probe_cfg=PERFECT)
    return {"simulator": sim, "demand": d, "controller": c, "seed": s, "mean_delay": m["mean_delay"],
            "mean_queue": m["mean_queue"], "throughput_vph": m["throughput_vph"], "mean_stops": m["mean_stops"]}


def _map_job(a):
    from trafficsync.sumo.realmap import run_realmap_episode

    sc, c, s, drop = a
    print(f"  map  {sc:11s} {c:11s} seed {s} drop {drop}", flush=True)
    m = run_realmap_episode(sc, c, s, {"comm_drop": drop}, work_dir=OUT / "work" / f"m_{sc}")
    m["drop"] = drop
    return m


def _paired(rows, key, ctrl, ref, **kw):
    sel = lambda c: {r["seed"]: r[key] for r in rows if r["controller"] == c and all(r[k] == v for k, v in kw.items())}  # noqa: E731
    a, b = sel(ctrl), sel(ref)
    s = sorted(set(a) & set(b))
    imp = percent_improvement([a[k] for k in s], [b[k] for k in s])
    return -imp["improvement_pct"], -imp["ci_hi"], -imp["ci_lo"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--grid-seeds", type=int, default=5)
    ap.add_argument("--map-seeds", type=int, default=10)
    ap.add_argument("--skip-runs", action="store_true", help="only the animations")
    ap.add_argument("--parts", default="S1,S2,S3,S4")
    a = ap.parse_args()
    parts = set(a.parts.split(","))
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    L = ["# SUMO experiments", ""]
    n = (2 * len(GRID_DEM) * len(GRID_CTRL) * a.grid_seeds * ("S1" in parts) + len(MAPS) * len(MAP_CTRL) * a.map_seeds * ("S2" in parts)
         + 4 * a.map_seeds * ("S3" in parts))
    print(f"{n} SUMO/TrafficSync runs + 4 animations on {a.workers} workers (about 25 min on 2 cores); "
          "to just watch a map use experiments/sumo_gui.py", flush=True)
    if not a.skip_runs:
        import csv

        def _load(name):
            rows = list(csv.DictReader(open(OUT / name)))
            for r in rows:
                for k, v in r.items():
                    try:
                        r[k] = int(v) if k in ("seed", "agents", "completed", "msgs", "msgs_dropped") else float(v)
                    except (TypeError, ValueError):
                        pass
            return rows

        with ProcessPoolExecutor(a.workers) as pool:
            if "S1" in parts:
                grid = list(pool.map(_grid_job, [(sim, d, c, s) for sim in ("trafficsync", "sumo") for d in GRID_DEM
                                                 for c in GRID_CTRL for s in range(a.grid_seeds)]))
                write_csv(OUT / "grid_validation.csv", grid)
            else:
                grid = _load("grid_validation.csv")
            if "S2" in parts:
                maps = list(pool.map(_map_job, [(sc, c, s, 0.0) for sc in MAPS for c in MAP_CTRL for s in range(a.map_seeds)]))
                write_csv(OUT / "real_maps.csv", maps)
            else:
                maps = _load("real_maps.csv")
            if "S3" in parts:
                drops = list(pool.map(_map_job, [(sc, c, s, p) for sc, c in (("cologne8", "comp"), ("cologne8", "cocycle"))
                                                 for p in (0.5, 1.0) for s in range(a.map_seeds)]))
                write_csv(OUT / "comp_message_loss.csv", drops)
            else:
                drops = _load("comp_message_loss.csv")
        L += [f"## S1 - validation: grid 3x3, local detectors, seeds 0-{a.grid_seeds - 1}", "",
              "Same vehicles (same seed, routes, arrival times) and the same controller code in both simulators. SUMO delay = "
              "time loss + insertion delay of vehicles generated after the 600 s warm-up (car-following, accelerations, "
              "a 60 m left-turn bay per approach); TrafficSync delay = waiting time in the queue model.", "",
              "| demand | controller | TrafficSync delay [s] | vs actuated | SUMO delay [s] | vs actuated |", "|---|---|---|---|---|---|"]
        for d in GRID_DEM:
            for c in GRID_CTRL:
                cells = []
                for sim in ("trafficsync", "sumo"):
                    rs = [r for r in grid if r["simulator"] == sim and r["demand"] == d]
                    ci = mean_ci([r["mean_delay"] for r in rs if r["controller"] == c])
                    ch = _paired(rs, "mean_delay", c, "actuated") if c != "actuated" else None
                    cells += [f"{ci['mean']:.1f} ± {ci['half_width']:.1f}", f"{ch[0]:+.1f}% [{ch[1]:+.1f}, {ch[2]:+.1f}]" if ch else "–"]
                L.append(f"| {d} | {c} | " + " | ".join(cells) + " |")
        L += ["", f"## S2 - real city maps, real demand (1 h, 10 min warm-up), SUMO seeds 0-{a.map_seeds - 1}", "",
              "| map | controller | delay [s] | vs city program (fixed) | waiting [s] | stops/veh | completed trips | switches / agent / h |",
              "|---|---|---|---|---|---|---|---|"]
        for sc in MAPS:
            rs = [r for r in maps if r["scenario"] == sc]
            for c in MAP_CTRL:
                cr = [r for r in rs if r["controller"] == c]
                ci = mean_ci([r["mean_delay"] for r in cr])
                ch = _paired(rs, "mean_delay", c, "fixed") if c != "fixed" else None
                L.append(f"| {sc} ({cr[0]['agents']} signals) | {c} | {ci['mean']:.1f} ± {ci['half_width']:.1f} | "
                         + (f"{ch[0]:+.1f}% [{ch[1]:+.1f}, {ch[2]:+.1f}]" if ch else "–")
                         + f" | {np.mean([r['mean_waiting'] for r in cr]):.1f} | {np.mean([r['mean_stops'] for r in cr]):.2f} | "
                           f"{np.mean([r['completed'] for r in cr]):.0f} | {np.mean([r['switches_per_agent_h'] for r in cr]):.0f} |")
        best = {}
        L += ["", "### Summary: mean delay [s] per city (best agent design in bold)", "",
              "| city | signals | " + " | ".join(MAP_LABEL[c] for c in MAP_CTRL) + " | best agent vs city program |",
              "|---|---|" + "---|" * (len(MAP_CTRL) + 1)]
        for sc in MAPS:
            rs = [r for r in maps if r["scenario"] == sc]
            mean = {c: float(np.mean([r["mean_delay"] for r in rs if r["controller"] == c])) for c in MAP_CTRL}
            b = min((c for c in MAP_CTRL if c != "fixed"), key=mean.get)
            best[sc] = b
            ch = _paired(rs, "mean_delay", b, "fixed")
            L.append(f"| {sc} | {rs[0]['agents']} | " + " | ".join((f"**{mean[c]:.1f}**" if c == b else f"{mean[c]:.1f}") for c in MAP_CTRL)
                     + f" | {MAP_LABEL[b]} {ch[0]:+.1f}% [{ch[1]:+.1f}, {ch[2]:+.1f}] |")
        L += ["", "## S3 - message loss on Cologne (CoMP and CoCycle)", "",
              "| controller | drop probability | delay [s] | vs local max-pressure |", "|---|---|---|---|"]
        mp = {r["seed"]: r["mean_delay"] for r in maps if r["scenario"] == "cologne8" and r["controller"] == "maxpressure"}
        for ctrl in ("comp", "cocycle"):
            allc = [{**r, "drop": 0.0} for r in maps if r["scenario"] == "cologne8" and r["controller"] == ctrl] + \
                   [r for r in drops if r["controller"] == ctrl]
            for p in (0.0, 0.5, 1.0):
                cr = [r for r in allc if float(r["drop"]) == p]
                ci = mean_ci([r["mean_delay"] for r in cr])
                imp = percent_improvement([r["mean_delay"] for r in cr], [mp[r["seed"]] for r in cr])
                L.append(f"| {MAP_LABEL[ctrl]} | {p:.1f} | {ci['mean']:.1f} ± {ci['half_width']:.1f} | {-imp['improvement_pct']:+.1f}% "
                         f"[{-imp['ci_hi']:+.1f}, {-imp['ci_lo']:+.1f}] |")
        _figures(grid, maps)
    if "S4" not in parts:
        if not a.skip_runs:
            (OUT / "summary.md").write_text("\n".join(L) + "\n")
            print("\n".join(L))
        return
    # S4 animations
    from trafficsync.sumo.grid import run_sumo_episode
    from trafficsync.sumo.realmap import run_realmap_episode
    from trafficsync.sumo.render import GridRecorder, MapRecorder

    anims = []
    for sc, c, title in [("cologne8", "fixed", "Cologne, real map and demand - city's fixed-time program"),
                         ("cologne8", "comp", "Cologne, real map and demand - TrafficSync CoMP agents"),
                         ("ingolstadt7", "comp", "Ingolstadt, real map and demand - TrafficSync CoMP agents"),
                         ("ingolstadt21", "cocycle", "Ingolstadt (21 signals), real demand - TrafficSync CoCycle agents"),
                         ("bologna_acosta", "maxpressure", "Bologna Acosta, real demand - max-pressure agents")]:
        rec = MapRecorder(every=3, rel_t0=1500, rel_len=300, title=title)
        run_realmap_episode(sc, c, 0, recorder=rec, end_offset_s=1800, work_dir=OUT / "work" / "anim")
        anims += rec.save(OUT / "figures" / f"{sc}_{c}", fps=8, dpi=80)
    rec = GridRecorder(every=3, t0=1500, t1=1800, title="SUMO grid 3x3, D_med - TrafficSync CoCycle agents")
    run_sumo_episode("grid3x3", "D_med", "cocycle", 0, probe_cfg=PERFECT, recorder=rec, work_dir=OUT / "work" / "anim_grid",
                     sim_cfg={"duration_s": 1800, "warmup_s": 600})
    anims += rec.save(OUT / "figures" / "sumo_grid3x3_cocycle", fps=8, dpi=80)
    if not a.skip_runs:
        L += ["", "## Animations", ""] + [f"* `figures/{p.name}`" for p in anims]
        L += ["", "Interactive browser replays of every city: `python experiments/make_web_viewer.py` -> `results/SUMO/web/index.html`."]
        (OUT / "summary.md").write_text("\n".join(L) + "\n")
        print("\n".join(L))


def _figures(grid, maps) -> None:
    from trafficsync.viz import plots

    plt = plots._plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, sim in zip(axes, ("trafficsync", "sumo")):
        w = 0.8 / len(GRID_CTRL)
        for k, c in enumerate(GRID_CTRL):
            ys = [mean_ci([r["mean_delay"] for r in grid if r["simulator"] == sim and r["demand"] == d and r["controller"] == c]) for d in GRID_DEM]
            ax.bar(np.arange(3) + (k - 1.5) * w, [y["mean"] for y in ys], w, yerr=[y["half_width"] for y in ys], capsize=2,
                   color=plots.color_for(c), label=plots.label_for(c))
        ax.set_xticks(range(3))
        ax.set_xticklabels(GRID_DEM)
        ax.set_ylabel("mean delay [s/veh]")
        ax.set_title("TrafficSync queue simulator" if sim == "trafficsync" else "SUMO (same vehicles, same controllers)", loc="left", fontsize=9)
        plots.style_axes(ax, "y")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    plots.save(fig, OUT / "figures" / "S1_grid_validation.png")
    cols = {"fixed": plots.color_for("fixed"), "actuated": plots.color_for("actuated"),
            "maxpressure": plots.color_for("maxpressure_local"), "comp": plots.color_for("comp"), "cocycle": plots.color_for("cocycle")}
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6))
    for ax, sc in zip(axes.ravel(), MAPS):
        ys = [mean_ci([r["mean_delay"] for r in maps if r["scenario"] == sc and r["controller"] == c]) for c in MAP_CTRL]
        ax.bar(range(len(MAP_CTRL)), [y["mean"] for y in ys], yerr=[y["half_width"] for y in ys], capsize=3,
               color=[cols[c] for c in MAP_CTRL])
        ax.set_xticks(range(len(MAP_CTRL)))
        ax.set_xticklabels([MAP_LABEL[c] for c in MAP_CTRL], fontsize=7.5, rotation=15)
        ax.set_ylabel("mean delay [s/veh]")
        ax.set_title(sc.replace("_", " ").title(), loc="left", fontsize=9)
        plots.style_axes(ax, "y")
    fig.suptitle("Real cities in SUMO, real demand, 1 h, 10 seeds (95% CI)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    plots.save(fig, OUT / "figures" / "S2_real_maps.png")


if __name__ == "__main__":
    main()
