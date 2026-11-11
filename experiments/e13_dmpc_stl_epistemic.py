"""E13 - CoMP vs DMPC vs DMPC+STL vs DMPC+STL+epistemic under message loss.

grid3x3, local detectors, 3600 s (metrics from 600 s).  Cells (where spillback can occur):
overload 0.25 veh/s on every entry with message drop 0 / 0.3 / 0.8, and D_high with drop 0 / 0.8.
Controllers: comp, comp+epistemic (veto replaced by the K_i Room guard), dmpc, dmpc+stl,
dmpc+stl+epistemic.  Seeds 0-4 (DMPC costs ~100 s per simulated hour on one core).

Metrics: mean delay, spillback events, STL robustness rho(Phi) and its parts (offline monitor),
guard blocks / fallbacks, QP solve time.  Paired comparisons vs CoMP in the same cell.

Outputs: results/E13_dmpc_stl_epistemic/{runs.csv, summary.csv, summary.md, figures/*.png}
Run:  python experiments/e13_dmpc_stl_epistemic.py [--seeds 5] [--workers 2]   (~1 h on 2 cores)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trafficsync.analysis.stats import mean_ci, paired_wilcoxon, percent_improvement  # noqa: E402
from trafficsync.runner import load_controller_params, run_episode, write_csv  # noqa: E402

OUT = ROOT / "results" / "E13_dmpc_stl_epistemic"
CONTROLLERS = ["comp", "comp+epistemic", "dmpc", "dmpc+stl", "dmpc+stl+epistemic"]
CELLS = [(0.25, 0.0), (0.25, 0.3), (0.25, 0.8), ("D_high", 0.0), ("D_high", 0.8)]
KEEP = ["mean_delay", "mean_travel_time", "throughput_vph", "spillback_events", "mean_queue", "max_queue", "gridlock",
        "stl_rho_Phi", "stl_rho_safe", "stl_rho_resp", "stl_rho_starve", "stl_rho_clear", "stl_safe_violated_links",
        "stl_resp_violated_links", "stl_starve_violated_phases", "guard_blocked", "guard_fallbacks", "guard_checks",
        "solve_ms_mean", "solver_fallbacks", "solve_failures", "comm_sent", "comm_dropped", "msgs_per_s_agent", "wall_s"]


def _dname(d) -> str:
    return d if isinstance(d, str) else f"uniform_{d:g}"


def _run(args):
    demand, drop, ctrl, seed, params = args
    m = run_episode("grid3x3", demand, ctrl, seed, probe_cfg={"mode": "perfect_sensor"}, comm_cfg={"drop_prob": drop},
                    controller_params=params, stl_monitor=True)
    return {"demand": _dname(demand), "drop": drop, "controller": ctrl, "seed": seed, **{k: m.get(k, "") for k in KEEP}}


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    for dem, drop in [(_dname(d), p) for d, p in CELLS]:
        ref = {r["seed"]: r for r in rows if r["demand"] == dem and r["drop"] == drop and r["controller"] == "comp"}
        for c in CONTROLLERS:
            rs = {r["seed"]: r for r in rows if r["demand"] == dem and r["drop"] == drop and r["controller"] == c}
            if not rs:
                continue
            s = sorted(set(rs) & set(ref))
            row = {"demand": dem, "drop": drop, "controller": c, "n": len(rs)}
            for met in ("mean_delay", "spillback_events", "stl_rho_Phi", "stl_rho_safe", "stl_rho_resp", "stl_rho_starve",
                        "max_queue", "throughput_vph"):
                ci = mean_ci([float(rs[k][met]) for k in rs])
                row[met], row[met + "_ci"] = ci["mean"], ci["half_width"]
            for met in ("guard_blocked", "guard_fallbacks", "solve_ms_mean", "solver_fallbacks", "stl_safe_violated_links",
                        "stl_resp_violated_links", "stl_starve_violated_phases"):
                vals = [float(rs[k][met]) for k in rs if rs[k][met] != ""]
                row[met] = float(np.mean(vals)) if vals else ""
            if c != "comp" and s:
                a = [float(rs[k]["mean_delay"]) for k in s]
                b = [float(ref[k]["mean_delay"]) for k in s]
                imp = percent_improvement(a, b)
                row["delay_vs_comp_pct"] = -imp["improvement_pct"]
                row["delay_vs_comp_ci"] = f"[{-imp['ci_hi']:+.1f}, {-imp['ci_lo']:+.1f}]"
                row["p_delay"] = paired_wilcoxon(a, b)["p_value"]
                sa = [float(rs[k]["spillback_events"]) for k in s]
                sb = [float(ref[k]["spillback_events"]) for k in s]
                row["p_spillback"] = paired_wilcoxon(sa, sb)["p_value"]
            out.append(row)
    return out


def figures(summ: list[dict]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fd = OUT / "figures"
    fd.mkdir(parents=True, exist_ok=True)
    cols = {"comp": "#2c3e50", "comp+epistemic": "#7f8c8d", "dmpc": "#e67e22", "dmpc+stl": "#2980b9",
            "dmpc+stl+epistemic": "#27ae60"}
    made = []
    for met, lab, fn in (("mean_delay", "mean delay [s/veh]", "E13_delay.png"),
                         ("spillback_events", "spillback events [steps x links]", "E13_spillback.png"),
                         ("stl_rho_Phi", r"STL robustness $\rho(\Phi)$ [veh or s]", "E13_stl_robustness.png")):
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [3, 2]})
        for ax, dem in zip(axes, ("uniform_0.25", "D_high")):
            drops = sorted({r["drop"] for r in summ if r["demand"] == dem})
            w = 0.16
            for k, c in enumerate(CONTROLLERS):
                ys = [next((r[met] for r in summ if r["demand"] == dem and r["drop"] == d and r["controller"] == c), np.nan) for d in drops]
                es = [next((r[met + "_ci"] for r in summ if r["demand"] == dem and r["drop"] == d and r["controller"] == c), 0) for d in drops]
                ax.bar(np.arange(len(drops)) + (k - 2) * w, ys, w, yerr=es, capsize=2, color=cols[c], label=c)
            ax.set_xticks(range(len(drops)))
            ax.set_xticklabels([f"drop {d:.0%}" for d in drops])
            ax.set_title(dem.replace("uniform_", "overload λ = ") + " veh/s" if "uniform" in dem else dem, fontsize=9)
            ax.set_ylabel(lab)
            ax.grid(axis="y", alpha=0.3)
            if met == "stl_rho_Phi":
                ax.axhline(0, color="#c0392b", lw=1)
        axes[0].legend(fontsize=7, ncol=2)
        fig.suptitle(f"E13 - {lab} (grid 3x3, 5 seeds, 95% CI)", fontsize=10)
        fig.tight_layout()
        fig.savefig(fd / fn, dpi=300)
        plt.close(fig)
        made.append(fn)
    return made


def summary_md(summ, figs) -> str:
    L = ["# E13 - CoMP vs DMPC vs DMPC+STL vs DMPC+STL+epistemic under message loss", "",
         "grid3x3, local detectors, seeds 0-4, 3600 s (metrics t >= 600 s).  rho(Phi) = min over agents of the "
         "traffic specifications (spillback safety and response on internal links, starvation freedom); "
         "positive = satisfied.  Delay change vs CoMP in the same cell, paired over seeds [95% bootstrap CI].", "",
         "| demand | drop | controller | delay [s] | vs CoMP | p | spillback | rho(Phi) | rho_safe | rho_resp | rho_starve | violated (safe/resp/starve) | guard blocks | fallbacks | QP ms |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in summ:
        vs = f"{r['delay_vs_comp_pct']:+.1f}% {r['delay_vs_comp_ci']}" if "delay_vs_comp_pct" in r else "–"
        pv = f"{r['p_delay']:.2g}" if "p_delay" in r else "–"
        gb = f"{r['guard_blocked']:.0f}" if r["guard_blocked"] != "" else "–"
        gf = f"{r['guard_fallbacks']:.0f}" if r["guard_fallbacks"] != "" else "–"
        ms = f"{r['solve_ms_mean']:.1f}" if r["solve_ms_mean"] not in ("", 0.0) else "–"
        L.append(f"| {r['demand']} | {r['drop']:.0%} | {r['controller']} | {r['mean_delay']:.1f} ± {r['mean_delay_ci']:.1f} | {vs} | {pv} | "
                 f"{r['spillback_events']:.0f} ± {r['spillback_events_ci']:.0f} | {r['stl_rho_Phi']:.1f} | {r['stl_rho_safe']:.1f} | "
                 f"{r['stl_rho_resp']:.1f} | {r['stl_rho_starve']:.1f} | {r['stl_safe_violated_links']:.1f}/{r['stl_resp_violated_links']:.1f}/"
                 f"{r['stl_starve_violated_phases']:.1f} | {gb} | {gf} | {ms} |")
    L += ["", "Figures: " + ", ".join(f"`figures/{f}`" for f in figs), ""]
    return "\n".join(L)


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    params = load_controller_params()
    # expensive runs first so the pool stays busy
    jobs = [(d, p, c, s, params) for c in reversed(CONTROLLERS) for d, p in CELLS for s in range(args.seeds)]
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for k, r in enumerate(ex.map(_run, jobs), 1):
            rows.append(r)
            if k % 5 == 0:
                print(f"  [E13] {k}/{len(jobs)}", flush=True)
                write_csv(OUT / "runs.csv", rows)
    write_csv(OUT / "runs.csv", rows)
    summ = summarise(rows)
    write_csv(OUT / "summary.csv", summ)
    figs = figures(summ)
    (OUT / "summary.md").write_text(summary_md(summ, figs))
    print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    main()
