"""E14 - stress scenarios: incident and tidal flow.

grid3x3, local detectors, 3600 s (metrics from 600 s).  Controllers: fixed, actuated, comp,
cocycle, comp+cycle (seeds 0-9) and dmpc+stl (seeds 0-4, expensive).

Incident (D_incident: the central eastbound link keeps 30% of its discharge capacity during
1200-2100 s):
* **recovery time** = seconds after the incident ends (2100 s) until the network mean queue
  (60 s trailing average) is back within 10% (and +0.2 veh) of its pre-incident level
  (mean over 600-1200 s); censored at 3600 s;
* **incident-link clearance** = seconds after 2100 s until the incident link's queue falls to
  q_thr = 0.2 C and stays there for 60 s;
* peak network mean queue and peak incident-link queue, spillback events, delay.

Tidal (D_tidal: E-W 0.22 veh/s, N-S 0.06 veh/s):
* **queue-clearance time** = durations of episodes in which a stop-line queue exceeds
  q_hi = 0.6 C until it is back below q_thr = 0.2 C (mean and max over links and episodes);
* mean queue of main (E/W) vs cross (N/S) approaches, delay, max red time, STL rho_clear / rho_resp.

Outputs: results/E14_incident_tidal/{runs.csv, summary.csv, summary.md, figures/*.png}
Run:  python experiments/e14_incident_tidal.py [--seeds 10] [--dmpc-seeds 5] [--workers 2]
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

OUT = ROOT / "results" / "E14_incident_tidal"
CONTROLLERS = ["fixed", "actuated", "comp", "cocycle", "comp+cycle", "dmpc+stl"]
INC_START, INC_END = 1200, 2100


def _trailing(x: np.ndarray, w: int = 60) -> np.ndarray:
    c = np.r_[0.0, np.cumsum(x)]
    out = np.empty_like(x, dtype=float)
    for t in range(x.size):
        a = max(0, t - w + 1)
        out[t] = (c[t + 1] - c[a]) / (t + 1 - a)
    return out


def incident_metrics(sim) -> dict:
    lg = sim.logger
    T = sim.duration
    qn = _trailing(lg.q.mean(axis=1).astype(float))
    base = float(qn[600:INC_START].mean())
    tol = max(0.1 * base, 0.2)
    rec = next((t - INC_END for t in range(INC_END, T) if qn[t] <= base + tol), None)
    link = sim.demand.incidents[0][0]
    k = sim.qcol[link]
    ql = lg.q[:, k].astype(float)
    thr = 0.2 * sim.cap[link]
    clear = None
    for t in range(INC_END, T - 60):
        if (ql[t:t + 60] <= thr).all():
            clear = t - INC_END
            break
    return {"recovery_s": float(rec) if rec is not None else float(T - INC_END), "recovery_censored": rec is None,
            "incident_link_clear_s": float(clear) if clear is not None else float(T - INC_END),
            "clear_censored": clear is None, "pre_incident_queue": base,
            "peak_network_queue": float(qn[INC_START:].max()), "peak_incident_link_queue": float(ql[INC_START:].max())}


def clearance_episodes(sim, t0: int = 600) -> list[int]:
    lg = sim.logger
    out = []
    for k, l in enumerate(sim.qlinks):
        C = sim.cap[l]
        q = lg.q[t0:, k]
        hi, lo = 0.6 * C, 0.2 * C
        t = 0
        while t < q.size:
            if q[t] > hi:
                s = t
                while t < q.size and q[t] > lo:
                    t += 1
                out.append(t - s)
            t += 1
    return out


def tidal_metrics(sim) -> dict:
    eps = clearance_episodes(sim)
    lg = sim.logger
    net = sim.network
    main = [k for k, l in enumerate(sim.qlinks) if net.links[l].heading in ("E", "W")]
    cross = [k for k, l in enumerate(sim.qlinks) if net.links[l].heading in ("N", "S")]
    return {"clear_episodes": len(eps), "clear_mean_s": float(np.mean(eps)) if eps else 0.0,
            "clear_max_s": float(np.max(eps)) if eps else 0.0,
            "queue_main": float(lg.q[600:, main].mean()), "queue_cross": float(lg.q[600:, cross].mean())}


def _run(args):
    demand, ctrl, seed, params = args
    m = run_episode("grid3x3", demand, ctrl, seed, probe_cfg={"mode": "perfect_sensor"}, controller_params=params,
                    stl_monitor=True, return_sim=True)
    sim = m.pop("_sim")
    m.pop("_controllers", None)
    extra = incident_metrics(sim) if demand == "D_incident" else tidal_metrics(sim)
    keep = ["mean_delay", "mean_travel_time", "throughput_vph", "spillback_events", "mean_queue", "max_queue",
            "max_red_time", "gridlock", "stl_rho_Phi", "stl_rho_safe", "stl_rho_resp", "stl_rho_clear", "stl_rho_starve"]
    series = _trailing(sim.logger.q.mean(axis=1).astype(float)) if demand == "D_incident" else None
    return {"demand": demand, "controller": ctrl, "seed": seed, **{k: m.get(k) for k in keep}, **extra}, series


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    for dem in ("D_incident", "D_tidal"):
        ref = {r["seed"]: r for r in rows if r["demand"] == dem and r["controller"] == "actuated"}
        mets = (["mean_delay", "recovery_s", "incident_link_clear_s", "peak_network_queue", "peak_incident_link_queue",
                 "spillback_events", "stl_rho_Phi"] if dem == "D_incident" else
                ["mean_delay", "clear_mean_s", "clear_max_s", "clear_episodes", "queue_main", "queue_cross", "max_red_time",
                 "stl_rho_clear", "stl_rho_resp", "stl_rho_Phi"])
        for c in CONTROLLERS:
            rs = {r["seed"]: r for r in rows if r["demand"] == dem and r["controller"] == c}
            if not rs:
                continue
            row = {"demand": dem, "controller": c, "n": len(rs)}
            for met in mets:
                ci = mean_ci([float(rs[k][met]) for k in rs])
                row[met], row[met + "_ci"] = ci["mean"], ci["half_width"]
            if dem == "D_incident":
                row["recovery_censored"] = int(sum(bool(rs[k]["recovery_censored"]) for k in rs))
            s = sorted(set(rs) & set(ref))
            if c != "actuated" and s:
                for met in ("mean_delay", "recovery_s" if dem == "D_incident" else "clear_mean_s"):
                    a = [float(rs[k][met]) for k in s]
                    b = [float(ref[k][met]) for k in s]
                    if np.mean(b) > 0:
                        imp = percent_improvement(a, b)
                        row[f"{met}_vs_actuated_pct"] = -imp["improvement_pct"]
                        row[f"{met}_vs_actuated_ci"] = f"[{-imp['ci_hi']:+.1f}, {-imp['ci_lo']:+.1f}]"
                    row[f"{met}_p"] = paired_wilcoxon(a, b)["p_value"]
            out.append(row)
    return out


def figures(series: dict, summ: list[dict]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fd = OUT / "figures"
    fd.mkdir(parents=True, exist_ok=True)
    cols = {"fixed": "#95a5a6", "actuated": "#2c3e50", "comp": "#8e44ad", "cocycle": "#16a085", "comp+cycle": "#27ae60",
            "dmpc+stl": "#2980b9"}
    fig, ax = plt.subplots(figsize=(8, 4))
    for c, S in series.items():
        S = np.array(S)
        t = np.arange(S.shape[1])
        m = S.mean(0)
        ax.plot(t, m, color=cols[c], label=f"{c} (n={S.shape[0]})", lw=1.4)
        if S.shape[0] > 1:
            h = 1.96 * S.std(0, ddof=1) / np.sqrt(S.shape[0])
            ax.fill_between(t, m - h, m + h, color=cols[c], alpha=0.15, lw=0)
    ax.axvspan(INC_START, INC_END, color="#c0392b", alpha=0.08, label="incident (30% capacity)")
    ax.set_xlim(600, 3600)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("network mean queue, 60 s average [veh/link]")
    ax.set_title("E14 - D_incident, grid 3x3 (mean over seeds, 95% CI band)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fd / "E14_incident_queue.png", dpi=300)
    plt.close(fig)
    made = ["E14_incident_queue.png"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, (dem, met, lab) in zip(axes, (("D_incident", "recovery_s", "recovery time after the incident [s]"),
                                          ("D_tidal", "clear_mean_s", "mean queue-clearance time (q > 0.6C -> < 0.2C) [s]"))):
        rs = [r for r in summ if r["demand"] == dem]
        ax.bar(range(len(rs)), [r[met] for r in rs], yerr=[r[met + "_ci"] for r in rs], capsize=3,
               color=[cols[r["controller"]] for r in rs])
        ax.set_xticks(range(len(rs)))
        ax.set_xticklabels([r["controller"] for r in rs], rotation=25, fontsize=8)
        ax.set_ylabel(lab, fontsize=8)
        ax.set_title(dem, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fd / "E14_recovery_clearance.png", dpi=300)
    plt.close(fig)
    made.append("E14_recovery_clearance.png")
    return made


def summary_md(summ, figs) -> str:
    L = ["# E14 - incident and tidal stress scenarios (grid3x3, local detectors)", "",
         "Seeds 0-9 (dmpc+stl: 0-4).  Changes are vs actuated control, paired over seeds [95% bootstrap CI].", "",
         "## D_incident (central eastbound link at 30% capacity, 1200-2100 s)", "",
         "| controller | delay [s] | delay vs actuated | recovery [s] | recovery vs actuated | censored | incident-link clearance [s] | peak net queue | peak link queue | spillback | rho(Phi) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in summ:
        if r["demand"] != "D_incident":
            continue
        dv = f"{r['mean_delay_vs_actuated_pct']:+.1f}% {r['mean_delay_vs_actuated_ci']}" if "mean_delay_vs_actuated_pct" in r else "–"
        rv = f"{r['recovery_s_vs_actuated_pct']:+.1f}% {r['recovery_s_vs_actuated_ci']}" if "recovery_s_vs_actuated_pct" in r else "–"
        L.append(f"| {r['controller']} | {r['mean_delay']:.1f} ± {r['mean_delay_ci']:.1f} | {dv} | {r['recovery_s']:.0f} ± {r['recovery_s_ci']:.0f} | {rv} | "
                 f"{r['recovery_censored']}/{r['n']} | {r['incident_link_clear_s']:.0f} ± {r['incident_link_clear_s_ci']:.0f} | "
                 f"{r['peak_network_queue']:.2f} | {r['peak_incident_link_queue']:.1f} | {r['spillback_events']:.0f} | {r['stl_rho_Phi']:.1f} |")
    L += ["", "## D_tidal (E-W 0.22 veh/s, N-S 0.06 veh/s)", "",
          "| controller | delay [s] | delay vs actuated | clearance mean [s] | clearance max [s] | episodes | queue main / cross [veh] | max red [s] | rho_clear | rho_resp |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in summ:
        if r["demand"] != "D_tidal":
            continue
        dv = f"{r['mean_delay_vs_actuated_pct']:+.1f}% {r['mean_delay_vs_actuated_ci']}" if "mean_delay_vs_actuated_pct" in r else "–"
        L.append(f"| {r['controller']} | {r['mean_delay']:.1f} ± {r['mean_delay_ci']:.1f} | {dv} | {r['clear_mean_s']:.0f} ± {r['clear_mean_s_ci']:.0f} | "
                 f"{r['clear_max_s']:.0f} | {r['clear_episodes']:.1f} | {r['queue_main']:.2f} / {r['queue_cross']:.2f} | {r['max_red_time']:.0f} | "
                 f"{r['stl_rho_clear']:.1f} | {r['stl_rho_resp']:.1f} |")
    L += ["", "Figures: " + ", ".join(f"`figures/{f}`" for f in figs), ""]
    return "\n".join(L)


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--dmpc-seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    params = load_controller_params()
    jobs = [(d, c, s, params) for c in reversed(CONTROLLERS) for d in ("D_incident", "D_tidal")
            for s in range(args.dmpc_seeds if c.startswith("dmpc") else args.seeds)]
    rows, series = [], {}
    with ProcessPoolExecutor(args.workers) as ex:
        for k, (r, sr) in enumerate(ex.map(_run, jobs), 1):
            rows.append(r)
            if sr is not None:
                series.setdefault(r["controller"], []).append(sr)
            if k % 10 == 0:
                print(f"  [E14] {k}/{len(jobs)}", flush=True)
    write_csv(OUT / "runs.csv", rows)
    summ = summarise(rows)
    write_csv(OUT / "summary.csv", summ)
    figs = figures({c: series[c] for c in CONTROLLERS if c in series}, summ)
    (OUT / "summary.md").write_text(summary_md(summ, figs))
    print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    main()
