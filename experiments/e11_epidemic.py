"""E11 - congestion spreading as an epidemic process (RQ5, H6).

1. Record the true per-intersection load ``rho_i(t) = mean_{l in I_i} n_l / C_l`` every second
   (observer only) for grid3x3 under every demand scenario, fixed-time and actuated control,
   10 seeds, 3600 s (analysis window t >= 600 s).
2. "Congested" = rho_i > 0.7 for more than 60 s (rho_i = mean over incoming links).
   In the standard scenarios that state almost never occurs (only under fixed-time control
   with the incident), because the 4 approaches of an intersection are rarely full together;
   so the analysis also uses the most loaded approach (> 0.7 and > 0.5), which is what
   spillback depends on.  An overload level of
   0.25 veh/s on every entry is added to the standard scenarios.
3. Pooled MLE of beta (neighbour-driven), beta_0 (background) and gamma per scenario;
   spectral threshold ratio R = beta lambda_max(A) / gamma; likelihood-ratio test of beta > 0.
4. Prediction vs observation: R > 1 (spreading predicted) vs the observed congested fraction,
   and the mean-field SIS trajectory with the estimated rates vs the observed mean fraction.
5. H6: (a) eigenvector centrality vs each node's attributed spreading contribution;
   (b) 'vaccination' - CoMP with a priority weight on the most central intersection vs the
   same weight on a corner (control), D_high and an overload level 0.25 veh/s, 10 seeds.

Outputs: results/E11_epidemic/{rates.csv, observed.csv, centrality.csv, prioritisation_runs.csv,
prioritisation_summary.csv, summary.md, figures/*.png}.

Run:  python experiments/e11_epidemic.py [--seeds 10] [--workers 2]   (~6 min on 2 cores)
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trafficsync.analysis.epidemic import (congestion_states, estimate_rates, mean_field, observed_spreading,  # noqa: E402
                                           spectral_epidemic, spreading_contribution)
from trafficsync.analysis.stats import mean_ci, paired_wilcoxon, percent_improvement  # noqa: E402
from trafficsync.network import load_network  # noqa: E402
from trafficsync.runner import _deep_merge, load_controller_params, run_episode, write_csv  # noqa: E402

OUT = ROOT / "results" / "E11_epidemic"
NET = "grid3x3"
DEMANDS = ["D_low", "D_med", "D_ramp", "D_tidal", "D_incident", "D_hotspot", "D_high", 0.25]
CONTROLLERS = ["fixed", "actuated"]
WARMUP = 600
# (node-load aggregate over incoming links, threshold): the standard definition first
DEFS = [("mean", 0.7), ("max", 0.7), ("max", 0.5)]


def _dname(d) -> str:
    return d if isinstance(d, str) else f"uniform_{d:g}"


def _tag(agg: str, thr: float) -> str:
    return f"{agg}>{thr}"


class LoadRecorder:
    """Observer: true node loads every second (never seen by agents)."""

    needs_trace = False

    def start(self, sim, ctrls, bus):
        net = sim.network
        self.inc = [net.incoming_links(i) for i in range(net.N)]
        self.cap = sim.cap
        self.L = {"mean": np.zeros((sim.duration, net.N), dtype=np.float32),
                  "max": np.zeros((sim.duration, net.N), dtype=np.float32)}

    def step(self, sim, info, ctrls, bus):
        t = info["t"]
        for i, ls in enumerate(self.inc):
            r = [sim.ls[l].n / self.cap[l] for l in ls]
            self.L["mean"][t, i] = sum(r) / len(r)
            self.L["max"][t, i] = max(r)

    def finish(self, sim, metrics):
        pass


def _episode(args):
    demand, ctrl, seed, params = args
    rec = LoadRecorder()
    m = run_episode(NET, demand, ctrl, seed, probe_cfg={"mode": "perfect_sensor"}, controller_params=params,
                    recorder=rec)
    return _dname(demand), ctrl, seed, {k: m[k] for k in ("mean_delay", "throughput_vph", "spillback_events", "gridlock")}, rec.L


def _lr_test(states, A, est) -> float:
    """p-value of H0: beta = 0 (likelihood ratio, boundary -> chi2_1 / 2)."""
    from scipy.stats import chi2


    if est.n_onsets == 0:
        return float("nan")
    n_on = est.n_onsets
    b0 = n_on / est.susceptible_time_s
    ll0 = n_on * np.log(b0) - b0 * est.susceptible_time_s
    stat = max(0.0, 2 * (est.loglik - ll0))
    return float(0.5 * chi2.sf(stat, 1)) if stat > 0 else 1.0


def analyse(results: dict, A: np.ndarray) -> tuple[list, list, list, dict]:
    sp = spectral_epidemic(A)
    rate_rows, obs_rows, cent_rows = [], [], []
    traces = {}
    for agg, thr in DEFS:
        tag = _tag(agg, thr)
        contrib_all = np.zeros(A.shape[0])
        cong_time = np.zeros(A.shape[0])
        for (demand, ctrl), runs in sorted(results.items()):
            S_list = []
            for seed, (m, L) in sorted(runs.items()):
                S, ev = congestion_states(L[agg], threshold=thr, min_on_s=60, merge_gap_s=30)
                S_list.append(S)
                o = observed_spreading(S, A, t_from=WARMUP)
                obs_rows.append({"definition": tag, "demand": demand, "controller": ctrl, "seed": seed, **o,
                                 "mean_delay": m["mean_delay"]})
            est = estimate_rates(S_list, A, t_from=WARMUP)
            obs_frac = float(np.mean([S[WARMUP:].mean() for S in S_list]))
            mf = mean_field(A, est.beta, est.gamma, 0.0, len(S_list[0]) - WARMUP, beta0=est.beta0, model="SIS")
            mf_sir = mean_field(A, est.beta, est.gamma, 0.0, len(S_list[0]) - WARMUP, beta0=est.beta0, model="SIR")
            mf_nb = mean_field(A, 0.0, est.gamma, 0.0, len(S_list[0]) - WARMUP, beta0=est.beta0, model="SIS")
            rate_rows.append({"definition": tag, "demand": demand, "controller": ctrl, **est.as_dict(),
                              "p_beta_gt_0": _lr_test(S_list, A, est),
                              "observed_mean_fraction": obs_frac,
                              "observed_peak_fraction": float(np.mean([S[WARMUP:].mean(1).max() for S in S_list])),
                              "meanfield_SIS_mean_fraction": float(mf["fraction"].mean()),
                              "meanfield_SIR_mean_fraction": float(mf_sir["fraction"].mean()),
                              "meanfield_no_contagion_mean_fraction": float(mf_nb["fraction"].mean()),
                              "notes": "; ".join(est.notes)})
            traces[(tag, demand, ctrl)] = (np.mean([S[WARMUP:].mean(1) for S in S_list], axis=0), mf["fraction"])
            c = spreading_contribution(S_list, A, t_from=WARMUP)
            contrib_all += c
            cong_time += np.sum([S[WARMUP:].sum(0) for S in S_list], axis=0)
        from scipy.stats import spearmanr

        rho_c = spearmanr(sp["v_max"], contrib_all).correlation if contrib_all.any() else float("nan")
        per_time = np.divide(contrib_all, cong_time, out=np.zeros_like(contrib_all), where=cong_time > 0) * 3600
        rho_t = spearmanr(sp["v_max"], per_time).correlation if per_time.any() else float("nan")
        for i in range(A.shape[0]):
            cent_rows.append({"definition": tag, "node": i, "v_max": float(sp["v_max"][i]),
                              "attributed_spread": float(contrib_all[i]), "congested_node_s": float(cong_time[i]),
                              "spread_per_congested_hour": float(per_time[i]),
                              "spearman_vmax_spread": float(rho_c), "spearman_vmax_spread_per_hour": float(rho_t)})
    return rate_rows, obs_rows, cent_rows, traces


def _prio_episode(args):
    demand, name, seed, params = args
    rec = LoadRecorder()
    m = run_episode(NET, demand, name, seed, probe_cfg={"mode": "perfect_sensor"}, controller_params=params, recorder=rec)
    S, _ = congestion_states(rec.L["max"], threshold=0.5)
    return {"demand": _dname(demand), "condition": name, "seed": seed, "mean_delay": m["mean_delay"],
            "throughput_vph": m["throughput_vph"], "spillback_events": m["spillback_events"],
            "congested_node_s_0.5": int(S[WARMUP:].sum()),
            "congested_node_s_0.7": int(congestion_states(rec.L["max"], 0.7)[0][WARMUP:].sum())}


def prioritisation(seeds: list[int], workers: int, central: int, corner: int) -> tuple[list, list]:
    base = load_controller_params()
    conds = {"comp": {}}
    for g in (1.5, 0.67):
        conds[f"comp-central-g{g}"] = {"base": "comp", "priority": {"agents": [central], "gamma": g}}
        conds[f"comp-corner-g{g}"] = {"base": "comp", "priority": {"agents": [corner], "gamma": g}}
    params = _deep_merge(base, conds)
    jobs = [(d, c, s, params) for d in ("D_high", 0.25) for c in conds for s in seeds]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_prio_episode, jobs))
    summ = []
    for d in ("D_high", "uniform_0.25"):
        ref = {r["seed"]: r for r in rows if r["demand"] == d and r["condition"] == "comp"}
        for c in conds:
            rs = {r["seed"]: r for r in rows if r["demand"] == d and r["condition"] == c}
            s = sorted(set(ref) & set(rs))
            row = {"demand": d, "condition": c}
            for met in ("mean_delay", "congested_node_s_0.5", "spillback_events"):
                a = [rs[k][met] for k in s]
                b = [ref[k][met] for k in s]
                ci = mean_ci(a)
                row[met] = ci["mean"]
                row[met + "_ci"] = ci["half_width"]
                if c != "comp" and np.mean(b) > 0:
                    imp = percent_improvement(a, b)
                    row[met + "_change_pct"] = -imp["improvement_pct"]
                    row[met + "_change_ci"] = f"[{-imp['ci_hi']:+.1f}, {-imp['ci_lo']:+.1f}]"
                    row[met + "_p"] = paired_wilcoxon(a, b)["p_value"]
            summ.append(row)
    return rows, summ


def figures(rate_rows, traces, cent_rows) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fd = OUT / "figures"
    fd.mkdir(parents=True, exist_ok=True)
    made = []
    names = [_dname(d) for d in DEMANDS]
    for agg, thr in DEFS:
        tag = _tag(agg, thr)
        rows = [r for r in rate_rows if r["definition"] == tag]
        if not any(r["n_onsets"] for r in rows):
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        ax = axes[0]
        x = np.arange(len(names))
        for k, ctrl in enumerate(CONTROLLERS):
            R = [next((r["R_spectral"] for r in rows if r["demand"] == d and r["controller"] == ctrl), np.nan) for d in names]
            ax.bar(x + (k - 0.5) * 0.38, R, width=0.38, label=ctrl)
        ax.axhline(1.0, color="#c0392b", ls="--", lw=1.2, label=r"threshold $\beta\lambda_{max}/\gamma = 1$")
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("uniform_", "λ=") for n in names], rotation=30, fontsize=8)
        ax.set_ylabel(r"$\hat\beta\,\lambda_{max}(A)\,/\,\hat\gamma$ [–]")
        ax.set_title(f"Spectral threshold ratio per scenario ({agg} approach load > {thr})", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        ax = axes[1]
        for ctrl, mk in zip(CONTROLLERS, ("o", "s")):
            rr = [r for r in rows if r["controller"] == ctrl]
            ax.scatter([r["meanfield_SIS_mean_fraction"] for r in rr], [r["observed_mean_fraction"] for r in rr], marker=mk, label=ctrl)
            for r in rr:
                ax.annotate(r["demand"].replace("D_", "").replace("uniform_", "λ="), (r["meanfield_SIS_mean_fraction"], r["observed_mean_fraction"]), fontsize=6)
        lim = max(0.05, max([r["observed_mean_fraction"] for r in rows] + [r["meanfield_SIS_mean_fraction"] for r in rows]) * 1.1)
        ax.plot([0, lim], [0, lim], color="0.5", lw=1, ls=":")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("mean-field SIS prediction, mean congested fraction [–]")
        ax.set_ylabel("observed mean congested fraction [–]")
        ax.set_title("Prediction vs simulation (grid 3x3, 10 seeds)", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fn = f"F11_epidemic_threshold_{agg}{str(thr).replace('.', '')}.png"
        fig.savefig(fd / fn, dpi=300)
        plt.close(fig)
        made.append(fn)
    # time series for the most congested scenarios
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, d in zip(axes, ("D_high", "uniform_0.25", "D_incident")):
        obs, mf = traces[("max>0.5", d, "actuated")]
        t = np.arange(obs.size) + WARMUP
        ax.plot(t, obs, label="observed (mean of seeds)")
        ax.plot(t[: mf.size], mf[: obs.size], ls="--", label="mean-field SIS")
        if d == "D_incident":
            ax.axvspan(1200, 2100, color="#c0392b", alpha=0.08, label="incident")
        ax.set_title(f"{d}, actuated, max approach load > 0.5", fontsize=9)
        ax.set_xlabel("time [s]")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("congested fraction of intersections [–]")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fd / "E11_congested_fraction_timeseries.png", dpi=300)
    plt.close(fig)
    made.append("E11_congested_fraction_timeseries.png")
    rows = [r for r in cent_rows if r["definition"] == "max>0.5"]
    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.scatter([r["v_max"] for r in rows], [r["spread_per_congested_hour"] for r in rows])
    for r in rows:
        ax.annotate(str(r["node"]), (r["v_max"], r["spread_per_congested_hour"]), fontsize=7)
    ax.set_xlabel("eigenvector centrality v_max,i [–]")
    ax.set_ylabel("attributed neighbour onsets per congested hour")
    ax.set_title(f"H6: centrality vs spreading (max load > 0.5), Spearman {rows[0]['spearman_vmax_spread_per_hour']:+.2f}", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fd / "E11_centrality_vs_spreading.png", dpi=300)
    plt.close(fig)
    made.append("E11_centrality_vs_spreading.png")
    return made


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    net = load_network(NET)
    A = (net.road_A > 0).astype(float)
    params = load_controller_params()
    jobs = [(d, c, s, params) for d in DEMANDS for c in CONTROLLERS for s in range(args.seeds)]
    results: dict = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for k, (d, c, s, m, L) in enumerate(ex.map(_episode, jobs), 1):
            results.setdefault((d, c), {})[s] = (m, L)
            if k % 20 == 0:
                print(f"  [E11] {k}/{len(jobs)}", flush=True)
    rate_rows, obs_rows, cent_rows, traces = analyse(results, A)
    write_csv(OUT / "rates.csv", rate_rows)
    write_csv(OUT / "observed.csv", obs_rows)
    write_csv(OUT / "centrality.csv", cent_rows)
    sp = spectral_epidemic(A)
    central, corner = sp["ranking"][0], sp["ranking"][-1]
    prio_rows, prio_sum = prioritisation(list(range(args.seeds)), args.workers, central, corner)
    write_csv(OUT / "prioritisation_runs.csv", prio_rows)
    write_csv(OUT / "prioritisation_summary.csv", prio_sum)
    figs = figures(rate_rows, traces, cent_rows)
    (OUT / "summary.md").write_text(summary_md(sp, rate_rows, cent_rows, prio_sum, central, corner, figs))
    print((OUT / "summary.md").read_text())


def summary_md(sp, rate_rows, cent_rows, prio_sum, central, corner, figs) -> str:
    L = ["# E11 - congestion spreading as an epidemic (grid3x3)", "",
         f"lambda_max(A) = {sp['lambda_max_A']:.3f}; eigenvector centrality v_max = "
         + ", ".join(f"{i}: {v:.3f}" for i, v in enumerate(sp["v_max"])), ""]
    for agg, thr in DEFS:
        tag = _tag(agg, thr)
        what = "mean load of the incoming links" if agg == "mean" else "load of the most loaded incoming link"
        L += [f"## Congested = {what} > {thr} for > 60 s{' (standard definition)' if agg == 'mean' else ' (spillback-oriented variant)'}", "",
              "| demand | controller | onsets | beta [1/s] | beta_0 [1/s] | 1/gamma [s] | R = beta lambda_max / gamma | p(beta>0) | spreading predicted | observed mean frac. | peak frac. | MF-SIS frac. | MF no-contagion frac. |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rate_rows:
            if r["definition"] != tag:
                continue
            L.append(f"| {r['demand']} | {r['controller']} | {r['n_onsets']} | {r['beta']:.2e} | {r['beta0']:.2e} | "
                     f"{r['mean_duration_s']:.0f} | {r['R_spectral']:.2f} | {r['p_beta_gt_0']:.2g} | "
                     f"{'yes' if r['spreading_predicted'] else 'no'} | {r['observed_mean_fraction']:.3f} | "
                     f"{r['observed_peak_fraction']:.2f} | {r['meanfield_SIS_mean_fraction']:.3f} | {r['meanfield_no_contagion_mean_fraction']:.3f} |")
        c = [r for r in cent_rows if r["definition"] == tag]
        if c:
            L += ["", f"Centrality vs attributed spreading (all scenarios pooled): Spearman(v_max, onsets attributed) = "
                  f"{c[0]['spearman_vmax_spread']:+.2f}; per congested hour = {c[0]['spearman_vmax_spread_per_hour']:+.2f}.", ""]
    L += ["## H6 prioritisation (CoMP; priority weight gamma on one intersection; 10 seeds; change vs plain CoMP)", "",
          f"Most central intersection = {central}; control = corner {corner}.", "",
          "| demand | condition | delay [s] | delay change | p | congested node-s (max load > 0.5) | change | p |", "|---|---|---|---|---|---|---|---|"]
    for r in prio_sum:
        L.append(f"| {r['demand']} | {r['condition']} | {r['mean_delay']:.1f} ± {r['mean_delay_ci']:.1f} | "
                 f"{r.get('mean_delay_change_pct', 0):+.1f}% {r.get('mean_delay_change_ci', '')} | {r.get('mean_delay_p', float('nan')):.2g} | "
                 f"{r['congested_node_s_0.5']:.0f} | {r.get('congested_node_s_0.5_change_pct', 0):+.1f}% {r.get('congested_node_s_0.5_change_ci', '')} | "
                 f"{r.get('congested_node_s_0.5_p', float('nan')):.2g} |")
    L += ["", "Figures: " + ", ".join(f"`figures/{f}`" for f in figs), ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
