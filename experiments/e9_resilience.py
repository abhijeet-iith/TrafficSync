"""E9 - resilience to faulty reporters: (r, s)-robustness, W-MSR vs plain averaging (RQ4, H5).

Part A  robustness of the communication graphs (exact for N <= 12, sampled above) and the
        number of extra communication links needed for F = 1 (greedy, k-hop, and an
        exhaustive minimum for grid3x3).
Part B  network-level consensus on loads: every F-local placement of one faulty node,
        fault types constant / noise / drift, plain averaging vs W-MSR, on the road graph
        and on the augmented graphs (20 random initial load vectors each).
Part C  closed loop on the simulator: CoCycle (and CoMP) on grid3x3 with agent 4 (centre)
        reporting false consensus values; averaging vs W-MSR; road vs augmented graph.

Outputs: results/E9_resilience/{robustness.csv, augmentation.csv, consensus.csv, closed_loop_runs.csv,
closed_loop_summary.csv, summary.md, figures/*.png}.

Run:  python experiments/e9_resilience.py [--seeds 10] [--workers 2]   (~8 min on 2 cores)
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trafficsync.analysis.stats import mean_ci, paired_wilcoxon, percent_improvement  # noqa: E402
from trafficsync.control.wmsr import FaultModel, is_f_local, simulate_resilient  # noqa: E402
from trafficsync.network import load_network  # noqa: E402
from trafficsync.network.robustness import (augment_to_robustness, is_r_robust, is_rs_robust, khop_adjacency,  # noqa: E402
                                            robustness_profile)
from trafficsync.runner import RunSpec, load_controller_params, run_many, write_csv  # noqa: E402

OUT = ROOT / "results" / "E9_resilience"
NETS = ["corridor3", "grid2x2", "grid3x3", "grid4x4", "grid5x5"]
FAULTS = {
    "constant": FaultModel("constant", value=1.0),
    "noise": FaultModel("noise", sigma=0.5),
    "drift": FaultModel("drift", rate=0.01),
}
# closed loop: what agent 4 claims (per consensus channel)
CL_FAULTS = {
    "constant": {"model": {"kind": "constant", "value": 1.0},
                 "channels": {"rho": {"kind": "constant", "value": 1.0},
                              "w": {"kind": "constant", "value": 0.9},
                              "cw": {"kind": "constant", "value": 180.0}}},     # claims a 200 s Webster cycle
    "noise": {"model": {"kind": "noise", "sigma": 0.3},
              "channels": {"cw": {"kind": "noise", "sigma": 40.0}}},
}


def exhaustive_min_links(A: np.ndarray, target: str, k_max: int = 4) -> tuple[int | None, list]:
    N = A.shape[0]
    non = [(i, j) for i in range(N) for j in range(i + 1, N) if A[i, j] == 0]
    for k in range(1, k_max + 1):
        for comb in itertools.combinations(non, k):
            B = A.copy()
            for i, j in comb:
                B[i, j] = B[j, i] = 1
            if target == "F_local" and B.sum(1).min() < 3:
                continue
            ok = is_rs_robust(B, 2, 2) if target == "F_total" else is_r_robust(B, 3)
            if ok:
                return k, list(comb)
    return None, []


def part_a() -> tuple[list, list, dict]:
    rob, aug, graphs = [], [], {}
    for name in NETS:
        net = load_network(name)
        A = (net.road_A > 0).astype(float)
        prof = robustness_profile(A, F=1, n_samples=80000)
        rob.append({"network": name, "graph": "road", **prof})
        for target in ("F_total", "F_local"):
            for method in ("greedy", "khop"):
                res = augment_to_robustness(A, F=1, target=target, method=method, n_samples=5000, verify_samples=60000)
                row = {"network": name, "target": target, "method": method, "satisfied": res.satisfied,
                       "links_added": res.n_added, "k_hop": res.k_hop if method == "khop" else "",
                       "road_edges": res.original_edges, "added": json.dumps(res.added)}
                key = "F_total_ok" if target == "F_total" else "F_local_sufficient"
                row["independent_check"] = (bool(robustness_profile(res.A, F=1, n_samples=80000, seed=7)[key])
                                            if res.satisfied else False)
                if name == "grid3x3" and method == "greedy":
                    kmin, comb = exhaustive_min_links(A, target)
                    row["exhaustive_min"] = kmin
                    row["exhaustive_example"] = json.dumps(comb)
                aug.append(row)
                if method == "greedy":
                    graphs[(name, target)] = (res.A, res.added)
                    p2 = robustness_profile(res.A, F=1, n_samples=80000, seed=7)
                    rob.append({"network": name, "graph": f"road+greedy({target})", **p2})
        Ak = khop_adjacency(A, 2)
        rob.append({"network": name, "graph": "2-hop", **robustness_profile(Ak, F=1, n_samples=80000, seed=7)})
    return rob, aug, graphs


def part_b(graphs: dict, n_init: int = 20) -> list[dict]:
    rows = []
    for name in ("corridor3", "grid2x2", "grid3x3"):
        net = load_network(name)
        variants = {"road": (net.road_A > 0).astype(float),
                    "augmented F_total": graphs[(name, "F_total")][0],
                    "augmented F_local": graphs[(name, "F_local")][0]}
        for gname, A in variants.items():
            for f in range(net.N):
                if not is_f_local(A, [f], 1):
                    continue
                for kind, fm in FAULTS.items():
                    for rule in ("average", "wmsr"):
                        spreads, biases, inv = [], [], []
                        for s in range(n_init):
                            x0 = np.random.default_rng(s).uniform(0.1, 0.6, net.N)
                            fm2 = FaultModel(fm.kind, fm.value, fm.relative, fm.sigma, fm.rate, seed=s)
                            run = simulate_resilient(A, x0, [f], fm2, F=1, rule=rule, iters=300)
                            spreads.append(run.final_spread)
                            biases.append(abs(run.final_bias))
                            inv.append(run.interval_invariant)
                        rows.append({"network": name, "graph": gname, "faulty": f, "fault": kind, "rule": rule,
                                     "final_spread_mean": float(np.mean(spreads)), "final_spread_max": float(np.max(spreads)),
                                     "abs_bias_mean": float(np.mean(biases)), "abs_bias_max": float(np.max(biases)),
                                     "interval_invariant_share": float(np.mean(inv)), "agreement_share": float(np.mean(np.array(spreads) < 1e-4))})
    return rows


def closed_loop_specs(graphs: dict, seeds: list[int]) -> tuple[list[RunSpec], dict]:
    base = load_controller_params()
    added = graphs[("grid3x3", "F_total")][1]
    sim = {"duration_s": 3600, "warmup_s": 600}
    probe = {"mode": "perfect_sensor"}
    comm_road = {}
    comm_aug = {"topology": {"topology": "road", "add_links": [list(e) for e in added]}}
    conds = {}
    aggs = {"average": ("average", "dynamic"), "wmsr-dynamic": ("wmsr", "dynamic"), "wmsr-static": ("wmsr", "static")}
    for ctrl, demand in (("cocycle", "D_med"), ("comp", "D_high")):
        for agg, (aggregation, mode) in aggs.items():
            for graph, comm in (("road", comm_road), ("augmented", comm_aug)):
                for fault in ("none", "constant", "noise"):
                    if ctrl == "comp" and (graph == "augmented" or fault == "noise"):
                        continue
                    name = f"{ctrl}|{agg}|{graph}|{fault}"
                    over = {"consensus": {"aggregation": aggregation, "mode": mode, "F": 1}}
                    if fault != "none":
                        over["faults"] = {"agents": [4], **CL_FAULTS[fault]}
                    conds[name] = (ctrl, demand, comm, {**base, name: {"base": ctrl, **over}})
    specs = [RunSpec("E9", "grid3x3", d, name, s, "perfect_sensor", sim, probe, comm, params)
             for name, (ctrl, d, comm, params) in conds.items() for s in seeds]
    return specs, conds


def summarize_closed(rows: list[dict]) -> list[dict]:
    by = {}
    for r in rows:
        by.setdefault(r["controller"], {})[r["seed"]] = r
    out = []
    for name, runs in by.items():
        ctrl, agg, graph, fault = name.split("|")
        ref = by.get(f"{ctrl}|average|road|none", {})  # fault-free, plain (dynamic) averaging
        seeds = sorted(set(runs) & set(ref))
        d = [runs[s]["mean_delay"] for s in seeds]
        d0 = [ref[s]["mean_delay"] for s in seeds]
        ci = mean_ci(d)
        imp = percent_improvement(d, d0) if name != f"{ctrl}|average|road|none" else {"improvement_pct": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
        w = paired_wilcoxon(d, d0)
        row = {"controller": ctrl, "aggregation": agg, "graph": graph, "fault": fault, "n": len(seeds),
               "mean_delay": ci["mean"], "delay_ci": ci["half_width"],
               "vs_faultfree_avg_pct": -imp["improvement_pct"], "ci_lo": -imp["ci_hi"], "ci_hi": -imp["ci_lo"],
               "p_wilcoxon": w["p_value"],
               "throughput_vph": float(np.mean([runs[s]["throughput_vph"] for s in seeds])),
               "gridlock_runs": int(sum(bool(runs[s]["gridlock"]) for s in seeds))}
        if ctrl == "cocycle":
            row["mean_cycle_benign"] = float(np.mean([np.mean([c for i, c in enumerate(runs[s]["final_cycles"]) if i != 4])
                                                       for s in seeds]))
            row["cycle_spread_benign"] = float(np.mean([np.ptp([c for i, c in enumerate(runs[s]["final_cycles"]) if i != 4])
                                                         for s in seeds]))
        out.append(row)
    order = {"none": 0, "constant": 1, "noise": 2}
    # benign estimate error of the agreed cycle vs the fault-free reference is in mean_cycle_benign
    return sorted(out, key=lambda r: (r["controller"], order[r["fault"]], r["graph"], r["aggregation"]))


def figures(graphs: dict, cl_summary: list[dict]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fd = OUT / "figures"
    fd.mkdir(parents=True, exist_ok=True)
    made = []
    # F08: benign trajectories, plain averaging vs W-MSR, constant attacker at the centre
    A = graphs[("grid3x3", "F_total")][0]
    x0 = np.random.default_rng(3).uniform(0.1, 0.6, 9)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    runs = [("Plain averaging (augmented graph)", simulate_resilient(A, x0, [4], FAULTS["constant"], 1, "average", 120)),
            ("W-MSR, F = 1 (road graph, not robust)", simulate_resilient(load_network("grid3x3").road_A, x0, [4], FAULTS["constant"], 1, "wmsr", 120)),
            ("W-MSR, F = 1 (augmented, (2,2)-robust)", simulate_resilient(A, x0, [4], FAULTS["constant"], 1, "wmsr", 120))]
    for ax, (title, run) in zip(axes, runs):
        for i in range(9):
            if i == 4:
                ax.plot(run.X[:, i], color="#c0392b", lw=2, ls="--", label="faulty agent (reports 1.0)")
            else:
                ax.plot(run.X[:, i], color="#2c3e50", lw=1, alpha=0.8, label="benign agents" if i == 0 else None)
        ax.axhspan(*run.interval0, color="#27ae60", alpha=0.12, label="initial benign interval")
        ax.axhline(run.target, color="#27ae60", lw=1, ls=":", label="benign average")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("consensus iteration")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("load estimate x_i [–]")
    axes[0].legend(fontsize=7, loc="center right")
    fig.suptitle("F08 - W-MSR vs plain averaging under a constant attacker (grid 3x3, agent 4 faulty)", fontsize=10)
    fig.tight_layout()
    fig.savefig(fd / "F08_wmsr_vs_average.png", dpi=300)
    plt.close(fig)
    made.append("F08_wmsr_vs_average.png")

    rows = [r for r in cl_summary if r["controller"] == "cocycle"]
    labels = [f"{r['fault']}\n{r['graph']}\n{r['aggregation']}" for r in rows]
    fig, ax = plt.subplots(figsize=(13, 4.2))
    cmap = {"average": "#2c3e50", "wmsr-dynamic": "#e67e22", "wmsr-static": "#2980b9"}
    cols = [cmap[r["aggregation"]] for r in rows]
    ax.bar(range(len(rows)), [r["mean_delay"] for r in rows], yerr=[r["delay_ci"] for r in rows], color=cols, capsize=3)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("mean delay [s/veh]")
    ax.set_title("E9 closed loop - CoCycle, grid 3x3, D_med, agent 4 faulty "
                 "(dark: averaging, orange: W-MSR dynamic, blue: W-MSR static); 95% CI", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fd / "E9_closed_loop_delay.png", dpi=300)
    plt.close(fig)
    made.append("E9_closed_loop_delay.png")
    return made


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--reuse", action="store_true", help="reuse parts A and B from an earlier run (closed loop only)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.reuse and (OUT / "augmentation.csv").exists():
        import csv

        rob = list(csv.DictReader(open(OUT / "robustness.csv")))
        aug = list(csv.DictReader(open(OUT / "augmentation.csv")))
        cons = list(csv.DictReader(open(OUT / "consensus.csv")))
        for r in rob:
            for k in ("F_total_ok", "F_local_sufficient", "exact"):
                r[k] = r[k] == "True"
            r["max_r"] = None if r["max_r"] in ("", "None") else int(r["max_r"])
        for r in aug:
            r["satisfied"] = r["satisfied"] == "True"
            r["independent_check"] = r["independent_check"] == "True"
        for r in cons:
            for k in ("interval_invariant_share", "agreement_share", "abs_bias_mean"):
                r[k] = float(r[k])
        graphs = {}
        for r in aug:
            if r["method"] == "greedy":
                net = load_network(r["network"])
                A = (net.road_A > 0).astype(float)
                for i, j in json.loads(r["added"]):
                    A[i, j] = A[j, i] = 1.0
                graphs[(r["network"], r["target"])] = (A, [tuple(e) for e in json.loads(r["added"])])
    else:
        print("Part A: robustness")
        rob, aug, graphs = part_a()
        write_csv(OUT / "robustness.csv", rob)
        write_csv(OUT / "augmentation.csv", aug)
        print("Part B: network-level consensus")
        cons = part_b(graphs)
        write_csv(OUT / "consensus.csv", cons)
    print("Part C: closed loop")
    specs, _ = closed_loop_specs(graphs, list(range(args.seeds)))
    rows = run_many(specs, args.workers, "E9")
    keep = ["controller", "seed", "mean_delay", "throughput_vph", "gridlock", "spillback_events", "cycle_spread",
            "final_cycles", "msgs_per_s_agent"]
    write_csv(OUT / "closed_loop_runs.csv", [{k: (json.dumps(r[k]) if isinstance(r.get(k), list) else r.get(k)) for k in keep}
                                             for r in rows])
    cl = summarize_closed(rows)
    write_csv(OUT / "closed_loop_summary.csv", cl)
    figs = figures(graphs, cl)
    (OUT / "summary.md").write_text(summary_md(rob, aug, cons, cl, figs))
    print((OUT / "summary.md").read_text())


def summary_md(rob, aug, cons, cl, figs) -> str:
    L = ["# E9 - resilience (W-MSR, robustness)", ""]
    L += ["## A. Robustness of communication graphs (F = 1)", "",
          "| network | graph | edges | min deg | max r | (2,2)-robust (F-total, iff) | 3-robust (F-local, suff.) | exact |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rob:
        L.append(f"| {r['network']} | {r['graph']} | {r['edges']} | {r['min_degree']} | {r['max_r'] if r['max_r'] is not None else '–'} | "
                 f"{'yes' if r['F_total_ok'] else 'no'} | {'yes' if r['F_local_sufficient'] else 'no'} | {'exact' if r['exact'] else 'sampled'} |")
    L += ["", "Links to add (F = 1).  For N > 12 the check is sampled: 'search' = no counterexample in the search's "
          "own samples, 'independent' = none in 80,000 further samples (a 'no' there refutes the search).", "",
          "| network | target | method | satisfied (search) | independent check | links added | exhaustive minimum (<= 4 links) |",
          "|---|---|---|---|---|---|---|"]
    for r in aug:
        L.append(f"| {r['network']} | {r['target']} | {r['method']}{' (k=' + str(r['k_hop']) + ')' if r['k_hop'] else ''} | "
                 f"{'yes' if r['satisfied'] else 'no'} | {'yes' if r['independent_check'] else 'no'} | "
                 f"{r['links_added'] if r['satisfied'] else '–'} (road has {r['road_edges']}) | {r.get('exhaustive_min', '') or ''} |")
    L += ["", "## B. Network-level consensus (mean over all F-local placements of one faulty node, 20 initial states)", "",
          "| network | graph | fault | rule | interval kept | benign agreement | mean abs. bias |", "|---|---|---|---|---|---|---|"]
    agg = {}
    for r in cons:
        agg.setdefault((r["network"], r["graph"], r["fault"], r["rule"]), []).append(r)
    for k, rs in agg.items():
        L.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {np.mean([x['interval_invariant_share'] for x in rs]):.0%} | "
                 f"{np.mean([x['agreement_share'] for x in rs]):.0%} | {np.mean([x['abs_bias_mean'] for x in rs]):.3f} |")
    L += ["", "## C. Closed loop (grid3x3, agent 4 faulty, local detectors)", "",
          "Change of mean delay vs the fault-free controller with plain averaging on the road graph "
          "(positive = worse), paired over seeds, [95% paired-bootstrap CI].", "",
          "| controller | fault | graph | aggregation | delay [s] | change | p | benign cycle [s] | benign cycle spread [s] |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in cl:
        L.append(f"| {r['controller']} | {r['fault']} | {r['graph']} | {r['aggregation']} | {r['mean_delay']:.1f} ± {r['delay_ci']:.1f} | "
                 f"{r['vs_faultfree_avg_pct']:+.1f}% [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}] | {r['p_wilcoxon']:.3g} | "
                 f"{r.get('mean_cycle_benign', float('nan')):.1f} | {r.get('cycle_spread_benign', float('nan')):.1f} |")
    L += ["", "Figures: " + ", ".join(f"`figures/{f}`" for f in figs), ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
