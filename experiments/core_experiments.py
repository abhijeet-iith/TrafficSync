"""Core experiments E0-E8 and the ablations at the full budget (30 seeds for the
headline comparison, 10 seeds for every sweep; evaluation seeds start at 0, tuning used 100-102).

E0   simulator validity: saturation flow 1/2/3 lanes, conservation over all scenarios, Webster delay, determinism
E1   baseline sanity: fixed vs actuated vs local max-pressure on corridor3 and grid2x2, D_low/med/high, 10 seeds
E2   main comparison: 6 controllers x 5 demands x {local detectors, probes 20%} x 30 seeds (grid3x3)
E3   network size: corridor3, 2x2, 3x3, 4x4, 5x5 x {D_med, D_high} x 4 controllers x 10 seeds
E4   demand sweep: uniform lambda 0.03-0.25 (9 levels) x 4 controllers x 10 seeds
E5   information: perfect sensor vs probe penetration {0.05, 0.1, 0.2, 0.5, 1.0}; pi = 0.2 without noise /
     without latency / without both; 4 controllers x 10 seeds
E6   communication graph: road, ring, complete, 2-hop (10 seeds) and Erdos-Renyi p in {0.1, 0.2, 0.4, 0.8}
     x 10 random graphs (3 seeds each); CoCycle, D_med; delay vs lambda_2 and vs mean degree
E7   communication failures: drop probability, latency, message period, 10-min outage; CoMP and CoCycle vs
     actuated (the independent fallback), D_med, 10 seeds
E8   consensus validation (matrix level): path example, epsilon bound, rate vs lambda_2, static vs dynamic
ABL  ablations (configs/experiment_ABL.yaml): downstream term, kappa, gamma, veto, offsets, K_c, cycle rule

Every result goes to results/<E*>/{runs.csv, summary.md, figures/}.
Run:  python experiments/core_experiments.py [--only E2,E4] [--workers 2]   (about 3 h on 2 cores)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trafficsync.analysis.stats import mean_ci, paired_wilcoxon, percent_improvement  # noqa: E402
from trafficsync.runner import RunSpec, load_controller_params, run_many, write_csv  # noqa: E402
from trafficsync.viz import plots  # noqa: E402

RES = ROOT / "results"
PERFECT = {"mode": "perfect_sensor"}
SWEEP_SEEDS = 10
HEADLINE_SEEDS = 30


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _spec(exp, net, dem, ctrl, seed, probe=None, comm=None, params=None, sim=None):
    return RunSpec(exp, net, dem, ctrl, seed, (probe or PERFECT).get("mode", "probe"), sim or {}, probe or PERFECT,
                   comm or {}, params or load_controller_params())


def _keep(rows, extra=()):
    keys = ["network", "demand", "info_mode", "controller", "seed", "mean_delay", "mean_travel_time", "mean_queue",
            "throughput_vph", "mean_stops", "spillback_events", "max_red_time", "jain_fairness", "gridlock",
            "msgs_per_s_agent", "bytes_per_s_agent", "stl_rho_safe", *extra]
    return [{k: r.get(k) for k in keys} for r in rows]


def _cell(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def _paired(rows, ref, ctrl, key="mean_delay", **kw):
    a = {r["seed"]: r[key] for r in _cell(rows, controller=ctrl, **kw)}
    b = {r["seed"]: r[key] for r in _cell(rows, controller=ref, **kw)}
    s = sorted(set(a) & set(b))
    A, B = [a[k] for k in s], [b[k] for k in s]
    imp = percent_improvement(A, B)
    return -imp["improvement_pct"], -imp["ci_hi"], -imp["ci_lo"], paired_wilcoxon(A, B)["p_value"]


# --------------------------------------------------------------------- E0
def e0(workers):
    from trafficsync.network import corridor
    from trafficsync.runner import run_episode
    from trafficsync.sim import Simulator

    out = RES / "E0_simulator_validity"
    out.mkdir(parents=True, exist_ok=True)
    L = ["# E0 - simulator validity", ""]
    rows = []
    for lanes in (1, 2, 3):
        net = corridor(1, lanes=lanes, link_length=2000)
        sim = Simulator(net, {"name": "sat", "rate": 0.0, "entry_rates": [{"entry": {"node": 0, "approach": "W"}, "rate": 4.0}]},
                        probe_cfg=PERFECT, seed=0, sim_cfg={"duration_s": 700, "warmup_s": 0, "signal": {"g_max": 10000}})
        W = [l for l in net.incoming_links(0) if net.links[l].approach == "W"][0]
        obs = sim.reset()
        cnt = {}
        while not sim.done:
            obs, _ = sim.step({0: 0 if sim.t < 400 else 1})
            cnt[sim.t] = sum(1 for v in sim.vehicles.values() if v.idx > 0 and v.route[0] == W)
        rate = (cnt[640] - cnt[440]) / 200
        rows.append({"lanes": lanes, "measured_veh_s": rate, "s_l": 0.5 * lanes, "error_pct": 100 * (rate - 0.5 * lanes) / (0.5 * lanes)})
    L += ["## Saturation flow (continuous green after a 400 s red, 200 s measured)", "", "| lanes | s_l [veh/s] | measured | error |", "|---|---|---|---|"]
    L += [f"| {r['lanes']} | {r['s_l']:.1f} | {r['measured_veh_s']:.3f} | {r['error_pct']:+.1f}% |" for r in rows]
    # conservation over every scenario
    viol, steps = 0, 0
    for net in ("corridor3", "grid2x2", "grid3x3"):
        for dem in ("D_low", "D_med", "D_high", "D_tidal", "D_ramp", "D_incident", "D_hotspot"):
            def cb(sim, info):
                nonlocal viol, steps
                st = sim.true_state()
                steps += 1
                viol += int(st["generated"] != st["n_in_network"] + st["completed"])
                viol += int(any(d["q"] < 0 or (sim.network.links[l].kind != "exit" and d["n"] > d["capacity"])
                                for l, d in st["links"].items()))
            run_episode(net, dem, "actuated", 0, probe_cfg=PERFECT, callback=cb)
    L += ["", "## Conservation and storage", "", f"generated = in network + completed, q >= 0, n <= C checked at "
          f"{steps:,} network-steps (3 networks x 7 scenarios x 3600 s, actuated): **{viol} violations**."]
    # Webster delay on an isolated intersection (fixed 60 s cycle, equal greens)
    wrows = []
    for lam in (0.04, 0.08, 0.12, 0.16):
        ds = [run_episode(corridor(1), lam, "fixed", s, probe_cfg=PERFECT,
                          controller_params={**load_controller_params(), "fixed": {"c_min": 60, "c_max": 60}})["mean_delay"]
              for s in range(5)]
        C, g = 60.0, (60.0 - 12) / 2
        lam_eff = lam
        lam_ap = lam_eff
        s = 0.5
        x = lam_ap * C / (s * g)
        u = g / C
        dw = C * (1 - u) ** 2 / (2 * (1 - u * x)) + x ** 2 / (2 * lam_ap * (1 - x)) - 0.65 * (C / lam_ap ** 2) ** (1 / 3) * x ** (2 + 5 * u)
        wrows.append({"lambda": lam, "x": x, "sim_delay": float(np.mean(ds)), "webster": dw})
    L += ["", "## Isolated intersection, 60 s cycle (24 s effective green per phase): simulated vs Webster delay", "",
          "Simulated delay counts every movement (through + turns) per vehicle; corridor1 has 4 approaches at lambda each.", "",
          "| lambda [veh/s] | x | simulated [s] | Webster [s] | difference |", "|---|---|---|---|---|"]
    L += [f"| {r['lambda']:.2f} | {r['x']:.2f} | {r['sim_delay']:.1f} | {r['webster']:.1f} | {100*(r['sim_delay']-r['webster'])/r['webster']:+.1f}% |" for r in wrows]
    (out / "summary.md").write_text("\n".join(L) + "\n" + e0_determinism())
    write_csv(out / "saturation.csv", rows)
    write_csv(out / "webster.csv", wrows)


# --------------------------------------------------------------------- E0 add-on: determinism
def e0_determinism() -> str:
    from trafficsync.runner import run_episode

    a = run_episode("grid3x3", "D_med", "comp", 7)
    b = run_episode("grid3x3", "D_med", "comp", 7)
    timing = ("compute_ms_per_decision", "wall_s")  # wall-clock measurements, not simulation output
    same = repr({k: v for k, v in a.items() if k not in timing}) == repr({k: v for k, v in b.items() if k not in timing})
    arrivals = {}
    for c in ("fixed", "actuated", "cocycle"):
        seen = []

        def cb(sim, info, seen=seen):
            if sim.t == sim.duration:
                seen.extend(sorted((v.t_gen, tuple(v.route)) for v in sim.vehicles.values()))
        run_episode("grid3x3", "D_med", c, 7, callback=cb)
        arrivals[c] = seen
    ident = arrivals["fixed"] == arrivals["actuated"] == arrivals["cocycle"] and len(arrivals["fixed"]) > 0
    return ("\n## Determinism\n\nSame seed and config run twice (CoMP, grid3x3, D_med, seed 7): all simulation metrics "
            f"{'identical' if same else 'DIFFERENT'}.  Fixed, actuated and CoCycle on seed 7 see "
            f"{'identical' if ident else 'DIFFERENT'} vehicle generation times and routes ({len(arrivals['fixed']):,} vehicles).\n")


# --------------------------------------------------------------------- E1
def e1(workers):
    sys.path.insert(0, str(ROOT / "experiments"))
    import run_experiment as rx
    from trafficsync.runner import load_experiment, run_experiment

    exp = load_experiment("E1")
    res = run_experiment("E1", workers=workers)
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in res["rows"]]
    (res["out_dir"] / "summary.md").write_text(rx.summary_md(exp, rows))
    rx.figures(exp, rows, res["out_dir"])


# --------------------------------------------------------------------- E2
E2_CTRL = ["fixed", "actuated", "maxpressure_local", "comp", "cocycle", "comp+cycle"]
E2_DEM = ["D_low", "D_med", "D_high", "D_tidal", "D_ramp"]


def e2(workers, seeds=HEADLINE_SEEDS):
    out = RES / "E2_main_comparison"
    specs = [_spec("E2", "grid3x3", d, c, s, probe=p) for p in (PERFECT, {"mode": "probe", "penetration": 0.2})
             for d in E2_DEM for c in E2_CTRL for s in range(seeds)]
    rows = _keep(run_many(specs, workers, "E2"))
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "runs.csv", rows)
    L = [f"# E2 - main comparison (grid3x3, seeds 0-{seeds - 1})", "",
         "Delay change vs fixed and vs actuated: paired over seeds, 95% paired-bootstrap CI, Wilcoxon signed-rank p; "
         "Holm-adjusted p over the 4 non-baseline controllers x 5 demands per information mode in `p_holm`.", ""]
    summ = []
    from trafficsync.analysis.stats import holm

    for mode in ("perfect_sensor", "probe"):
        block = []
        for d in E2_DEM:
            for c in E2_CTRL:
                rs = _cell(rows, demand=d, controller=c, info_mode=mode)
                ci = mean_ci([r["mean_delay"] for r in rs])
                vf = _paired(rows, "fixed", c, demand=d, info_mode=mode) if c != "fixed" else None
                va = _paired(rows, "actuated", c, demand=d, info_mode=mode) if c != "actuated" else None
                block.append({"info_mode": mode, "demand": d, "controller": c, "delay": ci["mean"], "ci": ci["half_width"],
                              "vs_fixed": vf[0] if vf else 0.0, "vs_act": va[0] if va else 0.0,
                              "vs_act_lo": va[1] if va else 0.0, "vs_act_hi": va[2] if va else 0.0, "p_act": va[3] if va else 1.0,
                              "queue": float(np.mean([r["mean_queue"] for r in rs])), "stops": float(np.mean([r["mean_stops"] for r in rs])),
                              "throughput": float(np.mean([r["throughput_vph"] for r in rs])),
                              "max_red": float(np.max([r["max_red_time"] for r in rs]))})
        test = [k for k, s in enumerate(block) if s["controller"] not in ("fixed", "actuated")]
        adj = holm([block[k]["p_act"] for k in test])
        for k, b in enumerate(block):
            b["p_holm"] = adj[test.index(k)] if k in test else 1.0
        summ += block
        L += [f"## information: {'local detectors' if mode == 'perfect_sensor' else 'probe vehicles, 20% penetration'}", "",
              "| demand | controller | delay [s] | vs fixed | vs actuated [95% CI] | p | p_holm | queue [veh] | stops/veh | throughput [veh/h] | max red [s] |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in block:
            c = s["controller"]
            L.append(f"| {s['demand']} | {c} | {s['delay']:.1f} ± {s['ci']:.1f} | {s['vs_fixed']:+.1f}% | "
                     + (f"{s['vs_act']:+.1f}% [{s['vs_act_lo']:+.1f}, {s['vs_act_hi']:+.1f}] | {s['p_act']:.2g} | {s['p_holm']:.2g}"
                        if c != "actuated" else "– | – | –")
                     + f" | {s['queue']:.2f} | {s['stops']:.2f} | {s['throughput']:.0f} | {s['max_red']:.0f} |")
        L.append("")
    (out / "summary.md").write_text("\n".join(L))
    write_csv(out / "summary.csv", summ)
    plt = plots._plt()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
    for ax, mode in zip(axes, ("perfect_sensor", "probe")):
        w = 0.8 / len(E2_CTRL)
        for k, c in enumerate(E2_CTRL):
            cell = {s["demand"]: s for s in summ if s["info_mode"] == mode and s["controller"] == c}
            ax.bar(np.arange(len(E2_DEM)) + (k - (len(E2_CTRL) - 1) / 2) * w, [cell[d]["delay"] for d in E2_DEM], w,
                   yerr=[cell[d]["ci"] for d in E2_DEM], capsize=2, color=plots.color_for(c), label=plots.label_for(c),
                   error_kw={"lw": 0.8})
        ax.set_xticks(range(len(E2_DEM)))
        ax.set_xticklabels(E2_DEM)
        ax.set_ylabel("mean delay [s/veh]")
        ax.set_title("local detectors" if mode == "perfect_sensor" else "probe vehicles (π = 0.2)", loc="left", fontsize=9)
        plots.style_axes(ax, "y")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"E2 — grid 3×3, {seeds} seeds, mean with 95% CI", x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout()
    plots.save(fig, out / "figures" / "E2_delay.png")


# --------------------------------------------------------------------- E3 / E4 / E5
def e3(workers, seeds=SWEEP_SEEDS):
    out = RES / "E3_network_size"
    nets = ["corridor3", "grid2x2", "grid3x3", "grid4x4", "grid5x5"]
    ctrls = ["fixed", "actuated", "comp", "cocycle"]
    rows = _keep(run_many([_spec("E3", n, d, c, s) for n in nets for d in ("D_med", "D_high") for c in ctrls for s in range(seeds)], workers, "E3"))
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "runs.csv", rows)
    L = [f"# E3 - network size (local detectors, seeds 0-{seeds - 1})", "",
         "| demand | network | N | " + " | ".join(ctrls) + " | CoCycle vs actuated | CoMP vs actuated |",
         "|---|---|---|" + "---|" * (len(ctrls) + 2)]
    Ns = {"corridor3": 3, "grid2x2": 4, "grid3x3": 9, "grid4x4": 16, "grid5x5": 25}
    for d in ("D_med", "D_high"):
        for n in nets:
            cells = [mean_ci([r["mean_delay"] for r in _cell(rows, network=n, demand=d, controller=c)]) for c in ctrls]
            g1 = _paired(rows, "actuated", "cocycle", network=n, demand=d)
            g2 = _paired(rows, "actuated", "comp", network=n, demand=d)
            L.append(f"| {d} | {n} | {Ns[n]} | " + " | ".join(f"{v['mean']:.1f} ± {v['half_width']:.1f}" for v in cells)
                     + f" | {g1[0]:+.1f}% [{g1[1]:+.1f}, {g1[2]:+.1f}] | {g2[0]:+.1f}% [{g2[1]:+.1f}, {g2[2]:+.1f}] |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    series = {}
    for c in ("cocycle", "comp"):
        for d in ("D_med", "D_high"):
            g = [_paired(rows, "actuated", c, network=n, demand=d) for n in nets]
            series[f"{plots.label_for(c)} vs actuated, {d}"] = ([x[0] for x in g], [x[1] for x in g], [x[2] for x in g])
    plots.gain_vs_size([Ns[n] for n in nets], series, f"E3 — gain of coordination vs network size ({seeds} seeds, paired 95% CI)",
                       out / "figures" / "E3_gain_vs_size.png")


E4_LAMBDAS = [0.03, 0.05, 0.08, 0.11, 0.14, 0.17, 0.20, 0.23, 0.25]


def e4(workers, seeds=SWEEP_SEEDS, resummarise=False):
    out = RES / "E4_demand_sweep"
    lams = E4_LAMBDAS
    ctrls = ["fixed", "actuated", "comp", "cocycle"]
    if resummarise and (out / "runs.csv").exists():
        import csv

        rows = [{k: (float(v) if k not in ("network", "demand", "info_mode", "controller", "gridlock") else v)
                 for k, v in r.items() if v != ""} for r in csv.DictReader(open(out / "runs.csv"))]
        for r in rows:
            r["seed"] = int(r["seed"])
            r["demand"] = float(r["demand"])
    else:
        rows = _keep(run_many([_spec("E4", "grid3x3", l, c, s) for l in lams for c in ctrls for s in range(seeds)], workers, "E4"))
        out.mkdir(parents=True, exist_ok=True)
        write_csv(out / "runs.csv", rows)
    L = [f"# E4 - demand sweep (grid3x3, local detectors, seeds 0-{seeds - 1})", "",
         "| lambda [veh/s] | " + " | ".join(ctrls) + " | CoCycle vs actuated | CoMP vs actuated | throughput (actuated) [veh/h] | gridlock runs (actuated) |",
         "|---|" + "---|" * (len(ctrls) + 4)]
    series = {}
    for c in ctrls:
        cis = [mean_ci([r["mean_delay"] for r in _cell(rows, demand=float(l), controller=c)]) for l in lams]
        series[c] = ([x["mean"] for x in cis], [x["half_width"] for x in cis])
    for k, l in enumerate(lams):
        g = _paired(rows, "actuated", "cocycle", demand=float(l))
        g2 = _paired(rows, "actuated", "comp", demand=float(l))
        act = _cell(rows, demand=float(l), controller="actuated")
        thr = np.mean([r["throughput_vph"] for r in act])
        gl = sum(1 for r in act if str(r.get("gridlock")) in ("True", "1", "true"))
        L.append(f"| {l:.2f} | " + " | ".join(f"{series[c][0][k]:.1f}" for c in ctrls)
                 + f" | {g[0]:+.1f}% [{g[1]:+.1f}, {g[2]:+.1f}] | {g2[0]:+.1f}% [{g2[1]:+.1f}, {g2[2]:+.1f}] | {thr:.0f} | {gl} |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    plots.lines_ci(lams, series, "entry rate λ at every entry [veh/s]", "mean delay [s/veh]",
                   f"E4 — delay vs demand, grid 3×3 ({seeds} seeds, 95% CI)", out / "figures" / "E4_delay_vs_demand.png")


def e5(workers, seeds=SWEEP_SEEDS):
    out = RES / "E5_information"
    pens = [0.05, 0.1, 0.2, 0.5, 1.0]
    ctrls = ["fixed", "actuated", "comp", "cocycle"]
    cases = [(f"probes π = {p}", {"mode": "probe", "penetration": p}) for p in pens]
    cases += [("probes π = 0.2, no noise", {"mode": "probe", "penetration": 0.2, "noise_std": 0.0, "pos_noise_std": 0.0}),
              ("probes π = 0.2, no latency", {"mode": "probe", "penetration": 0.2, "delay_steps": 0}),
              ("probes π = 0.2, no noise, no latency", {"mode": "probe", "penetration": 0.2, "noise_std": 0.0,
                                                         "pos_noise_std": 0.0, "delay_steps": 0}),
              ("local detectors (perfect)", PERFECT)]
    specs, tags = [], []
    for name, pr in cases:
        for c in ctrls:
            for s in range(seeds):
                specs.append(_spec("E5", "grid3x3", "D_med", c, s, probe=pr))
                tags.append(name)
    rows = _keep(run_many(specs, workers, "E5"))
    for r, t in zip(rows, tags):
        r["information"] = t
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "runs.csv", rows)
    L = [f"# E5 - information quality (grid3x3, D_med, seeds 0-{seeds - 1})", "",
         "Default probe model: report period 5 s, speed noise 1 m/s, position noise 5 m, delay 2 s, EMA 0.3.", "",
         "| information | " + " | ".join(ctrls) + " |", "|---|" + "---|" * len(ctrls)]
    series = {c: ([], []) for c in ctrls}
    for name, _ in cases:
        vals = []
        for c in ctrls:
            ci = mean_ci([r["mean_delay"] for r in rows if r["information"] == name and r["controller"] == c])
            vals.append(f"{ci['mean']:.1f} ± {ci['half_width']:.1f}")
            if name.startswith("probes π =") and "," not in name:
                series[c][0].append(ci["mean"])
                series[c][1].append(ci["half_width"])
        L.append(f"| {name} | " + " | ".join(vals) + " |")
    perf = {c: float(np.mean([r["mean_delay"] for r in rows if r["information"] == cases[-1][0] and r["controller"] == c]))
            for c in ("actuated", "cocycle")}
    (out / "summary.md").write_text("\n".join(L) + "\n")
    plots.lines_ci(pens, series, "probe penetration π [–]", "mean delay [s/veh]",
                   f"E5 — information quality, D_med ({seeds} seeds, 95% CI; dashed = local detectors)",
                   out / "figures" / "E5_delay_vs_penetration.png", ref=perf, logx=True)


# --------------------------------------------------------------------- E6 / E7
def e6(workers, seeds=SWEEP_SEEDS, er_graphs=10, er_seeds=3):
    from scipy.stats import spearmanr

    from trafficsync.network import comm_adjacency, load_network
    from trafficsync.network.spectral import spectral_summary

    out = RES / "E6_comm_graph"
    net = load_network("grid3x3")
    graphs = {"road": ({"topology": "road"}, seeds), "ring": ({"topology": "ring"}, seeds),
              "complete": ({"topology": "complete"}, seeds), "2-hop": ({"topology": "k_hop", "k": 2}, seeds)}
    for p in (0.1, 0.2, 0.4, 0.8):
        for g in range(er_graphs):
            graphs[f"ER p={p} #{g}"] = ({"topology": "erdos_renyi", "p": p, "seed": 1000 * g + int(100 * p)}, er_seeds)
    specs, names, meta = [], [], {}
    road = net.road_A > 0
    for name, (topo, ns) in graphs.items():
        A = comm_adjacency(net, topo)
        sp = spectral_summary(A)
        meta[name] = {"lambda2": sp["lambda2"], "mean_degree": sp["mean_degree"], "connected": sp["connected"],
                      "road_coverage": float((A[road] > 0).mean())}
        for s in range(ns):
            specs.append(RunSpec("E6", "grid3x3", "D_med", "cocycle", s, "perfect_sensor", {}, PERFECT, {"topology": topo},
                                 load_controller_params()))
            names.append(name)
    rows = _keep(run_many(specs, workers, "E6"))
    for r, n in zip(rows, names):
        r["graph"] = n
        r.update(meta[n])
    act_rows = _keep(run_many([_spec("E6", "grid3x3", "D_med", "actuated", s) for s in range(seeds)], workers, "E6-ref"))
    act = [r["mean_delay"] for r in act_rows]
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "runs.csv", rows + [{**r, "graph": "none (actuated)"} for r in act_rows])
    per = {n: mean_ci([r["mean_delay"] for r in rows if r["graph"] == n]) for n in graphs}
    L = ["# E6 - communication graph (CoCycle, grid3x3, D_med, local detectors)", "",
         f"Deterministic graphs: seeds 0-{seeds - 1}; Erdos-Renyi: {er_graphs} random graphs per p, seeds 0-{er_seeds - 1} each. "
         f"Actuated (no communication) on seeds 0-{seeds - 1}: {np.mean(act):.1f} s.", "",
         "| graph | lambda2 | mean degree | connected | road-link coverage | CoCycle delay [s] |", "|---|---|---|---|---|---|"]
    for n in graphs:
        m = meta[n]
        L.append(f"| {n} | {m['lambda2']:.2f} | {m['mean_degree']:.2f} | {m['connected']} | {m['road_coverage']:.2f} | "
                 f"{per[n]['mean']:.1f} ± {per[n]['half_width'] if np.isfinite(per[n]['half_width']) else 0:.1f} |")
    L += ["", "| ER p | graphs connected | mean lambda2 | mean degree | CoCycle delay (mean over graphs) [s] |", "|---|---|---|---|---|"]
    for p in (0.1, 0.2, 0.4, 0.8):
        ks = [n for n in graphs if n.startswith(f"ER p={p} ")]
        L.append(f"| {p} | {sum(meta[k]['connected'] for k in ks)}/{len(ks)} | {np.mean([meta[k]['lambda2'] for k in ks]):.2f} | "
                 f"{np.mean([meta[k]['mean_degree'] for k in ks]):.2f} | {np.mean([per[k]['mean'] for k in ks]):.1f} |")
    x2 = [meta[n]["lambda2"] for n in graphs]
    deg = [meta[n]["mean_degree"] for n in graphs]
    cov = [meta[n]["road_coverage"] for n in graphs]
    y = [per[n]["mean"] for n in graphs]
    L += ["", f"Spearman over the {len(graphs)} graphs: (lambda2, delay) = {spearmanr(x2, y).correlation:+.2f}; "
          f"(mean degree, delay) = {spearmanr(deg, y).correlation:+.2f}; (road-link coverage, delay) = {spearmanr(cov, y).correlation:+.2f}."]
    (out / "summary.md").write_text("\n".join(L) + "\n")
    plots.perf_vs_lambda2(x2, deg, y, [per[n]["half_width"] if np.isfinite(per[n]["half_width"]) else 0 for n in graphs],
                          float(np.mean(act)), f"E6 — CoCycle delay vs communication graph (grid 3×3, D_med; {len(graphs)} graphs)",
                          out / "figures" / "E6_delay_vs_lambda2.png")


def e7(workers, seeds=SWEEP_SEEDS):
    out = RES / "E7_comm_failures"
    levels = {"drop": ("drop_prob", [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]), "latency": ("latency_steps", [0, 1, 2, 5, 10]),
              "period": ("period_s", [1, 5, 10, 30])}
    ctrls = ["comp", "cocycle"]
    specs, tags = [], []
    for c in ctrls:
        for f, (key, vals) in levels.items():
            for v in vals:
                for s in range(seeds):
                    specs.append(RunSpec("E7", "grid3x3", "D_med", c, s, "perfect_sensor", {}, PERFECT, {key: v}, load_controller_params()))
                    tags.append((f, v))
        for s in range(seeds):
            specs.append(RunSpec("E7", "grid3x3", "D_med", c, s, "perfect_sensor", {}, PERFECT,
                                 {"outage_start_s": 1800, "outage_end_s": 2400}, load_controller_params()))
            tags.append(("outage", "10 min (1800-2400 s)"))
    for s in range(seeds):
        specs.append(_spec("E7", "grid3x3", "D_med", "actuated", s))
        tags.append(("ref", 0))
    rows = _keep(run_many(specs, workers, "E7"))
    for r, t in zip(rows, tags):
        r["factor"], r["level"] = t
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "runs.csv", rows)
    act_rows = [r for r in rows if r["factor"] == "ref"]
    act = float(np.mean([r["mean_delay"] for r in act_rows]))
    L = [f"# E7 - communication failures (grid3x3, D_med, local detectors, seeds 0-{seeds - 1})", "",
         f"Actuated (no communication, the independent fallback): {act:.1f} s.  Default: period 5 s, latency 1 step, no drops.", "",
         "| factor | level | CoMP delay [s] | CoMP vs actuated | CoCycle delay [s] | CoCycle vs actuated |", "|---|---|---|---|---|---|"]
    series = {f: {c: ([], []) for c in ctrls} for f in levels}
    for f, lvls in [(f, v[1]) for f, v in levels.items()] + [("outage", ["10 min (1800-2400 s)"])]:
        for lv in lvls:
            cells = []
            for c in ctrls:
                rs = [r for r in rows if r["factor"] == f and r["level"] == lv and r["controller"] == c]
                ci = mean_ci([r["mean_delay"] for r in rs])
                a = {r["seed"]: r["mean_delay"] for r in act_rows}
                imp = percent_improvement([r["mean_delay"] for r in rs], [a[r["seed"]] for r in rs])
                cells.append(f"{ci['mean']:.1f} ± {ci['half_width']:.1f} | {-imp['improvement_pct']:+.1f}% [{-imp['ci_hi']:+.1f}, {-imp['ci_lo']:+.1f}]")
                if f in series:
                    series[f][c][0].append(ci["mean"])
                    series[f][c][1].append(ci["half_width"])
            L.append(f"| {f} | {lv} | " + " | ".join(cells) + " |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    lab = {"drop": "message drop probability [–]", "latency": "message latency [steps of 1 s]", "period": "message period T_c [s]"}
    fn = {"drop": "E7_delay_vs_drop.png", "latency": "E7_delay_vs_latency.png", "period": "E7_delay_vs_period.png"}
    for f in levels:
        plots.lines_ci(levels[f][1], series[f], lab[f], "mean delay [s/veh]",
                       f"E7 — {f} sweep, grid 3×3, D_med ({seeds} seeds, 95% CI)", out / "figures" / fn[f], ref={"actuated": act})


# --------------------------------------------------------------------- ablations
def abl(workers):
    sys.path.insert(0, str(ROOT / "experiments"))
    import run_experiment as rx
    from trafficsync.runner import load_experiment, run_experiment

    exp = load_experiment("ABL")
    res = run_experiment("ABL", workers=workers)
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in res["rows"]]
    (res["out_dir"] / "summary.md").write_text(rx.summary_md(exp, rows))
    rx.figures(exp, rows, res["out_dir"])


# --------------------------------------------------------------------- E8
def e8(workers):
    from scipy.stats import spearmanr

    from trafficsync.control.consensus import (dynamic_consensus, epsilon_bound, laplacian_weights, metropolis_weights,
                                               static_consensus)
    from trafficsync.network import erdos_renyi, grid, ring
    from trafficsync.network.spectral import is_connected

    out = RES / "E8_consensus"
    out.mkdir(parents=True, exist_ok=True)
    A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)
    H = static_consensus([2, 5, 8], laplacian_weights(A), 100)
    L = ["# E8 - consensus validation", "", f"Path graph, x(0) = [2, 5, 8]: x(100) = {np.round(H[-1], 6).tolist()}, "
         f"max |1'x - 15| over iterations = {np.abs(H.sum(1) - 15).max():.1e}.", ""]
    R = ring(8)
    b = epsilon_bound(R)
    res = []
    for f in (0.5, 0.9, 0.99, 1.01, 1.1):
        x = static_consensus(np.arange(8.0), laplacian_weights(R, f * b), 400)[-1]
        res.append((f, float(np.ptp(x))))
    L += ["Ring of 8, eps = f x 2/lambda_max: spread after 400 iterations " + ", ".join(f"f={f}: {v:.1e}" for f, v in res), ""]
    lam2, its, curves = [], [], {}
    for p in (0.15, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0):
        for g in range(4):
            Aer = erdos_renyi(20, p, 100 * g + int(100 * p))
            if not is_connected(Aer):
                continue
            Lp = np.diag(Aer.sum(1)) - Aer
            l2 = np.sort(np.linalg.eigvalsh(Lp))[1]
            x0 = np.random.default_rng(g).uniform(0, 1, 20)
            Hh = static_consensus(x0, laplacian_weights(Aer), 2000)
            err = np.linalg.norm(Hh - x0.mean(), axis=1)
            k = int(np.argmax(err < 1e-6 * err[0])) if (err < 1e-6 * err[0]).any() else 2000
            lam2.append(l2)
            its.append(k)
            if g == 0:
                curves[f"G(20, {p})  λ2 = {l2:.2f}"] = err
    L += [f"Erdos-Renyi G(20, p), {len(lam2)} connected graphs: Spearman(lambda2, iterations to 1e-6) = {spearmanr(lam2, its).correlation:+.2f}.", ""]
    rng = np.random.default_rng(0)
    U = np.cumsum(rng.normal(0, 0.01, (400, 9)), 0) + rng.uniform(0, 1, 9)
    W = metropolis_weights(grid(3, 3).A)
    Xd = dynamic_consensus(U, W, K=5)
    Xs = np.array([static_consensus(U[k], W, 5)[-1] for k in range(400)])
    ed, es = np.abs(Xd - U.mean(1, keepdims=True)).mean(), np.abs(Xs - U.mean(1, keepdims=True)).mean()
    L += [f"Tracking a drifting average on grid 3x3 (K = 5 rounds per step): dynamic consensus error {ed:.4f}, restarted static {es:.4f}."]
    (out / "summary.md").write_text("\n".join(L) + "\n")
    plots.consensus_error(curves, "E8 — consensus error vs iteration, Erdős–Rényi G(20, p), ε = 1/(d_max+1)",
                          out / "figures" / "E8_consensus_error.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="E0,E8,E1,E2,E3,E4,E5,E6,E7,ABL")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resummarise-e4", action="store_true")
    args = ap.parse_args()
    if args.resummarise_e4:
        e4(args.workers, resummarise=True)
        return
    fns = {"E0": e0, "E1": e1, "ABL": abl, "E2": e2, "E3": e3, "E4": e4, "E5": e5, "E6": e6, "E7": e7, "E8": e8}
    for k in args.only.split(","):
        print(f"== {k}", flush=True)
        fns[k](args.workers)


if __name__ == "__main__":
    main()
