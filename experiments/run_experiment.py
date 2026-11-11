"""Run one experiment from a YAML config and save it to ``results/<exp_id>/``.

Every (network, demand, information mode, controller, seed) combination runs the standard
main loop (:func:`trafficsync.runner.run_episode`).  Outputs:

* ``runs.csv``      one row per episode (every scalar metric);
* ``summary.csv``   mean, 95% CI per cell and metric;
* ``summary.md``    delay table with paired changes vs the reference controller and vs actuated
                    (paired bootstrap 95% CI, Wilcoxon p);
* ``figures/F01_<exp>_delay.png``  bar chart with CI per scenario and controller;
* ``meta.json``     versions, wall time.

    python experiments/run_experiment.py configs/experiment_E1.yaml --workers 2
    python experiments/run_experiment.py E2 --seeds 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trafficsync.analysis.stats import mean_ci, paired_wilcoxon, percent_improvement  # noqa: E402
from trafficsync.runner import load_experiment, run_experiment  # noqa: E402


def paired(rows, ctrl, ref, key="mean_delay"):
    a = {r["seed"]: r[key] for r in rows if r["controller"] == ctrl}
    b = {r["seed"]: r[key] for r in rows if r["controller"] == ref}
    s = sorted(set(a) & set(b))
    if not s or ctrl == ref:
        return None
    A, B = [a[k] for k in s], [b[k] for k in s]
    imp = percent_improvement(A, B)
    return -imp["improvement_pct"], -imp["ci_hi"], -imp["ci_lo"], paired_wilcoxon(A, B)["p_value"]


def summary_md(exp: dict, rows: list[dict]) -> str:
    ref = exp["reference"]
    second = "actuated" if ("actuated" in exp["controllers"] and ref != "actuated") else None
    fmt = lambda v: f"{v[0]:+.1f}% [{v[1]:+.1f}, {v[2]:+.1f}]" if v else "–"  # noqa: E731
    L = [f"# {exp['id']} — {exp['description']}", "",
         f"{len(exp['seeds'])} seeds ({exp['seeds'][0]}–{exp['seeds'][-1]}); delay change vs `{ref}`"
         + (" and vs `actuated`" if second else "") + ", paired over seeds [95% paired-bootstrap CI], Wilcoxon p.", ""]
    for net in exp["networks"]:
        for mode in exp["info_modes"]:
            L += [f"## {net}, {'local detectors' if mode == 'perfect_sensor' else 'probe vehicles'}", "",
                  "| demand | controller | delay [s] | queue [veh] | throughput [veh/h] | vs " + ref + f" | p (vs {ref})"
                  + (" | vs actuated | p (vs actuated)" if second else "") + " |",
                  "|---|---|---|---|---|---|---|" + ("---|---|" if second else "")]
            for d in exp["demands"]:
                cell = [r for r in rows if r["network"] == net and r["info_mode"] == mode and str(r["demand"]) == str(d)]
                for c in exp["controllers"]:
                    rs = [r for r in cell if r["controller"] == c]
                    ci = mean_ci([r["mean_delay"] for r in rs])
                    vr = paired(cell, c, ref)
                    line = (f"| {d} | {c} | {ci['mean']:.1f} ± {ci['half_width']:.1f} | {np.mean([r['mean_queue'] for r in rs]):.2f} | "
                            f"{np.mean([r['throughput_vph'] for r in rs]):.0f} | {fmt(vr)} | {f'{vr[3]:.2g}' if vr else '–'} |")
                    if second:
                        va = paired(cell, c, second)
                        line += f" {fmt(va)} | {f'{va[3]:.2g}' if va else '–'} |"
                    L.append(line)
            L.append("")
    return "\n".join(L)


def figures(exp: dict, rows: list[dict], out: Path) -> None:
    from trafficsync.viz import plots

    for net in exp["networks"]:
        for mode in exp["info_modes"]:
            mean, ci = {}, {}
            for d in exp["demands"]:
                for c in exp["controllers"]:
                    m = mean_ci([r["mean_delay"] for r in rows if r["network"] == net and r["info_mode"] == mode
                                 and str(r["demand"]) == str(d) and r["controller"] == c])
                    mean[(str(d), c)], ci[(str(d), c)] = m["mean"], m["half_width"]
            tag = f"{net}_{'detectors' if mode == 'perfect_sensor' else 'probes'}"
            plots.bar_ci([str(d) for d in exp["demands"]], exp["controllers"], mean, ci, "mean delay [s/veh]",
                         f"{exp['id']} — {net}, {'local detectors' if mode == 'perfect_sensor' else 'probe vehicles'} "
                         f"({len(exp['seeds'])} seeds, 95% CI)", out / "figures" / f"F01_{exp['id']}_{tag}_delay.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="path or id (E1 -> configs/experiment_E1.yaml)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=None, help="override the number of seeds (0..n-1)")
    ap.add_argument("--resummarise", action="store_true", help="rebuild summary.md and figures from runs.csv")
    a = ap.parse_args()
    exp = load_experiment(a.config)
    if a.resummarise:
        import csv

        out = ROOT / "results" / exp["id"]
        rows = []
        for r in csv.DictReader(open(out / "runs.csv")):
            for k, v in list(r.items()):
                try:
                    r[k] = float(v) if k not in ("network", "demand", "info_mode", "controller") else v
                except ValueError:
                    pass
            r["seed"] = int(r["seed"])
            rows.append(r)
        (out / "summary.md").write_text(summary_md(exp, rows))
        figures(exp, rows, out)
        print((out / "summary.md").read_text())
        return
    cfg = a.config
    if a.seeds is not None:
        from trafficsync.config import load_yaml, resolve_config

        raw = load_yaml(resolve_config(a.config, prefix="experiment_"))
        raw["experiment"]["seeds"] = {"start": 0, "count": a.seeds}
        cfg, exp = raw, load_experiment(raw)
    res = run_experiment(cfg, workers=a.workers)
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in res["rows"]]
    out = res["out_dir"]
    (out / "summary.md").write_text(summary_md(exp, rows))
    figures(exp, rows, out)
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
