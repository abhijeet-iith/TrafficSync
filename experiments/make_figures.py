"""Write the summary figure list F01-F11 into ``results/figures/``.

Every experiment script draws its own figures with :mod:`trafficsync.viz.plots` (300 dpi, shared
colours and labels, units on axes, 95% CI bands) into ``results/<E*>/figures/``.  This script

1. writes the summary figure list F01-F11 under fixed names ``results/figures/F01_*.png`` ...,
2. draws the two figures that need no experiment run: F10 (offset time-space diagram, fixed vs
   actuated vs CoCycle on the middle eastbound arterial of grid 3x3, D_med, seed 0, 1200-1800 s)
   and the 3-node consensus example used in the slides.

    python experiments/make_figures.py            # ~1 min (F10 runs three episodes)
    python experiments/make_figures.py --no-f10
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trafficsync.viz import plots  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "results" / "figures"

# figure list -> source figure
FLIST = {
    "F01_bar_ci_main_comparison": "E2_main_comparison/figures/E2_delay.png",
    "F02_delay_vs_demand": "E4_demand_sweep/figures/E4_delay_vs_demand.png",
    "F03_gain_vs_network_size": "E3_network_size/figures/E3_gain_vs_size.png",
    "F04_performance_vs_lambda2": "E6_comm_graph/figures/E6_delay_vs_lambda2.png",
    "F05_performance_vs_drop": "E7_comm_failures/figures/E7_delay_vs_drop.png",
    "F06_consensus_error": "E8_consensus/figures/E8_consensus_error.png",
    "F07_optimizer_convergence": "E10_distributed_optimisation/figures/F07_optimizer_convergence.png",
    "F08_wmsr_vs_average": "E9_resilience/figures/F08_wmsr_vs_average.png",
    "F09_incident_queue": "E14_incident_tidal/figures/E14_incident_queue.png",
    "F10_time_space": None,  # drawn here
    "F11_epidemic_threshold": "E11_epidemic/figures/F11_epidemic_threshold_max05.png",
}


def f10(path: Path) -> Path:
    from trafficsync.viz.recording import experiment_settings, record_time_space

    kw = experiment_settings("E2")
    kw["probe_cfg"] = {**kw["probe_cfg"], "mode": "perfect_sensor"}
    datas = [record_time_space("grid3x3", "D_med", c, 0, t0=1200, t1=1500, **kw) for c in ("fixed", "actuated", "cocycle")]
    for d in datas:
        d.save(RES / "demo" / f"time_space_{d.meta['controller']}.npz")
    return plots.time_space(datas, path, "F10 — time–space diagram, middle eastbound arterial of grid 3×3, D_med, seed 0 "
                                        "(green/yellow/red bars = signal at each stop line)")


def consensus_3node(path: Path) -> Path:
    from trafficsync.control.consensus import laplacian_weights, static_consensus

    A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)
    H = static_consensus([2.0, 5.0, 8.0], laplacian_weights(A), 30)
    plt = plots._plt()
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    for i, c in enumerate(("#0969da", "#8250df", "#1a7f37")):
        ax.plot(H[:, i], "o-", ms=2.5, lw=1.3, color=c, label=f"$x_{i + 1}$")
    ax.axhline(5.0, color=plots.MUTED, ls=":", lw=1, label="average 5")
    ax.set_xlabel("iteration k [–]")
    ax.set_ylabel("state [–]")
    ax.set_title("3-node path, x(0) = [2, 5, 8]", loc="left", fontsize=9)
    plots.style_axes(ax)
    ax.legend(fontsize=7, ncol=4)
    return plots.save(fig, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-f10", action="store_true")
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    missing = []
    for name, src in FLIST.items():
        if src is None:
            if not a.no_f10:
                (RES / "demo").mkdir(parents=True, exist_ok=True)
                f10(FIG / f"{name}.png")
            continue
        if (RES / src).exists():
            shutil.copy(RES / src, FIG / f"{name}.png")
        else:
            missing.append(f"{name} <- results/{src}")
    consensus_3node(FIG / "consensus_3node.png")
    print(f"wrote the F01-F11 figures to {FIG.relative_to(ROOT)}")
    if missing:
        print("missing (run the experiment first):\n  " + "\n  ".join(missing))


if __name__ == "__main__":
    main()
