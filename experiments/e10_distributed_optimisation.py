"""E10 - distributed optimisation of the common cycle length.

Problem instance: for every network and demand scenario, each agent's local Webster cycle
``C_i* = (1.5 L_tot + 5) / (1 - Y_i)`` (unclipped, L_tot = 12 s) with ``Y_i`` from the
fixed-time plans (design flows, never run-time data), and load weight ``w_i = Y_i``.
Ground truth: ``z* = sum w_i C_i* / sum w_i``.

Outputs in ``results/E10_distributed_optimisation/``:
* ``instances.csv``   - C_i*, w_i, z* per (network, demand);
* ``solvers.csv``     - final error, iterations to 1e-3 / 1e-6, messages per solver and instance;
* ``rate_vs_lambda2.csv`` - iterations to 1e-6 vs lambda_2(L) on Erdos-Renyi G(N, p) graphs;
* ``local_z_rounds.csv``  - error of the course "ADMM hack" vs consensus rounds;
* ``figures/`` - convergence (F07), rate vs lambda_2, ADMM residuals, local-z rounds;
* ``summary.md``.

Run:  python experiments/e10_distributed_optimisation.py   (~1 min)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trafficsync.control.admm import (CycleAgreementProblem, admm_consensus, admm_decentralized,  # noqa: E402
                                      admm_neighbour_z, iterations_to)
from trafficsync.control.dgd import dgd, dgd_bias_bound, gradient_tracking, metropolis_weights, mixing_rate  # noqa: E402

L_TOT = 12.0          # 2 phases x (startup 2 + yellow 3 + all-red 1)
NETWORKS = ["corridor3", "grid2x2", "grid3x3"]
INSTANCES = [("D_med", "design"), ("D_tidal", "design"), ("D_hotspot", "design"),
             ("D_med", "measured"), ("D_high", "measured"), ("D_hotspot", "measured")]
MAIN = ("grid3x3", "D_hotspot", "measured")    # instance used for the convergence figures
RHO = 1.0             # default
ALPHA = 0.2           # DGD / GT step (both inside the stable range for w_i <= 1)


def _lambda2(A: np.ndarray) -> float:
    L = np.diag(A.sum(1)) - A
    return float(np.sort(np.linalg.eigvalsh(L))[1])


def instance(network: str, demand: str, source: str = "design") -> tuple[CycleAgreementProblem, np.ndarray, dict]:
    """Local Webster cycles ``C_i*`` and weights ``w_i = Y_i`` of every intersection.

    ``source="design"``: Y_i from the fixed-time plan (scenario average entry rates propagated
    with the turning ratios).  With *uniform* entry rates every link carries the same design flow
    (uniform flow is a fixed point of the turning ratios), so all C_i* coincide and the problem
    is trivial; it is kept in the table for completeness.
    ``source="measured"``: Y_i from the link flows actually realised in one simulated hour
    (seed 0, actuated control, t >= 600 s) - heterogeneous, as an agent would see them.
    Y is capped at 0.9 (as in CoCycle).
    """
    from trafficsync.control import make_agent_views
    from trafficsync.control.fixed_time import webster_plan
    from trafficsync.runner import build_network, run_episode

    net = build_network(network)
    A = (np.asarray(net.A) > 0).astype(float)
    if source == "design":
        m = run_episode(network, demand, "fixed", 0, sim_cfg={"duration_s": 60, "warmup_s": 0})
        Y = np.array([m["fixed_plans"][i]["Y"] for i in range(net.N)], float)
    else:
        m = run_episode(network, demand, "actuated", 0, probe_cfg={"mode": "perfect_sensor"}, return_sim=True)
        sim = m["_sim"]
        w0, T = sim.cfg.warmup_s, sim.duration
        counts = {l: 0 for l in net.links}
        for v in sim.vehicles.values():
            if v.t_gen >= w0:
                for l in v.route[: v.idx + 1]:
                    counts[l] += 1
        flows = {l: c / (T - w0) for l, c in counts.items()}
        views = make_agent_views(net, sim.phases, sim.cfg.signal, sim.cfg.turning)
        Y = np.array([webster_plan(views[i], flows).Y for i in range(net.N)], float)
    Y = np.minimum(Y, 0.9)
    c_star = (1.5 * L_TOT + 5.0) / (1.0 - Y)
    w = np.maximum(Y, 1e-3)
    return CycleAgreementProblem(c_star, w), A, {"Y": Y.tolist(), "source": source}


def run_solvers(prob: CycleAgreementProblem, A: np.ndarray, iters: int = 2000) -> dict[str, object]:
    W = metropolis_weights(A)
    return {
        "ADMM (central z)": admm_consensus(prob, rho=RHO, tol=1e-3, run_to=iters),
        "ADMM (decentralised)": admm_decentralized(prob, A, rho=RHO, iters=iters),
        "Gradient tracking": gradient_tracking(prob, W, alpha=ALPHA, iters=iters),
        f"DGD (alpha={ALPHA})": dgd(prob, W, alpha=ALPHA, iters=iters),
        "DGD (alpha=0.02)": dgd(prob, W, alpha=0.02, iters=iters),
        "DGD (alpha=0.5/(k+1)^0.6)": dgd(prob, W, alpha=lambda k: 0.5 / (k + 1) ** 0.6, iters=iters),
        "ADMM (local z, 1 round)": admm_neighbour_z(prob, A, rho=RHO, iters=iters, rounds=1),
    }


def rate_vs_lambda2(prob_seed: int = 7, N: int = 16, reps: int = 4) -> list[dict]:
    import networkx as nx

    rng = np.random.default_rng(prob_seed)
    prob = CycleAgreementProblem(rng.uniform(40, 110, N), rng.uniform(0.1, 0.9, N))
    rows = []
    for p in (0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0):
        for rep in range(reps):
            G = nx.erdos_renyi_graph(N, p, seed=1000 * rep + int(100 * p))
            if not nx.is_connected(G):
                continue
            A = nx.to_numpy_array(G)
            W = metropolis_weights(A)
            gt = {a: iterations_to(gradient_tracking(prob, W, a, 4000).err, 1e-6) or np.nan for a in (0.1, 0.2, 0.4)}
            ad = {r: iterations_to(admm_decentralized(prob, A, r, 4000).err, 1e-6) or np.nan
                  for r in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)}
            dg = dgd(prob, W, ALPHA, 3000)
            rows.append({"p": p, "rep": rep, "lambda2": _lambda2(A), "mean_degree": float(A.sum() / N),
                         "sigma2_W": mixing_rate(W),
                         "gt_best_iters": np.nanmin(list(gt.values())), "admm_rho1_iters": ad[1.0],
                         "admm_best_iters": np.nanmin(list(ad.values())),
                         "admm_best_rho": min(ad, key=lambda r: (np.nan_to_num(ad[r], nan=1e9), r)),
                         "dgd_bias": dg.final_error, "dgd_bias_bound": dgd_bias_bound(prob, W, ALPHA)})
    return rows


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def figures(out: Path, conv: dict, inst_label: str, rate_rows: list[dict], rounds_rows: list[dict]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    made = []
    # F07 - convergence
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for k, r in conv.items():
        ax.semilogy(np.arange(1, len(r.err) + 1), np.maximum(r.err, 1e-16), label=k, lw=1.4)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"max$_i$ |C$_i$ − z*|  [s]")
    ax.set_title(f"F07 - cycle agreement, {inst_label}")
    ax.set_ylim(1e-14, None)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figdir / "F07_optimizer_convergence.png", dpi=300)
    plt.close(fig)
    made.append("F07_optimizer_convergence.png")

    r = conv["ADMM (central z)"]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.semilogy(r.primal, label=r"primal $\|r^k\|_2$")
    ax.semilogy(np.maximum(r.dual, 1e-16), label=r"dual $\|s^k\|_2$")
    ax.axhline(1e-3, color="0.4", ls="--", lw=1, label="stop (1e-3)")
    ax.set_xlim(0, max(60, r.iterations * 3))
    ax.set_xlabel("iteration")
    ax.set_ylabel("residual norm [s]")
    ax.set_title(f"Consensus ADMM residuals, {inst_label}")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "E10_admm_residuals.png", dpi=300)
    plt.close(fig)
    made.append("E10_admm_residuals.png")

    lam = np.array([x["lambda2"] for x in rate_rows])
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.scatter(lam, [x["gt_best_iters"] for x in rate_rows], s=18, label="gradient tracking (best α)")
    ax.scatter(lam, [x["admm_best_iters"] for x in rate_rows], s=18, marker="s", label="decentralised ADMM (best ρ)")
    ax.scatter(lam, [x["admm_rho1_iters"] for x in rate_rows], s=18, marker="x", label="decentralised ADMM (ρ = 1)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda_2(L)$")
    ax.set_ylabel(r"iterations to error < $10^{-6}$ s")
    ax.set_title("Rate vs algebraic connectivity, ER G(16, p)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "E10_rate_vs_lambda2.png", dpi=300)
    plt.close(fig)
    made.append("E10_rate_vs_lambda2.png")

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.semilogy([x["rounds"] for x in rounds_rows], [max(x["final_error"], 1e-16) for x in rounds_rows], "o-")
    ax.set_xlabel("consensus rounds per ADMM iteration, K")
    ax.set_ylabel("final max error [s]")
    ax.set_title(f"ADMM with local z (course 'hack'), {inst_label}")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "E10_local_z_rounds.png", dpi=300)
    plt.close(fig)
    made.append("E10_local_z_rounds.png")
    return made


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "results" / "E10_distributed_optimisation"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    inst_rows, solver_rows = [], []
    conv_main = None
    for net in NETWORKS:
        for dem, src in INSTANCES:
            prob, A, extra = instance(net, dem, src)
            inst_rows.append({"network": net, "demand": dem, "source": src, "z_star": prob.z_star, "lambda2": _lambda2(A),
                              "C_star_min": float(prob.c_star.min()), "C_star_max": float(prob.c_star.max()),
                              "C_star": json.dumps([round(c, 3) for c in prob.c_star]),
                              "w": json.dumps([round(x, 4) for x in prob.w])})
            res = run_solvers(prob, A)
            if (net, dem, src) == MAIN:
                conv_main = res
            for name, r in res.items():
                solver_rows.append({"network": net, "demand": dem, "source": src, "solver": name, "z_star": prob.z_star,
                                    "final_error": r.final_error,
                                    "iters_to_1e-3": iterations_to(r.err, 1e-3),
                                    "iters_to_1e-6": iterations_to(r.err, 1e-6),
                                    "stopped_at": r.iterations if name == "ADMM (central z)" else "",
                                    "z_error_at_stop": (abs(admm_consensus(prob, rho=RHO, tol=1e-3).z - prob.z_star)
                                                        if name == "ADMM (central z)" else ""),
                                    "messages_to_1e-6": (r.messages / len(r.err)) * (iterations_to(r.err, 1e-6) or np.nan)})
    _write(out / "instances.csv", inst_rows)
    _write(out / "solvers.csv", solver_rows)

    rate_rows = rate_vs_lambda2()
    _write(out / "rate_vs_lambda2.csv", rate_rows)

    prob, A, _ = instance(*MAIN)
    rounds_rows = [{"rounds": K, "final_error": admm_neighbour_z(prob, A, RHO, 3000, rounds=K).final_error,
                    "disagreement": admm_neighbour_z(prob, A, RHO, 3000, rounds=K).disagreement[-1]}
                   for K in (1, 2, 3, 5, 10, 20, 40, 80)]
    _write(out / "local_z_rounds.csv", rounds_rows)

    made = figures(out, conv_main, "grid3x3, D_hotspot (measured flows)", rate_rows, rounds_rows)
    (out / "summary.md").write_text(summary(inst_rows, solver_rows, rate_rows, rounds_rows, made))
    print((out / "summary.md").read_text())


def summary(inst_rows, solver_rows, rate_rows, rounds_rows, figs) -> str:
    from scipy.stats import spearmanr

    lines = ["# E10 - distributed optimisation of the common cycle", "",
             "Instances: local Webster cycles from the fixed-time design flows, w_i = Y_i; "
             "z* = sum w_i C_i* / sum w_i.  rho = 1, DGD/GT step 0.2, Metropolis weights.", "",
             "With uniform entry rates the *design* flows are equal on every link, so all C_i* coincide "
             "(trivial instance, solved at iteration 1); *measured* flows come from one simulated hour.", "",
             "| network | demand | flows | z* [s] | C_i* range [s] | lambda2 |", "|---|---|---|---|---|---|"]
    for r in inst_rows:
        lines.append(f"| {r['network']} | {r['demand']} | {r['source']} | {r['z_star']:.2f} | "
                     f"{r['C_star_min']:.1f} – {r['C_star_max']:.1f} | {r['lambda2']:.3f} |")
    lines += ["", "## Solvers (grid3x3, non-trivial instances)", "",
              "| demand | flows | solver | final error after 2000 it. [s] | iters to 1e-3 | iters to 1e-6 |",
              "|---|---|---|---|---|---|"]
    for r in solver_rows:
        if r["network"] == "grid3x3" and not (r["demand"] == "D_med" and r["source"] == "design"):
            lines.append(f"| {r['demand']} | {r['source']} | {r['solver']} | {r['final_error']:.2e} | "
                         f"{r['iters_to_1e-3'] or '–'} | {r['iters_to_1e-6'] or '–'} |")
    adm = [r for r in solver_rows if r["solver"] == "ADMM (central z)"]
    adm = [r for r in adm if not (r["demand"] == "D_med" and r["source"] == "design")] or adm
    lines += ["", f"Central ADMM with the default stop (both residual norms < 1e-3): stopped after "
              f"{min(r['stopped_at'] for r in adm)}–{max(r['stopped_at'] for r in adm)} iterations, "
              f"|z − z*| ≤ {max(r['z_error_at_stop'] for r in adm):.1e} s over all {len(adm)} non-trivial instances."]
    lam = [r["lambda2"] for r in rate_rows]
    lines += ["", "## Rate vs lambda_2 (Erdos-Renyi G(16, p))", "",
              f"* gradient tracking (best step): Spearman(lambda2, iterations) = "
              f"{spearmanr(lam, [r['gt_best_iters'] for r in rate_rows]).correlation:+.2f}",
              f"* decentralised ADMM, rho tuned per graph: "
              f"{spearmanr(lam, [r['admm_best_iters'] for r in rate_rows]).correlation:+.2f}",
              f"* decentralised ADMM, fixed rho = 1: "
              f"{spearmanr(lam, [r['admm_rho1_iters'] for r in rate_rows]).correlation:+.2f}",
              f"* DGD constant-step bias: Spearman(lambda2, bias) = "
              f"{spearmanr(lam, [r['dgd_bias'] for r in rate_rows]).correlation:+.2f}", "",
              f"## ADMM with local z (course 'hack'), {MAIN[0]} {MAIN[1]} ({MAIN[2]} flows)", "", "| rounds K | final error [s] | disagreement [s] |",
              "|---|---|---|"]
    for r in rounds_rows:
        lines.append(f"| {r['rounds']} | {r['final_error']:.2e} | {r['disagreement']:.2e} |")
    lines += ["", "Figures: " + ", ".join(f"`figures/{f}`" for f in figs), ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
