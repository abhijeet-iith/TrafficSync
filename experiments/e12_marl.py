"""E12 - learned signal control: IQL vs IQL+comm vs VDN vs QMIX vs the best handcrafted controller.

Training (laptop budget, ~25 min on 2 cores): grid3x3, local detectors, 4 variants x 3 training
seeds, <= 120 episodes of 1500 s with uniform demand lambda ~ U[0.05, 0.20], early stopping on
validation delay (validation seeds 200-201).  See ``trafficsync/marl/train.py``.

Held-out test: seeds 0-9, 3600 s (metrics from 600 s), through the same runner as every other
controller.  Scenarios *seen* in kind (uniform demand inside the training range): D_med, D_high;
*unseen*: D_tidal (asymmetric), D_hotspot (one heavy entry), D_incident (capacity drop).
Handcrafted: fixed, actuated, comp, cocycle, comp+cycle; "best handcrafted" = lowest mean delay
per scenario.  A MARL variant's delay per test seed is the mean over its 3 trained policies
(paired with the handcrafted runs on that seed); the spread across training seeds is reported.

Outputs: results/E12_marl/{checkpoints/*.pt, checkpoints/*_log.csv, training_log.csv, test_runs.csv,
summary.csv, summary.md, figures/*.png}
Run:  python experiments/e12_marl.py [--train-seeds 3] [--episodes 120] [--workers 2] [--skip-training]
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trafficsync.analysis.stats import mean_ci, paired_wilcoxon, percent_improvement  # noqa: E402
from trafficsync.runner import write_csv  # noqa: E402

OUT = ROOT / "results" / "E12_marl"
CK = OUT / "checkpoints"
VARIANTS = ["iql", "iql+comm", "vdn", "qmix"]
HANDCRAFTED = ["fixed", "actuated", "comp", "cocycle", "comp+cycle"]
SEEN = ["D_med", "D_high"]
UNSEEN = ["D_tidal", "D_hotspot", "D_incident"]
TEST_SEEDS = list(range(10))


def _train(args):
    variant, seed, episodes = args
    from trafficsync.marl.train import TrainConfig, train

    r = train(variant, seed, CK, TrainConfig(episodes=episodes, min_episodes=min(40, episodes)),
              log=lambda s: print(s, flush=True))
    r.pop("log")
    return r


def _eval_marl(args):
    path, variant, tseed, demand, seed = args
    from trafficsync.marl.train import evaluate_policy

    m = evaluate_policy(path, "grid3x3", demand, seed)
    return {"controller": variant, "train_seed": tseed, "demand": demand, "seed": seed,
            **{k: m[k] for k in ("mean_delay", "mean_travel_time", "throughput_vph", "mean_queue", "spillback_events",
                                 "mean_stops", "max_red_time", "gridlock", "msgs_per_s_agent")}}


def _eval_hand(args):
    ctrl, demand, seed = args
    from trafficsync.runner import run_episode

    m = run_episode("grid3x3", demand, ctrl, seed, probe_cfg={"mode": "perfect_sensor"})
    return {"controller": ctrl, "train_seed": "", "demand": demand, "seed": seed,
            **{k: m[k] for k in ("mean_delay", "mean_travel_time", "throughput_vph", "mean_queue", "spillback_events",
                                 "mean_stops", "max_red_time", "gridlock", "msgs_per_s_agent")}}


def summarise(rows: list[dict], train_res: list[dict]) -> tuple[list[dict], dict]:
    out, best_hand = [], {}
    for d in SEEN + UNSEEN:
        hand = {c: {r["seed"]: r for r in rows if r["controller"] == c and r["demand"] == d} for c in HANDCRAFTED}
        best = min(HANDCRAFTED, key=lambda c: np.mean([r["mean_delay"] for r in hand[c].values()]))
        best_hand[d] = best
        ref = hand[best]
        for c in HANDCRAFTED + VARIANTS:
            if c in HANDCRAFTED:
                per_seed = {s: r["mean_delay"] for s, r in hand[c].items()}
                spread = ""
            else:
                rs = [r for r in rows if r["controller"] == c and r["demand"] == d]
                per_seed = {s: float(np.mean([r["mean_delay"] for r in rs if r["seed"] == s])) for s in TEST_SEEDS}
                by_policy = [np.mean([r["mean_delay"] for r in rs if r["train_seed"] == t]) for t in sorted({r["train_seed"] for r in rs})]
                spread = float(np.std(by_policy, ddof=1)) if len(by_policy) > 1 else 0.0
            s = sorted(set(per_seed) & set(ref))
            ci = mean_ci([per_seed[k] for k in s])
            row = {"demand": d, "seen": d in SEEN, "controller": c, "mean_delay": ci["mean"], "delay_ci": ci["half_width"],
                   "std_across_training_seeds": spread, "best_handcrafted": best}
            if c != best:
                a = [per_seed[k] for k in s]
                b = [ref[k]["mean_delay"] for k in s]
                imp = percent_improvement(a, b)
                row["vs_best_hand_pct"] = -imp["improvement_pct"]
                row["vs_best_hand_ci"] = f"[{-imp['ci_hi']:+.1f}, {-imp['ci_lo']:+.1f}]"
                row["p"] = paired_wilcoxon(a, b)["p_value"]
            if c not in HANDCRAFTED:
                rs = [r for r in rows if r["controller"] == c and r["demand"] == d]
                row["throughput_vph"] = float(np.mean([r["throughput_vph"] for r in rs]))
                row["max_red_time"] = float(np.max([r["max_red_time"] for r in rs]))
                row["gridlock_runs"] = int(sum(bool(r["gridlock"]) for r in rs))
            else:
                rs = list(hand[c].values())
                row["throughput_vph"] = float(np.mean([r["throughput_vph"] for r in rs]))
                row["max_red_time"] = float(np.max([r["max_red_time"] for r in rs]))
                row["gridlock_runs"] = int(sum(bool(r["gridlock"]) for r in rs))
            out.append(row)
    return out, best_hand


def _validation_reference(ctrl: str) -> float:
    """Handcrafted controller on exactly the validation episodes used for early stopping."""
    from trafficsync.marl.train import TrainConfig
    from trafficsync.runner import run_episode

    c = TrainConfig()
    return float(np.mean([run_episode(c.network, d, ctrl, s, probe_cfg={"mode": "perfect_sensor"},
                                      sim_cfg={"duration_s": c.val_s, "warmup_s": c.val_warmup_s})["mean_delay"]
                          for d in c.val_demands for s in c.val_seeds]))


def figures(train_res: list[dict], summ: list[dict]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fd = OUT / "figures"
    fd.mkdir(parents=True, exist_ok=True)
    cols = {"iql": "#e67e22", "iql+comm": "#d35400", "vdn": "#2980b9", "qmix": "#8e44ad"}
    logs = {}
    for f in sorted(CK.glob("*_log.csv")):
        rows = list(csv.DictReader(open(f)))
        logs.setdefault(rows[0]["variant"], []).append(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for v, runs in logs.items():
        # validation curve: episodes where val_delay is set
        eps = sorted({int(r["episode"]) for run in runs for r in run if r["val_delay"]})
        M = np.full((len(runs), len(eps)), np.nan)
        for k, run in enumerate(runs):
            d = {int(r["episode"]): float(r["val_delay"]) for r in run if r["val_delay"]}
            for j, e in enumerate(eps):
                M[k, j] = d.get(e, np.nan)
        m, s = np.nanmean(M, 0), np.nanstd(M, 0)
        axes[0].plot(eps, m, "o-", color=cols[v], label=f"{v} ({len(runs)} seeds)", ms=3)
        axes[0].fill_between(eps, m - s, m + s, color=cols[v], alpha=0.15)
        L = max(len(run) for run in runs)
        R = np.full((len(runs), L), np.nan)
        for k, run in enumerate(runs):
            ret = np.array([float(r["team_return"]) for r in run])
            R[k, : ret.size] = np.convolve(ret, np.ones(5) / 5, mode="same")
        m, s = np.nanmean(R, 0), np.nanstd(R, 0)
        axes[1].plot(np.arange(1, L + 1), m, color=cols[v], label=v)
        axes[1].fill_between(np.arange(1, L + 1), m - s, m + s, color=cols[v], alpha=0.15)
    for c, ls in (("actuated", "--"), ("fixed", ":")):
        ref = _validation_reference(c)
        axes[0].axhline(ref, color="0.3", ls=ls, lw=1, label=f"{c} on the same validation set ({ref:.1f} s)")
    axes[0].set_xlabel("training episode")
    axes[0].set_ylabel("validation mean delay [s/veh]")
    axes[0].set_title("Validation delay (seeds 200-201, λ = 0.12 / 0.20, 1500 s); mean ± std over training seeds", fontsize=8)
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3, which="both")
    axes[1].set_xlabel("training episode")
    axes[1].set_ylabel("team return per episode (5-episode average)")
    axes[1].set_title("Training return (ε-greedy, random λ per episode); mean ± std", fontsize=8)
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fd / "E12_training_curves.png", dpi=300)
    plt.close(fig)
    made = ["E12_training_curves.png"]

    order = HANDCRAFTED + VARIANTS
    cc = {"fixed": "#95a5a6", "actuated": "#2c3e50", "comp": "#7f8c8d", "cocycle": "#16a085", "comp+cycle": "#27ae60", **cols}
    dems = SEEN + UNSEEN
    fig, ax = plt.subplots(figsize=(12, 4))
    w = 0.09
    for k, c in enumerate(order):
        ys = [next(r["mean_delay"] for r in summ if r["demand"] == d and r["controller"] == c) for d in dems]
        es = [next(r["delay_ci"] for r in summ if r["demand"] == d and r["controller"] == c) for d in dems]
        ax.bar(np.arange(len(dems)) + (k - len(order) / 2 + 0.5) * w, ys, w, yerr=es, capsize=2, color=cc[c], label=c)
    ax.set_xticks(range(len(dems)))
    ax.set_xticklabels([f"{d}\n({'seen' if d in SEEN else 'unseen'})" for d in dems])
    ax.set_ylabel("mean delay [s/veh]")
    ax.set_yscale("log")
    ax.set_title("E12 held-out test (seeds 0-9, 3600 s); MARL = mean of 3 trained policies; 95% CI over test seeds", fontsize=9)
    ax.legend(fontsize=7, ncol=3)
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(fd / "E12_test_delay.png", dpi=300)
    plt.close(fig)
    made.append("E12_test_delay.png")
    return made


def summary_md(train_res, summ, figs) -> str:
    L = ["# E12 - MARL vs the best handcrafted controller (grid3x3, local detectors)", "",
         "## Training", "", "| variant | training seed | episodes run | gradient steps | best validation delay [s] | wall [s] |",
         "|---|---|---|---|---|---|"]
    for r in sorted(train_res, key=lambda x: (VARIANTS.index(x["variant"]), x["seed"])):
        L.append(f"| {r['variant']} | {r['seed']} | {r['episodes']} | {r['updates']} | {r['best_val_delay']:.1f} | {r['wall_s']:.0f} |")
    L += ["", "## Held-out test (seeds 0-9, 3600 s, metrics from 600 s)", "",
          "Change vs the best handcrafted controller of that scenario, paired over test seeds [95% bootstrap CI]; "
          "MARL rows = mean of the 3 trained policies per test seed; 'spread' = std of the 3 policies' mean delays.", "",
          "| scenario | controller | delay [s] | vs best handcrafted | p | spread over training seeds | throughput [veh/h] | max red [s] | gridlock runs |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in summ:
        tag = " **(best handcrafted)**" if r["controller"] == r["best_handcrafted"] else ""
        vs = f"{r['vs_best_hand_pct']:+.1f}% {r['vs_best_hand_ci']}" if "vs_best_hand_pct" in r else "–"
        p = f"{r['p']:.2g}" if "p" in r else "–"
        sp = f"{r['std_across_training_seeds']:.1f}" if r["std_across_training_seeds"] != "" else "–"
        L.append(f"| {r['demand']}{' (unseen)' if not r['seen'] else ''} | {r['controller']}{tag} | {r['mean_delay']:.1f} ± {r['delay_ci']:.1f} | "
                 f"{vs} | {p} | {sp} | {r['throughput_vph']:.0f} | {r['max_red_time']:.0f} | {r['gridlock_runs']} |")
    L += ["", "Figures: " + ", ".join(f"`figures/{f}`" for f in figs), ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-seeds", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--replot", action="store_true", help="figures and summary.md from the saved CSVs only")
    args = ap.parse_args()
    if args.replot:
        def num(r):
            out = {}
            for k, v in r.items():
                if v == "" and k != "std_across_training_seeds":
                    continue                    # optional fields absent for the reference row
                try:
                    out[k] = float(v) if v not in ("", "True", "False") else (v == "True" if v in ("True", "False") else v)
                except ValueError:
                    out[k] = v
            return out
        summ = [num(r) for r in csv.DictReader(open(OUT / "summary.csv"))]
        for r in summ:
            r["seen"] = r["seen"] in (True, "True", 1.0)
            if r["std_across_training_seeds"] == "":
                r["std_across_training_seeds"] = ""
        import torch

        train_res = []
        for f in sorted(CK.glob("*_log.csv")):
            log = list(csv.DictReader(open(f)))
            ck = torch.load(str(f).replace("_log.csv", ".pt"), weights_only=False)
            train_res.append({"variant": log[0]["variant"], "seed": int(log[0]["seed"]), "best_val_delay": ck["val_delay"],
                              "episodes": len(log), "updates": (len(log) * 300 - 999), "wall_s": float(log[-1]["wall_s"])})
        figs = figures(train_res, summ)
        (OUT / "summary.md").write_text(summary_md(train_res, summ, figs))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    CK.mkdir(parents=True, exist_ok=True)
    if args.skip_training:
        import torch

        train_res = []
        for v in VARIANTS:
            for s in range(args.train_seeds):
                p = CK / f"{v.replace('+', '_')}_s{s}.pt"
                ck = torch.load(p, weights_only=False)
                log = list(csv.DictReader(open(CK / f"{v.replace('+', '_')}_s{s}_log.csv")))
                train_res.append({"variant": v, "seed": s, "best_val_delay": ck["val_delay"], "checkpoint": str(p),
                                  "episodes": len(log), "updates": "", "wall_s": float(log[-1]["wall_s"])})
    else:
        jobs = [(v, s, args.episodes) for s in range(args.train_seeds) for v in VARIANTS]
        with ProcessPoolExecutor(args.workers) as ex:
            train_res = list(ex.map(_train, jobs))
    write_csv(OUT / "training_log.csv", [r for f in sorted(CK.glob("*_log.csv")) for r in csv.DictReader(open(f))])
    jobs_m = [(r["checkpoint"], r["variant"], r["seed"], d, s) for r in train_res for d in SEEN + UNSEEN for s in TEST_SEEDS]
    jobs_h = [(c, d, s) for c in HANDCRAFTED for d in SEEN + UNSEEN for s in TEST_SEEDS]
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for k, r in enumerate(ex.map(_eval_hand, jobs_h), 1):
            rows.append(r)
        print(f"  [E12] handcrafted done ({len(jobs_h)} runs)", flush=True)
        for k, r in enumerate(ex.map(_eval_marl, jobs_m), 1):
            rows.append(r)
        print(f"  [E12] MARL test done ({len(jobs_m)} runs)", flush=True)
    write_csv(OUT / "test_runs.csv", rows)
    summ, _ = summarise(rows, train_res)
    write_csv(OUT / "summary.csv", summ)
    figs = figures(train_res, summ)
    (OUT / "summary.md").write_text(summary_md(train_res, summ, figs))
    print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    main()
