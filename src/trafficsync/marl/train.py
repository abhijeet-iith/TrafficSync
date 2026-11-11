"""Training loop for IQL / IQL+comm / VDN / QMIX.

* Training demand: every episode draws a uniform entry rate ``lambda ~ U[lam_lo, lam_hi]``
  (default 0.05-0.20 veh/s) and a training seed (1000 + episode); episodes of ``train_s``
  seconds (1500) on grid3x3 with local detectors.
* epsilon-greedy from 1.0 to 0.05 over ``eps_frac`` (60%) of the step budget; replay buffer
  1e5; batch 64; one gradient step per decision step after ``learn_start`` transitions;
  target update every 200 steps; lr 1e-3 (IQL) / 5e-4 (VDN, QMIX); hidden 64; gamma 0.99.
* Validation every ``val_every`` episodes: greedy policy through the normal evaluation path
  (``run_episode`` with the ``marl`` controller), validation seeds 200-201 x lambda {0.12, 0.20},
  1500 s with 300 s warm-up; **early stopping** on the mean validation delay (patience in
  validations), only after ``min_episodes`` *and* after epsilon has reached its floor - an
  earlier version stopped at 40-50 episodes while epsilon was still ~0.5.  The best
  checkpoint is kept.
* Logging: one CSV row per episode (team return, training delay, epsilon, loss, validation delay).

Held-out testing (seeds 0-9, 3600 s, unseen scenarios) is done by ``experiments/e12_marl.py``.
"""
from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

__all__ = ["TrainConfig", "VARIANTS", "train", "evaluate_policy"]

VARIANTS = {            # name -> (algorithm, neighbour observations)
    "iql": ("iql", False),
    "iql+comm": ("iql", True),
    "vdn": ("vdn", False),
    "qmix": ("qmix", False),
}


@dataclass
class TrainConfig:
    network: str = "grid3x3"
    episodes: int = 80
    min_episodes: int = 30
    train_s: int = 1500
    lam_lo: float = 0.05
    lam_hi: float = 0.20
    buffer: int = 100_000
    batch: int = 64
    learn_start: int = 1000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_frac: float = 0.6
    gamma: float = 0.99
    hidden: int = 64
    target_update: int = 200
    val_every: int = 10
    val_seeds: tuple = (200, 201)
    val_demands: tuple = (0.12, 0.20)
    val_s: int = 1500
    val_warmup_s: int = 300
    patience: int = 3
    lr: float | None = None          # None: 1e-3 (IQL) / 5e-4 (VDN, QMIX)


def evaluate_policy(path: str, network: str, demand, seed: int, duration_s: int = 3600, warmup_s: int = 600,
                    comm_cfg: dict | None = None) -> dict:
    """Greedy evaluation through the standard runner (same metrics as every other controller)."""
    from ..control import marl_policy
    from ..runner import load_controller_params, run_episode

    marl_policy._POLICY_CACHE.pop(str(path), None)
    params = {**load_controller_params(), "marl": {"policy": str(path)}}
    return run_episode(network, demand, "marl", seed, probe_cfg={"mode": "perfect_sensor"}, controller_params=params,
                       sim_cfg={"duration_s": duration_s, "warmup_s": warmup_s}, comm_cfg=comm_cfg)


def train(variant: str, seed: int, out_dir: str | Path, cfg: TrainConfig | None = None, log=print) -> dict:
    import torch

    from .buffer import ReplayBuffer
    from .env import TrafficSignalEnv
    from .learners import make_learner

    torch.set_num_threads(1)
    cfg = cfg or TrainConfig()
    algo, comm = VARIANTS[variant]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = TrafficSignalEnv(cfg.network, comm=comm, duration_s=cfg.train_s)
    N, D = env.N, env.obs_dim
    learner = make_learner(algo, N, D, hidden=cfg.hidden, gamma=cfg.gamma, target_update=cfg.target_update, seed=seed,
                           lr=cfg.lr)
    buf = ReplayBuffer(cfg.buffer, N, D, seed=seed)
    rng = np.random.default_rng(10_000 + seed)
    steps_per_ep = cfg.train_s // env.dt
    eps_steps = max(1, int(cfg.eps_frac * cfg.episodes * steps_per_ep))
    tag = f"{variant.replace('+', '_')}_s{seed}"
    ckpt_best = out / f"{tag}.pt"
    tmp = out / f"{tag}_tmp.pt"
    rows: list[dict] = []
    best, bad, total = np.inf, 0, 0
    t0 = time.perf_counter()
    for ep in range(cfg.episodes):
        lam = float(rng.uniform(cfg.lam_lo, cfg.lam_hi))
        obs = env.reset(seed=1000 + 97 * seed + ep, demand=lam)
        o = np.stack([obs[a] for a in env.possible_agents])
        ret, losses = 0.0, []
        while True:
            eps = max(cfg.eps_end, cfg.eps_start - (cfg.eps_start - cfg.eps_end) * total / eps_steps)
            a = learner.act(o, eps, rng)
            obs2, rew, term, trunc, info = env.step({f"a{i}": int(a[i]) for i in range(N)})
            o2 = np.stack([obs2[ag] for ag in env.possible_agents])
            r = np.array([rew[f"a{i}"] for i in range(N)], np.float32)
            R = float(r.mean())
            done = all(trunc.values())
            buf.add(o, a, r, R, o2, done)
            ret += R
            total += 1
            if buf.size >= cfg.learn_start:
                losses.append(learner.update(buf.sample(cfg.batch)))
            o = o2
            if done:
                break
        m = env.metrics()
        row = {"variant": variant, "seed": seed, "episode": ep + 1, "lambda": round(lam, 4), "team_return": ret,
               "train_delay": m["mean_delay"], "epsilon": eps, "loss": float(np.mean(losses)) if losses else np.nan,
               "val_delay": "", "wall_s": round(time.perf_counter() - t0, 1)}
        if (ep + 1) % cfg.val_every == 0:
            torch.save(learner.checkpoint(comm, {"variant": variant, "seed": seed, "episode": ep + 1}), tmp)
            vals = [evaluate_policy(str(tmp), cfg.network, d, s, cfg.val_s, cfg.val_warmup_s)["mean_delay"]
                    for d in cfg.val_demands for s in cfg.val_seeds]
            v = float(np.mean(vals))
            row["val_delay"] = v
            if v < best - 1e-9:
                best, bad = v, 0
                torch.save(learner.checkpoint(comm, {"variant": variant, "seed": seed, "episode": ep + 1, "val_delay": v,
                                                     "train_cfg": asdict(cfg)}), ckpt_best)
            else:
                bad += 1
            log(f"[{tag}] ep {ep + 1:3d}  eps {eps:.2f}  return {ret:8.1f}  train delay {m['mean_delay']:6.1f}  "
                f"val delay {v:6.1f}  (best {best:6.1f})  {row['wall_s']:.0f} s")
        rows.append(row)
        explore_done = ep + 1 >= int(np.ceil(cfg.eps_frac * cfg.episodes))   # never stop while still exploring
        if bad >= cfg.patience and ep + 1 >= cfg.min_episodes and explore_done:
            log(f"[{tag}] early stop at episode {ep + 1} (no validation improvement in {cfg.patience} checks)")
            break
    if tmp.exists():
        tmp.unlink()
    with open(out / f"{tag}_log.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return {"variant": variant, "seed": seed, "best_val_delay": best, "checkpoint": str(ckpt_best), "episodes": len(rows),
            "updates": learner.steps, "wall_s": time.perf_counter() - t0, "log": rows}
