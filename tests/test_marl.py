"""MARL: environment API, information isolation, learners, checkpoint -> controller round trip."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from trafficsync.marl.buffer import ReplayBuffer  # noqa: E402
from trafficsync.marl.env import TrafficSignalEnv  # noqa: E402
from trafficsync.marl.learners import make_learner  # noqa: E402
from trafficsync.marl.nets import QMixer  # noqa: E402
from trafficsync.marl.train import TrainConfig, evaluate_policy, train  # noqa: E402


def test_env_parallel_api_and_rewards():
    env = TrafficSignalEnv("grid2x2", comm=True, duration_s=100)
    obs = env.reset(seed=0, demand=0.15)
    assert set(obs) == {"a0", "a1", "a2", "a3"} and obs["a0"].shape == (env.obs_dim,)
    assert env.state().shape == (env.state_dim,)
    steps = 0
    while env.agents:
        obs, rew, term, trunc, info = env.step({a: 1 for a in env.possible_agents})
        steps += 1
        assert all(r <= 0 for r in rew.values())
        assert info["a0"]["team_reward"] == pytest.approx(np.mean(list(rew.values())))
    assert steps == 20 and all(trunc.values())


def test_env_matches_runner_metrics():
    """Holding a constant action in the env equals the same requests through the simulator directly."""
    env = TrafficSignalEnv("grid2x2", comm=False, duration_s=300)
    env.reset(seed=3, demand="D_med")
    while env.agents:
        env.step({a: 0 if (env.sim.t // 30) % 2 == 0 else 1 for a in env.possible_agents})
    from trafficsync.network import load_network
    from trafficsync.sim import Simulator

    sim = Simulator(load_network("grid2x2"), "D_med", probe_cfg={"mode": "perfect_sensor"}, seed=3,
                    sim_cfg={"duration_s": 300, "warmup_s": 0})
    obs = sim.reset()
    while not sim.done:
        a = 0 if ((sim.t // 5 * 5) // 30) % 2 == 0 else 1
        obs, _ = sim.step({i: a for i in obs})
    assert env.metrics()["mean_delay"] == pytest.approx(sim.metrics()["mean_delay"])


def test_local_observation_has_no_neighbour_information():
    env = TrafficSignalEnv("grid2x2", comm=False, duration_s=50)
    o = env.reset(seed=0)
    assert env.obs_dim == 12 + 4 and o["a0"][-4:].tolist() == [1, 0, 0, 0]


def test_qmix_is_monotonic():
    torch.manual_seed(0)
    mix = QMixer(3, 6)
    s = torch.randn(64, 6)
    q = torch.randn(64, 3, requires_grad=True)
    mix(q, s).sum().backward()
    assert (q.grad >= 0).all()


@pytest.mark.parametrize("algo", ["iql", "vdn", "qmix"])
def test_learners_fit_a_fixed_target(algo):
    """One-step bandit: reward 1 for action 1 at every agent -> greedy policy picks 1."""
    N, D = 3, 5
    L = make_learner(algo, N, D, target_update=50, seed=0)
    buf = ReplayBuffer(5000, N, D)
    rng = np.random.default_rng(0)
    for _ in range(2000):
        o = rng.normal(size=(N, D)).astype(np.float32)
        a = rng.integers(0, 2, N)
        r = a.astype(np.float32)
        buf.add(o, a, r, float(r.mean()), o, True)
    for _ in range(600):
        L.update(buf.sample(64))
    o = rng.normal(size=(N, D)).astype(np.float32)
    assert (L.act(o, 0.0, rng) == 1).all()


def test_checkpoint_runs_as_controller(tmp_path):
    cfg = TrainConfig(network="grid2x2", episodes=2, min_episodes=1, train_s=200, val_every=2, val_s=300,
                      val_warmup_s=100, learn_start=50, val_seeds=(200,), val_demands=(0.12,))
    r = train("iql+comm", 0, tmp_path, cfg, log=lambda s: None)
    assert np.isfinite(r["best_val_delay"])
    m = evaluate_policy(r["checkpoint"], "grid2x2", "D_med", 0, duration_s=400, warmup_s=100)
    assert m["mean_delay"] > 0 and m["msgs_per_s_agent"] > 0         # comm variant really sends messages
    assert (tmp_path / "iql_comm_s0_log.csv").exists()
