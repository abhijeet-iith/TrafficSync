"""Multi-agent traffic-signal environment, PettingZoo parallel-API style.

* agents: one per intersection (``"a0" .. "a{N-1}"``); action ``a_i in {0, 1}`` = requested phase,
  applied for one decision step (5 s); min/max green, yellow and all-red are enforced by the
  signal state machine (requests during clearance are ignored, a request before g_min waits).
* observation: :class:`trafficsync.control.marl_policy.ObsBuilder` (local, or local + neighbour
  messages through the communication bus).
* rewards: ``r_i = -mean_{5 s} sum_{l in I_i} q_l / C_l`` (local) and the team reward
  ``r = mean_i r_i`` (in ``infos["team_reward"]``).  The global state for QMIX is the
  concatenation of all observations (``env.state()``).
* episode: ``duration_s`` seconds (default 1500 for training), demand sampled per reset.

``reset(seed, demand)`` -> ``obs`` dict; ``step(actions)`` -> ``obs, rewards, terminations,
truncations, infos`` (PettingZoo ``ParallelEnv`` signature, without the dependency).
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..comm.bus import CommBus
from ..config import SimConfig
from ..control import make_agent_views
from ..control.marl_policy import ObsBuilder, obs_dim
from ..network import load_network
from ..sim.simulator import Simulator

__all__ = ["TrafficSignalEnv"]


class TrafficSignalEnv:
    metadata = {"name": "trafficsync_v0"}

    def __init__(self, network: str = "grid3x3", comm: bool = False, duration_s: int = 1500,
                 decision_s: int = 5, probe_cfg: Mapping | None = None, comm_cfg: Mapping | None = None) -> None:
        self.net = load_network(network)
        self.comm = bool(comm)
        self.duration = int(duration_s)
        self.dt = int(decision_s)
        self.probe_cfg = dict(probe_cfg or {"mode": "perfect_sensor"})
        self.comm_cfg = dict(comm_cfg or {})
        self.N = self.net.N
        self.possible_agents = [f"a{i}" for i in range(self.N)]
        self.agents = list(self.possible_agents)
        self.n_actions = 2
        self.obs_dim = obs_dim(self.N, self.comm)
        self.state_dim = self.obs_dim * self.N
        self.sim: Simulator | None = None

    # PettingZoo-style spaces (shape only; gymnasium is not required)
    def observation_space(self, agent: str) -> tuple[int, ...]:
        return (self.obs_dim,)

    def action_space(self, agent: str) -> int:
        return self.n_actions

    def reset(self, seed: int = 0, demand: Any = "D_med") -> dict[str, np.ndarray]:
        cfg = SimConfig.from_dict({"duration_s": self.duration, "warmup_s": 0})
        self.sim = Simulator(self.net, demand, probe_cfg=self.probe_cfg, seed=seed, sim_cfg=cfg)
        views = make_agent_views(self.net, self.sim.phases, cfg.signal, cfg.turning)
        self.builders = [ObsBuilder(views[i], self.comm) for i in range(self.N)]
        self.bus = CommBus(self.net, self.comm_cfg, seed=seed)
        self.obs = self.sim.reset()
        self.agents = list(self.possible_agents)
        self.caps = [np.array([self.sim.cap[l] for l in self.net.incoming_links(i)], float) for i in range(self.N)]
        self.cols = [np.array([self.sim.qcol[l] for l in self.net.incoming_links(i)]) for i in range(self.N)]
        self._exchange()
        self._last_vec = self._vectors()
        return self._last_vec

    def _exchange(self) -> None:
        t = self.sim.t
        inbox = self.bus.deliver(t)
        for i, b in enumerate(self.builders):
            b.receive(inbox.get(i, []))
        for i, b in enumerate(self.builders):
            self.bus.send_many(b.outbox(self.obs[i], t))

    def _vectors(self) -> dict[str, np.ndarray]:
        return {f"a{i}": b.vector(self.obs[i], self.sim.t) for i, b in enumerate(self.builders)}

    def state(self) -> np.ndarray:
        return np.concatenate([self._last_vec[a] for a in self.possible_agents])

    def step(self, actions: Mapping[str, int]):
        sim = self.sim
        acts = {i: int(actions[f"a{i}"]) for i in range(self.N)}
        r = np.zeros(self.N)
        k = 0
        for _ in range(self.dt):
            if sim.done:
                break
            self.obs, _ = sim.step(acts)
            q = sim.logger.q[sim.t - 1]
            for i in range(self.N):
                r[i] -= float((q[self.cols[i]] / self.caps[i]).sum())
            k += 1
            self._exchange()
        r /= max(1, k)
        vec = self._vectors()
        self._last_vec = vec
        done = sim.done
        rewards = {f"a{i}": float(r[i]) for i in range(self.N)}
        terms = {a: False for a in self.agents}
        truncs = {a: done for a in self.agents}
        infos = {a: {"team_reward": float(r.mean())} for a in self.agents}
        if done:
            self.agents = []
        return vec, rewards, terms, truncs, infos

    def metrics(self) -> dict:
        return self.sim.metrics()
