"""Recording episodes for the animation, the dashboard and the time-space diagram.

The recorders are *observers*: they are handed to :func:`trafficsync.runner.run_episode`
(``recorder=``), read the simulator's ground truth, the communication bus trace and the
controllers' consensus states after every step, and never change the run.  Agents never
see any of this (information isolation); only visualisation code does.

A :class:`Recording` holds

* per **frame** (every ``stride`` steps in ``[t_start, t_end)``): link occupancy
  ``n_l / C_l``, stop-line queue ``q_l``, signal state / phase / seconds in state per
  intersection, which incoming links the displayed phase serves, and the messages sent
  on every directed communication pair since the previous frame (delivered vs dropped);
* per **step** (the whole episode): network mean queue [veh/link], vehicles generated and
  completed, vehicles in the network and the consensus error
  ``||x - mean(rho) 1||_2`` of the agents' load-consensus states (NaN for controllers
  without consensus).

Recordings are saved as one compressed ``.npz`` so videos can be re-rendered without
re-simulating.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..sim.observation import ALLRED, GREEN, YELLOW

__all__ = [
    "STATE_CODE", "Recording", "EpisodeRecorder", "record_episode", "record_side_by_side",
    "experiment_settings", "consensus_error", "TimeSpaceData", "TimeSpaceRecorder", "record_time_space",
    "corridor_links",
]

STATE_CODE = {GREEN: 0, YELLOW: 1, ALLRED: 2}


def consensus_error(controllers: Mapping[int, Any]) -> float:
    """``||x - mean(rho) 1||_2`` over agents that run load consensus, else NaN
    (see :func:`trafficsync.analysis.metrics.consensus_error`)."""
    from ..analysis.metrics import consensus_error as _ce

    return _ce(controllers)


# ----------------------------------------------------------------------------
# Episode recording
# ----------------------------------------------------------------------------
@dataclass
class Recording:
    """Everything needed to draw one controller's episode (see module docstring)."""

    meta: dict[str, Any]
    t: np.ndarray              # [F] time of each frame (s, end of the step)
    link_ids: np.ndarray       # [L] queue links in column order
    occ: np.ndarray            # [F, L] n_l / C_l
    queue: np.ndarray          # [F, L] stop-line queue (veh)
    sig_state: np.ndarray      # [F, N] 0 green, 1 yellow, 2 all-red
    sig_phase: np.ndarray      # [F, N]
    sig_tis: np.ndarray        # [F, N] seconds already in the state
    served: np.ndarray         # [F, L] link served by the displayed phase at its downstream node
    comm_pairs: np.ndarray     # [E, 2] directed (sender, receiver)
    msg_ok: np.ndarray         # [F, E] messages sent and not dropped since the previous frame
    msg_drop: np.ndarray       # [F, E] messages dropped since the previous frame
    series: dict[str, np.ndarray] = field(default_factory=dict)   # per step, length T

    @property
    def n_frames(self) -> int:
        return int(self.t.size)

    @property
    def label(self) -> str:
        return str(self.meta.get("label") or self.meta.get("controller", "?"))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: getattr(self, k) for k in ("t", "link_ids", "occ", "queue", "sig_state", "sig_phase",
                                                 "sig_tis", "served", "comm_pairs", "msg_ok", "msg_drop")}
        arrays.update({f"series__{k}": v for k, v in self.series.items()})
        np.savez_compressed(path, meta=np.array(json.dumps(self.meta)), **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Recording":
        with np.load(Path(path), allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            kw = {k: z[k] for k in ("t", "link_ids", "occ", "queue", "sig_state", "sig_phase", "sig_tis",
                                    "served", "comm_pairs", "msg_ok", "msg_drop")}
            series = {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith("series__")}
        return cls(meta=meta, series=series, **kw)

    def frame_index(self, t: float) -> int:
        """Index of the last frame at or before time ``t``."""
        return int(np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, self.n_frames - 1))


class EpisodeRecorder:
    """Recorder for :func:`run_episode`; ``.recording`` is filled by ``finish``."""

    def __init__(self, stride: int = 5, t_start: int = 0, t_end: int | None = None,
                 meta: Mapping[str, Any] | None = None) -> None:
        if stride < 1:
            raise ValueError("stride >= 1")
        self.stride = int(stride)
        self.t_start = int(t_start)
        self.t_end = t_end
        self.meta = dict(meta or {})
        self.recording: Recording | None = None

    # ------------------------------------------------------------ run hooks
    def start(self, sim, controllers, bus) -> None:
        net = sim.network
        self.net = net
        self.T = int(sim.duration)
        self.t_end_eff = min(self.T, int(self.t_end) if self.t_end is not None else self.T)
        self.links = list(net.queue_links)
        self.cap = np.array([net.links[l].capacity for l in self.links], dtype=float)
        self.col = {l: c for c, l in enumerate(self.links)}
        pairs = sorted({(i, j) for i in range(net.N) for j in net.neighbors(i)})
        self.pairs = np.array(pairs, dtype=np.int32).reshape(-1, 2)
        self.pair_idx = {p: k for k, p in enumerate(pairs)}
        # incoming links of every phase at every node (for the signal icon)
        self.phase_links = [[sorted({m.in_link for m in ph}) for ph in sim.phases[i]] for i in range(net.N)]
        self.frames: dict[str, list] = {k: [] for k in ("t", "occ", "queue", "sig_state", "sig_phase", "sig_tis",
                                                       "served", "msg_ok", "msg_drop")}
        self.cons = np.full(self.T, np.nan)
        self.n_net = np.zeros(self.T, dtype=np.int32)
        self._ptr = 0
        self._bus = bus

    def step(self, sim, info, controllers, bus) -> None:
        k = int(info["t"])
        self.cons[k] = consensus_error(controllers)
        st = None
        if (k + 1) % self.stride == 0 and self.t_start <= k < self.t_end_eff:
            st = sim.true_state()
            self._frame(k + 1, st, bus)
        self.n_net[k] = st["n_in_network"] if st is not None else _n_in_network(sim)

    def _frame(self, t: int, st: dict, bus) -> None:
        F = self.frames
        links = st["links"]
        n = np.array([links[l]["n"] for l in self.links], dtype=float)
        F["t"].append(t)
        F["occ"].append((n / self.cap).astype(np.float32))
        F["queue"].append(np.array([links[l]["q"] for l in self.links], dtype=np.int16))
        N = self.net.N
        state = np.zeros(N, dtype=np.int8)
        phase = np.zeros(N, dtype=np.int8)
        tis = np.zeros(N, dtype=np.int16)
        served = np.zeros(len(self.links), dtype=bool)
        for i in range(N):
            s = st["signals"][i]
            state[i] = STATE_CODE[s["state"]]
            phase[i] = s["phase"]
            tis[i] = s["t_in_state"]
            for l in self.phase_links[i][s["phase"]]:
                if l in self.col:
                    served[self.col[l]] = True
        F["sig_state"].append(state)
        F["sig_phase"].append(phase)
        F["sig_tis"].append(tis)
        F["served"].append(served)
        ok = np.zeros(len(self.pairs), dtype=np.int16)
        drop = np.zeros(len(self.pairs), dtype=np.int16)
        trace = bus.trace or []
        for snd, rcv, _t, dropped in trace[self._ptr:]:
            e = self.pair_idx[(snd, rcv)]
            if dropped:
                drop[e] += 1
            else:
                ok[e] += 1
        self._ptr = len(trace)
        F["msg_ok"].append(ok)
        F["msg_drop"].append(drop)

    def finish(self, sim, metrics: dict) -> None:
        lg = sim.logger
        T = self.T
        F = self.frames
        L, N, E = len(self.links), self.net.N, len(self.pairs)

        def stack(key, shape, dtype):
            return np.array(F[key], dtype=dtype) if F[key] else np.zeros((0, *shape), dtype=dtype)

        meta = dict(self.meta)
        meta.update(network=self.net.name, stride=self.stride, t_start=self.t_start, t_end=self.t_end_eff,
                    duration_s=T, warmup_s=int(sim.cfg.warmup_s), seed=int(sim.seed),
                    mean_delay=float(metrics.get("mean_delay", math.nan)),
                    throughput_vph=float(metrics.get("throughput_vph", math.nan)),
                    comm_sent=int(metrics.get("comm_sent", 0)), comm_dropped=int(metrics.get("comm_dropped", 0)))
        self.recording = Recording(
            meta=meta,
            t=np.array(F["t"], dtype=np.int32),
            link_ids=np.array(self.links, dtype=np.int32),
            occ=stack("occ", (L,), np.float32),
            queue=stack("queue", (L,), np.int16),
            sig_state=stack("sig_state", (N,), np.int8),
            sig_phase=stack("sig_phase", (N,), np.int8),
            sig_tis=stack("sig_tis", (N,), np.int16),
            served=stack("served", (L,), bool),
            comm_pairs=self.pairs,
            msg_ok=stack("msg_ok", (E,), np.int16),
            msg_drop=stack("msg_drop", (E,), np.int16),
            series={
                "mean_queue": lg.q[:T].mean(axis=1).astype(np.float32),
                "generated": lg.gen[:T].astype(np.int32),
                "completed": lg.done[:T].astype(np.int32),
                "n_in_network": self.n_net,
                "consensus_error": self.cons.astype(np.float32),
            },
        )


def _n_in_network(sim) -> int:
    return int(sim.in_network())


# ----------------------------------------------------------------------------
# Convenience runners
# ----------------------------------------------------------------------------
def experiment_settings(name: str | None) -> dict[str, Any]:
    """``sim_cfg``/``probe_cfg``/``comm_cfg``/``controller_params`` of an experiment config
    (e.g. ``"E2"``: tuned parameters, 3600 s with 600 s warm-up); ``{}`` for ``None``."""
    if not name:
        return {}
    from ..runner import load_experiment

    exp = load_experiment(name)
    return {"sim_cfg": exp["sim"], "probe_cfg": exp["probe"], "comm_cfg": exp["comm"],
            "controller_params": exp["controller_params"]}


def record_episode(network: str, demand: Any, controller: str, seed: int = 0, *, stride: int = 5,
                   t_start: int = 0, t_end: int | None = None, label: str | None = None,
                   **run_kw: Any) -> Recording:
    """Run one episode with an :class:`EpisodeRecorder` and return the recording.

    ``run_kw`` are passed to :func:`run_episode` (``sim_cfg``, ``probe_cfg``, ``comm_cfg``,
    ``controller_params``); :func:`experiment_settings` builds them from an experiment config.
    """
    from ..runner import run_episode

    meta = {"controller": controller, "label": label or controller,
            "demand": demand if isinstance(demand, str) else str(getattr(demand, "id", "custom"))}
    rec = EpisodeRecorder(stride=stride, t_start=t_start, t_end=t_end, meta=meta)
    run_episode(network, demand, controller, seed, recorder=rec, **run_kw)
    assert rec.recording is not None
    return rec.recording


def record_side_by_side(network: str, demand: Any, controllers: Sequence[str], seed: int = 0,
                        **kw: Any) -> list[Recording]:
    """The same demand seed under several controllers (identical arrivals)."""
    return [record_episode(network, demand, c, seed, **kw) for c in controllers]


# ----------------------------------------------------------------------------
# Time-space diagram along a corridor
# ----------------------------------------------------------------------------
def corridor_links(net, row: int | None = None, heading: str = "E") -> list[int]:
    """Entry and internal links of one straight arterial, in driving order.

    For a grid the arterial is the given ``row`` (default: the middle row) travelled with
    ``heading`` (``E`` or ``W``); for a corridor network ``row`` is ignored.
    """
    if heading not in ("E", "W"):
        raise ValueError("heading must be E or W")
    rows = int(net.meta.get("rows", 1))
    cols = int(net.meta.get("cols", net.N))
    r = rows // 2 if row is None else int(row)
    nodes = [r * cols + c for c in range(cols)]
    if heading == "W":
        nodes = nodes[::-1]
    out = [l for l in net.entry_links if net.links[l].to_node == nodes[0] and net.links[l].heading == heading]
    for a_, b_ in zip(nodes[:-1], nodes[1:]):
        l = net.link_between(a_, b_)
        if l is None or net.links[l].heading != heading:
            raise ValueError("nodes of the row are not connected in this heading")
        out.append(l)
    return out


@dataclass
class TimeSpaceData:
    """Trajectories and signal indications along a corridor (positions in m from its start)."""

    meta: dict[str, Any]
    node_pos: np.ndarray                    # [M] stop-line position of each corridor intersection
    t: np.ndarray                           # [T] times
    signal: np.ndarray                      # [T, M] 0 green, 1 yellow, 2 red (for the corridor approach)
    trajectories: dict[int, np.ndarray]     # vid -> [k, 2] (t, x)
    speed: float                            # free-flow speed (m/s)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        vids = sorted(self.trajectories)
        lens = np.array([len(self.trajectories[v]) for v in vids], dtype=np.int32)
        flat = np.concatenate([self.trajectories[v] for v in vids]) if vids else np.zeros((0, 2))
        np.savez_compressed(path, meta=np.array(json.dumps(self.meta)), node_pos=self.node_pos, t=self.t,
                            signal=self.signal, vids=np.array(vids, dtype=np.int64), lens=lens, flat=flat,
                            speed=np.array(self.speed))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TimeSpaceData":
        with np.load(Path(path), allow_pickle=False) as z:
            vids, lens, flat = z["vids"], z["lens"], z["flat"]
            cuts = np.cumsum(np.r_[0, lens])
            traj = {int(v): flat[cuts[k]:cuts[k + 1]] for k, v in enumerate(vids)}
            return cls(meta=json.loads(str(z["meta"])), node_pos=z["node_pos"], t=z["t"], signal=z["signal"],
                       trajectories=traj, speed=float(z["speed"]))


class TimeSpaceRecorder:
    """Positions of every vehicle on the corridor links in ``[t0, t1)``.

    The simulator is a point-queue model; for the diagram a queued vehicle is drawn at
    ``(stop line) - j * l_veh / lanes`` (``j`` = place in the queue) and a travelling one at
    free-flow progress, but never past the back of the queue and never moving backwards.
    """

    def __init__(self, links: Sequence[int], t0: int, t1: int, meta: Mapping[str, Any] | None = None,
                 vehicle_length: float = 7.5) -> None:
        self.links = list(links)
        self.t0, self.t1 = int(t0), int(t1)
        self.meta = dict(meta or {})
        self.lveh = float(vehicle_length)
        self.data: TimeSpaceData | None = None

    def start(self, sim, controllers, bus) -> None:
        net = sim.network
        self.net = net
        self.start_pos: dict[int, float] = {}
        pos = 0.0
        node_pos = []
        for l in self.links:
            lk = net.links[l]
            self.start_pos[l] = pos
            pos += lk.length
            node_pos.append(pos)
        self.node_pos = np.array(node_pos)
        self.nodes = [net.links[l].to_node for l in self.links]
        self.phase_links = {i: [sorted({m.in_link for m in ph}) for ph in sim.phases[i]] for i in self.nodes}
        self.times: list[int] = []
        self.sig: list[list[int]] = []
        self.traj: dict[int, list[tuple[int, float]]] = {}
        self.speed = float(np.mean([net.links[l].free_flow_speed for l in self.links]))

    def step(self, sim, info, controllers, bus) -> None:
        k = int(info["t"])
        if not self.t0 <= k < self.t1:
            return
        t = k + 1
        self.times.append(t)
        st = sim.true_state()["signals"]
        row = []
        for l, i in zip(self.links, self.nodes):
            s = st[i]
            on = l in self.phase_links[i][s["phase"]]
            row.append(0 if (on and s["state"] == GREEN) else 1 if (on and s["state"] == YELLOW) else 2)
        self.sig.append(row)
        for l in self.links:
            ls = sim.ls[l]
            lk = self.net.links[l]
            x0 = self.start_pos[l]
            end = x0 + lk.length
            gap = self.lveh / max(lk.lanes, 1)
            for j, vid in enumerate(ls.queue):
                self._put(vid, t, end - j * gap)
            tail = end - len(ls.queue) * gap
            for _due, vid in ls.travel:
                v = sim.vehicles[vid]
                frac = min(1.0, (t - v.t_link_enter) / max(lk.tau, 1))
                self._put(vid, t, min(x0 + frac * lk.length, tail))

    def _put(self, vid: int, t: int, x: float) -> None:
        # the drawn queue tail can move upstream when a vehicle joins ahead; never draw a car reversing
        pts = self.traj.setdefault(vid, [])
        if pts:
            x = max(x, pts[-1][1])
        pts.append((t, x))

    def finish(self, sim, metrics: dict) -> None:
        meta = dict(self.meta)
        meta.update(network=self.net.name, links=self.links, nodes=self.nodes, t0=self.t0, t1=self.t1,
                    seed=int(sim.seed), mean_delay=float(metrics.get("mean_delay", math.nan)))
        self.data = TimeSpaceData(meta=meta, node_pos=self.node_pos, t=np.array(self.times, dtype=np.int32),
                                  signal=np.array(self.sig, dtype=np.int8).reshape(-1, len(self.links)),
                                  trajectories={v: np.array(p, dtype=float) for v, p in self.traj.items()},
                                  speed=self.speed)


def record_time_space(network: str, demand: Any, controller: str, seed: int = 0, *, t0: int = 1200,
                      t1: int = 1800, row: int | None = None, heading: str = "E", label: str | None = None,
                      **run_kw: Any) -> TimeSpaceData:
    """Run one episode and record the time-space data of one arterial."""
    from ..runner import build_network, run_episode
    from ..comm.bus import CommConfig

    comm = run_kw.get("comm_cfg")
    comm = comm if isinstance(comm, CommConfig) else CommConfig.from_dict(comm)
    net = build_network(network, comm)
    links = corridor_links(net, row=row, heading=heading)
    rec = TimeSpaceRecorder(links, t0, t1, meta={"controller": controller, "label": label or controller,
                                                 "demand": demand if isinstance(demand, str) else "custom",
                                                 "heading": heading})
    run_episode(net, demand, controller, seed, recorder=rec, **run_kw)
    assert rec.data is not None
    return rec.data
