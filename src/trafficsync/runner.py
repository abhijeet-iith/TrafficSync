"""Experiment runner: one episode and YAML experiments.

``run_episode`` implements::

    obs = sim.reset()
    for t in range(T):
        inboxes = bus.deliver(t)
        actions = {i: ctrl[i].act(obs[i], inboxes.get(i, []), t) for i in agents}
        for i in agents: bus.send_many(ctrl[i].outbox(obs[i], t))
        obs, info = sim.step(actions)

Extensions used by later stages (no change to the loop):

* ``recorder`` - observer with ``start(sim, ctrls, bus)``, ``step(sim, info, ctrls, bus)``,
  ``finish(sim, metrics)`` (visualisation, E11 load traces); it never feeds agents.
* ``faults`` (controller parameter, E9) - ``{agents: [i, ...], model: {...}, channels: {...}}``:
  the listed agents become faulty *reporters* (:class:`control.wmsr.FaultInjector` rewrites
  the consensus values in their outgoing messages; their own signals run normally).
* controller *variants*: a section with a ``base`` key, e.g. ``cocycle-wmsr: {base: cocycle, ...}``.
"""
from __future__ import annotations

import copy
import csv
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .analysis.stats import summarize
from .comm.bus import CommBus, CommConfig
from .config import ProbeConfig, SimConfig, config_dir, load_yaml, project_root, resolve_config
from .control import CONTROLLERS, expected_link_flows, make_agent_views, make_controller
from .control.wmsr import FaultInjector
from .network import apply_communication, load_network
from .network.graph import Network
from .sim.simulator import Simulator

__all__ = ["run_episode", "run_experiment", "load_experiment", "build_network", "load_controller_params",
           "controller_cfg", "resolve_controller", "results_dir", "HEADLINE_METRICS"]

HEADLINE_METRICS = ["mean_delay", "mean_travel_time", "mean_travel_time_completed", "mean_queue", "max_queue",
                    "throughput_vph", "mean_stops", "spillback_events", "max_red_time", "jain_fairness",
                    "msgs_per_s_agent", "bytes_per_s_agent", "stl_rho_safe"]
HIGHER_IS_BETTER = {"throughput_vph", "jain_fairness", "stl_rho_safe"}


def results_dir() -> Path:
    p = project_root() / "results"
    p.mkdir(exist_ok=True)
    return p


def _deep_merge(base: Mapping[str, Any], over: Mapping[str, Any] | None) -> dict[str, Any]:
    out = copy.deepcopy(dict(base))
    for k, v in dict(over or {}).items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_controller_params(path: str | Path | None = None) -> dict[str, Any]:
    p = resolve_config(path) if path else config_dir() / "controllers_tuned.yaml"
    if not Path(p).exists():
        p = config_dir() / "controllers_default.yaml"
    return load_yaml(p).get("controllers", {})


def resolve_controller(name: str, params: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """``(registered class name, merged cfg)``; follows ``base`` keys of variants."""
    sec = dict(params.get(name, {}) or {})
    chain = [sec]
    base = sec.get("base")
    while base is not None and base not in CONTROLLERS:
        nxt = dict(params.get(base, {}) or {})
        chain.append(nxt)
        base = nxt.get("base")
    cls = base or name
    if cls not in CONTROLLERS:
        raise KeyError(f"unknown controller {name!r}; known: {sorted(CONTROLLERS)} or variants with 'base'")
    cfg = _deep_merge(params.get("common", {}), params.get(cls, {}) if cls != name else {})
    for s in reversed(chain):
        cfg = _deep_merge(cfg, {k: v for k, v in s.items() if k != "base"})
    return cls, cfg


def controller_cfg(name: str, params: Mapping[str, Any]) -> dict[str, Any]:
    return resolve_controller(name, params)[1]


@lru_cache(maxsize=64)
def _cached_network(name: str, topology_json: str) -> Network:
    net = load_network(name)
    topo = json.loads(topology_json)
    return apply_communication(net, topo) if topo else net


def build_network(name: str, comm_cfg: CommConfig | None = None) -> Network:
    topo = comm_cfg.topology if comm_cfg is not None else None
    return _cached_network(str(name), json.dumps(topo or {}, sort_keys=True))


class _FaultyController:
    """Wraps a controller: identical behaviour, falsified consensus values in its outbox."""

    def __init__(self, inner: Any, injector: FaultInjector) -> None:
        self.inner, self.injector = inner, injector

    def __getattr__(self, k: str) -> Any:
        return getattr(self.inner, k)

    def act(self, obs, inbox, t):
        return self.inner.act(obs, inbox, t)

    def outbox(self, obs, t):
        return self.injector.rewrite(self.inner.outbox(obs, t))


def run_episode(
    network: Network | str,
    demand: Any,
    controller: str,
    seed: int,
    *,
    sim_cfg: SimConfig | Mapping[str, Any] | None = None,
    probe_cfg: ProbeConfig | Mapping[str, Any] | None = None,
    comm_cfg: CommConfig | Mapping[str, Any] | None = None,
    controller_params: Mapping[str, Any] | None = None,
    return_sim: bool = False,
    callback: Any = None,
    recorder: Any = None,
    stl_monitor: bool = False,
) -> dict[str, Any]:
    """Run one episode and return its metrics (plus bookkeeping)."""
    t_wall = time.perf_counter()
    sim_cfg = sim_cfg if isinstance(sim_cfg, SimConfig) else SimConfig.from_dict(sim_cfg)
    probe_cfg = probe_cfg if isinstance(probe_cfg, ProbeConfig) else ProbeConfig.from_dict(probe_cfg)
    comm_cfg = comm_cfg if isinstance(comm_cfg, CommConfig) else CommConfig.from_dict(comm_cfg)
    net = build_network(network, comm_cfg) if isinstance(network, str) else network
    params = dict(controller_params) if controller_params is not None else load_controller_params()

    sim = Simulator(net, demand, probe_cfg=probe_cfg, seed=seed, sim_cfg=sim_cfg)
    turning = dict(sim.scenario.turning or sim_cfg.turning)
    cls, ccfg = resolve_controller(controller, params)
    ccfg.setdefault("comm_period_s", comm_cfg.period_s)
    ccfg.setdefault("missing_policy", comm_cfg.missing_policy)
    views = make_agent_views(net, sim.phases, sim_cfg.signal, turning,
                             decision_interval=int(ccfg.get("decision_interval_s", sim_cfg.decision_interval_s)))
    if cls == "fixed":
        entry_rates = {e: float(sim.demand.rate_vector(e).mean()) for e in net.entry_links}
        ccfg["design_flows"] = expected_link_flows(net, entry_rates, turning)
    faults = dict(ccfg.pop("faults", {}) or {})
    ctrls: dict[int, Any] = {i: make_controller(cls, i, views[i], ccfg) for i in sim.agents}
    for k, i in enumerate(faults.get("agents", []) or []):
        ctrls[int(i)] = _FaultyController(ctrls[int(i)], FaultInjector(faults.get("model"), faults.get("channels"),
                                                                       seed=seed * 101 + k))
    bus = CommBus(net, comm_cfg, seed=seed)
    if recorder is not None and getattr(recorder, "needs_trace", True) and hasattr(bus, "enable_trace"):
        bus.enable_trace()

    obs = sim.reset()
    logger = sim.logger
    if recorder is not None:
        recorder.start(sim, ctrls, bus)
    clock = time.perf_counter
    from .analysis.metrics import consensus_error, per_intersection_queues

    cons_err: list[float] = []
    while not sim.done:
        t = sim.t
        inboxes = bus.deliver(t)
        actions = {}
        for i, c in ctrls.items():
            t0 = clock()
            actions[i] = c.act(obs[i], inboxes.get(i, []), t)
            if c.last_was_decision:
                logger.record_compute_time(clock() - t0)
        for i, c in ctrls.items():
            bus.send_many(c.outbox(obs[i], t))
        n_msgs, n_bytes = bus.pop_step_counts()
        if n_msgs:
            logger.record_comm(t, n_msgs, n_bytes)
        obs, info = sim.step(actions)
        if t >= sim.cfg.warmup_s and t % 5 == 0:
            cons_err.append(consensus_error(ctrls))
        if callback is not None:
            callback(sim, info)
        if recorder is not None:
            recorder.step(sim, info, ctrls, bus)

    m = sim.metrics()
    ce = [c for c in cons_err if np.isfinite(c)]
    m["consensus_error_mean"] = float(np.mean(ce)) if ce else float("nan")
    piq = per_intersection_queues(sim)
    m["queue_per_intersection"] = [round(x, 3) for x in piq["mean"]]
    m["max_queue_per_intersection"] = piq["max"]
    st = bus.stats()
    m.update(comm_sent=st["sent"], comm_dropped=st["dropped"], comm_bytes=st["bytes_sent"],
             decisions=int(sum(c.n_decisions for c in ctrls.values())))
    if cls == "fixed":
        plans = [c.plan for c in ctrls.values()]
        m["fixed_cycle_mean"] = float(np.mean([p.cycle for p in plans]))
        m["fixed_Y_max"] = float(max(p.Y for p in plans))
        m["fixed_plans"] = {i: {"cycle": p.cycle, "greens": list(p.greens), "Y": round(p.Y, 6),
                                "cycle_webster": p.cycle_webster} for i, p in zip(ctrls, plans)}
    if stl_monitor:
        from .logic.stl import STLParams, TraceMonitor

        mon = TraceMonitor(net, [len(ph) for ph in sim.phases], STLParams.from_dict(sim_cfg.stl))
        m.update({f"stl_{k}": v for k, v in mon.evaluate(sim.stl_trace(), t0=int(sim_cfg.warmup_s)).items()})
    stats = [getattr(c, "dmpc_stats", None) or getattr(c, "epistemic_stats", None) for c in ctrls.values()]
    stats = [f() for f in stats if f is not None]
    if stats:
        for k in stats[0]:
            vals = [s_[k] for s_ in stats]
            m[k] = float(np.mean(vals)) if k.endswith("_mean") else float(np.sum(vals))
    if cls in ("cocycle", "comp+cycle"):
        cs = [c.consensus_state() for c in ctrls.values()]
        m["final_cycles"] = [round(s["C"], 2) for s in cs]
        m["cycle_spread"] = float(max(s["C"] for s in cs) - min(s["C"] for s in cs))
    m["wall_s"] = time.perf_counter() - t_wall
    if recorder is not None:
        recorder.finish(sim, m)
    if return_sim:
        m["_sim"] = sim
        m["_controllers"] = ctrls
    return m


# ----------------------------------------------------------------------------
# Experiments (YAML)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class RunSpec:
    exp_id: str
    network: str
    demand: str
    controller: str
    seed: int
    info_mode: str
    sim: dict = field(default_factory=dict)
    probe: dict = field(default_factory=dict)
    comm: dict = field(default_factory=dict)
    controller_params: dict = field(default_factory=dict)


def _as_list(x: Any) -> list:
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _seeds(spec: Any) -> list[int]:
    if isinstance(spec, Mapping):
        return list(range(int(spec.get("start", 0)), int(spec.get("start", 0)) + int(spec["count"])))
    return [int(s) for s in _as_list(spec)]


def load_experiment(cfg: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    raw = load_yaml(resolve_config(cfg, prefix="experiment_")) if not isinstance(cfg, Mapping) else dict(cfg)
    exp = dict(raw.get("experiment", {}))
    if "id" not in exp:
        raise ValueError("experiment.id is required")
    sim_d = _deep_merge(load_yaml(config_dir() / "sim_default.yaml")["sim"], raw.get("sim"))
    if raw.get("signal"):
        sim_d["signal"] = _deep_merge(sim_d.get("signal", {}), raw["signal"])
    for k in ("duration_s", "warmup_s"):
        if k in exp:
            sim_d[k] = exp[k]
    probe_d = _deep_merge(load_yaml(config_dir() / "probe_default.yaml")["probe"], raw.get("probe"))
    comm_d = _deep_merge(load_yaml(config_dir() / "comm_default.yaml")["comm"], raw.get("comm"))
    ctrl_d = _deep_merge(load_controller_params(raw.get("controllers_config")), raw.get("controller_params"))
    SimConfig.from_dict(sim_d), ProbeConfig.from_dict(probe_d), CommConfig.from_dict(comm_d)
    controllers = _as_list(exp.get("controllers"))
    for c in controllers:
        resolve_controller(c, ctrl_d)
    return {"id": str(exp["id"]), "description": str(exp.get("description", "")),
            "networks": _as_list(exp.get("networks", exp.get("network"))),
            "demands": _as_list(exp.get("demands", exp.get("demand"))),
            "controllers": controllers, "seeds": _seeds(exp.get("seeds", [0])),
            "info_modes": _as_list(exp.get("info_modes")) or [probe_d.get("mode", "probe")],
            "reference": exp.get("reference", "fixed"), "sim": sim_d, "probe": probe_d, "comm": comm_d,
            "controller_params": ctrl_d}


def _run_spec(spec: RunSpec) -> dict[str, Any]:
    m = run_episode(spec.network, spec.demand, spec.controller, spec.seed, sim_cfg=spec.sim,
                    probe_cfg=spec.probe, comm_cfg=spec.comm, controller_params=spec.controller_params)
    return {"network": spec.network, "demand": spec.demand, "info_mode": spec.info_mode,
            "controller": spec.controller, "seed": spec.seed, **m}


def _scalar(v: Any) -> bool:
    return v is None or isinstance(v, (bool, int, float, str, np.integer, np.floating))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def run_many(specs: Sequence[RunSpec], workers: int = 1, label: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for k, r in enumerate(pool.map(_run_spec, specs, chunksize=2), 1):
                rows.append(r)
                if k % 25 == 0 or k == len(specs):
                    print(f"  [{label}] {k}/{len(specs)} runs", flush=True)
    else:
        for k, s in enumerate(specs, 1):
            rows.append(_run_spec(s))
            if k % 25 == 0 or k == len(specs):
                print(f"  [{label}] {k}/{len(specs)} runs", flush=True)
    return rows


def run_experiment(cfg: str | Path | Mapping[str, Any], out_dir: str | Path | None = None, workers: int = 1) -> dict:
    exp = load_experiment(cfg)
    out = Path(out_dir) if out_dir is not None else results_dir() / exp["id"]
    out.mkdir(parents=True, exist_ok=True)
    specs = [RunSpec(exp["id"], n, d, c, s, mode, exp["sim"], {**exp["probe"], "mode": mode}, exp["comm"],
                     exp["controller_params"])
             for n in exp["networks"] for d in exp["demands"] for mode in exp["info_modes"]
             for c in exp["controllers"] for s in exp["seeds"]]
    t0 = time.perf_counter()
    rows = run_many(specs, workers, exp["id"])
    flat = [{k: v for k, v in r.items() if _scalar(v)} for r in rows]
    write_csv(out / "runs.csv", flat)
    metrics = [m for m in HEADLINE_METRICS if m in flat[0]]
    summary = summarize(flat, ["network", "demand", "info_mode", "controller"], metrics)
    write_csv(out / "summary.csv", summary)
    meta = {"exp_id": exp["id"], "n_runs": len(rows), "wall_s": round(time.perf_counter() - t0, 2),
            "python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return {"exp": exp, "rows": rows, "summary": summary, "out_dir": out}
