"""Configuration dataclasses and YAML helpers."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import yaml

__all__ = ["SignalTiming", "SimConfig", "ProbeConfig", "config_dir", "load_yaml", "resolve_config", "project_root"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    return project_root() / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def resolve_config(name: str | Path, prefix: str = "") -> Path:
    """``name`` may be a path, a file in ``configs/`` or a bare id (``E1`` -> ``experiment_E1.yaml``)."""
    p = Path(name)
    if p.exists():
        return p
    for cand in (config_dir() / str(name), config_dir() / f"{name}.yaml", config_dir() / f"{prefix}{name}.yaml"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"config {name!r} not found in {config_dir()}")


def _from_dict(cls, data: Mapping[str, Any] | None):
    data = dict(data or {})
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**data)


@dataclass(frozen=True)
class SignalTiming:
    g_min: int = 5
    g_max: int = 40
    yellow: int = 3
    allred: int = 1
    startup_lost: int = 2

    @property
    def lost_per_phase(self) -> int:
        """Startup + yellow + all-red = 6 s (Webster lost time per phase)."""
        return self.startup_lost + self.yellow + self.allred

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "SignalTiming":
        return _from_dict(cls, d)


@dataclass(frozen=True)
class SimConfig:
    dt: float = 1.0
    duration_s: int = 3600
    warmup_s: int = 600
    decision_interval_s: int = 5
    phase_scheme: str = "2phase"
    queue_mode: str = "link"
    drive_side: str = "right"
    initial_phase: int = 0
    turning: dict = field(default_factory=lambda: {"through": 0.70, "left": 0.15, "right": 0.15})
    travel_time_rule: str = "include_unfinished"
    gridlock_window_s: int = 300
    gridlock_ratio: float = 0.05
    signal: SignalTiming = field(default_factory=SignalTiming)
    stl: dict = field(default_factory=lambda: {"delta_room": 2, "T_c": 300, "T_r": 180, "q_thr_frac": 0.2,
                                               "q_hi_frac": 0.6, "R_max": 120})

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "SimConfig":
        d = dict(d or {})
        if "sim" in d and len(d) == 1:
            d = dict(d["sim"])
        base = load_yaml(config_dir() / "sim_default.yaml").get("sim", {}) if (config_dir() / "sim_default.yaml").exists() else {}
        merged = {**base, **d}
        merged["signal"] = SignalTiming.from_dict({**base.get("signal", {}), **dict(d.get("signal", {}) or {})}) \
            if not isinstance(d.get("signal"), SignalTiming) else d["signal"]
        if merged.get("queue_mode", "link") != "link":
            raise ValueError("only queue_mode: link is implemented")
        if merged.get("dt", 1.0) != 1.0:
            raise ValueError("dt is fixed to 1 s")
        return _from_dict(cls, merged)


@dataclass(frozen=True)
class ProbeConfig:
    mode: str = "probe"
    penetration: float = 0.2
    report_period_s: int = 5
    noise_std: float = 1.0
    pos_noise_std: float = 5.0
    delay_steps: int = 2
    ema: float = 0.3
    v_stop: float = 1.0
    stop_noise_correction: bool = True
    count_ema: bool = False
    record_batches: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ("probe", "perfect_sensor"):
            raise ValueError("probe mode must be probe | perfect_sensor")
        if not 0.0 < self.penetration <= 1.0:
            raise ValueError("penetration must be in (0, 1]")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "ProbeConfig":
        d = dict(d or {})
        if "probe" in d and len(d) == 1:
            d = dict(d["probe"])
        base = load_yaml(config_dir() / "probe_default.yaml").get("probe", {}) if (config_dir() / "probe_default.yaml").exists() else {}
        return _from_dict(cls, {**base, **d})
