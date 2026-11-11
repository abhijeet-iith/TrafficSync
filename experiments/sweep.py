"""Parameter sweeps on top of an experiment config.

``sweep(config, key, values)`` re-runs the experiment once per value of a dotted config key
(``demand``, ``probe.penetration``, ``comm.drop_prob``, ``comm.latency_steps``,
``comm.period_s``, ``comm.topology``, ``controller_params.comp.kappa`` ...), stores every run in
``results/<exp_id>/sweep_<key>.csv`` (column ``value``) and returns the rows.

    python experiments/sweep.py E4 --key demand --values 0.03,0.05,0.08,0.11,0.14,0.17,0.20,0.23,0.25
    python experiments/sweep.py E7 --key comm.drop_prob --values 0,0.1,0.3,0.5,0.8,1.0
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trafficsync.config import load_yaml, resolve_config  # noqa: E402
from trafficsync.runner import RunSpec, load_experiment, results_dir, run_many, write_csv  # noqa: E402


def _set(d: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def specs_for(config: str | dict, key: str, values: Sequence[Any], seeds: Sequence[int] | None = None,
              controllers: Sequence[str] | None = None) -> list[tuple[Any, RunSpec]]:
    raw = copy.deepcopy(config) if isinstance(config, dict) else load_yaml(resolve_config(config, prefix="experiment_"))
    out = []
    for v in values:
        r = copy.deepcopy(raw)
        if key == "demand":
            r["experiment"]["demands"] = [v]
        else:
            _set(r, key, v)
        if seeds is not None:
            r["experiment"]["seeds"] = list(seeds)
        if controllers is not None:
            r["experiment"]["controllers"] = list(controllers)
        exp = load_experiment(r)
        for n in exp["networks"]:
            for d in exp["demands"]:
                for mode in exp["info_modes"]:
                    for c in exp["controllers"]:
                        for s in exp["seeds"]:
                            out.append((v, RunSpec(exp["id"], n, d, c, s, mode, exp["sim"], {**exp["probe"], "mode": mode},
                                                   exp["comm"], exp["controller_params"])))
    return out


def sweep(config: str | dict, key: str, values: Sequence[Any], workers: int = 2, seeds: Sequence[int] | None = None,
          controllers: Sequence[str] | None = None, tag: str | None = None) -> list[dict]:
    pairs = specs_for(config, key, values, seeds, controllers)
    rows = run_many([s for _, s in pairs], workers, f"sweep {key}")
    out = []
    for (v, sp), r in zip(pairs, rows):
        row = {k: x for k, x in r.items() if not k.startswith("_") and not isinstance(x, (list, dict))}
        row["value"] = json.dumps(v) if isinstance(v, (dict, list)) else v
        row["sweep_key"] = key
        out.append(row)
    exp_id = pairs[0][1].exp_id if pairs else "sweep"
    write_csv(results_dir() / exp_id / f"sweep_{tag or key.replace('.', '_')}.csv", out)
    return out


def _parse(v: str):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--key", required=True)
    ap.add_argument("--values", required=True, help="comma-separated; JSON allowed per value")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    vals = [_parse(v) for v in a.values.split(",")]
    rows = sweep(a.config, a.key, vals, a.workers)
    print(f"{len(rows)} runs written to results/<exp_id>/sweep_{a.key.replace('.', '_')}.csv")


if __name__ == "__main__":
    main()
