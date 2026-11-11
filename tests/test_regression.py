"""Regression test: the headline experiment on a tiny config runs in under
60 s and reproduces the stored reference numbers in ``tests/regression_reference.json``.

Tiny config: grid 2x2, D_med and D_high, fixed / actuated / CoMP / CoCycle, seeds 0-1, 900 s
(300 s warm-up), local detectors, tuned parameters.  After an intended behaviour change,
regenerate the reference with

    TRAFFICSYNC_UPDATE_REGRESSION=1 pytest tests/test_regression.py
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import pytest

from trafficsync.runner import run_episode

REF = Path(__file__).with_name("regression_reference.json")
CONTROLLERS = ["fixed", "actuated", "comp", "cocycle"]
DEMANDS = ["D_med", "D_high"]
SEEDS = [0, 1]
KEYS = ["mean_delay", "mean_travel_time", "mean_queue", "throughput_vph", "mean_stops", "completed"]
SIM = {"duration_s": 900, "warmup_s": 300}


def tiny_headline() -> dict[str, dict[str, float]]:
    out = {}
    for d in DEMANDS:
        for c in CONTROLLERS:
            for s in SEEDS:
                m = run_episode("grid2x2", d, c, s, sim_cfg=SIM, probe_cfg={"mode": "perfect_sensor"})
                out[f"{d}/{c}/{s}"] = {k: float(m[k]) for k in KEYS if k in m}
    return out


@pytest.mark.slow
def test_regression_tiny_headline():
    t0 = time.perf_counter()
    got = tiny_headline()
    wall = time.perf_counter() - t0
    assert wall < 60.0, f"tiny headline experiment took {wall:.1f} s (limit 60 s)"
    if os.environ.get("TRAFFICSYNC_UPDATE_REGRESSION") == "1" or not REF.exists():
        REF.write_text(json.dumps({"config": {"network": "grid2x2", "demands": DEMANDS, "controllers": CONTROLLERS,
                                              "seeds": SEEDS, "sim": SIM, "info": "perfect_sensor"},
                                   "results": got}, indent=1, sort_keys=True))
        pytest.skip("reference written")
    ref = json.loads(REF.read_text())["results"]
    assert set(ref) == set(got)
    bad = [(k, m, ref[k][m], got[k][m]) for k in ref for m in ref[k]
           if not math.isclose(ref[k][m], got[k][m], rel_tol=1e-9, abs_tol=1e-9)]
    assert not bad, "regression mismatch (key, metric, reference, now): " + repr(bad[:8])
