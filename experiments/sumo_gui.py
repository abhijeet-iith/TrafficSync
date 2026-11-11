"""Watch the TrafficSync agents in sumo-gui.

    python experiments/sumo_gui.py                                  # Cologne, CoMP agents
    python experiments/sumo_gui.py --map ingolstadt7 --controller maxpressure
    python experiments/sumo_gui.py --map cologne8 --controller fixed          # the city's own signal program
    python experiments/sumo_gui.py --grid grid3x3 --demand D_med --controller cocycle   # TrafficSync grid in SUMO
    python experiments/sumo_gui.py --map path/to/your.sumocfg       # any SUMO scenario (e.g. from osmWebWizard)

The simulation starts immediately at 100 ms per simulated second (--delay; 0 = as fast as possible).
In the sumo-gui window: mouse wheel = zoom, drag = pan, the "Delay" box changes speed, right-click a
traffic light -> "Show phases" for its signal plan.  Closing the window ends the run; the metrics are printed.
Real maps: cologne3, cologne8, ingolstadt7, ingolstadt21, bologna_acosta, bologna_pasubio.
Real-map controllers: fixed (city program), actuated, maxpressure, comp, cocycle.
Grid controllers: fixed, actuated, maxpressure_local, comp, cocycle, comp+cycle, dmpc, ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default="cologne8", help="cologne8, ingolstadt7 or a .sumocfg path")
    ap.add_argument("--grid", default=None, help="run a TrafficSync network instead (corridor3, grid2x2, grid3x3, ...)")
    ap.add_argument("--demand", default="D_med", help="TrafficSync demand scenario for --grid")
    ap.add_argument("--controller", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delay", type=int, default=100, help="ms per simulation step in the GUI")
    ap.add_argument("--minutes", type=int, default=None, help="stop after this many simulated minutes")
    a = ap.parse_args()
    gui_args = ["--start", "true", "--delay", str(a.delay), "--quit-on-end", "true"]
    from trafficsync.sumo import common

    common.GUI_ARGS[:] = gui_args
    if a.grid:
        from trafficsync.sumo.grid import run_sumo_episode

        sim = {"duration_s": 60 * a.minutes, "warmup_s": min(600, 30 * a.minutes)} if a.minutes else None
        m = run_sumo_episode(a.grid, a.demand, a.controller or "cocycle", a.seed, sim_cfg=sim,
                             probe_cfg={"mode": "perfect_sensor"}, gui=True, work_dir=ROOT / "results" / "SUMO" / "gui_grid")
    else:
        from trafficsync.sumo.realmap import run_realmap_episode

        m = run_realmap_episode(a.map, a.controller or "comp", a.seed, gui=True,
                                end_offset_s=60 * a.minutes if a.minutes else None,
                                cfg={"warmup_s": min(600, 30 * a.minutes)} if a.minutes else None)
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in m.items()})


if __name__ == "__main__":
    main()
