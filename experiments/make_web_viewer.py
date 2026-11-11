"""Interactive browser replays of the SUMO runs (one self-contained HTML file per map).

    python experiments/make_web_viewer.py            # Cologne, Ingolstadt, SUMO grid 3x3  (~5 min)
    python experiments/make_web_viewer.py --maps cologne8 --minutes 10

Each file holds several controllers on the same map, seed and vehicles; open it in Chrome / Edge / Firefox
(double-click, no server needed; the street-map background needs internet).  Output: results/SUMO/web/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "results" / "SUMO" / "web"
MAP_RUNS = [("fixed", "Fixed-time (city signals)", "#f59e0b", "City fixed"), ("actuated", "Actuated agents", "#3fb950", "Actuated"),
            ("maxpressure", "Max-pressure agents", "#a78bfa", "Max-press."), ("comp", "TrafficSync CoMP agents", "#22d3ee", "CoMP"),
            ("cocycle", "TrafficSync CoCycle agents", "#f472b6", "CoCycle")]
GRID_RUNS = [("comp", "CoMP agents", "#22d3ee", "CoMP"), ("cocycle", "CoCycle agents", "#a78bfa", "CoCycle"),
             ("actuated", "Actuated agents", "#3fb950", "Actuated"), ("fixed", "Fixed-time (Webster)", "#f59e0b", "Fixed")]
TITLES = {
    "cologne3": ("Cologne (3 junctions) · AI traffic signals", "3 agent-controlled junctions · real OpenStreetMap network · real morning demand (TAPASCologne)"),
    "cologne8": ("Cologne · AI traffic signals", "8 agent-controlled junctions · real OpenStreetMap network · real morning demand (TAPASCologne)"),
    "ingolstadt7": ("Ingolstadt · AI traffic signals", "7 agent-controlled junctions · real OpenStreetMap network · real afternoon demand (InTAS)"),
    "ingolstadt21": ("Ingolstadt (21 junctions) · AI traffic signals", "21 agent-controlled junctions · real OpenStreetMap network · real afternoon demand (InTAS)"),
    "bologna_acosta": ("Bologna · Andrea Costa · AI traffic signals", "7 agent-controlled junctions · real network, real demand incl. buses (iTETRIS) · city signal plans"),
    "bologna_pasubio": ("Bologna · Pasubio · AI traffic signals", "8 agent-controlled junctions · real network, real demand incl. buses (iTETRIS) · city signal plans"),
}
GRID_ROWS = [("Avg delay / veh", "mean_delay", "s", True), ("Avg stopped time / veh", "mean_stopped_delay", "s", True),
             ("Avg trip time", "mean_travel_time", "s", True), ("Trips completed", "completed", "", False),
             ("Stops / veh", "mean_stops", "", True), ("Mean queue / link", "mean_queue", "veh", True),
             ("Messages / agent / s", "msgs_per_s_agent", "", None)]


def main() -> None:
    from trafficsync.sumo.grid import run_sumo_episode
    from trafficsync.sumo.realmap import run_realmap_episode
    from trafficsync.sumo.webviz import WebRecorder, write_html

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", default="cologne8,ingolstadt7,ingolstadt21,cologne3,bologna_acosta,bologna_pasubio,grid3x3")
    ap.add_argument("--minutes", type=int, default=20, help="recorded minutes (after a 5-minute warm-up)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repack", action="store_true", help="rebuild existing pages with the current template (no simulation)")
    a = ap.parse_args()
    if a.repack:
        from trafficsync.sumo.webviz import repack_html

        for p in sorted(OUT.glob("*.html")):
            if p.name != "index.html":
                repack_html(p, ["fixed", "actuated", "maxpressure", "comp", "cocycle"])
                print("repacked", p.relative_to(ROOT))
        return
    warm, rec_s = 300, 60 * a.minutes
    index = []
    for name in a.maps.split(","):
        runs, scene = [], None
        if name.startswith(("grid", "corridor")):
            for ctrl, label, color, short in GRID_RUNS:
                rec = WebRecorder(rel_t0=warm, rel_len=rec_s)
                m = run_sumo_episode(name, "D_med", ctrl, a.seed, sim_cfg={"duration_s": warm + rec_s, "warmup_s": warm},
                                     probe_cfg={"mode": "perfect_sensor"}, recorder=rec, work_dir=OUT / "work" / name)
                scene = rec.scene
                runs.append(rec.pack(ctrl, label, m, comm=ctrl in ("comp", "cocycle"), color=color, short=short))
                print(f"{name} {ctrl}: delay {m['mean_delay']:.1f} s, {len(rec.frames)} frames", flush=True)
            title = f"TrafficSync {name} in SUMO · AI traffic signals"
            sub = f"{len(scene['agents'])} agents · synthetic grid, demand D_med · seed {a.seed} · same vehicles in every run"
            p = write_html(OUT / f"{name}.html", title, sub, scene, runs, ("comp", "fixed"), GRID_ROWS)
        else:
            for ctrl, label, color, short in MAP_RUNS:
                rec = WebRecorder(rel_t0=warm, rel_len=rec_s)
                m = run_realmap_episode(name, ctrl, a.seed, {"warmup_s": warm}, recorder=rec, end_offset_s=warm + rec_s,
                                        work_dir=OUT / "work" / name)
                scene = rec.scene
                runs.append(rec.pack(ctrl, label, m, comm=ctrl in ("comp", "cocycle"), color=color, short=short))
                print(f"{name} {ctrl}: delay {m['mean_delay']:.1f} s, {len(rec.frames)} frames", flush=True)
            title, sub = TITLES.get(name, (name, ""))
            p = write_html(OUT / f"{name}.html", title, f"{sub} · SUMO seed {a.seed}", scene, runs, ("comp", "fixed"))
        print(f"wrote {p.relative_to(ROOT)} ({p.stat().st_size / 1e6:.1f} MB)", flush=True)
        index.append({"file": p.name, "title": title.replace(" · AI traffic signals", ""), "subtitle": sub,
                      "rows": [(r["label"], r["metrics"]["mean_delay"], r["color"]) for r in runs]})
    from trafficsync.sumo.webviz import write_index

    old = OUT / "index.json"
    import json

    merged = {e["file"]: e for e in (json.loads(old.read_text()) if old.exists() else [])}
    merged.update({e["file"]: e for e in index})
    order = [f"{m}.html" for m in ("cologne8", "ingolstadt21", "bologna_acosta", "bologna_pasubio", "ingolstadt7", "cologne3", "grid3x3")]
    entries = sorted(merged.values(), key=lambda e: order.index(e["file"]) if e["file"] in order else 99)
    old.write_text(json.dumps(entries, indent=1))
    print("wrote", write_index(OUT / "index.html", entries).relative_to(ROOT))
    import shutil

    shutil.rmtree(OUT / "work", ignore_errors=True)


if __name__ == "__main__":
    main()
