"""Turn any OpenStreetMap area into a TrafficSync city scenario (e.g. an Indian city).

1. Get the map: on https://www.openstreetmap.org zoom to the area (about 1-2 km wide, e.g. Gachibowli or
   Kukatpally in Hyderabad), click "Export" -> "Manually select a different area" -> "Export" (saves
   ``map.osm``).  Or use SUMO's OSM Web Wizard and point ``--map`` of the other scripts at its ``osm.sumocfg``.
2. Build the scenario:

       python experiments/import_osm_city.py map.osm --name hyderabad_gachibowli --lefthand --mix india

   writes ``sumo_scenarios/<name>/`` with the network (traffic lights from OSM, joined clusters), one hour of
   random demand (``--veh-per-hour``) and, with ``--mix india``, a mixed fleet (two-wheelers, auto-rickshaws,
   cars, buses, trucks).  ``--lefthand`` = drive on the left (India, UK, Japan, ...).
3. Use it like the bundled cities:

       python experiments/sumo_gui.py --map hyderabad_gachibowli --controller comp
       python experiments/make_web_viewer.py --maps hyderabad_gachibowli

Demand is synthetic (random origin-destination trips weighted to the map edges), unlike the bundled cities,
which have measured demand; use the results as a demonstration, not as a calibrated study.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MIXES = {
    # id, vClass, share, length, max speed [m/s], accel, colour
    "india": [("two_wheeler", "motorcycle", 0.45, 2.0, 16.7, 3.0, "0.75,0.52,0.98"),
              ("auto_rickshaw", "passenger", 0.20, 2.8, 11.1, 1.8, "0.98,0.80,0.08"),
              ("car", "passenger", 0.27, 4.3, 16.7, 2.6, "0.9,0.9,0.95"),
              ("bus", "bus", 0.04, 11.0, 13.9, 1.2, "0.94,0.27,0.27"),
              ("truck", "truck", 0.04, 8.0, 13.9, 1.0, "0.38,0.65,0.98")],
    "europe": [("car", "passenger", 0.92, 4.5, 16.7, 2.6, "0.9,0.9,0.95"), ("bus", "bus", 0.03, 12.0, 13.9, 1.2, "0.94,0.27,0.27"),
               ("truck", "truck", 0.05, 8.0, 13.9, 1.0, "0.38,0.65,0.98")],
}


def main() -> None:
    from trafficsync.sumo.common import _sumo_home, tool_binary

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("osm", help="map.osm exported from openstreetmap.org")
    ap.add_argument("--name", required=True)
    ap.add_argument("--lefthand", action="store_true", help="left-hand traffic (India, UK, ...)")
    ap.add_argument("--mix", choices=sorted(MIXES), default="europe")
    ap.add_argument("--veh-per-hour", type=float, default=2400)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = ROOT / "sumo_scenarios" / a.name
    out.mkdir(parents=True, exist_ok=True)
    home = _sumo_home()
    typemap = home / "data" / "typemap" / "osmNetconvert.typ.xml"
    net = out / f"{a.name}.net.xml"
    cmd = [tool_binary("netconvert"), "--osm-files", str(Path(a.osm).resolve()), "-o", str(net), "--type-files", str(typemap),
           "--geometry.remove", "--ramps.guess", "--junctions.join", "--tls.guess-signals", "--tls.discard-simple",
           "--tls.join", "--tls.default-type", "static", "--remove-edges.isolated", "--keep-edges.by-vclass", "passenger",
           "--no-turnarounds.except-deadend", "--output.street-names", "--no-warnings"]
    if a.lefthand:
        cmd += ["--lefthand"]
    subprocess.run(cmd, check=True)
    vt = ["<additional>", '  <vTypeDistribution id="mix">']
    for vid, vcl, share, length, vmax, acc, col in MIXES[a.mix]:
        vt.append(f'    <vType id="{vid}" vClass="{vcl}" probability="{share}" length="{length}" maxSpeed="{vmax}" '
                  f'accel="{acc}" decel="4.5" sigma="0.5" color="{col}"'
                  + (' latAlignment="arbitrary" minGapLat="0.3"' if vcl == "motorcycle" else "") + "/>")
    vt += ["  </vTypeDistribution>", "</additional>"]
    (out / f"{a.name}.vtypes.add.xml").write_text("\n".join(vt) + "\n")
    period = 3600.0 / a.veh_per_hour
    subprocess.run([sys.executable, str(home / "tools" / "randomTrips.py"), "-n", str(net), "-o", str(out / f"{a.name}.trips.xml"),
                    "-r", str(out / f"{a.name}.rou.xml"), "-b", "0", "-e", "3600", "-p", f"{period:.3f}", "--fringe-factor", "5",
                    "--min-distance", "300", "--seed", str(a.seed), "--validate", "--trip-attributes", 'type="mix"',
                    "--additional-files", str(out / f"{a.name}.vtypes.add.xml")], check=True)
    (out / f"{a.name}.sumocfg").write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<!-- vehicle types are written into the route file by randomTrips/duarouter -->
<!-- {a.name}: OpenStreetMap network (c) OpenStreetMap contributors, ODbL; synthetic random demand ({a.veh_per_hour:.0f} veh/h, {a.mix} fleet). -->
<configuration>
    <input>
        <net-file value="{net.name}"/>
        <route-files value="{a.name}.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
    </time>
</configuration>
""")
    import sumolib

    n = sumolib.net.readNet(str(net))
    print(f"wrote {out.relative_to(ROOT)}: {len(n.getTrafficLights())} traffic lights, {len(n.getEdges())} edges")


if __name__ == "__main__":
    main()
