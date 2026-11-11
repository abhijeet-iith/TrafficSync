"""Interactive browser replay of SUMO runs: one self-contained HTML file, double-click to open.

``WebRecorder`` is passed as ``recorder=`` to :func:`~trafficsync.sumo.realmap.run_realmap_episode` or
:func:`~trafficsync.sumo.grid.run_sumo_episode`.  Every simulated second it stores, for every vehicle, its id,
position, heading, speed and class; the state of every signal head; the queue at every agent; and network-wide
numbers (vehicles in the network, stopped vehicles, mean speed, arrivals, average waiting so far).

``write_html`` packs several recorded runs of the *same* scenario and seed (e.g. the city's fixed-time program
vs TrafficSync agents) plus the static scene into one HTML file built from ``webviz_template.html``:

* real maps are drawn on a street basemap (Esri World Gray Canvas tiles, no API key; needs internet while
  viewing, otherwise only the SUMO roads are drawn); SUMO coordinates are converted to latitude / longitude from
  the network's UTM projection;
* every recorded controller side by side (or any subset, or one at a time) with synchronised pan / zoom, smooth
  vehicle motion, signal heads,
  agent rings showing the active phase, queue halos and pulsing message links between communicating agents;
* live comparison table, charts of stopped vehicles and average waiting time, final results with % change,
  play / pause, 1-32x speed, timeline scrubbing.

Vehicle data are stored as zlib-compressed binary (base64), about 9 bytes per vehicle per second.
"""
from __future__ import annotations

import base64
import json
import math
import re
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = ["WebRecorder", "write_html", "write_index", "repack_html", "utm_to_lonlat"]

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "webviz_template.html"
SIG_CODE = {"G": 0, "g": 1, "y": 2, "Y": 2, "u": 2, "r": 3, "R": 3, "s": 3, "o": 4, "O": 4}
VCLASS = {"passenger": 0, "private": 0, "taxi": 0, "motorcycle": 1, "moped": 1, "bicycle": 1, "bus": 3, "coach": 3,
          "truck": 4, "trailer": 4, "delivery": 4, "emergency": 0}
ROAD_W = [("motorway", 5), ("trunk", 4), ("primary", 4), ("secondary", 3), ("tertiary", 2), ("residential", 1),
          ("living_street", 1), ("unclassified", 1), ("service", 0)]


# ------------------------------------------------------------------------------------------- geo
def utm_to_lonlat(e: float, n: float, zone: int, north: bool = True) -> tuple[float, float]:
    """Inverse UTM (WGS84), accurate to centimetres; avoids a pyproj dependency."""
    a, f, k0 = 6378137.0, 1 / 298.257223563, 0.9996
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    x, y = e - 500000.0, n if north else n - 10000000.0
    m = y / k0
    mu = m / (a * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256))
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    p1 = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * math.sin(2 * mu) + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * math.sin(4 * mu)
          + (151 * e1 ** 3 / 96) * math.sin(6 * mu) + (1097 * e1 ** 4 / 512) * math.sin(8 * mu))
    c1, t1 = ep2 * math.cos(p1) ** 2, math.tan(p1) ** 2
    n1 = a / math.sqrt(1 - e2 * math.sin(p1) ** 2)
    r1 = a * (1 - e2) / (1 - e2 * math.sin(p1) ** 2) ** 1.5
    d = x / (n1 * k0)
    lat = p1 - (n1 * math.tan(p1) / r1) * (d ** 2 / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * ep2) * d ** 4 / 24
                                          + (61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * ep2 - 3 * c1 ** 2) * d ** 6 / 720)
    lon = (d - (1 + 2 * t1 + c1) * d ** 3 / 6 + (5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * ep2 + 24 * t1 ** 2) * d ** 5 / 120) / math.cos(p1)
    return math.degrees(lon) + (zone - 1) * 6 - 180 + 3, math.degrees(lat)


def _geo_affine(netfile: str, box: tuple[float, float, float, float]) -> list[float] | None:
    """Affine map local (x, y) -> (lat, lon) fitted on a grid of points (the area is a few km, so it is exact
    to well below a metre).  None if the network has no UTM projection."""
    head = open(netfile, encoding="utf-8").read(20000)
    m = re.search(r'<location netOffset="([-\d.]+),([-\d.]+)"[^>]*projParameter="([^"]*)"', head)
    if not m or "+proj=utm" not in m.group(3):
        return None
    ox, oy = float(m.group(1)), float(m.group(2))
    zone = int(re.search(r"\+zone=(\d+)", m.group(3)).group(1))
    north = "+south" not in m.group(3)
    x0, y0, x1, y1 = box
    rows, rhs = [], []
    for gx in np.linspace(x0, x1, 5):
        for gy in np.linspace(y0, y1, 5):
            lon, lat = utm_to_lonlat(gx - ox, gy - oy, zone, north)
            rows.append([1.0, gx - x0, gy - y0])
            rhs.append([lat, lon])
    coef, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    return [float(v) for v in coef.T.ravel()]   # lat = c0 + c1 x + c2 y ; lon = c3 + c4 x + c5 y


# ------------------------------------------------------------------------------------------- recorder
class WebRecorder:
    """Records a SUMO run for the browser viewer (works for real maps and the TrafficSync grid)."""

    def __init__(self, t0: float | None = None, t1: float | None = None, rel_t0: int | None = None,
                 rel_len: int | None = None) -> None:
        self.t0, self.t1, self.rel_t0, self.rel_len = t0, t1, rel_t0, rel_len
        self.frames: list[bytes] = []
        self.sig: list[bytes] = []
        self.queues: list[bytes] = []
        self.stats: list[list[float]] = []
        self.times: list[int] = []
        self._ids: dict[str, int] = {}
        self._cls: dict[str, int] = {}
        self._wait = 0.0
        self._departed = 0
        self._arrived = 0

    # ------------------------------------------------------------ hooks
    def start(self, conn_or_sim, netfile: str | None = None, agents: Any = None) -> None:
        if netfile is None:                       # grid co-simulation: the hook receives the simulator
            self.sim = conn_or_sim
            conn, netfile = conn_or_sim.conn, str(conn_or_sim.netfile)
        else:
            self.sim, conn = None, conn_or_sim
        self.netfile = netfile
        self.scene = build_scene(conn, netfile)
        self.lanes_in = {tls: sorted({c[0][0] for c in conn.trafficlight.getControlledLinks(tls) if c})
                         for tls in self.scene["tls_ids"]}
        b = conn.simulation.getTime()
        if self.rel_t0 is not None:
            self.t0 = b + self.rel_t0
            self.t1 = self.t0 + (self.rel_len or 10 ** 9)

    def step(self, conn_or_sim, t: float | None = None) -> None:
        conn = conn_or_sim.conn if self.sim is not None else conn_or_sim
        t = conn.simulation.getTime() if t is None else t
        self._departed += conn.simulation.getDepartedNumber()
        self._arrived += conn.simulation.getArrivedNumber()
        vids = conn.vehicle.getIDList()
        spd = np.array([conn.vehicle.getSpeed(v) for v in vids], float)
        stopped = int((spd < 0.1).sum())
        self._wait += stopped
        if self.t0 is not None and t < self.t0:
            return
        if self.t1 is not None and t >= self.t1:
            return
        x0, y0 = self.scene["origin"]
        n = len(vids)
        rec = np.zeros(n, dtype=[("id", "<u2"), ("x", "<i2"), ("y", "<i2"), ("a", "u1"), ("s", "u1"), ("c", "u1")])
        for k, v in enumerate(vids):
            if v not in self._ids:
                self._ids[v] = len(self._ids) % 65535
                try:
                    tid = conn.vehicle.getTypeID(v).lower()
                    self._cls[v] = 2 if ("auto" in tid or "rickshaw" in tid) else VCLASS.get(conn.vehicle.getVehicleClass(v), 0)
                except Exception:  # noqa: BLE001
                    self._cls[v] = 0
            x, y = conn.vehicle.getPosition(v)
            rec[k] = (self._ids[v], int(round((x - x0) * 2)), int(round((y - y0) * 2)),
                      int(conn.vehicle.getAngle(v) / 360 * 256) % 256, min(255, int(spd[k] * 10)), self._cls[v])
        self.frames.append(rec.tobytes())
        sig = bytearray()
        for tls in self.scene["tls_ids"]:
            sig += bytes(SIG_CODE.get(ch, 3) for ch in conn.trafficlight.getRedYellowGreenState(tls))
        self.sig.append(bytes(sig))
        self.queues.append(bytes(min(255, sum(conn.lane.getLastStepHaltingNumber(ln) for ln in self.lanes_in[tls]))
                                 for tls in self.scene["tls_ids"]))
        self.stats.append([n, stopped, round(float(spd.mean() * 3.6) if n else 0.0, 1), self._arrived,
                           round(self._wait / max(1, self._departed), 1)])
        self.times.append(int(t))

    # ------------------------------------------------------------ export
    def pack(self, key: str, label: str, metrics: Mapping[str, Any], comm: bool = False, color: str = "#22d3ee",
             short: str | None = None) -> dict:
        lens = [len(f) // 9 for f in self.frames]
        blob = zlib.compress(b"".join(self.frames), 9)
        sig = zlib.compress(b"".join(self.sig), 9)
        q = zlib.compress(b"".join(self.queues), 9)
        return {"key": key, "label": label, "short": short or label, "color": color, "comm": comm, "times": self.times, "counts": lens,
                "veh": base64.b64encode(blob).decode(), "sig": base64.b64encode(sig).decode(),
                "queue": base64.b64encode(q).decode(), "stats": self.stats,
                "metrics": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items()
                            if isinstance(v, (int, float, str))}}


def build_scene(conn, netfile: str) -> dict:
    """Static picture: lanes (with road class), junction outlines, signal heads, agents, neighbour pairs."""
    import sumolib

    net = sumolib.net.readNet(netfile, withInternal=False)
    (x0, y0), (x1, y1) = net.getBBoxXY()
    lanes = []
    for e in net.getEdges():
        if e.isSpecial():
            continue
        t = (e.getType() or "").lower()
        w = next((c for k, c in ROAD_W if k in t), 2 if not t else 1)
        if "footway" in t or "path" in t or "cycleway" in t:
            continue
        for ln in e.getLanes():
            pts = [v for p in ln.getShape() for v in (round(p[0] - x0, 1), round(p[1] - y0, 1))]
            lanes.append({"p": pts, "w": w, "n": e.getLaneNumber()})
    juncs = []
    for j in net.getNodes():
        shp = j.getShape()
        if len(shp) >= 3:
            juncs.append([v for p in shp for v in (round(p[0] - x0, 1), round(p[1] - y0, 1))])
    tls_ids = list(conn.trafficlight.getIDList())
    heads, agents = [], []
    for tls in tls_ids:
        pts = []
        for c in conn.trafficlight.getControlledLinks(tls):
            if c:
                shp = conn.lane.getShape(c[0][0])
                (ax, ay), (bx, by) = shp[-2], shp[-1]
                L = math.hypot(bx - ax, by - ay) or 1.0
                pts.append([round(bx - x0 - (bx - ax) / L * 1.5, 1), round(by - y0 - (by - ay) / L * 1.5, 1)])
            else:
                pts.append(None)
        heads.append(pts)
        if net.hasNode(tls):                   # usually the signal has the id of its junction
            xy = np.array(net.getNode(tls).getCoord(), float)
        else:                                  # joined signals: centre of their stop lines
            xy = np.array([p for p in pts if p], float).mean(axis=0) + [x0, y0]
        agents.append([round(float(xy[0] - x0), 1), round(float(xy[1] - y0), 1)])
    # neighbour pairs = agents joined by a road (searching up to 4 edges downstream), used for message pulses
    from collections import deque

    approach = {}
    for k, tls in enumerate(tls_ids):
        for c in conn.trafficlight.getControlledLinks(tls):
            if c:
                approach[c[0][0].rsplit("_", 1)[0]] = k
    pairs = set()
    for k, tls in enumerate(tls_ids):
        outs = {c[0][1].rsplit("_", 1)[0] for c in conn.trafficlight.getControlledLinks(tls) if c}
        for e0 in outs:
            if not net.hasEdge(e0):
                continue
            frontier, seen = deque([(net.getEdge(e0), 0)]), set()
            while frontier:
                e, h = frontier.popleft()
                if e.getID() in seen or h > 4:
                    continue
                seen.add(e.getID())
                j = approach.get(e.getID())
                if j is not None and j != k:
                    pairs.add((min(j, k), max(j, k)))
                    continue
                frontier.extend((nx, h + 1) for nx in e.getOutgoing())
    pairs = sorted([list(p) for p in pairs])
    return {"origin": [x0, y0], "extent": [round(x1 - x0, 1), round(y1 - y0, 1)], "lanes": lanes, "juncs": juncs,
            "tls_ids": tls_ids, "heads": heads, "agents": agents, "pairs": pairs,
            "geo": _geo_affine(netfile, (x0, y0, x1, y1))}


def repack_html(path: str | Path, order: Sequence[str] | None = None) -> Path:
    """Rebuild an existing replay with the current template (no simulation re-run); optionally reorder runs."""
    html = Path(path).read_text(encoding="utf-8")
    m = re.search(r"const D = (\{.*?\});\nconst S = D\.scene", html, re.S)
    if not m:
        raise ValueError(f"no replay data in {path}")
    d = json.loads(m.group(1))
    runs = d["runs"]
    if order:
        runs = sorted(runs, key=lambda r: order.index(r["key"]) if r["key"] in order else 99)
    return write_html(path, d["title"], d["subtitle"], d["scene"], runs, tuple(d["pair"]), [tuple(r) for r in d["rows"]])


def write_html(path: str | Path, title: str, subtitle: str, scene: dict, runs: Sequence[dict],
               default_pair: tuple[str, str] | None = None, metric_rows: Sequence[tuple] | None = None) -> Path:
    """Write the self-contained viewer.  ``runs`` come from :meth:`WebRecorder.pack`."""
    tpl = TEMPLATE.read_text(encoding="utf-8")
    vendor = HERE / "vendor"
    payload = {"title": title, "subtitle": subtitle, "scene": {k: v for k, v in scene.items() if k != "origin"},
               "runs": list(runs), "pair": list(default_pair or (runs[0]["key"], runs[-1]["key"])),
               "rows": list(metric_rows or DEFAULT_ROWS)}
    html = (tpl.replace("/*LEAFLET_CSS*/", (vendor / "leaflet.css").read_text(encoding="utf-8"))
               .replace("/*LEAFLET_JS*/", (vendor / "leaflet.js").read_text(encoding="utf-8"))
               .replace("__TITLE__", title)
               .replace("/*DATA*/null", json.dumps(payload, separators=(",", ":"))))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return p


# label, metric key, unit, lower-is-better
DEFAULT_ROWS = [("Avg delay / veh", "mean_delay", "s", True), ("Avg waiting / veh", "mean_waiting", "s", True),
                ("Avg trip time", "mean_travel_time", "s", True), ("Trips completed", "completed", "", False),
                ("Stops / veh", "mean_stops", "", True), ("Mean queue / lane", "mean_queue_per_lane", "veh", True),
                ("Signal switches / agent / h", "switches_per_agent_h", "", None)]


def write_index(path: str | Path, entries: Sequence[Mapping[str, Any]], title: str = "TrafficSync in real cities") -> Path:
    """Landing page linking every city replay.  ``entries``: dicts with file, title, subtitle and a list of
    (label, delay, color) rows."""
    cards = []
    for e in entries:
        rows = "".join(f'<tr><td><i style="background:{c}"></i>{lab}</td><td>{d:.1f} s</td></tr>' for lab, d, c in e["rows"])
        cards.append(f'<a class="card" href="{e["file"]}"><h2>{e["title"]}</h2><p>{e["subtitle"]}</p>'
                     f'<table><tr><th>controller</th><th>avg delay</th></tr>{rows}</table><span>Open replay &rarr;</span></a>')
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
body{{margin:0;background:#0b0f17;color:#e6edf3;font:14px/1.45 Inter,ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:36px 24px}} h1{{margin:0 0 6px;font-size:26px}} .lead{{color:#9aa7b6;margin:0 0 26px;max-width:860px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:16px}}
.card{{display:block;background:#111723;border:1px solid #232b3a;border-radius:14px;padding:16px 18px;color:inherit;text-decoration:none;transition:border-color .15s}}
.card:hover{{border-color:#22d3ee}} .card h2{{margin:0 0 4px;font-size:17px}} .card p{{margin:0 0 12px;color:#9aa7b6;font-size:12.5px}}
table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:13px}} th{{text-align:left;color:#6b7686;font-weight:500;font-size:11px}}
th:last-child,td:last-child{{text-align:right}} td{{padding:3px 0}} td i{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}}
.card span{{display:inline-block;margin-top:12px;color:#22d3ee;font-weight:600;font-size:12.5px}}
footer{{margin-top:28px;color:#6b7686;font-size:12px}}</style></head><body><main>
<h1>{title}</h1><p class="lead">Every traffic light is a TrafficSync agent that sees only its own approaches and talks to its
neighbours. Each replay shows the same vehicles under the city's own signal program and under all our agent designs
(actuated, max-pressure, CoMP, CoCycle) side by side, on the real street map. Delays are averages over the replayed 10 minutes (same seed); the multi-seed
results are in <code>results/SUMO/summary.md</code>.</p>
<div class="grid">{''.join(cards)}</div>
<footer>Simulation: Eclipse SUMO. Networks and demand: RESCO benchmark (Cologne: TAPASCologne, CC BY-SA; Ingolstadt: InTAS, GPL-3)
and DLR sumo-scenarios (Bologna, EPL-2.0). Basemap &copy; Esri, &copy; OpenStreetMap contributors.</footer></main></body></html>"""
    p = Path(path)
    p.write_text(html, encoding="utf-8")
    return p
