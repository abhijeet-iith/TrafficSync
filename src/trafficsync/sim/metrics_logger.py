"""All episode metrics, computed for ``t >= warmup``."""
from __future__ import annotations

import math

import numpy as np

__all__ = ["MetricsLogger", "jain_index"]


def jain_index(x) -> float:
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if x.size == 0 or (x ** 2).sum() == 0:
        return 1.0
    return float(x.sum() ** 2 / (x.size * (x ** 2).sum()))


class MetricsLogger:
    def __init__(self, T: int, queue_links: list[int], n_agents: int, warmup: int) -> None:
        self.T, self.warmup, self.N = int(T), int(warmup), int(n_agents)
        self.qlinks = list(queue_links)
        L = len(self.qlinks)
        self.q = np.zeros((T, L), dtype=np.float32)
        self.n = np.zeros((T, L), dtype=np.float32)
        self.gen = np.zeros(T, dtype=np.int32)
        self.done = np.zeros(T, dtype=np.int32)
        self.spill = np.zeros(T, dtype=np.int32)
        self.blocked_entry = np.zeros(T, dtype=np.int32)
        self.msgs = np.zeros(T, dtype=np.int64)
        self.bytes = np.zeros(T, dtype=np.int64)
        self.compute: list[float] = []
        self.max_red = 0
        self.tsg: np.ndarray | None = None        # (T, N, P) seconds since each phase was last green
        self.link_wait_sum: dict[int, float] = {}
        self.link_wait_n: dict[int, int] = {}

    def record_step(self, t: int, q: np.ndarray, n: np.ndarray, gen: int, done: int, spill: int, blocked: int) -> None:
        self.q[t], self.n[t] = q, n
        self.gen[t], self.done[t], self.spill[t], self.blocked_entry[t] = gen, done, spill, blocked

    def record_comm(self, t: int, n_msgs: int, n_bytes: int) -> None:
        self.msgs[t] += n_msgs
        self.bytes[t] += n_bytes

    def record_compute_time(self, s: float) -> None:
        self.compute.append(float(s))

    def record_discharge_wait(self, t: int, l: int, wait: float) -> None:
        if t >= self.warmup:
            self.link_wait_sum[l] = self.link_wait_sum.get(l, 0.0) + wait
            self.link_wait_n[l] = self.link_wait_n.get(l, 0) + 1

    def metrics(self, vehicles: dict, phase_links: list[list[list[int]]], caps: np.ndarray, delta_room: float,
                gridlock_window: int, gridlock_ratio: float, rule: str) -> dict:
        T, w = self.T, self.warmup
        tt, tt_done, delay, stops = [], [], [], []
        for v in vehicles.values():
            if v.t_gen < w:
                continue
            end = v.t_done if v.t_done is not None else T
            tt_v = end - v.t_gen
            if v.t_done is not None:
                tt_done.append(tt_v)
            if v.t_done is not None or rule == "include_unfinished":
                tt.append(tt_v)
                delay.append(v.wait)
                stops.append(v.stops)
        win = slice(w, T)
        span = max(1, T - w)
        # gridlock: some window with completions < ratio * arrivals
        gl = False
        g, d = self.gen[win].astype(float), self.done[win].astype(float)
        if g.size >= gridlock_window:
            cg, cd = np.r_[0, np.cumsum(g)], np.r_[0, np.cumsum(d)]
            Wg = cg[gridlock_window:] - cg[:-gridlock_window]
            Wd = cd[gridlock_window:] - cd[:-gridlock_window]
            gl = bool(((Wg > 0) & (Wd < gridlock_ratio * Wg)).any())
        # fairness: mean wait per (node, phase)
        per_phase = []
        for node in phase_links:
            for links in node:
                vals = [self.link_wait_sum[l] / self.link_wait_n[l] for l in links if self.link_wait_n.get(l)]
                if vals:
                    per_phase.append(float(np.mean(vals)))
        internal_mask = caps > 0
        rob = (float((caps[None, internal_mask] - delta_room - self.n[win][:, internal_mask]).min())
               if T > w and internal_mask.any() else math.nan)          # no internal link: undefined
        return {
            "mean_travel_time": float(np.mean(tt)) if tt else math.nan,
            "mean_travel_time_completed": float(np.mean(tt_done)) if tt_done else math.nan,
            "mean_delay": float(np.mean(delay)) if delay else math.nan,
            "mean_waiting": float(np.mean(delay)) if delay else math.nan,
            "mean_stops": float(np.mean(stops)) if stops else math.nan,
            "mean_queue": float(self.q[win].mean()) if T > w else math.nan,
            "max_queue": float(self.q[win].max()) if T > w else math.nan,
            "throughput_vph": float(self.done[win].sum()) * 3600.0 / span,
            "completed": int(self.done[win].sum()),
            "generated": int(self.gen[win].sum()),
            "spillback_events": int(self.spill[win].sum()),
            "entry_blocked_events": int(self.blocked_entry[win].sum()),
            "max_red_time": int(self.max_red),
            "jain_fairness": jain_index(per_phase),
            "gridlock": gl,
            "msgs_per_s_agent": float(self.msgs[win].sum()) / span / max(1, self.N),
            "bytes_per_s_agent": float(self.bytes[win].sum()) / span / max(1, self.N),
            "compute_ms_per_decision": 1000.0 * float(np.mean(self.compute)) if self.compute else 0.0,
            "stl_rho_safe": rob,
        }
