"""Probe-vehicle information model.

Every ``T_p`` seconds each probe on a link reports its speed (0 if queued, free-flow if
travelling) with Gaussian noise; the report reaches the agent ``delay_steps`` later.  The
agent of the link's downstream intersection forms::

    q_hat = #{probes with v < v_stop} / pi / Phi(v_stop / sigma_v)   (noise correction)
    n_hat = #{probes} / pi,   rho_hat = n_hat / C
    v_hat = EMA_eta(mean probe speed),   lambda_hat = EMA_eta(probe entries / (pi T_p))

``perfect_sensor``: true ``q, n`` every step; ``lambda_hat`` = EMA of true entries per ``T_p``.
"""
from __future__ import annotations

import math

import numpy as np

from ..config import ProbeConfig
from .observation import LinkEstimate

__all__ = ["ProbeModel"]


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class ProbeModel:
    def __init__(self, cfg: ProbeConfig, links: dict, queue_links: list[int], seed: int) -> None:
        self.cfg = cfg
        self.links = links
        self.qlinks = list(queue_links)
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xBEEF]))
        self.corr = _phi(cfg.v_stop / cfg.noise_std) if (cfg.stop_noise_correction and cfg.noise_std > 0) else 1.0
        self.reset()

    def reset(self) -> None:
        self.est: dict[int, LinkEstimate] = {}
        for l in self.qlinks:
            lk = self.links[l]
            self.est[l] = LinkEstimate(l, 0.0, 0.0, 0.0, lk.free_flow_speed, 0.0, 0, lk.capacity, lk.sat_flow)
        self.pending: list[tuple[int, dict]] = []
        self.entries = {l: 0 for l in self.qlinks}          # true entries since last report
        self.probe_entries = {l: 0 for l in self.qlinks}
        self.lam = {l: 0.0 for l in self.qlinks}
        self.v = {l: self.links[l].free_flow_speed for l in self.qlinks}

    def on_enter(self, l: int, is_probe: bool) -> None:
        if l in self.entries:
            self.entries[l] += 1
            if is_probe:
                self.probe_entries[l] += 1

    def update(self, t: int, state: dict) -> None:
        """``state[l] = (q, n, probe_queued, probe_travelling)`` true values after step ``t``."""
        c = self.cfg
        eta, Tp = c.ema, max(1, c.report_period_s)
        if c.mode == "perfect_sensor":
            report = t % Tp == 0
            for l in self.qlinks:
                q, n, _, _ = state[l]
                lk = self.links[l]
                if report:
                    self.lam[l] = (1 - eta) * self.lam[l] + eta * self.entries[l] / Tp
                    self.entries[l] = 0
                    v = lk.free_flow_speed * (1 - q / n) if n > 0 else lk.free_flow_speed
                    self.v[l] = (1 - eta) * self.v[l] + eta * v
                self.est[l] = LinkEstimate(l, float(q), float(n), n / lk.capacity, self.v[l], self.lam[l], t,
                                           lk.capacity, lk.sat_flow)
            return
        if t % Tp == 0:
            batch = {}
            for l in self.qlinks:
                _, _, pq, pt = state[l]
                vf = self.links[l].free_flow_speed
                speeds = np.r_[np.zeros(pq), np.full(pt, vf)]
                if speeds.size and c.noise_std > 0:
                    speeds = speeds + self.rng.normal(0.0, c.noise_std, speeds.size)
                batch[l] = (speeds, self.probe_entries[l])
                self.probe_entries[l] = 0
            self.pending.append((t + c.delay_steps, batch))
        while self.pending and self.pending[0][0] <= t:
            _, batch = self.pending.pop(0)
            pi = c.penetration
            for l, (speeds, ent) in batch.items():
                lk = self.links[l]
                n_hat = speeds.size / pi
                q_hat = min(n_hat, float((speeds < c.v_stop).sum()) / pi / self.corr)
                if speeds.size:
                    self.v[l] = (1 - eta) * self.v[l] + eta * float(np.clip(speeds.mean(), 0, None))
                self.lam[l] = (1 - eta) * self.lam[l] + eta * ent / (pi * Tp)
                old = self.est[l]
                if c.count_ema:
                    q_hat = (1 - eta) * old.q_hat + eta * q_hat
                    n_hat = (1 - eta) * old.n_hat + eta * n_hat
                self.est[l] = LinkEstimate(l, q_hat, n_hat, n_hat / lk.capacity, self.v[l], self.lam[l], t,
                                           lk.capacity, lk.sat_flow)
