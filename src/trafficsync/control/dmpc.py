"""Distributed MPC with predicted-flow exchange, optional STL
constraints and an epistemic guard.

Local problem of agent i at time t (step Delta = 5 s, horizon H = 12, i.e. 60 s)::

    min  sum_h sum_l w_q q_l[h+1]^2 + w_u ||g[h] - g[h-1]||^2 + w_room * slack
    s.t. q_l[h+1] = q_l[h] + a_l[h] - d_l[h],      q >= 0, d >= 0
         d_l[h] <= s_l Delta sum_p share_lp g_p[h]          (green fraction of the phase)
         sum_p g_p[h] <= 1,  u_min <= g <= 1
         n_m[h] = n_m0 + sum_{k<=h} (sum_l r_lm d_l[k] - out_m[k]) <= C_m - delta_room + slack_m[h]

Store-and-forward (link) model with green *fractions* (relaxation of binary phases);
the phase with the largest planned fraction at ``h = 0`` is requested through the shared
switching rule (min/max green, yellow, all-red).

Information (only messages and own observation):

* ``a_l[h]``: vehicles already travelling on l (``n_hat - q_hat``) arrive uniformly over the
  next ``ceil(tau_l / Delta)`` steps; later arrivals come from the upstream neighbour's
  *predicted departure profile* onto l, shifted by ``tau_l`` (entry links / missing profile:
  ``lambda_hat Delta``).
* ``n_m0``: downstream occupancy reported by m's owner (extrapolated by age); ``out_m[h]``:
  the owner's *planned discharge* of m (missing: half its saturation flow).
* After solving, i sends each downstream neighbour its departure profile and each upstream
  neighbour its planned discharge (the course's neighbour predicted trajectories).

Consistency (``consistency``):

* ``sequential`` (default): a **sequential sweep upstream-to-downstream**
  in every decision interval.  Agent i has a static stage ``s_i`` = hop distance from
  intersection 0 (the north-west corner / west end); it solves at ``t mod 5 = min(s_i, 4)``
  using the profiles that its lower-stage (upstream) neighbours sent one second earlier
  (latency 1 s), applies its first move and broadcasts its own profile, which the next
  stage uses.  Consecutive decisions repeat the sweep, so information also flows back.
* ``negotiation`` (option (b)): every agent solves at ``t mod 5 = 0``, exchanges, re-solves at
  ``t mod 5 = 2`` with the fresh profiles and applies only the second solution.

``stl: true`` adds the STL specifications as constraints: spillback safety (the room
constraint with a 1000x penalty instead of 10x), a terminal clearance condition
``q_l[H] <= q_thr`` and a response condition ``q_l[h] <= q_hi`` for ``h >= T_r / Delta`` (soft,
heavily penalised), and starvation freedom (a phase whose time since last green reaches
``R_max`` within the horizon must get at least one full step of green before then).
``epistemic: true`` forbids flow at ``h = 0`` into any link m without ``K_i Room_m`` and filters
the requested phase through the epistemic feasible set.
"""
from __future__ import annotations

import math
import time
from typing import Any, Mapping, Sequence

import numpy as np

from ..comm.messages import Message
from ..sim.observation import GREEN, Observation
from .base import AgentView, Controller, local_demand, select_phase
from .comp import NeighbourBoard
from .epistemic import EpistemicGuard

__all__ = ["DMPCController", "LocalQP"]


class LocalQP:
    """The local QP of one agent, built once with cvxpy Parameters (DPP) and re-solved."""

    def __init__(self, view: AgentView, H: int, dt: float, cfg: Mapping[str, Any], stl: bool, epistemic: bool) -> None:
        import cvxpy as cp

        self.view = view
        self.eps = float(cfg.get("solver_eps", 1e-3))
        self.H, self.dt = int(H), float(dt)
        self.links = list(view.incoming)
        self.outs = [m for m in view.outgoing if view.downstream_node.get(m) is not None]
        L, P, M, H = len(self.links), view.num_phases, len(self.outs), self.H
        share = np.zeros((L, P))
        for p in range(P):
            for l, sh in view.phase_link_shares(p).items():
                share[self.links.index(l), p] = sh
        R = np.zeros((M, L))                       # turning ratio l -> m
        for (l, m), r in view.turning_ratio.items():
            if m in self.outs and l in self.links:
                R[self.outs.index(m), self.links.index(l)] = r
        self.R = R
        s = np.array([view.links[l].sat_flow for l in self.links])
        w_q = float(cfg.get("w_q", 1.0))
        w_u = float(cfg.get("w_u", float(cfg.get("w_switch", 5.0))))
        w_room = float(cfg.get("w_room_stl" if stl else "w_room", 1000.0 if stl else 10.0))
        w_lin = float(cfg.get("w_lin", 0.0))            # linear queue term (= total delay)
        w_sw = float(cfg.get("w_switch_l1", 0.0))       # L1 change of green fractions (switching / lost time)

        self.q0 = cp.Parameter(L, nonneg=True)
        self.a = cp.Parameter((L, H), nonneg=True)
        self.gprev = cp.Parameter(P, nonneg=True)
        self.room = cp.Parameter((M, H)) if M else None       # C - delta - n0 + cumulative outflow
        self.g = cp.Variable((P, H), nonneg=True)
        self.d = cp.Variable((L, H), nonneg=True)
        self.q = cp.Variable((L, H + 1), nonneg=True)
        cons = [self.q[:, 0] == self.q0,
                self.q[:, 1:] == self.q[:, :-1] + self.a - self.d,
                self.d <= cp.multiply(np.tile((s * self.dt)[:, None], (1, H)), share @ self.g),
                cp.sum(self.g, axis=0) <= 1, self.g <= 1, self.g >= float(cfg.get("u_min", 0.0))]
        obj = w_q * cp.sum_squares(self.q[:, 1:]) + w_u * cp.sum_squares(self.g[:, 0] - self.gprev)
        if H > 1:
            obj = obj + w_u * cp.sum_squares(self.g[:, 1:] - self.g[:, :-1])
        if w_lin:
            obj = obj + w_lin * cp.sum(self.q[:, 1:])
        if w_sw:
            obj = obj + w_sw * cp.sum(cp.abs(self.g[:, 0] - self.gprev))
            if H > 1:
                obj = obj + w_sw * cp.sum(cp.abs(self.g[:, 1:] - self.g[:, :-1]))
        if M:
            self.slack = cp.Variable((M, H), nonneg=True)
            inflow = R @ self.d                                   # (M, H)
            cum = inflow @ np.triu(np.ones((H, H)))               # cumulative over h
            cons.append(cum <= self.room + self.slack)
            obj = obj + w_room * cp.sum(self.slack)
        if epistemic and M:
            self.allow = cp.Parameter(M, nonneg=True)              # 0 = no K Room_m: no flow into m now
            cons.append((R @ self.d)[:, 0] <= self.allow)
        else:
            self.allow = None
        if stl:
            self.qthr = cp.Parameter(L, nonneg=True)
            self.qhi = cp.Parameter(L, nonneg=True)
            self.need = cp.Parameter(P, nonneg=True)              # 1 = phase must get a full step soon
            self.mask = cp.Parameter((P, H), nonneg=True)         # steps before its deadline
            sl_c = cp.Variable(L, nonneg=True)
            sl_r = cp.Variable((L, H), nonneg=True)
            sl_s = cp.Variable(P, nonneg=True)
            k_r = max(0, min(H - 1, int(math.ceil(float(cfg.get("T_r", 180)) / self.dt))))
            cons += [self.q[:, H] <= self.qthr + sl_c,
                     cp.sum(cp.multiply(self.mask, self.g), axis=1) >= self.need - sl_s]
            if k_r < H:
                cons.append(self.q[:, k_r + 1:] <= cp.reshape(self.qhi, (L, 1), order="F") @ np.ones((1, H - k_r)) + sl_r[:, k_r:])
            obj = obj + 100.0 * (cp.sum(sl_c) + cp.sum(sl_r)) + 1000.0 * cp.sum(sl_s)
        else:
            self.qthr = self.qhi = self.need = self.mask = None
        self.prob = cp.Problem(cp.Minimize(obj), cons)
        self.cp = cp

    def solve(self) -> bool:
        """OSQP (warm-started); if it stops at its iteration limit, CLARABEL (interior point)."""
        import warnings

        self.used_fallback = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.prob.solve(solver="OSQP", warm_start=True, eps_abs=self.eps, eps_rel=self.eps, max_iter=4000, verbose=False)
            except Exception:  # noqa: BLE001
                pass
            if self.prob.status not in ("optimal",) or self.g.value is None:
                self.used_fallback = True
                try:
                    self.prob.solve(solver="CLARABEL", verbose=False)
                except Exception:  # noqa: BLE001 - a failed solve falls back to the local rule
                    return False
        return self.prob.status in ("optimal", "optimal_inaccurate") and self.g.value is not None


class DMPCController(Controller):
    """``cfg``: ``horizon`` (12), ``step_s`` (5), ``w_q``, ``w_u``, ``w_room``, ``stl`` (bool),
    ``epistemic`` (bool), ``theta`` (switching margin on planned fractions), ``consistency``
    (``sequential`` | ``negotiation``), ``sweeps`` (2, negotiation rounds), ``u_min`` (0),
    ``delta_room``, ``max_info_age_s``, STL parameters (``T_c``, ``T_r``, ``q_thr_frac``, ``q_hi_frac``, ``R_max``)."""

    name = "dmpc"

    def __init__(self, agent_id: int, view: AgentView, cfg: Mapping[str, Any] | None = None) -> None:
        cfg = dict(cfg or {})
        cfg["decision_interval_s"] = 1
        super().__init__(agent_id, view, cfg)

    def reset(self) -> None:
        super().reset()
        c, v = self.cfg, self.view
        self.H = int(c.get("horizon", 12))
        self.dt = int(c.get("step_s", 5))
        self.stage = min(int(v.meta.get("sweep_stage", 0)), self.dt - 1)
        self.stl = bool(c.get("stl", False))
        self.epi = bool(c.get("epistemic", False))
        self.sweeps = int(c.get("sweeps", 2))
        self.consistency = str(c.get("consistency", "sequential"))
        if self.consistency not in ("sequential", "negotiation"):
            raise ValueError("consistency must be sequential or negotiation")
        self.theta = float(c.get("theta", 0.1))
        self.lookahead = int(c.get("decision_lookahead", 1))   # planned green summed over the first k steps
        self.delta_room = float(c.get("delta_room", 2.0))
        self.max_age = float(c.get("max_info_age_s", 30.0))
        self.qp = LocalQP(v, self.H, self.dt, c, self.stl, self.epi)
        self.board = NeighbourBoard()
        self.plans: dict[int, tuple[int, dict]] = {}          # sender -> (t_sent, plan)
        self.guard = EpistemicGuard(v, {**dict(c.get("epistemic_cfg", {}) or {}), "delta_room": self.delta_room,
                                        "max_info_age_s": float(c.get("max_info_age_s", 60.0))}) if self.epi else None
        self.up_of = {l: v.upstream_node[l] for l in v.incoming if v.upstream_node.get(l) is not None}
        self.dep: dict[int, list[float]] = {}                  # my planned departures per out link
        self.dis: dict[int, list[float]] = {}                  # my planned discharge per in link
        self.g0 = np.zeros(v.num_phases)
        self.solve_times: list[float] = []
        self.n_fail = 0
        self.n_solve = 0
        self.n_fallback = 0
        self.tsg = [0] * v.num_phases
        self._want = 0
        p = dict(c)
        self.q_thr = float(p.get("q_thr_frac", 0.2))
        self.q_hi = float(p.get("q_hi_frac", 0.6))
        self.R_max = float(p.get("R_max", 120))

    # ------------------------------------------------------------ messages
    def receive(self, inbox: Sequence[Message], t: int) -> None:
        self.board.receive(inbox)
        for m in inbox:
            if m.plan is not None:
                old = self.plans.get(m.sender)
                if old is None or m.t_sent >= old[0]:
                    self.plans[m.sender] = (m.t_sent, m.plan)

    def _fresh_plan(self, j: int | None, t: int) -> dict | None:
        if j is None or j not in self.plans:
            return None
        ts, plan = self.plans[j]
        return plan if t - ts <= self.max_age else None

    # ------------------------------------------------------------- QP data
    def _arrivals(self, obs: Observation, t: int) -> np.ndarray:
        v, H, dt = self.view, self.H, self.dt
        A = np.zeros((len(self.qp.links), H))
        for k, l in enumerate(self.qp.links):
            e = obs.incoming[l]
            tau = v.links[l].tau
            n_tau = max(1, int(math.ceil(tau / dt)))
            moving = max(0.0, float(e.n_hat) - float(e.q_hat))
            A[k, : min(H, n_tau)] += moving / n_tau
            plan = self._fresh_plan(self.up_of.get(l), t)
            prof = None
            if plan is not None:
                prof = (plan.get("dep") or {}).get(l, (plan.get("dep") or {}).get(str(l)))
            if prof is not None:
                ts = int(plan.get("t", t))
                for h2, c in enumerate(prof):
                    arr = ts + h2 * dt + tau            # departures in step h2 reach the stop line after tau
                    h = int((arr - t) // dt)
                    if 0 <= h < H and arr >= t:
                        A[k, h] += max(0.0, float(c))
                # beyond the profile: average rate
                end = int((int(plan.get("t", t)) + len(prof) * dt + tau - t) // dt)
                if end < H:
                    A[k, max(0, end):] += max(0.0, float(e.lambda_hat)) * dt
            else:
                A[k, min(H, n_tau):] += max(0.0, float(e.lambda_hat)) * dt
        return A

    def _room(self, t: int) -> np.ndarray:
        v, H, dt = self.view, self.H, self.dt
        out = np.zeros((len(self.qp.outs), H))
        for k, m in enumerate(self.qp.outs):
            C = float(v.links[m].capacity)
            j = v.downstream_node[m]
            inf = self.board.info.get(j)
            n0 = 0.5 * C                                         # unknown: assume half full
            if inf is not None and m in inf.occupancy and t - inf.t_data <= self.max_age:
                n0 = min(C, float(inf.occupancy[m]) * C)   # my own inflow since then is part of the QP
            plan = self._fresh_plan(j, t)
            dis = None
            if plan is not None:
                dis = (plan.get("dis") or {}).get(m, (plan.get("dis") or {}).get(str(m)))
            outflow = np.full(H, 0.5 * v.links[m].sat_flow * dt)
            if dis is not None:
                ts = int(plan.get("t", t))
                for h2, c in enumerate(dis):
                    h = int((ts + h2 * dt - t) // dt)
                    if 0 <= h < H:
                        outflow[h] = max(0.0, float(c))
            out[k] = C - self.delta_room - n0 + np.cumsum(outflow)
        return out

    def _solve(self, obs: Observation, t: int) -> bool:
        qp, v = self.qp, self.view
        qp.q0.value = np.array([max(0.0, float(obs.incoming[l].q_hat)) for l in qp.links])
        qp.a.value = self._arrivals(obs, t)
        g = np.zeros(v.num_phases)
        if obs.signal_state == GREEN:
            g[int(obs.phase)] = 1.0
        qp.gprev.value = g
        if qp.room is not None:
            qp.room.value = self._room(t)
        if qp.allow is not None:
            qp.allow.value = np.array([0.0 if not self.guard.knows_room(self.board, m, t) else 1e6 for m in qp.outs])
        if self.stl:
            caps = np.array([float(v.links[l].capacity) for l in qp.links])
            qp.qthr.value = self.q_thr * caps
            qp.qhi.value = self.q_hi * caps
            need = np.zeros(v.num_phases)
            mask = np.zeros((v.num_phases, self.H))
            for p in range(v.num_phases):
                left = self.R_max - self.tsg[p]                # seconds until the starvation deadline
                if left <= self.H * self.dt and not (obs.signal_state == GREEN and obs.phase == p):
                    need[p] = 1.0
                    mask[p, : max(1, int(left // self.dt))] = 1.0
            qp.need.value = need
            qp.mask.value = mask
        t0 = time.perf_counter()
        ok = qp.solve()
        self.solve_times.append(time.perf_counter() - t0)
        self.n_solve += 1
        self.n_fallback += int(qp.used_fallback)
        if not ok:
            self.n_fail += 1
            return False
        self.g0 = np.asarray(qp.g.value)[:, : max(1, self.lookahead)].sum(axis=1)
        d = np.maximum(np.asarray(qp.d.value), 0.0)
        R = qp.R
        self.dep = {m: (R[k] @ d).round(4).tolist() for k, m in enumerate(qp.outs)}
        self.dis = {l: d[k].round(4).tolist() for k, l in enumerate(qp.links)}
        return True

    # -------------------------------------------------------------- control
    def before_decision(self, obs: Observation, t: int) -> None:
        for p in range(self.view.num_phases):
            self.tsg[p] = 0 if (obs.signal_state == GREEN and obs.phase == p) else self.tsg[p] + 1

    def decide(self, obs: Observation, t: int) -> int:
        r = t % self.dt
        if self.consistency == "sequential":
            solve_now = commit = r == self.stage
        else:
            solve_now = r == 0 or (self.sweeps >= 2 and r == 2)
            commit = r == (2 if self.sweeps >= 2 else 0)
        if self._last is None:
            solve_now = commit = True
        if solve_now:
            ok = self._solve(obs, t)
            if commit:
                dem = local_demand(self.view, obs)
                scores = list(self.g0) if ok else dem
                p = select_phase(obs, scores, self.view.timing, self.theta if ok else 16.0, True, dem)
                if self.guard is not None and obs.signal_state == GREEN and obs.elapsed >= self.view.timing.g_min:
                    q = self.guard.choose(obs, p, self.board, t)
                    if not (q == obs.phase and obs.elapsed >= self.view.timing.g_max):
                        p = q
                self._want = p
        return self._want

    def outbox(self, obs: Observation, t: int) -> list[Message]:
        r = t % self.dt
        send = (r == self.stage) if self.consistency == "sequential" else (r == 0 or (self.sweeps >= 2 and r == 2))
        if not send:
            return []
        v = self.view
        msgs = []
        for j in v.neighbors:
            dep = {m: self.dep.get(m, []) for m in v.outgoing if v.downstream_node.get(m) == j}
            dis = {l: self.dis.get(l, []) for l in v.incoming if v.upstream_node.get(l) == j}
            inc = [l for l in v.incoming if v.upstream_node.get(l) == j]
            if not dep and not dis:
                continue
            msgs.append(Message(sender=self.agent_id, receiver=j, t_sent=t, phase=int(obs.phase),
                                queues={l: float(obs.incoming[l].q_hat) for l in inc},
                                occupancy={l: float(obs.incoming[l].rho_hat) for l in inc},
                                arrival_rate={l: float(obs.incoming[l].lambda_hat) for l in inc},
                                plan={"t": int(t), "dep": dep, "dis": dis}))
        return msgs

    def dmpc_stats(self) -> dict:
        st = {"solves": self.n_solve, "solve_failures": self.n_fail, "solver_fallbacks": self.n_fallback,
              "solve_ms_mean": 1000.0 * float(np.mean(self.solve_times)) if self.solve_times else 0.0}
        if self.guard is not None:
            st.update(guard_blocked=self.guard.blocked, guard_fallbacks=self.guard.fallbacks, guard_checks=self.guard.checks)
        return st
