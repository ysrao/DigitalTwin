"""Twin-assisted MPC controller (Stage D).

Constrained model-predictive allocation with receding horizon:
each decision epoch, the action-conditioned predictor scores candidate
budget vectors over a 6-epoch horizon; the best first action is applied,
then we replan. Falls back to the reactive policy when uncertain.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from ..learning.dataset_v2 import H, F, _log1p
from ..learning.models_v2 import ActionPredictor
from ..sim.baseline import BaselineSimulator

KPI_KEYS = ["served", "viol", "offered", "backlog", "delay_mean_slots",
            "prb_used"]


class TwinMPC:
    def __init__(self, scenario, n_ues: int = 60, n_slots: int = 1000,
                 seed: int = 0, ckpt: str = "checkpoints/predictor_v2.pt",
                 burst: tuple | None = None,
                 lam: float = 4.0, kappa: float = 0.5,
                 sd_fallback: float = 6.3, tilt: float = 0.15):
        """sd_fallback: p90 of per-sample mean-sd on the cal layouts —
        fall back to reactive only on unusually uncertain forecasts."""
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model_kind = ck.get("model_kind", "v2")
        if self.model_kind == "film":
            from ..learning.models_v3 import FiLMPredictor
            self.model = FiLMPredictor(ck["hist_dim"], ck["n_slices"])
        elif self.model_kind == "structured":
            from ..learning.models_structured import StructuredPredictor
            self.model = StructuredPredictor(ck["hist_dim"], ck["n_slices"])
            self.model.set_action_stats(ck["stats"]["a_mu"],
                                        ck["stats"]["a_sd"])
        else:
            self.model = ActionPredictor(ck["hist_dim"], ck["n_slices"])
        self.model.load_state_dict(ck["state"])
        self.model.eval()
        self.stats = ck["stats"]
        self.sigma_scale = ck.get("sigma_scale", 1.0)
        self.unc_fallback_thr = ck.get("fallback_thr", None)
        self.scenario = scenario
        self.seed = seed
        self.sim = BaselineSimulator(scenario, n_ues=n_ues, n_slots=n_slots,
                                     seed=seed, budget_policy="external")
        self.snaps = []
        self.sim.recorder = lambda ep, snap: self.snaps.append(snap)
        self.burst = burst
        self.lam, self.kappa = lam, kappa
        self.sd_fallback = sd_fallback
        self.tilt = tilt
        self.C = self.sim.n_cells
        self.S = len(self.sim._slice_names)
        self.names = self.sim._slice_names
        self.decision_ms = []
        self.fallbacks = 0
        self.decisions = 0
        self._static = self._build_static()

    def _build_static(self):
        cells = self.scenario.cells
        macro_sites = {c.site_id for c in cells if c.cell_type == "macro"}
        st = np.zeros((self.C, 8), np.float32)
        for ci, c in enumerate(cells):
            st[ci, 0] = len(c.carriers)
            st[ci, 1] = sum(x.bandwidth_mhz for x in c.carriers)
            # load_mean filled per decision from running history
            st[ci, 3] = self.seed % 2
            t = {"macro": 0, "pico": 1, "femto": 2}[c.cell_type]
            st[ci, 4 + t] = 1.0
            st[ci, 7] = 1.0 if c.site_id in macro_sites else 0.0
        return st

    def _hist(self, t):
        """History features for decision at epoch t: outcomes of t-H..t-1."""
        feats = np.zeros((self.C * self.S, H,
                          6 + 8 + self.S), np.float32)
        load = np.zeros(self.C, np.float32)
        for hi, e in enumerate(range(t - H + 1, t + 1)):
            s = self.snaps[e]
            k = np.stack([_log1p(s[k_]) for k_ in KPI_KEYS], -1)  # (C,S,6)
            load += s["offered"].mean(-1)
            for si in range(self.S):
                oh = np.zeros(self.S, np.float32)
                oh[si] = 1.0
                base = np.concatenate([k[:, si, :], self._static,
                                       np.tile(oh, (self.C, 1))], -1)
                feats[si::self.S, hi, :] = base
        load /= H
        st = self._static.copy()
        st[:, 2] = _log1p(load)
        for si in range(self.S):
            feats[si::self.S, :, 6:14] = st[:, None, :]
        mu, sd = self.stats["h_mu"], self.stats["h_sd"]
        return torch.from_numpy((feats - mu) / sd).float()

    def _candidates(self, ci, reactive_shares, demand_w):
        """Share vectors. Lesson from the oracle test: +-15% tilts around
        reactive move true 6-epoch outcomes by <1% -- below the model's
        noise floor. Candidates must span BOLD reallocations so the
        predictor can actually discriminate between them."""
        mins = np.array([self.sim.budgets[ci].min_share[n]
                         for n in self.names])
        S = self.S
        cands = [reactive_shares.copy()]
        # timid pair tilts (local refinement)
        for a in range(S):
            for b in range(S):
                if a == b:
                    continue
                v = reactive_shares.copy()
                move = min(self.tilt, v[b] - mins[b])
                if move > 1e-3:
                    v[b] -= move
                    v[a] += move
                    cands.append(v)
        # bold: 60% to one slice, rest by min-share proportions
        for s in range(S):
            v = mins.copy()
            flex = max(0.0, 1.0 - mins.sum())
            v[s] += flex * 0.8
            rest = [i for i in range(S) if i != s]
            w = mins[rest] / (mins[rest].sum() + 1e-9)
            v[rest] += flex * 0.2 * w
            cands.append(v / v.sum())
        # all-min (most conservative) and equal split
        cands.append(mins / mins.sum())
        cands.append(np.ones(S) / S)
        # demand-proportional
        flex = max(0.0, 1.0 - mins.sum())
        w = demand_w / (demand_w.sum() + 1e-9)
        cands.append(mins + flex * w)
        # normalize: floor min shares, rescale
        out = []
        for v in cands:
            v = np.maximum(v, mins)
            v = v / v.sum() * min(1.0, v.sum())
            out.append(v)
        return out

    def _decide(self, epoch):
        t0 = time.perf_counter()
        self.sim._update_budgets()  # reactive baseline -> candidate 0
        r_shares = np.array([[self.sim.budgets[ci].budgets[n] /
                              self.sim.cell_total_prbs[ci]
                              for n in self.names]
                             for ci in range(self.C)])
        last = self.snaps[-1]
        demand = last["backlog"] + last["offered"] + 1e-9  # (C, S)
        xh = self._hist(epoch)                            # (C*S, H, Ft)
        per_cell = []
        for ci in range(self.C):
            cands = self._candidates(ci, r_shares[ci], demand[ci])
            per_cell.append(cands)
        xa_list, xh_rows, idx, xa_raw_list = [], [], [], []
        for ci, cands in enumerate(per_cell):
            tot = self.sim.cell_total_prbs[ci]
            for k, sh in enumerate(cands):
                plan = np.tile((sh * tot)[None, :], (F, 1))  # (F, S) PRBs
                for si in range(self.S):
                    xa_list.append(plan.reshape(-1))
                    xa_raw_list.append(plan.reshape(-1))
                    xh_rows.append(ci * self.S + si)
                    idx.append((ci, si, k))
        xa = torch.from_numpy(
            (np.stack(xa_list) - self.stats["a_mu"]) / self.stats["a_sd"]
            ).float()
        xa_raw = torch.from_numpy(np.stack(xa_raw_list)).float()
        xh_rep = xh[torch.tensor(xh_rows)]
        st = self.stats
        with torch.no_grad():
            if self.model_kind == "structured":
                served, viol, dlv, _, _ = self.model(xh_rep, xa_raw)
                sds = torch.exp(0.5 * dlv) * st["y_sd"][0]
            else:
                mu, lv = self.model(xh_rep, xa)
                served = mu[:, :, 0] * st["y_sd"][0] + st["y_mu"][0]
                viol = mu[:, :, 1] * st["y_sd"][1] + st["y_mu"][1]
                sds = (torch.exp(0.5 * lv[:, :, 0]) * st["y_sd"][0] *
                       self.sigma_scale)
        scr = (served - self.lam * viol - self.kappa * sds).sum(-1)
        # regroup per (cell, candidate); K varies per cell
        pos_of = {}
        for p, (ci, si, k) in enumerate(idx):
            pos_of.setdefault((ci, k), []).append(p)
        best = np.zeros((self.C, self.S))
        for ci in range(self.C):
            ks = sorted({k for (c, k) in pos_of if c == ci})
            cand_pos = [k for k in range(len(per_cell[ci]))]
            scores, sdmeans = [], []
            for k in cand_pos:
                pp = pos_of[(ci, k)]
                scores.append(scr[pp].mean().item())
                sdmeans.append(sds[pp].mean().item())
            kstar = int(np.argmax(scores))
            self.decisions += 1
            thr = (self.unc_fallback_thr if self.unc_fallback_thr is not None
                   else self.sd_fallback)
            if sdmeans[kstar] > thr:
                kstar = 0  # uncertain -> reactive fallback
                self.fallbacks += 1
            best[ci] = (per_cell[ci][kstar] *
                        self.sim.cell_total_prbs[ci])
        self.sim.set_budgets(best)
        self.snaps[-1]["budgets"] = best.copy()
        self.decision_ms.append((time.perf_counter() - t0) * 1e3)

    def run(self):
        sim = self.sim
        dt_s = sim.slot_ms / 1e3
        epoch = 0
        for slot in range(sim.n_slots):
            if slot % sim.budget_epoch == 0:
                sim._epoch_snapshot(epoch)
                if self.burst and self.burst[0] <= epoch < self.burst[1]:
                    sim.traffic.scale["eMBB"] = 3.0
                else:
                    sim.traffic.scale["eMBB"] = 1.0
                if epoch < H:
                    sim._update_budgets()
                else:
                    self._decide(epoch)
                epoch += 1
            sim.ues.step(dt_s, sim.slot_ms)
            if slot % sim.coupling_refresh_slots == 0 and slot > 0:
                sim.ues.refresh_coupling(
                    sim.provider.compute(sim.ues.xy, sim.cells))
            for p in sim.traffic.arrivals(slot):
                sim._enqueue(p)
            for p in sim.prach.step(
                    slot, sim.slot_ms,
                    next(x.latency_ms for x in sim.scenario.slices
                         if x.name == "mIoT")):
                sim._enqueue(p)
            sim._schedule(slot)
            sim._expire(slot)
        rep = sim._report()
        rep["mpc"] = {
            "decisions": self.decisions,
            "fallbacks": self.fallbacks,
            "fallback_rate": self.fallbacks / max(1, self.decisions),
            "decision_ms_mean": float(np.mean(self.decision_ms)),
            "decision_ms_p95": float(np.percentile(self.decision_ms, 95)),
        }
        return rep
