"""Stage B reference baseline simulator.

Slot-by-slot downlink simulation over the HetNet layout:
mobility + A3 handover, Sionna-derived coupling loss (refreshed periodically),
per-slice queues, slice PRB budgets with reactive borrowing, PRACH for mIoT
access, and proportional-fair scheduling within each slice budget.

Abstractions (documented, per the brief's honesty rule):
- Sionna gives wideband coupling loss per UT-cell pair; per-slot SINR adds
  thermal noise and full (all-cells-transmitting) interference analytically.
- Spectral efficiency via Shannon with a 3 dB gap, capped at 7.5 bps/Hz.
- TDD duty cycle is folded into the average DL PRB budget (numerology.py).
- A packet not finished by its deadline is dropped and counted as a
  deadline violation (never hidden in an average).
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from ..scenarios.loader import Scenario, load
from .hetnet import build_layout
from .numerology import slot_duration_ms, scs_khz
from .channel import CouplingProvider, _TIER_SCENARIO, tier_scenario
from .mobility import UEs
from .traffic import TrafficGen
from .prach import Prach
from .slicing import SliceBudgets

_NOISE_FIGURE_DB = 7.0
_THERMAL_DBM_HZ = -174.0
_SE_CAP = 7.5
_GAP_LIN = 2.0  # ~3 dB Shannon gap


class BaselineSimulator:
    def __init__(self, scenario: Scenario, n_ues: int = 120,
                 n_slots: int = 2000, seed: int = 0,
                 coupling_refresh_slots: int = 400,
                 slice_mix: dict | None = None,
                 budget_policy: str = "reactive",
                 borrowing: bool = True,
                 trace_csv: str | None = None):
        """budget_policy: 'reactive' (Stage B baseline), 'random' (corpus
        exploration, 50/50), 'explore' (corpus v3, 100% randomized budgets),
        or 'external' (MPC sets budgets via set_budgets)."""
        self.scenario = scenario
        self._sla_target = {s.name: float(s.reliability)
                            for s in scenario.slices}
        self.rng = np.random.default_rng(seed)
        self.n_slots = n_slots
        self.slot_ms = slot_duration_ms(scenario.carriers[0].numerology)
        self.scs_hz = scs_khz(scenario.carriers[0].numerology) * 1e3

        self.cells = build_layout(scenario, self.rng)
        self.n_cells = len(self.cells)
        self.tx_pwr = np.array([c["tx_power_dbm"] for c in self.cells])
        self.cell_total_prbs = np.array([c["total_dl_prbs"] for c in self.cells])

        # --- UEs: positions, then per-carrier Sionna coupling ---
        # coupling shape: (n_carrier, n_ut, n_cell). Cell selection, mobility
        # and handover camp on the anchor carrier (carriers[0], the FR1
        # coverage layer); SINR/scheduling use every carrier at its own
        # frequency (required for honest FR3 pathloss).
        span = scenario.site_spacing_m * 1.2
        ut_xy = self.rng.uniform(-span, span, size=(n_ues, 2))
        self._freqs_hz = [c.freq_mhz * 1e6 for c in scenario.carriers]
        # tier-matched 38.901 scenarios: macro -> UMa, pico/femto -> UMi.
        # One persistent provider per (carrier, tier); seeds offset for
        # deterministic, independent realizations (matched-seed comparisons
        # need identical channels across policies).
        _by_kind = {}
        for i, c in enumerate(self.cells):
            _by_kind.setdefault(tier_scenario(c["type"], scenario.propagation), []).append(i)
        self._tier_groups = list(_by_kind.items())  # [(kind, [cell idx])]
        self._providers = {}
        for k, f in enumerate(self._freqs_hz):
            for t, (kind, _idx) in enumerate(self._tier_groups):
                self._providers[(k, t)] = CouplingProvider(
                    carrier_freq_hz=f, seed=seed + 7919 * k + 131 * t,
                    scenario_kind=kind)
        coupling = self._compute_coupling(ut_xy)
        self._coupling = coupling  # (n_cc, U, C) dB; refreshed periodically
        self.ues = UEs(scenario, self.cells, coupling[0], n_ues, self.rng,
                       xy=ut_xy)
        # per-carrier DL PRB-unit weights for the pooled scheduler
        units = np.array([[cc["dl_prb_units"] for cc in cell["carriers"]]
                          for cell in self.cells])          # (C, n_cc)
        self._cc_w = units / units.sum(axis=1, keepdims=True)  # (C, n_cc)
        self._scs_hz = np.array(
            [scs_khz(c.numerology) * 1e3 for c in scenario.carriers])

        # --- slice per UE ---
        mix = slice_mix or {"eMBB": 0.5, "URLLC": 0.2, "mIoT": 0.3}
        names, probs = zip(*mix.items())
        self.slice_of_ue = list(self.rng.choice(names, size=n_ues,
                                                p=np.array(probs)))
        self.ue_queues = [deque() for _ in range(n_ues)]  # per-UE FIFO
        self.avg_rate = np.ones(n_ues)  # PF averager (bits/slot)

        self.traffic = TrafficGen(scenario, n_ues, self.rng, self.slot_ms,
                                  self.slice_of_ue, trace_csv=trace_csv,
                                  n_slots=n_slots)
        self.prach = Prach(scenario, self.slice_of_ue, self.ues.serving,
                           self.rng)

        self.budgets = [SliceBudgets(scenario.slices, tot,
                                     epoch_ms=scenario.budget_decision_ms,
                                     borrowing=borrowing)
                        for tot in self.cell_total_prbs]
        self.budget_policy = budget_policy
        self.ue_served = np.zeros(n_ues)  # bits served per UE (p5 throughput)
        self.coupling_refresh_slots = coupling_refresh_slots
        self.budget_epoch = max(
            1, int(round(scenario.budget_decision_ms / self.slot_ms)))

        # --- KPI accumulators ---
        self.kpi = {s.name: defaultdict(float) for s in scenario.slices}
        self.delays = {s.name: [] for s in scenario.slices}
        self.prb_trace = []  # (slot, cell, allocated, budget_sum, total)
        self.arrived_bits = 0.0
        self.served_bits = 0.0
        # --- per-epoch per-cell recorder (Stage C corpus) ---
        self.recorder = None  # fn(epoch_idx, snapshot_dict)
        self._slice_names = [s.name for s in scenario.slices]
        self._reset_epoch_acc()

    def _compute_coupling(self, xy: np.ndarray) -> np.ndarray:
        """Assemble (n_cc, n_ut, n_cell) coupling from tier-matched providers.

        Each tier's cells are evaluated under their own 38.901 scenario
        (UMa for macros, UMi for pico/femto); columns are placed back in
        cell order so downstream code is unchanged.
        """
        out = np.empty((len(self._freqs_hz), len(xy), self.n_cells))
        for (k, t), prov in self._providers.items():
            _kind, idx = self._tier_groups[t]
            sub = [self.cells[i] for i in idx]
            out[k][:, idx] = prov.compute(xy, sub)
        return out

    def set_budgets(self, arr: np.ndarray):
        """External budget override, arr shape (n_cells, n_slices) in PRBs."""
        assert arr.shape == (self.n_cells, len(self._slice_names))
        for ci in range(self.n_cells):
            tot = self.cell_total_prbs[ci]
            b = {n: float(arr[ci, si]) for si, n in enumerate(self._slice_names)}
            assert sum(b.values()) <= tot + 1e-6, "external budget overflow"
            self.budgets[ci].budgets = b

    def _random_budgets(self):
        """Exploration policy for corpus v2: Dirichlet around min shares."""
        for ci in range(self.n_cells):
            tot = self.cell_total_prbs[ci]
            mins = np.array([self.budgets[ci].min_share[n]
                             for n in self._slice_names])
            flex = max(0.0, 1.0 - mins.sum())
            w = self.rng.dirichlet(np.ones(len(mins)))
            share = mins + flex * w
            self.budgets[ci].budgets = {
                n: tot * share[si] for si, n in enumerate(self._slice_names)}

    # ---------------- main loop ----------------
    def run(self) -> dict:
        dt_s = self.slot_ms / 1e3
        epoch = 0
        for slot in range(self.n_slots):
            if slot % self.budget_epoch == 0:
                if self.budget_policy == "reactive":
                    self._update_budgets()
                elif self.budget_policy == "random":
                    if self.rng.random() < 0.5:
                        self._update_budgets()
                    else:
                        self._random_budgets()
                elif self.budget_policy == "explore":
                    self._random_budgets()
                # 'external': budgets already set via set_budgets()
                self._epoch_snapshot(epoch)
                epoch += 1
            ho_events = self.ues.step(dt_s, self.slot_ms)
            if slot % self.coupling_refresh_slots == 0 and slot > 0:
                coupling = self._compute_coupling(self.ues.xy)
                self.ues.refresh_coupling(coupling[0])  # anchor carrier
                self._coupling = coupling
            for p in self.traffic.arrivals(slot):
                self._enqueue(p)
            for p in self.prach.step(
                    slot, self.slot_ms,
                    next(x.latency_ms for x in self.scenario.slices
                         if x.name == "mIoT")):
                self._enqueue(p)
            self._schedule(slot)
            self._expire(slot)
        return self._report()

    # ---------------- helpers ----------------
    def _reset_epoch_acc(self):
        self._eacc = {ci: {n: {"offered": 0.0, "served": 0.0, "viol": 0,
                               "delay_sum": 0.0, "delay_n": 0,
                               "prb": 0.0}
                            for n in self._slice_names}
                      for ci in range(self.n_cells)}

    def _cell_backlog(self):
        """Backlog bits per (cell, slice)."""
        bl = {ci: {n: 0.0 for n in self._slice_names}
              for ci in range(self.n_cells)}
        for u in range(self.ues.n):
            if not self.ue_queues[u]:
                continue
            ci = int(self.ues.serving[u])
            sl = self.slice_of_ue[u]
            bl[ci][sl] += sum(p["remaining_bits"] for p in self.ue_queues[u])
        return bl

    def _epoch_snapshot(self, epoch: int):
        import numpy as np
        C, S = self.n_cells, len(self._slice_names)
        snap = {}
        for key, fld in (("offered", "offered"), ("served", "served"),
                         ("viol", "viol"), ("prb_used", "prb")):
            arr = np.zeros((C, S))
            for ci in range(C):
                for si, n in enumerate(self._slice_names):
                    arr[ci, si] = self._eacc[ci][n][fld]
            snap[key] = arr
        dm = np.zeros((C, S))
        for ci in range(C):
            for si, n in enumerate(self._slice_names):
                a = self._eacc[ci][n]
                dm[ci, si] = a["delay_sum"] / a["delay_n"] if a["delay_n"] else 0.0
        snap["delay_mean_slots"] = dm
        bl = self._cell_backlog()
        snap["backlog"] = np.array(
            [[bl[ci][n] for n in self._slice_names] for ci in range(C)])
        snap["budgets"] = np.array(
            [[self.budgets[ci].budgets[n] for n in self._slice_names]
             for ci in range(C)])
        snap["epoch"] = epoch
        if self.recorder is not None:
            self.recorder(epoch, snap)
        self._reset_epoch_acc()

    def _enqueue(self, p: dict):
        self.ue_queues[p["ue"]].append(dict(p, remaining_bits=p["size_bits"]))
        self.arrived_bits += p["size_bits"]
        self.kpi[p["slice"]]["offered_bits"] += p["size_bits"]
        self.kpi[p["slice"]]["packets"] += 1
        ci = int(self.ues.serving[p["ue"]])
        self._eacc[ci][p["slice"]]["offered"] += p["size_bits"]

    def _update_budgets(self):
        bl = self._cell_backlog()
        for ci, cell in enumerate(self.cells):
            self.budgets[ci].update(bl[ci])

    def _sinr_per_ue_carrier(self) -> np.ndarray:
        """SINR (linear) per UE per carrier, shape (U, n_cc).

        Signal and interference in dBm per PRB-slot; interference is
        co-channel within each carrier (every cell carries every carrier).
        Noise scales with each carrier's PRB bandwidth (12 * SCS).
        """
        serving = self.ues.serving  # (U,)
        U = self.ues.n
        out = np.empty((U, len(self._freqs_hz)))
        for k in range(len(self._freqs_hz)):
            rx_dbm = self.tx_pwr[None, :] - self._coupling[k]  # (U, C)
            rx_lin = 10.0 ** ((rx_dbm - 30.0) / 10.0)  # watts-ish
            sig = rx_lin[np.arange(U), serving]
            interf = rx_lin.sum(axis=1) - sig
            noise_w = 10.0 ** ((self._noise_per_prb_dbm(k) - 30.0) / 10.0)
            out[:, k] = sig / (interf + noise_w)
        return out

    def _noise_per_prb_dbm(self, cc: int = 0) -> float:
        bw_hz = 12 * self._scs_hz[cc]
        return _THERMAL_DBM_HZ + 10 * np.log10(bw_hz) + _NOISE_FIGURE_DB

    def _bits_per_prb(self, sinr_lin: np.ndarray) -> np.ndarray:
        """Effective bits per pooled PRB-unit per grid slot, per UE.

        sinr_lin: (U, n_cc). Per-carrier spectral efficiency via truncated
        Shannon, pooled with the serving cell's DL PRB-unit weights. Bits
        per unit = SE * 180: 12 * SCS * slot_duration is numerology-
        invariant over a fixed grid slot (12*30e3*0.5e-3 = 12*60e3*0.25e-3).
        """
        se = np.minimum(np.log2(1.0 + sinr_lin / _GAP_LIN), _SE_CAP)
        w = self._cc_w[self.ues.serving]  # (U, n_cc)
        return (se * w).sum(axis=1) * 180.0

    def _schedule(self, slot: int):
        sinr = self._sinr_per_ue_carrier()  # (U, n_cc)
        bpp = self._bits_per_prb(sinr)
        for ci in range(self.n_cells):
            budgets = self.budgets[ci].budgets
            allocated = 0.0
            for sname in self.budgets[ci].names:
                budget = budgets[sname]
                if budget <= 0:
                    continue
                cands = [u for u in range(self.ues.n)
                         if self.ues.serving[u] == ci
                         and self.slice_of_ue[u] == sname
                         and self.ue_queues[u]]
                if not cands:
                    continue
                # proportional fair
                cands.sort(key=lambda u: bpp[u] / max(self.avg_rate[u], 1e-9),
                           reverse=True)
                used = 0.0
                for u in cands:
                    if used >= budget:
                        break
                    need_bits = sum(p["remaining_bits"]
                                    for p in self.ue_queues[u])
                    prbs = min(budget - used,
                               np.ceil(need_bits / max(bpp[u], 1e-9)))
                    if prbs <= 0:
                        continue
                    served = min(need_bits, prbs * bpp[u])
                    self._serve_ue(u, served, slot)
                    used += prbs
                    self.avg_rate[u] = (0.99 * self.avg_rate[u]
                                        + 0.01 * served)
                allocated += used
                self.kpi[sname]["prb_used"] += used
                self._eacc[ci][sname]["prb"] += used
            total = self.cell_total_prbs[ci]
            self.prb_trace.append(
                (slot, ci, allocated, sum(budgets.values()), total))
            assert allocated <= total + 1e-6, "PRB over-allocation"

    def _serve_ue(self, u: int, bits: float, slot: int):
        q = self.ue_queues[u]
        ci = int(self.ues.serving[u])
        sl = self.slice_of_ue[u]
        left = bits
        while left > 0 and q:
            p = q[0]
            take = min(p["remaining_bits"], left)
            p["remaining_bits"] -= take
            left -= take
            self.served_bits += take
            self.ue_served[u] += take
            self._eacc[ci][sl]["served"] += take
            if p["remaining_bits"] <= 0:
                q.popleft()
                self.delays[p["slice"]].append(slot - p["arrival_slot"])
                self.kpi[p["slice"]]["served_bits"] += p["size_bits"]
                self._eacc[ci][sl]["delay_sum"] += slot - p["arrival_slot"]
                self._eacc[ci][sl]["delay_n"] += 1

    def _expire(self, slot: int):
        for u in range(self.ues.n):
            q = self.ue_queues[u]
            while q and q[0]["deadline_slot"] < slot:
                p = q.popleft()
                self.kpi[p["slice"]]["violations"] += 1
                self.kpi[p["slice"]]["dropped_bits"] += p["remaining_bits"]
                ci = int(self.ues.serving[u])
                self._eacc[ci][p["slice"]]["viol"] += 1

    def queued_bits(self) -> float:
        return sum(p["remaining_bits"] for q in self.ue_queues for p in q)

    def _report(self) -> dict:
        rep = {"slots": self.n_slots, "slot_ms": self.slot_ms,
               "handovers": self.ues.handovers,
               "prach": {"attempts": self.prach.attempts,
                         "collisions": self.prach.collisions,
                         "successes": self.prach.successes,
                         "mean_access_delay_slots": float(np.mean(
                             self.prach.access_delays))
                         if self.prach.access_delays else 0.0}}
        for sname in self.kpi:
            d = np.array(self.delays[sname]) * self.slot_ms if self.delays[sname] else [0]
            npk = int(self.kpi[sname]["packets"])
            nviol = int(self.kpi[sname]["violations"])
            rel = 1.0 - nviol / max(1, npk)  # in-flight packets count as met
            tgt = self._sla_target.get(sname, 0.0)
            rep[sname] = {
                "offered_kb": self.kpi[sname]["offered_bits"] / 1e3,
                "served_kb": self.kpi[sname]["served_bits"] / 1e3,
                "violations": nviol,
                "packets": npk,
                "reliability": round(rel, 5),
                "sla_met": bool(rel >= tgt),
                "dropped_kb": self.kpi[sname]["dropped_bits"] / 1e3,
                "delay_ms_p50": float(np.median(d)),
                "delay_ms_p95": float(np.percentile(d, 95)),
                "prb_used": float(self.kpi[sname]["prb_used"]),
            }
        rep["queue_conservation"] = {
            "arrived_kb": self.arrived_bits / 1e3,
            "served_kb": self.served_bits / 1e3,
            "dropped_kb": sum(self.kpi[s]["dropped_bits"] for s in self.kpi) / 1e3,
            "queued_kb": self.queued_bits() / 1e3,
        }
        dur_s = self.n_slots * self.slot_ms / 1e3
        thr = self.ue_served / 1e3 / dur_s  # kbps per UE
        rep["ue_thr_kbps_p5"] = float(np.percentile(thr, 5))
        rep["ue_thr_kbps_mean"] = float(thr.mean())
        return rep


def demo() -> dict:
    scenario = load("configs/scenario_schema_v1.yaml")
    sim = BaselineSimulator(scenario, n_ues=60, n_slots=1000, seed=7)
    return sim.run()
