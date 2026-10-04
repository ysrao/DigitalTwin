"""Per-slice traffic generators.

Packets are dicts: {ue, slice, size_bits, arrival_slot, deadline_slot}.
mIoT payload enters via the PRACH model, not directly.
"""
from __future__ import annotations

import numpy as np


class TrafficGen:
    def __init__(self, scenario, n_ues: int, rng: np.random.Generator,
                 slot_ms: float, slice_of_ue: list, trace_csv: str | None = None,
                 n_slots: int = 1000):
        self.scenario = scenario
        self.n_ues = n_ues
        self.rng = rng
        self.slot_ms = slot_ms
        self.slice_of_ue = slice_of_ue
        # mean inter-arrival (slots) per slice — demo-scale rates
        self.rates = {"eMBB": 40.0, "URLLC": 200.0}
        self.sizes = {"eMBB": 12000, "URLLC": 256}  # bits, mean (exponential)
        self.scale = {"eMBB": 1.0, "URLLC": 1.0}  # burst-test multipliers
        # trace-driven (temporally correlated) mode: CSV with
        # timestamp,active_ues columns; diurnal load multiplier normalized
        # to mean 1.0, mapped uniformly over the run's slots
        self.trace_mult = None
        if trace_csv:
            import csv as _csv
            ues = []
            with open(trace_csv, newline="") as f:
                for row in _csv.DictReader(f):
                    ues.append(float(row["active_ues"]))
            mean = sum(ues) / len(ues)
            self.trace_mult = [u / mean for u in ues]
            self._n_trace = len(ues)
            self._n_slots = n_slots

    def _load_multiplier(self, slot: int) -> float:
        if self.trace_mult is None:
            return 1.0
        idx = min(int(slot / self._n_slots * self._n_trace), self._n_trace - 1)
        return self.trace_mult[idx]

    def arrivals(self, slot: int) -> list:
        pkts = []
        mult = self._load_multiplier(slot)
        for u in range(self.n_ues):
            sl = self.slice_of_ue[u]
            if sl == "mIoT":
                continue  # via PRACH
            if self.rng.random() < mult * self.scale[sl] / self.rates[sl]:
                size = int(self.rng.exponential(self.sizes[sl])) + 1
                deadline_ms = next(
                    x.latency_ms for x in self.scenario.slices if x.name == sl)
                pkts.append({
                    "ue": u, "slice": sl, "size_bits": size,
                    "arrival_slot": slot,
                    "deadline_slot": slot + max(
                        1, int(round(deadline_ms / self.slot_ms))),
                })
        return pkts
