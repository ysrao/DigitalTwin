"""Slice PRB budgets with reactive borrowing (the "strong reactive" baseline).

Per cell: each slice holds guaranteed PRBs = total * min_share. Leftover
capacity plus unused guarantees form a shared pool, redistributed every
budget_decision_ms proportionally to slice backlog (in bits). Borrowing is
the reactive allocation policy that Stage D's twin-assisted allocation must
beat. Invariant: sum of budgets <= total usable PRBs (checked by tests).
"""
from __future__ import annotations

import numpy as np


class SliceBudgets:
    def __init__(self, slices: list, total_prbs: float, epoch_ms: float = 10.0,
                 borrowing: bool = True):
        self.names = [s.name for s in slices]
        self.min_share = {s.name: s.min_share for s in slices}
        # latency-critical slices (deadline <= 2 epochs) keep their guarantee
        # even when idle: lending it would starve bursty arrivals mid-epoch
        self.hold = {s.name: s.latency_ms <= 2 * epoch_ms for s in slices}
        self.total = float(total_prbs)
        self.budgets = {n: self.total * self.min_share[n] for n in self.names}
        self.borrowed = {n: 0.0 for n in self.names}  # cumulative, for KPIs
        self.borrowing = borrowing  # False: strict guaranteed shares only

    def update(self, backlog_bits: dict) -> dict:
        """Redistribute. backlog_bits: slice -> bits waiting. Returns budgets."""
        guaranteed = {n: self.total * self.min_share[n] for n in self.names}
        if not self.borrowing:
            # ablation: hard partition, no pool redistribution
            self.budgets = dict(guaranteed)
            return self.budgets
        # slices with no backlog lend their whole guarantee to the pool,
        # unless latency-critical (held)
        pool = self.total - sum(guaranteed.values())
        demand = {}
        for n in self.names:
            if backlog_bits.get(n, 0.0) <= 0 and not self.hold[n]:
                pool += guaranteed[n]
                guaranteed[n] = 0.0
                demand[n] = 0.0
            else:
                # rough PRB need: assume 2 bps/Hz avg, 12 subcarriers/PRB
                need = backlog_bits[n] / (2.0 * 12.0)
                demand[n] = max(0.0, need - guaranteed[n])
        tot_demand = sum(demand.values())
        budgets = dict(guaranteed)
        if tot_demand > 0 and pool > 0:
            for n in self.names:
                extra = pool * demand[n] / tot_demand
                budgets[n] += extra
                self.borrowed[n] += extra
        # hard invariant
        s = sum(budgets.values())
        assert s <= self.total + 1e-6, f"budget overflow: {s} > {self.total}"
        self.budgets = budgets
        return budgets
