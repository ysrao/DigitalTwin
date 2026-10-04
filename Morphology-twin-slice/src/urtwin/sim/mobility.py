"""UE state, mixed mobility and A3 handover.

UEs are dropped uniformly in the layout area, assigned a mobility class
(stationary / pedestrian / vehicular) per the scenario shares, and move with
a random-direction model (direction re-picked on boundary hit). Handover is
A3-style: neighbor's RSRP exceeds serving by hysteresis for time-to-trigger.

RSRP here = tx_power - coupling_loss (dB), per carrier; handover uses the
FDD carrier (coverage layer). CRE bias (dB) can offload to small cells.
"""
from __future__ import annotations

import numpy as np


class UEs:
    def __init__(self, scenario, cells, coupling: np.ndarray,
                 n_ues: int, rng: np.random.Generator, cre_bias_db: float = 6.0,
                 xy: np.ndarray | None = None):
        self.scenario = scenario
        self.cells = cells
        self.cre_bias_db = cre_bias_db
        self.n = n_ues
        self.rng = rng

        span = scenario.site_spacing_m * 1.2
        self.xy = xy if xy is not None else rng.uniform(-span, span,
                                                       size=(n_ues, 2))

        # mobility class per UE from shares
        classes = scenario.mobility_classes
        shares = np.array([c.share for c in classes])
        idx = rng.choice(len(classes), size=n_ues, p=shares / shares.sum())
        self.cls = [classes[i].name for i in idx]
        self.speed = np.array([classes[i].speed_mps for i in idx])
        self.heading = rng.uniform(0, 2 * np.pi, size=n_ues)

        self.coupling = coupling  # (n_ut, n_cell) dB, refreshed externally
        self.serving = np.argmax(self._rsrp(), axis=1)  # RSRP: higher is better
        self._ttt_count = np.zeros(n_ues, dtype=int)
        self._ttt_target = np.full(n_ues, -1, dtype=int)
        self.handovers = 0

    def _rsrp(self) -> np.ndarray:
        tx = np.array([c["tx_power_dbm"] for c in self.cells])  # (C,)
        rsrp = tx[None, :] - self.coupling  # (U, C)
        bias = np.array([self.cre_bias_db if c["type"] != "macro" else 0.0
                         for c in self.cells])
        return rsrp + bias[None, :]

    def step(self, dt_s: float, slot_ms: float):
        """Move UEs; evaluate A3 handover. Returns list of (ue, old, new)."""
        # --- movement ---
        moving = self.speed > 0
        self.xy[moving, 0] += self.speed[moving] * np.cos(self.heading[moving]) * dt_s
        self.xy[moving, 1] += self.speed[moving] * np.sin(self.heading[moving]) * dt_s
        span = self.scenario.site_spacing_m * 1.2
        hit = (np.abs(self.xy) > span).any(axis=1) & moving
        self.heading[hit] = self.rng.uniform(0, 2 * np.pi, size=hit.sum())
        np.clip(self.xy, -span, span, out=self.xy)

        # --- A3 handover ---
        hys = self.scenario.handover.get("hysteresis_db", 3.0)
        ttt_ms = self.scenario.handover.get("time_to_trigger_ms", 160.0)
        ttt_slots = max(1, int(round(ttt_ms / slot_ms)))
        rsrp = self._rsrp()
        serving_rsrp = rsrp[np.arange(self.n), self.serving]
        best_nb = np.argmax(np.where(
            np.arange(len(self.cells))[None, :] == self.serving[:, None],
            -np.inf, rsrp), axis=1)
        best_rsrp = rsrp[np.arange(self.n), best_nb]
        cond = best_rsrp > serving_rsrp + hys

        new_target = np.where(cond, best_nb, -1)
        reset = new_target != self._ttt_target
        self._ttt_count[reset] = 0
        self._ttt_target = new_target
        self._ttt_count[cond] += 1

        fired = cond & (self._ttt_count >= ttt_slots)
        events = []
        for u in np.where(fired)[0]:
            events.append((int(u), int(self.serving[u]), int(best_nb[u])))
            self.serving[u] = best_nb[u]
            self._ttt_count[u] = 0
            self._ttt_target[u] = -1
            self.handovers += 1
        return events

    def refresh_coupling(self, coupling: np.ndarray):
        self.coupling = coupling
