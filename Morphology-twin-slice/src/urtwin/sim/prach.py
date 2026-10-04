"""PRACH access model for the mIoT slice.

Slotted access occasions (every `occasion_slots`). Each mIoT UE with a pending
payload attempts with probability p_access, picking one of 64 preambles
uniformly in its serving cell. Collision (same preamble, same cell, same
occasion) -> uniform backoff in [1, max_backoff] occasions. Success -> the
payload packet is admitted to the mIoT queue.

Kept separate from payload allocation per the brief: access success and
access delay are reported independently.
"""
from __future__ import annotations

import numpy as np

N_PREAMBLES = 64


class Prach:
    def __init__(self, scenario, slice_of_ue: list, serving_of_ue: np.ndarray,
                 rng: np.random.Generator,
                 occasion_slots: int = 10, p_access: float = 0.3,
                 max_backoff: int = 8, payload_bits: int = 800):
        self.rng = rng
        self.occasion_slots = occasion_slots
        self.p_access = p_access
        self.max_backoff = max_backoff
        self.payload_bits = payload_bits
        self.miot_ues = [u for u, s in enumerate(slice_of_ue) if s == "mIoT"]
        self.serving_of_ue = serving_of_ue  # live view, updated by simulator
        # per-UE state: 'idle' | attempts remaining backoff | payload ready
        self.backoff = {u: 0 for u in self.miot_ues}
        self.pending_since = {}  # u -> slot of first attempt (access delay)
        self.attempts = 0
        self.collisions = 0
        self.successes = 0
        self.access_delays = []  # slots from first attempt to success

    def step(self, slot: int, slot_ms: float, deadline_ms: float) -> list:
        """Run one slot. Returns admitted payload packets."""
        if slot % self.occasion_slots != 0:
            return []
        # who attempts this occasion
        attempters = []
        for u in self.miot_ues:
            if self.backoff[u] > 0:
                self.backoff[u] -= 1
                continue
            if self.rng.random() < self.p_access:
                attempters.append(u)
                self.pending_since.setdefault(u, slot)
                self.attempts += 1
        # resolve per cell
        admitted = []
        by_cell: dict = {}
        for u in attempters:
            by_cell.setdefault(int(self.serving_of_ue[u]), []).append(u)
        for cell, ues in by_cell.items():
            picks = self.rng.integers(0, N_PREAMBLES, size=len(ues))
            seen = {}
            for u, p in zip(ues, picks):
                seen.setdefault(int(p), []).append(u)
            for p, contenders in seen.items():
                if len(contenders) > 1:
                    self.collisions += len(contenders)
                    for u in contenders:
                        self.backoff[u] = int(
                            self.rng.integers(1, self.max_backoff + 1))
                else:
                    u = contenders[0]
                    self.successes += 1
                    self.access_delays.append(slot - self.pending_since.pop(u))
                    self.backoff[u] = 0
                    admitted.append({
                        "ue": u, "slice": "mIoT",
                        "size_bits": self.payload_bits,
                        "arrival_slot": slot,
                        "deadline_slot": slot + max(
                            1, int(round(deadline_ms / slot_ms))),
                    })
        return admitted
