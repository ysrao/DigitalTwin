"""A3/A5 event-triggered handover and 6G multi-access steering.

Handover is modelled as the actual 3GPP measurement-event state machine rather
than as a per-step re-association, because the quantities the operator cares
about — handover success rate, ping-pong rate, radio link failures, session
continuity — only exist if a handover has a duration, a trigger condition that
must hold for a time-to-trigger, and a chance of failing.

Events implemented (TS 38.331 §5.5.4):

    A3   neighbour becomes offset better than serving (the mobility workhorse)
    A5   serving becomes worse than threshold1 AND neighbour better than
         threshold2 (the coverage-driven escape, and how sessions fall back to
         the satellite tier when terrestrial coverage runs out)

On top of the events sits the AI-enabled 6G access-steering rule: which tiers a
session is even allowed to consider. Indoor sessions prefer Wi-Fi and indoor
cellular; outdoor sessions moving too fast for a tier's beam tracking, or with
no terrestrial tier in reach at all, are steered to the LEO overlay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from mobility import SessionType, SessionPopulation


# --------------------------------------------------------------------------
# Steering policy
# --------------------------------------------------------------------------

@dataclass
class SteeringPolicy:
    """Which access tiers a session may use, before radio conditions are considered.

    This is the 6G access-selection layer: an indoor UE should be on Wi-Fi or an
    indoor small cell rather than burning macro PRBs, and an outdoor UE that is
    either moving too fast for the terrestrial beam tiers or has no terrestrial
    coverage at all belongs on the satellite.
    """

    prefer_wifi_indoor: bool = True
    # Outdoor UEs above this speed are steered off the beam-tracking tiers.
    ntn_speed_threshold_kmh: float = 120.0
    # If the best terrestrial RSRP is below this, treat it as a coverage hole.
    ntn_coverage_hole_dbm: float = -118.0
    # Bias applied to Wi-Fi RSRP for indoor UEs, in dB, to express the preference.
    wifi_indoor_bias_db: float = 8.0
    ntn_bias_db: float = 0.0

    def eligibility(self, indoor: np.ndarray, speed_kmh: np.ndarray,
                    cell_indoor_only: np.ndarray, cell_outdoor_only: np.ndarray,
                    cell_is_ntn: np.ndarray, cell_is_wifi: np.ndarray,
                    cell_max_speed_kmh: np.ndarray,
                    best_terrestrial_rsrp_dbm: np.ndarray) -> np.ndarray:
        """Boolean (sessions x cells) mask of permitted associations."""
        n, m = indoor.shape[0], cell_is_ntn.shape[0]
        allowed = np.ones((n, m), dtype=bool)

        # Indoor-only cells (Wi-Fi APs, indoor small cells) serve indoor UEs only,
        # and outdoor-only cells (the satellite beam) serve outdoor UEs only.
        allowed &= ~(cell_indoor_only[None, :] & ~indoor[:, None])
        allowed &= ~(cell_outdoor_only[None, :] & indoor[:, None])

        # A tier cannot hold a session moving faster than it can track.
        allowed &= speed_kmh[:, None] <= cell_max_speed_kmh[None, :]

        # The satellite is reserved for the two cases that justify its latency:
        # outdoor high mobility, or no terrestrial coverage worth having.
        if cell_is_ntn.any():
            needs_ntn = (~indoor) & (
                (speed_kmh > self.ntn_speed_threshold_kmh)
                | (best_terrestrial_rsrp_dbm < self.ntn_coverage_hole_dbm))
            allowed[:, cell_is_ntn] &= needs_ntn[:, None]
        return allowed

    def rsrp_bias_db(self, indoor: np.ndarray, cell_is_wifi: np.ndarray,
                     cell_is_ntn: np.ndarray) -> np.ndarray:
        """Per-(session, cell) association bias expressing the steering preference."""
        n, m = indoor.shape[0], cell_is_wifi.shape[0]
        bias = np.zeros((n, m))
        if self.prefer_wifi_indoor and cell_is_wifi.any():
            bias[:, cell_is_wifi] += self.wifi_indoor_bias_db * indoor[:, None]
        if cell_is_ntn.any():
            bias[:, cell_is_ntn] += self.ntn_bias_db
        return bias


# --------------------------------------------------------------------------
# Handover engine
# --------------------------------------------------------------------------

@dataclass
class HandoverCounters:
    attempts: int = 0
    successes: int = 0
    failures_late: int = 0        # target degraded before execution completed
    failures_admission: int = 0   # target refused for lack of capacity
    failures_speed: int = 0       # UE too fast for the target tier
    ping_pongs: int = 0
    radio_link_failures: int = 0
    a3_triggers: int = 0
    a5_triggers: int = 0
    interruption_ms_total: float = 0.0
    continuity_breaks: int = 0    # interruption exceeded the class's tolerance

    def as_dict(self) -> Dict[str, float]:
        attempts = max(1, self.attempts)
        return {
            "attempts": self.attempts,
            "successes": self.successes,
            "success_rate": self.successes / attempts,
            "failure_rate": (self.attempts - self.successes) / attempts,
            "failures_late": self.failures_late,
            "failures_admission": self.failures_admission,
            "failures_speed": self.failures_speed,
            "ping_pongs": self.ping_pongs,
            "ping_pong_rate": self.ping_pongs / attempts,
            "radio_link_failures": self.radio_link_failures,
            "a3_triggers": self.a3_triggers,
            "a5_triggers": self.a5_triggers,
            "mean_interruption_ms": self.interruption_ms_total / max(1, self.successes),
            "continuity_breaks": self.continuity_breaks,
        }


class HandoverEngine:
    """A3/A5 state machine over a session population.

    Holds one time-to-trigger timer per session and the identity of the
    candidate it is timing, so a candidate that stops satisfying the entry
    condition resets the timer exactly as a real UE would.
    """

    def __init__(self, population: SessionPopulation, session_types: Sequence[SessionType],
                 n_cells: int, steering: Optional[SteeringPolicy] = None,
                 rng: Optional[np.random.Generator] = None,
                 execution_time_ms: float = 40.0,
                 ping_pong_window_s: float = 5.0):
        self.pop = population
        self.types = list(session_types)
        self.n_cells = n_cells
        self.steering = steering or SteeringPolicy()
        self.rng = rng or np.random.default_rng(0)
        self.execution_time_ms = execution_time_ms
        self.ping_pong_window_s = ping_pong_window_s

        n = population.n
        self.serving = np.full(n, -1, dtype=int)      # -1 = unattached
        self.previous = np.full(n, -1, dtype=int)
        self.time_since_handover_s = np.full(n, 1e9)
        self.ttt_timer_ms = np.zeros(n)
        self.ttt_candidate = np.full(n, -1, dtype=int)
        self.out_of_service_ms = np.zeros(n)
        self.counters = HandoverCounters()
        self.per_type = {t.name: HandoverCounters() for t in self.types}

        # Per-session A3/A5 parameters, gathered from the session type table.
        self.a3_offset = self._by_type("a3_offset_db")
        self.a3_hyst = self._by_type("a3_hysteresis_db")
        self.ttt_ms = self._by_type("a3_time_to_trigger_ms")
        self.a5_t1 = self._by_type("a5_threshold1_dbm")
        self.a5_t2 = self._by_type("a5_threshold2_dbm")
        self.interrupt_tolerance_ms = self._by_type("handover_interrupt_ms")
        self.min_rsrp = self._by_type("min_rsrp_dbm")

    def _by_type(self, attr: str) -> np.ndarray:
        values = np.array([getattr(t, attr) for t in self.types], dtype=float)
        return values[self.pop.type_index]

    # -- parameter override, for the threshold optimizer --------------------

    def set_thresholds(self, per_type: Dict[str, Dict[str, float]]) -> None:
        """Override A3/A5 parameters per session type (used by the optimizer)."""
        names = [t.name for t in self.types]
        for type_name, params in per_type.items():
            if type_name not in names:
                continue
            sel = self.pop.type_index == names.index(type_name)
            for key, target in (("a3_offset_db", self.a3_offset),
                                ("a3_hysteresis_db", self.a3_hyst),
                                ("a3_time_to_trigger_ms", self.ttt_ms),
                                ("a5_threshold1_dbm", self.a5_t1),
                                ("a5_threshold2_dbm", self.a5_t2)):
                if key in params:
                    target[sel] = float(params[key])

    # -- one measurement tick ----------------------------------------------

    def tick(self, rsrp_dbm: np.ndarray, allowed: np.ndarray, dt_ms: float,
             target_load: np.ndarray, cell_max_speed_kmh: np.ndarray) -> None:
        """Advance the A3/A5 state machine by one measurement period.

        `rsrp_dbm` is (sessions x cells) including the steering bias; `allowed`
        is the eligibility mask; `target_load` is per-cell utilization used for
        admission control.
        """
        n = self.pop.n
        masked = np.where(allowed, rsrp_dbm, -np.inf)

        # Attach anything unattached to its best permitted cell.
        detached = self.serving < 0
        if detached.any():
            best = np.argmax(masked[detached], axis=1)
            feasible = np.isfinite(masked[detached, best])
            idx = np.flatnonzero(detached)[feasible]
            self.serving[idx] = best[feasible]

        attached = self.serving >= 0
        serving_rsrp = np.full(n, -np.inf)
        if attached.any():
            serving_rsrp[attached] = rsrp_dbm[attached, self.serving[attached]]
            # A cell that has become ineligible (UE went indoors, sped up, the
            # satellite set) cannot keep the session.
            lost = attached & ~allowed[np.arange(n), np.maximum(self.serving, 0)]
            serving_rsrp[lost] = -np.inf

        # Radio link failure: serving level below the class's minimum.
        rlf = attached & (serving_rsrp < self.min_rsrp)
        if rlf.any():
            self.counters.radio_link_failures += int(rlf.sum())
            for i, t in enumerate(self.types):
                sel = rlf & (self.pop.type_index == i)
                self.per_type[t.name].radio_link_failures += int(sel.sum())
            self.serving[rlf] = -1
            self.out_of_service_ms[rlf] += dt_ms

        # Best permitted neighbour that is not the serving cell.
        neighbour_view = masked.copy()
        rows = np.arange(n)
        safe_serving = np.maximum(self.serving, 0)
        neighbour_view[rows, safe_serving] = -np.inf
        best_neighbour = np.argmax(neighbour_view, axis=1)
        best_neighbour_rsrp = neighbour_view[rows, best_neighbour]

        # --- entry conditions ---
        # A3: Mn - Hys > Ms + Off
        a3 = best_neighbour_rsrp - self.a3_hyst > serving_rsrp + self.a3_offset
        # A5: Ms + Hys < Thresh1 AND Mn - Hys > Thresh2
        a5 = ((serving_rsrp + self.a3_hyst < self.a5_t1)
              & (best_neighbour_rsrp - self.a3_hyst > self.a5_t2))
        entering = (a3 | a5) & np.isfinite(best_neighbour_rsrp) & (self.serving >= 0)

        # --- time-to-trigger ---
        # The timer only accumulates while the same candidate keeps satisfying
        # the condition; a change of candidate restarts it.
        same_candidate = self.ttt_candidate == best_neighbour
        self.ttt_timer_ms = np.where(entering & same_candidate,
                                     self.ttt_timer_ms + dt_ms,
                                     np.where(entering, dt_ms, 0.0))
        self.ttt_candidate = np.where(entering, best_neighbour, -1)

        fired = entering & (self.ttt_timer_ms >= self.ttt_ms)
        if fired.any():
            self.counters.a3_triggers += int((fired & a3).sum())
            self.counters.a5_triggers += int((fired & a5 & ~a3).sum())
            self._execute(fired, best_neighbour, best_neighbour_rsrp, serving_rsrp,
                          target_load, cell_max_speed_kmh)

        self.time_since_handover_s += dt_ms / 1000.0

    # -- execution ---------------------------------------------------------

    def _execute(self, fired: np.ndarray, target: np.ndarray,
                 target_rsrp: np.ndarray, serving_rsrp: np.ndarray,
                 target_load: np.ndarray, cell_max_speed_kmh: np.ndarray) -> None:
        """Attempt the handovers that fired, and account for how they ended."""
        idx = np.flatnonzero(fired)
        if idx.size == 0:
            return
        tgt = target[idx]
        self.counters.attempts += idx.size
        for i, t in enumerate(self.types):
            self.per_type[t.name].attempts += int((self.pop.type_index[idx] == i).sum())

        # Speed check: the target tier must be able to track this UE.
        too_fast = self.pop.speed_kmh[idx] > cell_max_speed_kmh[tgt]

        # Admission control: a target already at capacity refuses the handover.
        refused = (~too_fast) & (self.rng.random(idx.size) < np.clip(
            (target_load[tgt] - 0.9) / 0.1, 0, 1))

        # Late handover: during the execution interruption the UE keeps moving,
        # so a marginal trigger can decay below the usable level before the
        # handover completes. Higher speed and a thinner margin both hurt.
        margin_db = target_rsrp[idx] - serving_rsrp[idx]
        speed_factor = np.clip(self.pop.speed_kmh[idx] / 120.0, 0, 3)
        p_late = np.clip(0.01 + 0.10 * speed_factor - 0.01 * margin_db, 0.0, 0.6)
        late = (~too_fast) & (~refused) & (self.rng.random(idx.size) < p_late)

        ok = ~(too_fast | refused | late)

        # Interruption grows with the target's distance in the tier hierarchy;
        # a satellite target costs far more than a neighbouring macro cell.
        interruption = self.execution_time_ms * (
            1.0 + 2.0 * (cell_max_speed_kmh[tgt] > 1e5))
        broke = ok & (interruption > self.interrupt_tolerance_ms[idx])

        for local, session in enumerate(idx):
            type_name = self.types[self.pop.type_index[session]].name
            counters = self.per_type[type_name]
            if too_fast[local]:
                self.counters.failures_speed += 1
                counters.failures_speed += 1
                continue
            if refused[local]:
                self.counters.failures_admission += 1
                counters.failures_admission += 1
                continue
            if late[local]:
                self.counters.failures_late += 1
                counters.failures_late += 1
                # A late handover drops the session back to unattached.
                self.serving[session] = -1
                continue

            new_cell = int(tgt[local])
            # Ping-pong: straight back to the cell we just left, inside the window.
            if (new_cell == self.previous[session]
                    and self.time_since_handover_s[session] < self.ping_pong_window_s):
                self.counters.ping_pongs += 1
                counters.ping_pongs += 1

            self.previous[session] = self.serving[session]
            self.serving[session] = new_cell
            self.time_since_handover_s[session] = 0.0
            self.ttt_timer_ms[session] = 0.0
            self.ttt_candidate[session] = -1

            self.counters.successes += 1
            counters.successes += 1
            self.counters.interruption_ms_total += float(interruption[local])
            counters.interruption_ms_total += float(interruption[local])
            if broke[local]:
                self.counters.continuity_breaks += 1
                counters.continuity_breaks += 1

    # -- reporting ---------------------------------------------------------

    def session_continuity(self) -> Dict[str, float]:
        """Fraction of sessions currently attached, overall and per class."""
        out = {"overall": float((self.serving >= 0).mean())}
        for i, t in enumerate(self.types):
            sel = self.pop.type_index == i
            if sel.any():
                out[t.name] = float((self.serving[sel] >= 0).mean())
        return out

    def report(self) -> Dict:
        return {
            "overall": self.counters.as_dict(),
            "by_session_type": {k: v.as_dict() for k, v in self.per_type.items()},
            "session_continuity": self.session_continuity(),
        }
