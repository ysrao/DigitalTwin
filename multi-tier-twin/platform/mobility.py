"""UE sessions, velocity vectors and mobility limits.

The twin carries a population of individual sessions rather than an aggregate
Mbps figure, because handover success, ping-pong and session continuity are not
definable on an aggregate. Each session has a position, a velocity vector, a
session type (eMBB / URLLC / mIoT) with its own 3GPP-shaped QoS and A3/A5
parameters, and a serving cell.

Mobility limits are a first-class input. A tier is not simply "available" to a
fast-moving UE: mmWave beam tracking collapses when the channel coherence time
falls below the beam-refinement period, small cells generate handovers faster
than they can complete them, and mIoT devices are not expected to move at all.
Each layer therefore declares the speed beyond which it derates and the speed
beyond which it cannot hold a session.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import radio

SESSION_TYPES = ("eMBB", "URLLC", "mIoT")
N_TYPES = len(SESSION_TYPES)


# --------------------------------------------------------------------------
# Session types: QoS, mobility tolerance and handover thresholds
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SessionType:
    """One traffic class, with the QoS and handover parameters it is served under.

    A3/A5 parameters differ per class on purpose: URLLC trades ping-pong risk
    for a short time-to-trigger because it cannot absorb a late handover, while
    mIoT uses a long trigger and a large offset because its sessions are
    delay-tolerant and its devices are battery-limited.
    """

    name: str
    bitrate_mbps: float           # per-session offered rate when active
    latency_budget_ms: float      # end-to-end budget (5QI-style)
    reliability_target: float     # per-interval success target
    # Mobility
    max_speed_kmh: float          # above this the class is not offered
    # A3: neighbour becomes offset better than serving
    a3_offset_db: float
    a3_hysteresis_db: float
    a3_time_to_trigger_ms: float
    # A5: serving becomes worse than threshold1 AND neighbour better than threshold2
    a5_threshold1_dbm: float
    a5_threshold2_dbm: float
    # Continuity
    handover_interrupt_ms: float  # interruption this class can absorb
    min_rsrp_dbm: float           # below this the session drops


DEFAULT_SESSION_TYPES: Tuple[SessionType, ...] = (
    SessionType(
        name="eMBB", bitrate_mbps=5.0, latency_budget_ms=100.0,
        reliability_target=0.95, max_speed_kmh=500.0,
        a3_offset_db=3.0, a3_hysteresis_db=2.0, a3_time_to_trigger_ms=320.0,
        a5_threshold1_dbm=-110.0, a5_threshold2_dbm=-105.0,
        handover_interrupt_ms=50.0, min_rsrp_dbm=-124.0,
    ),
    SessionType(
        # Short TTT and a small offset: a late handover is a failed handover.
        name="URLLC", bitrate_mbps=1.0, latency_budget_ms=5.0,
        reliability_target=0.99999, max_speed_kmh=250.0,
        a3_offset_db=1.5, a3_hysteresis_db=1.0, a3_time_to_trigger_ms=40.0,
        a5_threshold1_dbm=-100.0, a5_threshold2_dbm=-98.0,
        handover_interrupt_ms=10.0, min_rsrp_dbm=-115.0,
    ),
    SessionType(
        # Long TTT and a large offset: ping-pong costs battery, latency does not.
        name="mIoT", bitrate_mbps=0.02, latency_budget_ms=1000.0,
        reliability_target=0.99, max_speed_kmh=30.0,
        a3_offset_db=6.0, a3_hysteresis_db=4.0, a3_time_to_trigger_ms=1280.0,
        a5_threshold1_dbm=-118.0, a5_threshold2_dbm=-114.0,
        handover_interrupt_ms=500.0, min_rsrp_dbm=-130.0,
    ),
)

SESSION_TYPE_BY_NAME = {s.name: s for s in DEFAULT_SESSION_TYPES}


# --------------------------------------------------------------------------
# Mobility classes
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MobilityClass:
    name: str
    speed_kmh_mean: float
    speed_kmh_std: float
    # Direction persistence: 1.0 keeps a straight line, 0.0 re-randomizes each step.
    heading_persistence: float = 0.98


MOBILITY_CLASSES: Tuple[MobilityClass, ...] = (
    MobilityClass("stationary", 0.0, 0.0, 1.0),
    MobilityClass("pedestrian", 4.0, 1.2, 0.90),
    MobilityClass("vehicular", 45.0, 12.0, 0.97),
    MobilityClass("highway", 110.0, 15.0, 0.995),
    MobilityClass("high_speed", 300.0, 30.0, 0.999),
)

# Mobility mix per session type: mIoT barely moves, URLLC skews vehicular
# (industrial AGVs, V2X), eMBB is the broadest mix.
MOBILITY_MIX: Dict[str, Tuple[float, ...]] = {
    #            stationary  pedestrian  vehicular  highway  high_speed
    "eMBB":     (0.45,       0.30,       0.18,      0.06,    0.01),
    "URLLC":    (0.30,       0.12,       0.44,      0.13,    0.01),
    "mIoT":     (0.92,       0.08,       0.00,      0.00,    0.00),
}


# --------------------------------------------------------------------------
# Layer mobility limits
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MobilityLimits:
    """Speed beyond which a tier degrades, and beyond which it cannot serve.

    `derate_speed_kmh` is where beam tracking / handover rate starts costing
    throughput; `max_speed_kmh` is where the tier stops being usable at all.
    `beam_refinement_ms` is compared against the channel coherence time: a beam
    that cannot be refined within a coherence time is a beam that is lost.
    """

    derate_speed_kmh: float
    max_speed_kmh: float
    beam_refinement_ms: float = 0.0   # 0 = no beam tracking dependency

    def derate(self, speed_kmh: float, freq_ghz: float) -> float:
        """Throughput retention factor in [0, 1] for a UE at this speed."""
        if speed_kmh >= self.max_speed_kmh:
            return 0.0
        factor = 1.0
        if speed_kmh > self.derate_speed_kmh:
            span = max(1e-6, self.max_speed_kmh - self.derate_speed_kmh)
            factor *= max(0.0, 1.0 - 0.7 * (speed_kmh - self.derate_speed_kmh) / span)
        if self.beam_refinement_ms > 0 and speed_kmh > 0:
            # Beam tracking fails once the coherence time drops below the
            # refinement period; this is what rules mmWave out at speed.
            coherence_ms = radio.coherence_time_s(freq_ghz, speed_kmh) * 1000.0
            if coherence_ms < self.beam_refinement_ms:
                factor *= max(0.05, coherence_ms / self.beam_refinement_ms)
        return float(np.clip(factor, 0.0, 1.0))


# --------------------------------------------------------------------------
# The session population
# --------------------------------------------------------------------------

class SessionPopulation:
    """A population of UE sessions with positions and velocity vectors.

    State is held as parallel NumPy arrays so a few thousand sessions can be
    stepped at one-second resolution without leaving vectorized code.
    """

    def __init__(self, n_sessions: int, area_km: Tuple[float, float],
                 type_mix: Optional[Dict[str, float]] = None,
                 session_types: Sequence[SessionType] = DEFAULT_SESSION_TYPES,
                 rng: Optional[np.random.Generator] = None):
        self.types = list(session_types)
        self.area_km = area_km
        self.rng = rng or np.random.default_rng(0)
        mix = type_mix or {"eMBB": 0.55, "URLLC": 0.15, "mIoT": 0.30}
        weights = np.array([mix.get(t.name, 0.0) for t in self.types], dtype=float)
        weights = weights / max(1e-9, weights.sum())

        self.n = int(n_sessions)
        self.type_index = self.rng.choice(len(self.types), size=self.n, p=weights)

        # Mobility class per session, drawn from its type's mix.
        self.mobility_index = np.zeros(self.n, dtype=int)
        for ti, stype in enumerate(self.types):
            sel = self.type_index == ti
            if not sel.any():
                continue
            probs = np.array(MOBILITY_MIX.get(stype.name, (1.0, 0, 0, 0, 0)), dtype=float)
            probs = probs / max(1e-9, probs.sum())
            self.mobility_index[sel] = self.rng.choice(
                len(MOBILITY_CLASSES), size=int(sel.sum()), p=probs)

        # Speeds, clipped to what each session type actually supports.
        means = np.array([m.speed_kmh_mean for m in MOBILITY_CLASSES])
        stds = np.array([m.speed_kmh_std for m in MOBILITY_CLASSES])
        self.speed_kmh = np.maximum(0.0, self.rng.normal(
            means[self.mobility_index], stds[self.mobility_index]))
        type_max = np.array([t.max_speed_kmh for t in self.types])
        self.speed_kmh = np.minimum(self.speed_kmh, type_max[self.type_index])

        self.heading_rad = self.rng.uniform(0, 2 * math.pi, self.n)
        self.x_km = self.rng.uniform(0, area_km[0], self.n)
        self.y_km = self.rng.uniform(0, area_km[1], self.n)
        self.active = np.ones(self.n, dtype=bool)

    # -- derived views -----------------------------------------------------

    @property
    def velocity_kmh(self) -> np.ndarray:
        """Velocity vectors as an (n, 2) array of km/h components."""
        return np.stack([self.speed_kmh * np.cos(self.heading_rad),
                         self.speed_kmh * np.sin(self.heading_rad)], axis=1)

    def bitrate_mbps(self) -> np.ndarray:
        rates = np.array([t.bitrate_mbps for t in self.types])
        return rates[self.type_index] * self.active

    def type_mask(self, name: str) -> np.ndarray:
        return self.type_index == [t.name for t in self.types].index(name)

    # -- dynamics ----------------------------------------------------------

    def step(self, dt_s: float) -> None:
        """Advance positions along the velocity vectors, reflecting at the edges."""
        persistence = np.array([m.heading_persistence for m in MOBILITY_CLASSES])[
            self.mobility_index]
        # Correlated random walk: heading drifts more for slower, less
        # committed movers and holds nearly straight on a highway.
        self.heading_rad = self.heading_rad + (1 - persistence) * self.rng.normal(
            0, 0.9, self.n)

        step_km = self.speed_kmh * (dt_s / 3600.0)
        self.x_km += step_km * np.cos(self.heading_rad)
        self.y_km += step_km * np.sin(self.heading_rad)

        # Reflect at the service-area boundary and mirror the heading with it.
        for coord, extent, axis in ((self.x_km, self.area_km[0], 0),
                                    (self.y_km, self.area_km[1], 1)):
            low, high = coord < 0, coord > extent
            np.abs(coord, out=coord)
            coord[high] = 2 * extent - coord[high]
            np.clip(coord, 0, extent, out=coord)
            flip = low | high
            if flip.any():
                if axis == 0:
                    self.heading_rad[flip] = math.pi - self.heading_rad[flip]
                else:
                    self.heading_rad[flip] = -self.heading_rad[flip]

    def line_of_sight_angle_deg(self, target_x_km: np.ndarray,
                                target_y_km: np.ndarray) -> np.ndarray:
        """Angle between each velocity vector and the bearing to its serving cell.

        Feeds the UE-motion Doppler term: a UE driving toward its serving cell
        sees the full shift, one crossing it broadside sees almost none.
        """
        bearing = np.arctan2(target_y_km - self.y_km, target_x_km - self.x_km)
        return np.degrees(np.abs(np.arctan2(np.sin(bearing - self.heading_rad),
                                            np.cos(bearing - self.heading_rad))))

    def summary(self) -> Dict:
        by_type = {}
        for ti, t in enumerate(self.types):
            sel = self.type_index == ti
            if not sel.any():
                continue
            by_type[t.name] = {
                "sessions": int(sel.sum()),
                "mean_speed_kmh": round(float(self.speed_kmh[sel].mean()), 2),
                "p95_speed_kmh": round(float(np.percentile(self.speed_kmh[sel], 95)), 2),
                "stationary_fraction": round(float((self.speed_kmh[sel] < 1).mean()), 3),
                "offered_mbps": round(float(self.bitrate_mbps()[sel].sum()), 2),
            }
        return {"total_sessions": self.n, "by_type": by_type}
