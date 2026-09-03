"""Multi-tier AI-enabled 6G RAN digital twin.

`engine.py` replays a single-cell trace. This module is the network model
underneath it: real cell coordinates and antenna heights, a population of UE
sessions with velocity vectors, 3GPP propagation, and an A3/A5 handover state
machine — so the questions an operator actually asks can be asked of it.

Tier stack (all inputs are configurable; these are the defaults):

    macro_low     0.7 GHz   10 MHz   NR      coverage anchor, high mobility
    macro_mid     3.5 GHz  100 MHz   NR      primary capacity layer
    umb_6g        7.5 GHz  200 MHz   6G      upper mid-band, the 6G workhorse
    mmwave        28  GHz  400 MHz   NR      hotspot capacity, blockage-limited
    wifi7_indoor  6.0 GHz  320 MHz   Wi-Fi 7 indoor offload
    ntn_leo       2.0 GHz   30 MHz   NR-NTN  LEO overlay, 600 km

Access selection is the AI-enabled 6G part: indoor sessions are steered to
Wi-Fi and indoor cellular, and outdoor sessions that are either moving too fast
for the beam-tracking tiers or sitting in a terrestrial coverage hole are
steered to the LEO overlay.

Two timescales, as in the paper: a fast tick (default 100 ms) carries mobility
and the A3/A5 measurement state machine, and a slow control interval (default
10 s) is where the RL policy acts on slice allocation and steering.

Screening-grade model. See LIMITATIONS.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field, asdict, replace
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import radio
from engine import SLICE_NAMES, TEMPLATES
from handover import HandoverEngine, SteeringPolicy
from mobility import (DEFAULT_SESSION_TYPES, MOBILITY_CLASSES, MobilityLimits,
                      SESSION_TYPES, SessionPopulation, SessionType)

N_SLICES = len(SESSION_TYPES)
_TRACE_CSV = Path(__file__).resolve().parents[1] / "synthetic_profiles" / "all_profiles.csv"


@lru_cache(maxsize=8)
def load_traffic_shape(profile: str) -> Optional[np.ndarray]:
    """Per-slice offered-load shape over time, normalised to its own peak.

    Returns a (3, T) array (eMBB, URLLC, mIoT) in [0, ~1] from the platform's
    established trace CSV, or None for "flat" / "diurnal" (handled analytically).
    A carrier trace would replace this file without touching anything else.
    """
    if profile in ("flat", "diurnal") or not _TRACE_CSV.exists():
        return None
    embb, urllc, iot = [], [], []
    with _TRACE_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("profile") != profile:
                continue
            embb.append(float(row.get("embb_offered_dl_mbps", 0) or 0))
            # URLLC and IoT: convert their native units to an offered-Mbps proxy.
            urllc.append(float(row.get("urllc_packets", 0) or 0)
                         * float(row.get("urllc_payload_bytes", 32) or 32) * 8
                         / max(1.0, float(row.get("interval_ms", 10) or 10)) / 1000.0)
            iot.append(float(row.get("iot_active_devices", 0) or 0)
                       * float(row.get("iot_min_rate_kbps_per_device", 0.2) or 0.2) / 1000.0)
    if not embb:
        raise ValueError(f"Unknown traffic profile: {profile}")
    arr = np.array([embb, urllc, iot], dtype=float)
    peak = arr.max(axis=1, keepdims=True)
    return arr / np.maximum(peak, 1e-9)

# Slice weights, SLA thresholds and violation costs, aligned with engine.py.
SLICE_WEIGHTS = np.array([0.27, 0.48, 0.25])
SLA_THRESHOLDS = np.array([0.95, 0.999, 0.99])
VIOLATION_COST = np.array([1.0, 4.0, 1.5])


# ==========================================================================
# Tiers and cells
# ==========================================================================

@dataclass(frozen=True)
class Tier:
    """A radio technology layer. Cells inherit their defaults from their tier."""

    name: str
    rat: str
    band_ghz: float                 # centre frequency of the spectrum block
    bandwidth_mhz: float            # per carrier
    scenario: str                  # TR 38.901 morphology: UMa | UMi | RMa | InH
    tx_power_dbm: float
    antenna_height_m: float
    antenna_gain_dbi: float
    # Spectrum / numerology. prb_bandwidth_mhz and prbs_per_cell are DERIVED
    # from these unless given explicitly (> 0), so changing the bandwidth or
    # the subcarrier spacing recomputes the resource grid automatically.
    numerology_khz: float = 30.0    # SCS: 15 / 30 / 60 / 120
    carriers: int = 1              # carrier-aggregation component carriers
    prb_bandwidth_mhz: float = 0.0  # 0 => 12 * SCS
    prbs_per_cell: int = 0          # 0 => floor(0.95 * BW * carriers / prb_bw)
    guard_fraction: float = 0.05   # spectrum lost to guard bands
    # MIMO. The per-stream spectral efficiency stays Shannon-bounded; MIMO
    # multiplies the cell's delivered capacity by layers * scheduling efficiency.
    mimo_layers: int = 4           # spatial layers (1 = SISO)
    mimo_mode: str = "MU"         # "SU" or "MU"
    mimo_efficiency: float = 0.75  # realised fraction of the ideal MIMO gain
    ue_antenna_gain_dbi: float = 0.0
    noise_figure_db: float = 7.0
    max_se_bps_hz: float = 6.0
    # Cell individual offset (TS 38.331 cellIndividualOffset): the association
    # bias that pulls load off the coverage layer onto the capacity layers.
    cio_db: float = 0.0
    slices: Tuple[str, ...] = SESSION_TYPES
    mobility: MobilityLimits = field(
        default_factory=lambda: MobilityLimits(500.0, 1000.0))
    one_way_latency_ms: float = 1.0
    energy_idle_w: float = 180.0
    energy_max_w: float = 900.0
    backhaul_mbps_per_cell: float = 4_000.0
    indoor_only: bool = False
    outdoor_only: bool = False
    blockage_sensitivity: float = 0.0
    # Co-channel interference coupling: the fraction of a neighbour cell's power
    # that lands as interference. 1.0 = omni reuse-1; beamformed tiers see far
    # less because narrow Tx/Rx beams rarely align (3GPP FD-MIMO / analog beam).
    interference_coupling: float = 1.0
    # TR 38.901 shadow-fading sigma (dB), NLOS-weighted, per morphology.
    shadow_sigma_db: float = 6.0
    shadow_decorrelation_m: float = 50.0
    # NTN fields; altitude_km > 0 marks the tier non-terrestrial.
    altitude_km: float = 0.0
    min_elevation_deg: float = 30.0
    pass_minutes: float = 10.0
    rain_fade_sensitivity: float = 0.0

    @property
    def is_ntn(self) -> bool:
        return self.altitude_km > 0.0

    @property
    def is_wifi(self) -> bool:
        return self.rat.lower().startswith("wi")

    def carries(self, slice_index: int) -> bool:
        return SESSION_TYPES[slice_index] in self.slices

    @property
    def resolved_prb_bandwidth_mhz(self) -> float:
        return self.prb_bandwidth_mhz or (12 * self.numerology_khz / 1000.0)

    @property
    def resolved_prbs_per_cell(self) -> int:
        if self.prbs_per_cell:
            return int(self.prbs_per_cell)
        usable = self.bandwidth_mhz * self.carriers * (1 - self.guard_fraction)
        return max(1, int(usable / self.resolved_prb_bandwidth_mhz))

    @property
    def mimo_capacity_gain(self) -> float:
        """Multiplier on delivered cell capacity from spatial multiplexing.

        SU-MIMO gains rank up to the layer count; MU-MIMO adds a modest
        multi-user scheduling bonus. Efficiency discounts both for correlation,
        overhead and imperfect CSI.
        """
        base = max(1, self.mimo_layers) * self.mimo_efficiency
        return base * (1.15 if self.mimo_mode.upper() == "MU" else 1.0)

    @property
    def pathloss_model(self) -> str:
        return {"UMa": "TR38.901-UMa", "UMi": "TR38.901-UMi",
                "RMa": "TR38.901-RMa", "InH": "TR38.901-InH"}.get(
                    self.scenario, "TR38.811-NTN" if self.is_ntn else self.scenario)


@dataclass(frozen=True)
class Cell:
    """One deployed cell: where it is, how high, how much power and bandwidth.

    These are the planning inputs an operator actually holds, so they are
    explicit rather than derived from a cell count.
    """

    cell_id: str
    tier: str
    x_km: float
    y_km: float
    antenna_height_m: float
    tx_power_dbm: float
    bandwidth_mhz: float
    azimuth_deg: float = 0.0        # reserved for sectorization
    enabled: bool = True


def default_tiers() -> List[Tier]:
    """Reference AI-enabled 6G stack: cellular + indoor Wi-Fi + LEO overlay."""
    return [
        Tier(
            name="macro_low", cio_db=0.0, rat="NR", band_ghz=0.7, bandwidth_mhz=10.0,
            # 10 MHz / 52 PRBs is the NR 15 kHz SCS resource grid.  Keep the
            # established grid and state its numerology explicitly so the
            # console does not inherit the Tier default of 30 kHz.
            numerology_khz=15.0, prb_bandwidth_mhz=0.18,
            prbs_per_cell=52, scenario="UMa",
            tx_power_dbm=46.0, antenna_height_m=30.0, antenna_gain_dbi=15.0,
            noise_figure_db=7.0, max_se_bps_hz=3.0,
            mobility=MobilityLimits(350.0, 500.0),
            one_way_latency_ms=4.0, energy_idle_w=160.0, energy_max_w=700.0,
            backhaul_mbps_per_cell=600.0,
            interference_coupling=1.0, shadow_sigma_db=6.0,
        ),
        Tier(
            name="macro_mid", cio_db=8.0, rat="NR", band_ghz=3.5, bandwidth_mhz=100.0,
            prb_bandwidth_mhz=0.36, prbs_per_cell=273, scenario="UMa",
            tx_power_dbm=49.0, antenna_height_m=25.0, antenna_gain_dbi=17.0,
            noise_figure_db=7.0, max_se_bps_hz=5.5,
            mobility=MobilityLimits(250.0, 500.0),
            one_way_latency_ms=2.0, energy_idle_w=220.0, energy_max_w=1_100.0,
            backhaul_mbps_per_cell=4_000.0,
            interference_coupling=1.0, shadow_sigma_db=6.0,
        ),
        Tier(
            # Upper mid-band: the 6G capacity layer. Beamformed, so it starts
            # derating well before the macro tiers do.
            name="umb_6g", cio_db=14.0, rat="6G-NR", band_ghz=7.5, bandwidth_mhz=200.0,
            numerology_khz=60.0, prb_bandwidth_mhz=0.72,
            prbs_per_cell=264, scenario="UMi",
            tx_power_dbm=44.0, antenna_height_m=12.0, antenna_gain_dbi=20.0,
            noise_figure_db=7.0, max_se_bps_hz=7.0,
            mobility=MobilityLimits(120.0, 250.0, beam_refinement_ms=1.0),
            one_way_latency_ms=1.0, energy_idle_w=90.0, energy_max_w=420.0,
            backhaul_mbps_per_cell=10_000.0, blockage_sensitivity=0.3,
            interference_coupling=0.35, shadow_sigma_db=7.8,
        ),
        Tier(
            name="mmwave", cio_db=18.0, rat="NR", band_ghz=28.0, bandwidth_mhz=400.0,
            numerology_khz=120.0, prb_bandwidth_mhz=1.44,
            prbs_per_cell=264, scenario="UMi",
            tx_power_dbm=40.0, antenna_height_m=10.0, antenna_gain_dbi=22.0,
            noise_figure_db=9.0, max_se_bps_hz=8.0,
            slices=("eMBB", "URLLC"),
            mobility=MobilityLimits(60.0, 120.0, beam_refinement_ms=0.5),
            one_way_latency_ms=1.0, energy_idle_w=70.0, energy_max_w=320.0,
            backhaul_mbps_per_cell=10_000.0, blockage_sensitivity=1.0,
            interference_coupling=0.08, shadow_sigma_db=7.8,
        ),
        Tier(
            # Wi-Fi 7 at 6 GHz: indoor only, and preferred there. No URLLC —
            # unlicensed contention cannot underwrite a 5 ms budget.
            name="wifi7_indoor", cio_db=0.0, rat="WiFi7", band_ghz=6.0, bandwidth_mhz=320.0,
            prb_bandwidth_mhz=2.0, prbs_per_cell=160, scenario="InH",
            tx_power_dbm=23.0, antenna_height_m=3.0, antenna_gain_dbi=5.0,
            noise_figure_db=8.0, max_se_bps_hz=7.5,
            slices=("eMBB", "mIoT"),
            mobility=MobilityLimits(10.0, 30.0),
            one_way_latency_ms=3.0, energy_idle_w=8.0, energy_max_w=25.0,
            backhaul_mbps_per_cell=2_000.0, indoor_only=True,
            interference_coupling=0.6, shadow_sigma_db=8.0,
        ),
        Tier(
            # LEO overlay. Outdoor only, no URLLC (25 ms RTT on the service
            # link alone), and reserved by the steering policy for high mobility
            # or terrestrial coverage holes.
            name="ntn_leo", cio_db=0.0, rat="NR-NTN", band_ghz=2.0, bandwidth_mhz=30.0,
            prb_bandwidth_mhz=0.36, prbs_per_cell=78, scenario="UMa",
            tx_power_dbm=64.0, antenna_height_m=0.0, antenna_gain_dbi=30.0,
            noise_figure_db=7.0, max_se_bps_hz=2.5,
            slices=("eMBB", "mIoT"),
            mobility=MobilityLimits(1000.0, 1200.0),
            one_way_latency_ms=12.0, energy_idle_w=120.0, energy_max_w=300.0,
            backhaul_mbps_per_cell=1_500.0, outdoor_only=True,
            altitude_km=600.0, min_elevation_deg=30.0, pass_minutes=10.0,
            rain_fade_sensitivity=0.3, interference_coupling=0.0, shadow_sigma_db=4.0,
        ),
    ]


def hex_layout(tier: Tier, count: int, area_km: Tuple[float, float],
               rng: np.random.Generator, jitter_km: float = 0.0) -> List[Cell]:
    """Place `count` cells of a tier on a jittered hex grid over the area."""
    if count <= 0:
        return []
    aspect = area_km[0] / max(1e-9, area_km[1])
    cols = max(1, int(round(math.sqrt(count * aspect))))
    rows = max(1, math.ceil(count / cols))
    cells: List[Cell] = []
    for k in range(count):
        r, c = divmod(k, cols)
        # Offset alternate rows to approximate a hex packing.
        x = (c + 0.5 + 0.5 * (r % 2)) / cols * area_km[0]
        y = (r + 0.5) / rows * area_km[1]
        if jitter_km:
            x += rng.normal(0, jitter_km)
            y += rng.normal(0, jitter_km)
        cells.append(Cell(
            cell_id=f"{tier.name}_{k:03d}", tier=tier.name,
            x_km=float(np.clip(x, 0, area_km[0])),
            y_km=float(np.clip(y, 0, area_km[1])),
            antenna_height_m=tier.antenna_height_m,
            tx_power_dbm=tier.tx_power_dbm,
            bandwidth_mhz=tier.bandwidth_mhz,
        ))
    return cells


DEFAULT_CELL_COUNTS = {
    "macro_low": 3, "macro_mid": 7, "umb_6g": 12,
    "mmwave": 16, "wifi7_indoor": 40, "ntn_leo": 1,
}


# ==========================================================================
# Configuration
# ==========================================================================

@dataclass
class TwinConfig:
    # Service area and deployment
    area_km: Tuple[float, float] = (1.2, 1.2)
    cell_counts: Dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_CELL_COUNTS))
    layout_jitter_km: float = 0.02

    # Session population
    n_sessions: int = 1200
    session_mix: Dict[str, float] = field(
        default_factory=lambda: {"eMBB": 0.55, "URLLC": 0.15, "mIoT": 0.30})
    indoor_fraction: float = 0.65
    high_loss_building_fraction: float = 0.35

    # Two timescales
    tick_seconds: float = 0.1          # mobility and A3/A5 measurement period
    control_interval_ticks: int = 100  # RL acts every 10 s by default
    episode_steps: int = 60            # control intervals per episode
    neighbour_refresh_ticks: int = 20  # rebuild the measurement set every 2 s

    # Environment
    rain_rate_mm_h: float = 0.0
    doppler_precompensation: float = 0.95
    handover_execution_ms: float = 40.0
    # Channel realism
    interference: bool = True          # co-channel inter-cell interference in SINR
    shadow_fading: bool = True         # spatially-correlated log-normal shadowing
    # Offered-load time profile: "flat" | "diurnal" | a CSV profile name
    #   (P1_balanced_busy_hour, P2_eMBB_hotspot, P3_URLLC_burst,
    #    P4_massive_IoT, P5_mixed_failure_stress) taken from
    #   synthetic_profiles/all_profiles.csv — the platform's established trace.
    traffic_profile: str = "diurnal"
    traffic_noise: float = 0.06       # AR(1) innovation on the load factor

    # Reward shaping
    energy_price: float = 0.03
    handover_price: float = 0.05
    continuity_price: float = 0.10
    seed: int = 3080

    tiers: List[Tier] = field(default_factory=default_tiers)
    cells: Optional[List[Cell]] = None      # explicit coordinates override layout
    steering: SteeringPolicy = field(default_factory=SteeringPolicy)
    # Per-tier field overrides applied on top of `tiers`, e.g.
    #   {"macro_mid": {"bandwidth_mhz": 200, "carriers": 2, "mimo_layers": 8,
    #                  "numerology_khz": 30, "scenario": "UMi"}}
    # so morphology, spectrum, numerology and MIMO can be changed per tier
    # without redefining the tier list. The resource grid is re-derived.
    tier_overrides: Dict[str, Dict] = field(default_factory=dict)


# Steering postures the policy can choose between, applied on top of the
# configured steering policy.
STEERING_POSTURES: Tuple[Tuple[str, Dict[str, float]], ...] = (
    ("balanced", {}),
    ("wifi_lean", {"wifi_indoor_bias_db": 14.0}),
    ("ntn_lean", {"ntn_speed_threshold_kmh": 60.0, "ntn_coverage_hole_dbm": -112.0,
                  "ntn_bias_db": 4.0}),
    ("terrestrial_hold", {"ntn_speed_threshold_kmh": 200.0,
                          "ntn_coverage_hole_dbm": -126.0}),
)

# The control is per tier: the single centralized policy emits one branch per
# tier (which slice-mix template that tier runs) plus one global steering
# branch. The action is therefore a vector of branch indices, length
# n_tiers + 1, not an index into a flat joint table — 9**n_tiers * 4 would be
# ~2.8M entries at six tiers. `branch_sizes(twin)` gives the per-branch cardinalities.
STEERING_BRANCH = len(STEERING_POSTURES)


def branch_sizes(n_tiers: int) -> List[int]:
    """[templates-per-tier] * n_tiers + [steering postures]."""
    return [len(TEMPLATES)] * n_tiers + [STEERING_BRANCH]


# ==========================================================================
# The twin
# ==========================================================================

class MultiTierTwin:
    """Stateful multi-tier 6G RAN model with a gym-shaped control interface."""

    def __init__(self, config: Optional[TwinConfig] = None):
        self.cfg = config or TwinConfig()
        applied = []
        for tier in self.cfg.tiers:
            over = dict(self.cfg.tier_overrides.get(tier.name) or {})
            if over:
                # If the spectrum or numerology is changed but the resource grid
                # is not given explicitly, re-derive it rather than keep the old
                # PRB count — while preserving the tier's original subcarrier
                # spacing unless that too was overridden.
                if ({"bandwidth_mhz", "numerology_khz", "carriers"} & over.keys()):
                    if "numerology_khz" not in over and tier.prb_bandwidth_mhz:
                        over["numerology_khz"] = tier.prb_bandwidth_mhz * 1000.0 / 12.0
                    over.setdefault("prbs_per_cell", 0)
                    over.setdefault("prb_bandwidth_mhz", 0.0)
                applied.append(replace(tier, **over))
            else:
                applied.append(tier)
        self.cfg.tiers = applied
        self.tiers = {t.name: t for t in applied}
        self.session_types = list(DEFAULT_SESSION_TYPES)
        self._build_cells()
        self._build_pathloss_tables()
        self.n_tiers = len(self.tier_order)
        self.obs_dim = 10 + 6 * self.n_tiers
        # Per-tier action: one template branch per tier + one steering branch.
        self.branch_sizes = branch_sizes(self.n_tiers)
        self.n_branches = len(self.branch_sizes)
        self.joint_action_count = int(np.prod(self.branch_sizes))
        self._steering_override: Dict[str, float] = {}
        self._min_rsrp_override: Dict[str, float] = {}
        self._threshold_override: Dict[str, Dict[str, float]] = {}
        self.reset()

    # -- optimizer hooks -------------------------------------------------------

    def set_cell_offsets(self, cio_db: Dict[str, float]) -> None:
        """Override the cell individual offset (dB) for named tiers, in place.

        Range expansion is the operator's lever for pulling load off the
        coverage layer onto the capacity layers; the throughput and coverage
        optimizers search over it. Survives reset (cells are not rebuilt).
        """
        for name, value in cio_db.items():
            if name in self.tier_index:
                self.cell_cio_db[self.cell_tier == self.tier_index[name]] = float(value)

    def set_steering_override(self, **overrides: float) -> None:
        """Persist steering-policy overrides across reset (latency/coverage search)."""
        self._steering_override = dict(overrides)
        self.steering = replace(self.cfg.steering, **self._steering_override)
        if hasattr(self, "ho"):
            self.ho.steering = self.steering

    def set_min_rsrp_override(self, per_type: Dict[str, float]) -> None:
        """Lower the drop threshold per session type (coverage search)."""
        self._min_rsrp_override = dict(per_type)
        if hasattr(self, "ho"):
            names = [t.name for t in self.session_types]
            for type_name, value in per_type.items():
                if type_name in names:
                    self.ho.min_rsrp[self.pop.type_index == names.index(type_name)] = value

    def set_threshold_override(self, per_type: Dict[str, Dict[str, float]]) -> None:
        """Persist A3/A5 threshold overrides across reset (handover search)."""
        self._threshold_override = {k: dict(v) for k, v in per_type.items()}
        if hasattr(self, "ho"):
            self.ho.set_thresholds(self._threshold_override)

    # -- deployment --------------------------------------------------------

    def _build_cells(self) -> None:
        rng = np.random.default_rng(self.cfg.seed)
        if self.cfg.cells is not None:
            cells = list(self.cfg.cells)
        else:
            cells = []
            for tier in self.cfg.tiers:
                count = self.cfg.cell_counts.get(tier.name, 0)
                if tier.is_ntn:
                    # One beam, centred on the service area.
                    cells += [Cell(f"{tier.name}_000", tier.name,
                                   self.cfg.area_km[0] / 2, self.cfg.area_km[1] / 2,
                                   0.0, tier.tx_power_dbm, tier.bandwidth_mhz)]
                else:
                    cells += hex_layout(tier, count, self.cfg.area_km, rng,
                                        self.cfg.layout_jitter_km)
        self.cells = [c for c in cells if c.enabled]
        self.n_cells = len(self.cells)
        if self.n_cells == 0:
            raise ValueError("No enabled cells in the deployment")

        self.tier_order = [t.name for t in self.cfg.tiers if any(
            c.tier == t.name for c in self.cells)]
        self.tier_index = {name: i for i, name in enumerate(self.tier_order)}

        self.cell_x = np.array([c.x_km for c in self.cells])
        self.cell_y = np.array([c.y_km for c in self.cells])
        self.cell_tier = np.array([self.tier_index[c.tier] for c in self.cells])
        self.cell_tx_dbm = np.array([c.tx_power_dbm for c in self.cells])
        self.cell_bw_mhz = np.array([c.bandwidth_mhz for c in self.cells])
        self.cell_height_m = np.array([c.antenna_height_m for c in self.cells])

        tiers = [self.tiers[n] for n in self.tier_order]
        self.cell_prbs = np.array(
            [tiers[t].resolved_prbs_per_cell for t in self.cell_tier], dtype=float)
        self.cell_prb_bw = np.array(
            [tiers[t].resolved_prb_bandwidth_mhz for t in self.cell_tier])
        self.cell_max_se = np.array([tiers[t].max_se_bps_hz for t in self.cell_tier])
        self.cell_mimo_gain = np.array(
            [tiers[t].mimo_capacity_gain for t in self.cell_tier])
        self.cell_is_ntn = np.array([tiers[t].is_ntn for t in self.cell_tier])
        self.cell_is_wifi = np.array([tiers[t].is_wifi for t in self.cell_tier])
        self.cell_indoor_only = np.array([tiers[t].indoor_only for t in self.cell_tier])
        self.cell_outdoor_only = np.array([tiers[t].outdoor_only for t in self.cell_tier])
        self.cell_max_speed = np.array(
            [tiers[t].mobility.max_speed_kmh for t in self.cell_tier])
        self.cell_latency_ms = np.array(
            [tiers[t].one_way_latency_ms for t in self.cell_tier])
        self.cell_backhaul = np.array(
            [tiers[t].backhaul_mbps_per_cell for t in self.cell_tier])
        self.cell_noise_figure = np.array(
            [tiers[t].noise_figure_db for t in self.cell_tier])
        # RSRP is defined per resource element, which is what makes levels
        # comparable across tiers with different PRB widths and carrier sizes.
        # Per-PRB power for the SINR calculation is this plus 10*log10(12).
        self.cell_eirp_per_re = (
            self.cell_tx_dbm - 10 * np.log10(np.maximum(1.0, self.cell_prbs * 12))
            + np.array([tiers[t].antenna_gain_dbi for t in self.cell_tier]))
        # Cell individual offset (3GPP cellIndividualOffset / range expansion):
        # the operator's lever for pulling load off the coverage layer and onto
        # the capacity layers. This is a primary target for the optimizer.
        self.cell_cio_db = np.array([tiers[t].cio_db for t in self.cell_tier])
        self.cell_slice_mask = np.array(
            [[tiers[t].carries(s) for s in range(N_SLICES)] for t in self.cell_tier])

    def _build_pathloss_tables(self) -> None:
        """Precompute path loss vs distance per tier, then interpolate.

        TR 38.901 is expensive to evaluate per session per cell per tick, and it
        depends only on 2-D distance once the tier fixes frequency and antenna
        height — so one table per tier is both exact and fast.
        """
        self.pl_distance_m = np.logspace(math.log10(1.0), math.log10(20_000), 400)
        self.pl_tables: Dict[str, np.ndarray] = {}
        self.o2i_db: Dict[str, Tuple[float, float]] = {}
        for name in self.tier_order:
            tier = self.tiers[name]
            if tier.is_ntn:
                self.pl_tables[name] = np.zeros_like(self.pl_distance_m)
                self.o2i_db[name] = (0.0, 0.0)
                continue
            self.pl_tables[name] = np.array([
                radio.pathloss(tier.scenario, float(d), tier.antenna_height_m, 1.5,
                               tier.band_ghz)
                for d in self.pl_distance_m])
            self.o2i_db[name] = (
                radio.o2i_penetration_loss_db(tier.band_ghz, False),
                radio.o2i_penetration_loss_db(tier.band_ghz, True),
            )

    # -- lifecycle ---------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        seed = self.cfg.seed if seed is None else seed
        self.rng = np.random.default_rng(seed)
        self.tick_index = 0
        self.step_index = 0
        self.rain = self.cfg.rain_rate_mm_h

        self.pop = SessionPopulation(
            self.cfg.n_sessions, self.cfg.area_km, self.cfg.session_mix,
            self.session_types, np.random.default_rng(seed + 1))
        self.indoor = self.rng.random(self.pop.n) < self.cfg.indoor_fraction
        # Indoor UEs move slowly; nobody drives through a building.
        self.pop.speed_kmh = np.where(
            self.indoor, np.minimum(self.pop.speed_kmh, 5.0), self.pop.speed_kmh)
        self.high_loss = self.rng.random(self.pop.n) < self.cfg.high_loss_building_fraction

        # Optimizer overrides (cell individual offsets, steering knobs, per
        # session-type minimum RSRP) survive reset so a search can hold them
        # fixed across evaluation seeds.
        self.steering = replace(self.cfg.steering, **self._steering_override)
        self.ho = HandoverEngine(
            self.pop, self.session_types, self.n_cells, self.steering,
            np.random.default_rng(seed + 2), self.cfg.handover_execution_ms)
        if self._min_rsrp_override:
            self.set_min_rsrp_override(self._min_rsrp_override)
        if self._threshold_override:
            self.ho.set_thresholds(self._threshold_override)

        self.candidates = self._refresh_candidates()
        self.blockage = np.ones(len(self.tier_order))
        self.cell_load = np.zeros(self.n_cells)
        self.last_info: Dict = {}
        self.last_link: Optional[radio.NTNLink] = None

        # Spatially-correlated log-normal shadow fading, one value per (session,
        # tier). Evolves as a Gudmundson AR(1) process with distance moved.
        sig = np.array([self.tiers[n].shadow_sigma_db for n in self.tier_order])
        self.shadow = (self.rng.standard_normal((self.pop.n, self.n_tiers))
                       * sig[None, :]) if self.cfg.shadow_fading else \
            np.zeros((self.pop.n, self.n_tiers))

        # Time-varying offered load: a per-slice multiplier on each session's rate.
        self.load_shape = load_traffic_shape(self.cfg.traffic_profile)
        self.load_residual = np.zeros(N_SLICES)
        self.load_factor = np.ones(N_SLICES)
        # The stack need not contain every tier: a deployment with no Wi-Fi and
        # no satellite is a valid configuration and runs unchanged.
        ntn_tiers = [n for n in self.tier_order if self.tiers[n].is_ntn]
        self.ntn_elevation_deg = (self.tiers[ntn_tiers[0]].min_elevation_deg
                                  if ntn_tiers else 0.0)
        return self._observe(np.zeros(N_SLICES), np.zeros(len(self.tier_order)))

    # -- radio -------------------------------------------------------------

    def _refresh_candidates(self, k: int = 10) -> np.ndarray:
        """The k nearest cells per session — the UE's measurement set."""
        dx = self.pop.x_km[:, None] - self.cell_x[None, :]
        dy = self.pop.y_km[:, None] - self.cell_y[None, :]
        d2 = dx * dx + dy * dy
        # The satellite beam covers everything, so it is always a candidate.
        d2[:, self.cell_is_ntn] = -1.0
        k = min(k, self.n_cells)
        return np.argpartition(d2, k - 1, axis=1)[:, :k]

    def _ntn_state(self) -> Tuple[float, Optional[radio.NTNLink]]:
        """Advance the LEO pass and return the current elevation and link."""
        ntn_tiers = [n for n in self.tier_order if self.tiers[n].is_ntn]
        if not ntn_tiers:
            return 0.0, None
        tier = self.tiers[ntn_tiers[0]]
        # Elevation sweeps min -> 90 -> min over one pass, then a new satellite.
        pass_ticks = max(1, int(tier.pass_minutes * 60 / self.cfg.tick_seconds))
        phase = (self.tick_index % pass_ticks) / pass_ticks
        elevation = tier.min_elevation_deg + (90.0 - tier.min_elevation_deg) * \
            math.sin(math.pi * phase)
        link = radio.ntn_link_budget(
            freq_ghz=tier.band_ghz, altitude_km=tier.altitude_km,
            elevation_deg=elevation, eirp_dbm=tier.tx_power_dbm + tier.antenna_gain_dbi,
            rx_gain_dbi=tier.ue_antenna_gain_dbi,
            bandwidth_hz=tier.resolved_prb_bandwidth_mhz * 1e6,
            noise_figure_db=tier.noise_figure_db, required_sinr_db=-3.0,
            rain_rate_mm_h=self.rain,
            subcarrier_spacing_hz=30_000.0,
            doppler_precompensation=self.cfg.doppler_precompensation,
        )
        return elevation, link

    def _rsrp(self, link: Optional[radio.NTNLink]) -> np.ndarray:
        """RSRP in dBm for each session over its candidate cells."""
        cand = self.candidates
        cx, cy = self.cell_x[cand], self.cell_y[cand]
        d_m = np.sqrt((self.pop.x_km[:, None] - cx) ** 2
                      + (self.pop.y_km[:, None] - cy) ** 2) * 1000.0
        d_m = np.maximum(d_m, 1.0)

        tier_of = self.cell_tier[cand]
        pl = np.zeros_like(d_m)
        for ti, name in enumerate(self.tier_order):
            sel = tier_of == ti
            if not sel.any():
                continue
            if self.tiers[name].is_ntn:
                pl[sel] = link.total_loss_db if link is not None else 1e6
                continue
            pl[sel] = np.interp(d_m[sel], self.pl_distance_m, self.pl_tables[name])
            # Outdoor cells serving indoor UEs pay building penetration.
            if not self.tiers[name].indoor_only:
                low, high = self.o2i_db[name]
                o2i = np.where(self.high_loss, high, low)[:, None] * self.indoor[:, None]
                pl[sel] += np.broadcast_to(o2i, d_m.shape)[sel]

        rsrp = self.cell_eirp_per_re[cand] - pl
        # Blockage and shadow fading on the beamformed / all tiers.
        for ti, name in enumerate(self.tier_order):
            m = tier_of == ti
            if not m.any():
                continue
            s = self.tiers[name].blockage_sensitivity
            if s > 0:
                rsrp[m] -= s * (1 - self.blockage[ti]) * 20.0
            if self.cfg.shadow_fading and not self.tiers[name].is_ntn:
                rsrp[m] -= np.broadcast_to(self.shadow[:, ti:ti + 1], rsrp.shape)[m]
        return rsrp

    def _scatter(self, values: np.ndarray) -> np.ndarray:
        """Expand a (sessions x candidates) array to (sessions x cells)."""
        full = np.full((self.pop.n, self.n_cells), -np.inf)
        rows = np.repeat(np.arange(self.pop.n), self.candidates.shape[1])
        full[rows, self.candidates.ravel()] = values.ravel()
        return full

    # -- fast tick ---------------------------------------------------------

    def _tick(self) -> None:
        dt_s = self.cfg.tick_seconds
        self.pop.step(dt_s)
        if self.tick_index % self.cfg.neighbour_refresh_ticks == 0:
            self.candidates = self._refresh_candidates()

        # Evolve shadow fading: Gudmundson AR(1), rho = exp(-d_moved / d_corr).
        if self.cfg.shadow_fading:
            moved_m = self.pop.speed_kmh / 3.6 * dt_s
            for ti, name in enumerate(self.tier_order):
                d_corr = max(1.0, self.tiers[name].shadow_decorrelation_m)
                sig = self.tiers[name].shadow_sigma_db
                rho = np.exp(-moved_m / d_corr)
                self.shadow[:, ti] = (rho * self.shadow[:, ti]
                                      + np.sqrt(np.maximum(0.0, 1 - rho ** 2))
                                      * self.rng.standard_normal(self.pop.n) * sig)

        # Slow-varying blockage on the beamformed tiers.
        self.blockage = np.clip(
            self.blockage + self.rng.normal(0, 0.01, len(self.tier_order)), 0.3, 1.0)

        elevation, link = self._ntn_state()
        self.ntn_elevation_deg = elevation
        rsrp_cand = self._rsrp(link)
        rsrp = self._scatter(rsrp_cand)

        terrestrial = ~self.cell_is_ntn
        best_terrestrial = np.max(np.where(terrestrial[None, :], rsrp, -np.inf), axis=1)

        allowed = self.steering.eligibility(
            self.indoor, self.pop.speed_kmh, self.cell_indoor_only,
            self.cell_outdoor_only, self.cell_is_ntn, self.cell_is_wifi,
            self.cell_max_speed, best_terrestrial)
        allowed &= np.isfinite(rsrp)
        # A cell only carries the slices its tier supports.
        allowed &= self.cell_slice_mask[:, self.pop.type_index].T

        # A3/A5 are evaluated on biased levels: the cell individual offset plus
        # the steering preference, exactly as Ocn/Ofn enter the event condition.
        biased = (rsrp + self.cell_cio_db[None, :]
                  + self.steering.rsrp_bias_db(self.indoor, self.cell_is_wifi,
                                               self.cell_is_ntn))

        self.ho.tick(biased, allowed, dt_s * 1000.0, self.cell_load,
                     self.cell_max_speed)
        self.last_rsrp = rsrp
        self.last_link = link
        self.tick_index += 1

    # -- control step ------------------------------------------------------

    def _parse_action(self, action) -> Tuple[np.ndarray, int]:
        """Return (per-tier template array (n_tiers, N_SLICES), posture index).

        Accepts:
          - a branch vector of length n_tiers + 1  (per-tier control, from RL)
          - a 2-vector (template_idx, posture_idx)  (a global baseline; the
            template is broadcast to every tier)
          - a dict {"tier_templates": [...], "posture": p}
          - explicit per-tier share vectors in place of template indices
        """
        if isinstance(action, dict):
            seq = list(action["tier_templates"]) + [action["posture"]]
        elif isinstance(action, (int, np.integer)):
            seq = [int(action), 0]
        else:
            seq = list(action)

        if len(seq) == self.n_tiers + 1:
            tier_choices, posture_idx = seq[:-1], seq[-1]
        elif len(seq) == 2:
            tier_choices, posture_idx = [seq[0]] * self.n_tiers, seq[1]
        else:
            raise ValueError(
                f"action must have length {self.n_tiers + 1} (per-tier) or 2 "
                f"(global); got {len(seq)}")

        rows = []
        for choice in tier_choices:
            if np.isscalar(choice) or isinstance(choice, (int, np.integer, float)):
                rows.append(TEMPLATES[int(choice) % len(TEMPLATES)])
            else:
                vec = np.asarray(choice, dtype=float)
                rows.append(vec / max(1e-9, vec.sum()))
        return np.stack(rows), int(posture_idx) % len(STEERING_POSTURES)

    def _refresh_load_factor(self) -> None:
        """Advance the per-slice offered-load multiplier for this control interval.

        "flat" holds it at 1. "diurnal" is a two-peak day plus an AR(1) residual.
        A named CSV profile follows that trace's own normalised shape, also with
        a small AR(1) residual so repeated passes over a short trace still vary.
        """
        self.load_residual = (0.85 * self.load_residual
                              + self.rng.normal(0, self.cfg.traffic_noise, N_SLICES))
        if self.load_shape is not None:
            t = self.step_index % self.load_shape.shape[1]
            base = self.load_shape[:, t]
        elif self.cfg.traffic_profile == "diurnal":
            phi = 2 * math.pi * (self.step_index % max(1, self.cfg.episode_steps)) \
                / max(1, self.cfg.episode_steps)
            shape = 0.55 + 0.32 * math.sin(phi - 1.9) + 0.18 * math.sin(2 * phi - 0.4)
            base = np.full(N_SLICES, max(0.1, shape))
            base[2] = max(0.25, 0.45 + 0.55 * shape)     # mIoT less diurnal
        else:  # "flat"
            base = np.ones(N_SLICES)
        self.load_factor = np.maximum(0.0, base * (1 + self.load_residual))

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        """Advance one control interval: many mobility ticks, then one allocation."""
        tier_templates, posture_idx = self._parse_action(action)

        # Apply the steering posture for this interval.
        posture_name, overrides = STEERING_POSTURES[posture_idx]
        self.steering = replace(self.cfg.steering, **overrides)
        self.ho.steering = self.steering

        self._refresh_load_factor()

        before = self.ho.counters.as_dict()
        for _ in range(self.cfg.control_interval_ticks):
            self._tick()
        after = self.ho.counters.as_dict()

        info = self._allocate(tier_templates)
        info["posture"] = posture_name
        info["tier_intent"] = {
            name: np.round(tier_templates[ti], 3).tolist()
            for ti, name in enumerate(self.tier_order)}
        info["handover_delta"] = {
            "attempts": after["attempts"] - before["attempts"],
            "successes": after["successes"] - before["successes"],
            "ping_pongs": after["ping_pongs"] - before["ping_pongs"],
            "radio_link_failures": (after["radio_link_failures"]
                                    - before["radio_link_failures"]),
        }
        attempts = max(1, info["handover_delta"]["attempts"])
        info["handover_success_rate"] = info["handover_delta"]["successes"] / attempts
        info["session_continuity"] = self.ho.session_continuity()

        reward = self._reward(info)
        info["reward"] = round(reward, 5)
        self.last_info = info

        self.step_index += 1
        done = self.step_index >= self.cfg.episode_steps
        util = np.array([info["tier_utilization"].get(n, 0.0) for n in self.tier_order])
        demand = np.array([info["offered_mbps"][s] for s in SESSION_TYPES])
        return self._observe(demand, util), reward, done, info

    # -- allocation and service -------------------------------------------

    def _allocate(self, tier_templates: np.ndarray) -> Dict:
        """Allocate PRBs per cell across slices and serve the attached sessions.

        `tier_templates` is (n_tiers, N_SLICES): each cell uses its own tier's
        slice-mix intent, so the low band can run a coverage-weighted split
        while the 6G capacity layer runs an eMBB-heavy one in the same interval.
        """
        serving = self.ho.serving
        attached = serving >= 0
        # Time-varying offered load: session base rate x the per-slice factor.
        demand = self.pop.bitrate_mbps() * self.load_factor[self.pop.type_index]

        n = self.pop.n
        idx = np.arange(n)
        cell_of = np.maximum(serving, 0)
        rsrp = np.where(attached, self.last_rsrp[idx, cell_of], -np.inf)

        # Per-PRB signal, thermal noise and (optionally) co-channel interference.
        noise_dbm = (-174 + 10 * np.log10(self.cell_prb_bw[cell_of] * 1e6)
                     + self.cell_noise_figure[cell_of])
        signal_prb_dbm = rsrp + 10 * np.log10(12.0)      # RSRP is per RE; 12 per PRB

        if self.cfg.interference:
            # Interference = sum of received power from co-channel cells (same
            # tier => same band) other than the server, each scaled by its load
            # (activity) and the tier's beam-coupling factor. `last_rsrp` covers
            # the candidate set, which holds the dominant interferers.
            same_tier = (self.cell_tier[None, :] == self.cell_tier[cell_of][:, None])
            not_server = np.ones((n, self.n_cells), dtype=bool)
            not_server[idx, cell_of] = False
            rx_lin = np.where(np.isfinite(self.last_rsrp),
                              np.power(10.0, self.last_rsrp / 10.0), 0.0)  # per RE
            kappa = np.array([self.tiers[nm].interference_coupling
                              for nm in self.tier_order])[self.cell_tier]
            weight = self.cell_load[None, :] * kappa[None, :] * same_tier * not_server
            interf_re = (rx_lin * weight).sum(axis=1)
            interf_prb_lin = 12.0 * interf_re
        else:
            interf_prb_lin = np.zeros(n)

        sinr_lin = np.where(
            attached,
            np.power(10.0, signal_prb_dbm / 10.0)
            / (np.power(10.0, noise_dbm / 10.0) + interf_prb_lin + 1e-30),
            10 ** (-5.0))
        se = np.minimum(np.log2(1 + np.clip(sinr_lin, 1e-6, 1e12)),
                        self.cell_max_se[cell_of])

        # Mobility derate: beam tracking and handover rate cost throughput.
        derate = np.ones(self.pop.n)
        for ti, name in enumerate(self.tier_order):
            tier = self.tiers[name]
            sel = attached & (self.cell_tier[cell_of] == ti)
            if not sel.any():
                continue
            speeds = self.pop.speed_kmh[sel]
            derate[sel] = [tier.mobility.derate(float(v), tier.band_ghz)
                           for v in speeds]
        se *= derate

        # NTN residual-Doppler ICI on the satellite tier.
        if self.last_link is not None and self.cell_is_ntn.any():
            on_ntn = attached & self.cell_is_ntn[cell_of]
            if on_ntn.any():
                se[on_ntn] = [radio.spectral_efficiency_with_ici(
                    float(v), self.last_link.ici_ratio) for v in se[on_ntn]]

        # PRB split per cell: the policy's intent, masked to the slices the tier
        # carries and renormalized, then shared equally within each slice.
        served = np.zeros(self.pop.n)
        cell_used = np.zeros(self.n_cells)
        cell_capacity = np.zeros(self.n_cells)
        for c in range(self.n_cells):
            on_cell = attached & (serving == c)
            if not on_cell.any():
                continue
            mask = self.cell_slice_mask[c].astype(float)
            intent = tier_templates[self.cell_tier[c]] * mask
            intent = intent / intent.sum() if intent.sum() > 1e-9 else mask / mask.sum()
            for s in range(N_SLICES):
                members = on_cell & (self.pop.type_index == s)
                n_members = int(members.sum())
                if n_members == 0:
                    continue
                prbs = intent[s] * self.cell_prbs[c]
                per_session = prbs / n_members * self.cell_prb_bw[c]
                # MIMO multiplies delivered capacity; per-stream SE stays
                # Shannon-bounded above.
                capacity = per_session * se[members] * self.cell_mimo_gain[c]
                served[members] = np.minimum(demand[members], capacity)
                cell_capacity[c] += float(capacity.sum())
            # Backhaul ceiling.
            total = float(served[on_cell].sum())
            if total > self.cell_backhaul[c]:
                served[on_cell] *= self.cell_backhaul[c] / max(1e-9, total)
            cell_used[c] = float(served[on_cell].sum())

        self.cell_load = np.divide(cell_used, np.maximum(cell_capacity, 1e-9))
        np.clip(self.cell_load, 0, 1, out=self.cell_load)

        offered = np.array([demand[self.pop.type_index == s].sum()
                            for s in range(N_SLICES)])
        got = np.array([served[self.pop.type_index == s].sum()
                        for s in range(N_SLICES)])
        satisfaction = np.clip(np.divide(got, np.maximum(offered, 1e-9)), 0, 1)
        sla = np.logical_or(offered <= 1e-9, satisfaction >= SLA_THRESHOLDS).astype(float)

        tier_util, tier_served, tier_sessions = {}, {}, {}
        for ti, name in enumerate(self.tier_order):
            sel_cells = self.cell_tier == ti
            tier_util[name] = round(float(self.cell_load[sel_cells].mean()), 4)
            tier_served[name] = round(float(cell_used[sel_cells].sum()), 2)
            tier_sessions[name] = int((attached & (self.cell_tier[cell_of] == ti)).sum())

        energy_w = 0.0
        for ti, name in enumerate(self.tier_order):
            tier = self.tiers[name]
            sel = self.cell_tier == ti
            energy_w += float((tier.energy_idle_w + (tier.energy_max_w - tier.energy_idle_w)
                               * self.cell_load[sel]).sum())
        energy_max = sum(self.tiers[self.tier_order[t]].energy_max_w
                         for t in self.cell_tier)

        latency = float(self.cell_latency_ms[cell_of][attached] @ served[attached]
                        / max(1e-9, served[attached].sum())) if attached.any() else 0.0
        fairness = float(satisfaction.sum() ** 2
                         / (N_SLICES * np.square(satisfaction).sum() + 1e-9))

        # Split the unmet demand by cause, so the feasibility check and the
        # coverage/throughput optimizers can tell a coverage hole (no eligible
        # tier reached the UE) from a capacity shortfall (the serving tier ran
        # out of PRBs). Thresholds can move the second; only spectrum or sites
        # move the first.
        unmet = demand - served
        coverage_gap = np.array([demand[(~attached) & (self.pop.type_index == s)].sum()
                                 for s in range(N_SLICES)])
        capacity_gap = np.array([unmet[attached & (self.pop.type_index == s)].sum()
                                 for s in range(N_SLICES)])

        return {
            "step": self.step_index,
            "offered_mbps": dict(zip(SESSION_TYPES, offered.round(2).tolist())),
            "served_mbps": dict(zip(SESSION_TYPES, got.round(2).tolist())),
            "satisfaction": dict(zip(SESSION_TYPES, satisfaction.round(4).tolist())),
            "sla": dict(zip(SESSION_TYPES, sla.tolist())),
            "attached_sessions": int(attached.sum()),
            "unattached_sessions": int((~attached).sum()),
            "coverage_gap_mbps": dict(zip(SESSION_TYPES, coverage_gap.round(2).tolist())),
            "capacity_gap_mbps": dict(zip(SESSION_TYPES, capacity_gap.round(2).tolist())),
            "tier_headroom_mbps": {
                name: round(float((cell_capacity - cell_used)[self.cell_tier == ti].sum()), 1)
                for ti, name in enumerate(self.tier_order)},
            "tier_utilization": tier_util,
            "tier_served_mbps": tier_served,
            "tier_sessions": tier_sessions,
            "network_utilization": round(float(self.cell_load.mean()), 4),
            "jain_fairness": round(fairness, 4),
            "mean_access_latency_ms": round(latency, 3),
            "mean_spectral_efficiency": round(float(se[attached].mean()), 3)
            if attached.any() else 0.0,
            "energy_w": round(energy_w, 1),
            "energy_fraction": round(energy_w / max(1e-9, energy_max), 4),
            "ntn_elevation_deg": round(float(self.ntn_elevation_deg), 2),
            "ntn_doppler_khz": round(self.last_link.doppler_shift_hz / 1e3, 2)
            if self.last_link else 0.0,
            "ntn_rain_attenuation_db": round(self.last_link.rain_attenuation_db, 3)
            if self.last_link else 0.0,
        }

    def _reward(self, info: Dict) -> float:
        satisfaction = np.array([info["satisfaction"][s] for s in SESSION_TYPES])
        sla = np.array([info["sla"][s] for s in SESSION_TYPES])
        violation = float(((1 - sla) * VIOLATION_COST).sum())
        continuity = info["session_continuity"]["overall"]
        ho = info["handover_delta"]
        churn = (ho["ping_pongs"] + ho["radio_link_failures"]) / max(1, self.pop.n)
        return float(
            SLICE_WEIGHTS @ satisfaction
            + 0.06 * info["network_utilization"]
            + 0.04 * info["jain_fairness"]
            - 0.12 * violation
            - self.cfg.energy_price * info["energy_fraction"]
            - self.cfg.handover_price * churn
            - self.cfg.continuity_price * (1 - continuity)
        )

    # -- observation -------------------------------------------------------

    def _observe(self, demand: np.ndarray, utilization: np.ndarray) -> np.ndarray:
        peak = np.array([
            sum(t.bitrate_mbps for t in self.session_types
                if t.name == s) * self.cfg.n_sessions for s in SESSION_TYPES])
        peak = np.maximum(peak, 1e-9)
        continuity = self.ho.session_continuity()["overall"] if hasattr(self, "ho") else 1.0
        attempts = max(1, self.ho.counters.attempts) if hasattr(self, "ho") else 1
        head = [
            min(2.0, demand[0] / peak[0]), min(2.0, demand[1] / peak[1]),
            min(2.0, demand[2] / peak[2]),
            self.step_index / max(1, self.cfg.episode_steps),
            float(np.clip(utilization.mean(), 0, 1)),
            continuity,
            self.ho.counters.successes / attempts if hasattr(self, "ho") else 1.0,
            self.ho.counters.ping_pongs / attempts if hasattr(self, "ho") else 0.0,
            self.ntn_elevation_deg / 90.0,
            float(np.clip(self.rain / 50.0, 0, 1)),
        ]
        per_tier: List[float] = []
        for ti, name in enumerate(self.tier_order):
            tier = self.tiers[name]
            sel = self.cell_tier == ti
            per_tier += [
                float(np.clip(utilization[ti] if ti < len(utilization) else 0.0, 0, 1)),
                float(sel.sum()) / self.n_cells,
                float(self.blockage[ti]),
                float(np.clip(tier.one_way_latency_ms / 20.0, 0, 1)),
                float(tier.is_ntn),
                float(tier.is_wifi),
            ]
        return np.array(head + per_tier, dtype=np.float64)

    # -- reporting ---------------------------------------------------------

    def plan_summary(self) -> Dict:
        area_km2 = self.cfg.area_km[0] * self.cfg.area_km[1]
        tiers = []
        for name in self.tier_order:
            tier = self.tiers[name]
            sel = self.cell_tier == self.tier_index[name]
            entry = {
                "tier": name, "rat": tier.rat, "band_ghz": tier.band_ghz,
                "bandwidth_mhz": tier.bandwidth_mhz, "carriers": tier.carriers,
                "aggregate_bandwidth_mhz": round(tier.bandwidth_mhz * tier.carriers, 1),
                "numerology_khz": tier.numerology_khz,
                "prb_bandwidth_mhz": round(tier.resolved_prb_bandwidth_mhz, 4),
                "prbs_per_cell": tier.resolved_prbs_per_cell,
                "mimo": f"{tier.mimo_layers}x {tier.mimo_mode}",
                "mimo_capacity_gain": round(tier.mimo_capacity_gain, 2),
                "morphology": tier.scenario,
                "pathloss_model": tier.pathloss_model,
                "cells": int(sel.sum()),
                "antenna_height_m": tier.antenna_height_m,
                "tx_power_dbm": tier.tx_power_dbm,
                "antenna_gain_dbi": tier.antenna_gain_dbi,
                "noise_figure_db": tier.noise_figure_db,
                "scenario": tier.scenario,
                "slices": list(tier.slices),
                "one_way_latency_ms": tier.one_way_latency_ms,
                "derate_speed_kmh": tier.mobility.derate_speed_kmh,
                "max_speed_kmh": tier.mobility.max_speed_kmh,
                "indoor_only": tier.indoor_only, "outdoor_only": tier.outdoor_only,
                "ntn": tier.is_ntn,
                "peak_cell_capacity_mbps": round(
                    tier.resolved_prbs_per_cell * tier.resolved_prb_bandwidth_mhz
                    * tier.max_se_bps_hz * tier.mimo_capacity_gain, 1),
            }
            if tier.is_ntn:
                link = radio.ntn_link_budget(
                    tier.band_ghz, tier.altitude_km, tier.min_elevation_deg,
                    tier.tx_power_dbm + tier.antenna_gain_dbi, tier.ue_antenna_gain_dbi,
                    tier.resolved_prb_bandwidth_mhz * 1e6, tier.noise_figure_db, -3.0,
                    self.cfg.rain_rate_mm_h)
                entry.update({
                    "altitude_km": tier.altitude_km,
                    "slant_range_km": round(link.slant_range_km, 1),
                    "free_space_loss_db": round(link.free_space_loss_db, 1),
                    "rain_attenuation_db": round(link.rain_attenuation_db, 2),
                    "doppler_khz": round(link.doppler_shift_hz / 1e3, 1),
                    "one_way_delay_ms": round(link.one_way_delay_ms, 2),
                    "link_margin_db": round(link.link_margin_db, 1),
                })
            else:
                mapl = radio.max_path_loss_db(
                    tier.tx_power_dbm, tier.antenna_gain_dbi, tier.ue_antenna_gain_dbi,
                    tier.resolved_prb_bandwidth_mhz * 1e6, tier.noise_figure_db, -3.0, 8.0)
                radius_m = radio.cell_radius_m(
                    tier.scenario, mapl, tier.antenna_height_m, 1.5, tier.band_ghz)
                entry["cell_radius_m"] = round(radius_m, 1)
                entry["max_path_loss_db"] = round(mapl, 1)

            # Capacity and coverage rollup for this tier.
            n_cells = int(entry["cells"])
            entry["tier_capacity_mbps"] = round(entry["peak_cell_capacity_mbps"] * n_cells, 1)
            entry["interference_coupling"] = tier.interference_coupling
            entry["shadow_sigma_db"] = tier.shadow_sigma_db

            if tier.is_ntn:
                entry["area_covered"] = 1.0 if entry.get("link_margin_db", 1) > 0 else 0.0
                entry["loaded_cell_capacity_mbps"] = entry["peak_cell_capacity_mbps"]
            else:
                # Nominal cell radius from the deployed density (equal-area
                # circles), not the link-budget max reach.
                r_deploy = math.sqrt(area_km2 / max(1, n_cells) / math.pi) * 1000.0
                entry["deploy_radius_m"] = round(r_deploy, 1)
                # Coverage: does the link close at the nominal cell edge, with a
                # shadow-fade allowance, and a small inter-cell edge outage.
                fade_margin = (1.28 * tier.shadow_sigma_db - 8.0
                               if self.cfg.shadow_fading else 0.0)
                link_radius = radio.cell_radius_m(
                    tier.scenario, entry["max_path_loss_db"] - fade_margin,
                    tier.antenna_height_m, 1.5, tier.band_ghz)
                entry["cell_edge_radius_m"] = round(link_radius, 1)
                reach = link_radius / max(1.0, r_deploy)
                outage = 0.05 if self.cfg.shadow_fading else 0.0
                entry["area_covered"] = round(min(1.0, reach ** 2) * (1 - outage), 3)

                # Interference-limited SE at ~50% neighbour load, hex reuse-1.
                se_loaded = self._loaded_se(tier, r_deploy) if self.cfg.interference \
                    else tier.max_se_bps_hz
                entry["loaded_se_bps_hz"] = round(se_loaded, 2)
                entry["loaded_cell_capacity_mbps"] = round(
                    tier.resolved_prbs_per_cell * tier.resolved_prb_bandwidth_mhz
                    * se_loaded * tier.mimo_capacity_gain, 1)
            entry["loaded_tier_capacity_mbps"] = round(
                entry["loaded_cell_capacity_mbps"] * n_cells, 1)

            tiers.append(entry)

        def _sum(field, pred=lambda t: True):
            return round(sum(t[field] for t in tiers if pred(t)), 1)

        by_slice = {s: _sum("tier_capacity_mbps", lambda t, s=s: s in t["slices"])
                    for s in SESSION_TYPES}
        by_slice_loaded = {s: _sum("loaded_tier_capacity_mbps", lambda t, s=s: s in t["slices"])
                           for s in SESSION_TYPES}
        terrestrial = [t for t in tiers if not t["ntn"]]
        return {
            "area_km": list(self.cfg.area_km),
            "area_km2": round(area_km2, 3),
            "total_cells": self.n_cells,
            "channel": {
                "interference": self.cfg.interference,
                "shadow_fading": self.cfg.shadow_fading,
                "traffic_profile": self.cfg.traffic_profile,
            },
            "tiers": tiers,
            "capacity": {
                "network_mbps": _sum("tier_capacity_mbps"),
                "network_loaded_mbps": _sum("loaded_tier_capacity_mbps"),
                "by_tier_mbps": {t["tier"]: t["tier_capacity_mbps"] for t in tiers},
                "by_tier_loaded_mbps": {t["tier"]: t["loaded_tier_capacity_mbps"] for t in tiers},
                "by_slice_mbps": by_slice,
                "by_slice_loaded_mbps": by_slice_loaded,
                "basis": "peak = full PRB, SE ceiling, no interference; "
                         "loaded = interference-limited SE at ~50% neighbour load",
            },
            "coverage": {
                "area_km2": round(area_km2, 3),
                "fade_margin": "90th-percentile (1.28 sigma) shadow fading"
                if self.cfg.shadow_fading else "fixed 8 dB",
                "by_tier": {t["tier"]: t["area_covered"] for t in tiers},
                "terrestrial_best": round(max((t["area_covered"] for t in terrestrial),
                                              default=0.0), 3),
                "ntn_backstop": any(t["ntn"] and t["area_covered"] > 0 for t in tiers),
            },
            "sessions": self.pop.summary() if hasattr(self, "pop") else {},
            "indoor_fraction": self.cfg.indoor_fraction,
        }

    def _loaded_se(self, tier: Tier, r_deploy_m: float) -> float:
        """Screening interference-limited spectral efficiency, hex reuse-1.

        UE at a cell-edge (0.7 r) and a mid-cell (0.35 r) point; six first-ring
        interferers at sqrt(3) r, each at 50% activity, scaled by the tier's
        beam-coupling factor. Static analogue of the per-session SINR the
        simulation computes under load.
        """
        pl_tbl = self.pl_tables.get(tier.name)
        if pl_tbl is None or r_deploy_m <= 1:
            return tier.max_se_bps_hz
        eirp_re = (tier.tx_power_dbm - 10 * math.log10(max(1, tier.resolved_prbs_per_cell * 12))
                   + tier.antenna_gain_dbi)
        noise_lin = 10 ** ((-174 + 10 * math.log10(tier.resolved_prb_bandwidth_mhz * 1e6)
                            + tier.noise_figure_db) / 10)
        d_int = 1.732 * r_deploy_m
        pl_int = float(np.interp(d_int, self.pl_distance_m, pl_tbl))
        i_lin = 12 * 6 * 10 ** ((eirp_re - pl_int) / 10) * tier.interference_coupling * 0.5
        ses = []
        for frac in (0.7, 0.35):
            d = max(1.0, r_deploy_m * frac)
            s_dbm = eirp_re + 10 * math.log10(12) - float(np.interp(d, self.pl_distance_m, pl_tbl))
            sinr = 10 ** (s_dbm / 10) / (noise_lin + i_lin)
            ses.append(min(math.log2(1 + max(1e-6, sinr)), tier.max_se_bps_hz))
        return sum(ses) / len(ses)


LIMITATIONS = [
    "TR 38.901 path loss is evaluated from a per-tier distance table with a "
    "LOS-probability blend; no per-link shadow fading realization or ray tracing.",
    "TR 38.811/38.821 NTN link budget is screening-grade: a single beam, a "
    "circular orbit over a non-rotating Earth, and no Earth-rotation Doppler term.",
    "Interference is not modelled explicitly; SINR is computed against thermal "
    "noise, so utilization and spectral efficiency are optimistic.",
    "Wi-Fi is modelled as a scheduled tier, not as CSMA/CA with contention.",
    "Handover failure probabilities are parametric, not derived from a "
    "measurement-report and RRC-signalling simulation.",
    "Synthetic session population; not a carrier trace.",
    "SLA figures are modeled interval compliance, not proof of five- or "
    "seven-nines reliability.",
]
