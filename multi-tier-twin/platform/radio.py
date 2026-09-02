"""Propagation and link-budget models for the multi-tier twin.

Terrestrial tiers use 3GPP TR 38.901 UMa/UMi/RMa path loss, which takes base
station and UE antenna heights explicitly, so site height is a real input
rather than a fudge factor folded into a margin.

Non-terrestrial tiers use the 3GPP TR 38.811 NTN channel model and the TR 38.821
system parameters: free-space loss over the slant range, gaseous absorption,
scintillation, elevation-dependent shadow fading, ITU-R P.618 rain attenuation
(with the P.838 coefficients and P.839 rain height), and the Doppler shift and
Doppler rate of a circular LEO orbit.

Everything here is a screening-grade implementation of the published models:
correct in form and parameterization, but not a ray-traced or measurement-
calibrated planning tool. Deployment decisions need a vendor planning model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

SPEED_OF_LIGHT = 299_792_458.0
EARTH_RADIUS_KM = 6371.0


# ==========================================================================
# Terrestrial: 3GPP TR 38.901
# ==========================================================================

TERRESTRIAL_SCENARIOS = ("UMa", "UMi", "RMa", "InH")


def _breakpoint_distance_m(h_bs_m: float, h_ut_m: float, fc_ghz: float,
                           h_e_m: float = 1.0) -> float:
    """TR 38.901 breakpoint distance d'_BP with effective antenna heights."""
    h_bs_eff = max(0.1, h_bs_m - h_e_m)
    h_ut_eff = max(0.1, h_ut_m - h_e_m)
    return 4 * h_bs_eff * h_ut_eff * (fc_ghz * 1e9) / SPEED_OF_LIGHT


def _d3d(d_2d_m: float, h_bs_m: float, h_ut_m: float) -> float:
    return math.sqrt(d_2d_m ** 2 + (h_bs_m - h_ut_m) ** 2)


def pathloss_uma(d_2d_m: float, h_bs_m: float, h_ut_m: float, fc_ghz: float,
                 los: bool) -> float:
    """TR 38.901 Table 7.4.1-1, Urban Macro."""
    d3 = _d3d(d_2d_m, h_bs_m, h_ut_m)
    d_bp = _breakpoint_distance_m(h_bs_m, h_ut_m, fc_ghz)
    if d_2d_m <= d_bp:
        pl_los = 28.0 + 22 * math.log10(d3) + 20 * math.log10(fc_ghz)
    else:
        pl_los = (28.0 + 40 * math.log10(d3) + 20 * math.log10(fc_ghz)
                  - 9 * math.log10(d_bp ** 2 + (h_bs_m - h_ut_m) ** 2))
    if los:
        return pl_los
    pl_nlos = (13.54 + 39.08 * math.log10(d3) + 20 * math.log10(fc_ghz)
               - 0.6 * (h_ut_m - 1.5))
    return max(pl_los, pl_nlos)


def pathloss_umi(d_2d_m: float, h_bs_m: float, h_ut_m: float, fc_ghz: float,
                 los: bool) -> float:
    """TR 38.901 Table 7.4.1-1, Urban Micro street canyon."""
    d3 = _d3d(d_2d_m, h_bs_m, h_ut_m)
    d_bp = _breakpoint_distance_m(h_bs_m, h_ut_m, fc_ghz)
    if d_2d_m <= d_bp:
        pl_los = 32.4 + 21 * math.log10(d3) + 20 * math.log10(fc_ghz)
    else:
        pl_los = (32.4 + 40 * math.log10(d3) + 20 * math.log10(fc_ghz)
                  - 9.5 * math.log10(d_bp ** 2 + (h_bs_m - h_ut_m) ** 2))
    if los:
        return pl_los
    pl_nlos = (35.3 * math.log10(d3) + 22.4 + 21.3 * math.log10(fc_ghz)
               - 0.3 * (h_ut_m - 1.5))
    return max(pl_los, pl_nlos)


def pathloss_rma(d_2d_m: float, h_bs_m: float, h_ut_m: float, fc_ghz: float,
                 los: bool, h_building_m: float = 5.0, w_street_m: float = 20.0) -> float:
    """TR 38.901 Table 7.4.1-1, Rural Macro."""
    d3 = _d3d(d_2d_m, h_bs_m, h_ut_m)
    d_bp = 2 * math.pi * h_bs_m * h_ut_m * (fc_ghz * 1e9) / SPEED_OF_LIGHT
    h = h_building_m

    def _pl1(d: float) -> float:
        return (20 * math.log10(40 * math.pi * d * fc_ghz / 3)
                + min(0.03 * h ** 1.72, 10) * math.log10(d)
                - min(0.044 * h ** 1.72, 14.77)
                + 0.002 * math.log10(h) * d)

    pl_los = _pl1(d3) if d_2d_m <= d_bp else _pl1(d_bp) + 40 * math.log10(d3 / d_bp)
    if los:
        return pl_los
    pl_nlos = (161.04 - 7.1 * math.log10(w_street_m) + 7.5 * math.log10(h)
               - (24.37 - 3.7 * (h / h_bs_m) ** 2) * math.log10(h_bs_m)
               + (43.42 - 3.1 * math.log10(h_bs_m)) * (math.log10(d3) - 3)
               + 20 * math.log10(fc_ghz)
               - (3.2 * (math.log10(11.75 * h_ut_m)) ** 2 - 4.97))
    return max(pl_los, pl_nlos)


def pathloss_inh(d_2d_m: float, h_bs_m: float, h_ut_m: float, fc_ghz: float,
                 los: bool) -> float:
    """TR 38.901 Table 7.4.1-1, Indoor Hotspot / office — the Wi-Fi and indoor
    small-cell tier."""
    d3 = max(1.0, _d3d(d_2d_m, h_bs_m, h_ut_m))
    pl_los = 32.4 + 17.3 * math.log10(d3) + 20 * math.log10(fc_ghz)
    if los:
        return pl_los
    pl_nlos = 38.3 * math.log10(d3) + 17.30 + 24.9 * math.log10(fc_ghz)
    return max(pl_los, pl_nlos)


def o2i_penetration_loss_db(fc_ghz: float, high_loss: bool = False,
                            indoor_depth_m: float = 8.0) -> float:
    """TR 38.901 7.4.3 outdoor-to-indoor loss: building entry plus indoor depth.

    The high-loss model (IRR glass, thermally efficient construction) is what
    makes an outdoor macro effectively useless deep inside a modern building —
    and therefore what justifies the indoor Wi-Fi tier.
    """
    l_glass = 2 + 0.2 * fc_ghz
    l_concrete = 5 + 4 * fc_ghz
    l_irr_glass = 23 + 0.3 * fc_ghz
    if high_loss:
        pl_tw = 5 - 10 * math.log10(0.7 * 10 ** (-l_irr_glass / 10)
                                    + 0.3 * 10 ** (-l_concrete / 10))
    else:
        pl_tw = 5 - 10 * math.log10(0.3 * 10 ** (-l_glass / 10)
                                    + 0.7 * 10 ** (-l_concrete / 10))
    return pl_tw + 0.5 * indoor_depth_m


def los_probability(scenario: str, d_2d_m: float, h_ut_m: float = 1.5) -> float:
    """TR 38.901 Table 7.4.2-1 line-of-sight probability."""
    d = max(1.0, d_2d_m)
    if scenario == "InH":
        if d <= 5:
            return 1.0
        if d <= 49:
            return math.exp(-(d - 5) / 70.8)
        return math.exp(-(d - 49) / 211.7) * 0.54
    if scenario == "RMa":
        return 1.0 if d <= 10 else math.exp(-(d - 10) / 1000)
    if scenario == "UMi":
        return 1.0 if d <= 18 else (18 / d + math.exp(-d / 36) * (1 - 18 / d))
    # UMa, with the h_UT correction for users above 13 m.
    if d <= 18:
        return 1.0
    c = 0.0 if h_ut_m < 13 else ((h_ut_m - 13) / 10) ** 1.5
    base = 18 / d + math.exp(-d / 63) * (1 - 18 / d)
    return min(1.0, base * (1 + c * 1.25 * (d / 100) ** 3 * math.exp(-d / 150)))


def pathloss(scenario: str, d_2d_m: float, h_bs_m: float, h_ut_m: float,
             fc_ghz: float) -> float:
    """LOS-probability-weighted path loss, averaged in the linear domain."""
    if scenario not in TERRESTRIAL_SCENARIOS:
        raise ValueError(f"Unsupported scenario: {scenario}")
    fn = {"UMa": pathloss_uma, "UMi": pathloss_umi, "RMa": pathloss_rma,
          "InH": pathloss_inh}[scenario]
    p_los = los_probability(scenario, d_2d_m, h_ut_m)
    gain_los = 10 ** (-fn(d_2d_m, h_bs_m, h_ut_m, fc_ghz, True) / 10)
    gain_nlos = 10 ** (-fn(d_2d_m, h_bs_m, h_ut_m, fc_ghz, False) / 10)
    return -10 * math.log10(p_los * gain_los + (1 - p_los) * gain_nlos)


def cell_radius_m(scenario: str, max_path_loss_db: float, h_bs_m: float,
                  h_ut_m: float, fc_ghz: float,
                  bounds_m: Tuple[float, float] = (10.0, 20_000.0)) -> float:
    """Largest 2-D distance whose path loss still fits the link budget."""
    lo, hi = bounds_m
    if pathloss(scenario, lo, h_bs_m, h_ut_m, fc_ghz) > max_path_loss_db:
        return lo
    if pathloss(scenario, hi, h_bs_m, h_ut_m, fc_ghz) <= max_path_loss_db:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if pathloss(scenario, mid, h_bs_m, h_ut_m, fc_ghz) <= max_path_loss_db:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def thermal_noise_dbm(bandwidth_hz: float, noise_figure_db: float) -> float:
    return -174 + 10 * math.log10(max(1.0, bandwidth_hz)) + noise_figure_db


def max_path_loss_db(tx_power_dbm: float, tx_gain_dbi: float, rx_gain_dbi: float,
                     bandwidth_hz: float, noise_figure_db: float,
                     required_sinr_db: float, margin_db: float,
                     feeder_loss_db: float = 0.0) -> float:
    """Link budget: how much path loss the link can absorb and still close."""
    sensitivity = thermal_noise_dbm(bandwidth_hz, noise_figure_db) + required_sinr_db
    return (tx_power_dbm + tx_gain_dbi + rx_gain_dbi
            - sensitivity - margin_db - feeder_loss_db)


# ==========================================================================
# Non-terrestrial: 3GPP TR 38.811 / TR 38.821
# ==========================================================================

def slant_range_km(altitude_km: float, elevation_deg: float) -> float:
    """TR 38.821 Eq. for slant range from elevation angle over a spherical Earth."""
    el = math.radians(elevation_deg)
    r = EARTH_RADIUS_KM
    return math.sqrt((r * math.sin(el)) ** 2 + 2 * r * altitude_km + altitude_km ** 2) \
        - r * math.sin(el)


def central_angle_deg(altitude_km: float, elevation_deg: float) -> float:
    """Geocentric angle between the satellite subpoint and the ground station."""
    el = math.radians(elevation_deg)
    r = EARTH_RADIUS_KM
    # From the sine rule on the Earth-centre / satellite / station triangle.
    return math.degrees(math.acos(min(1.0, (r / (r + altitude_km)) * math.cos(el))) - el)


def elevation_from_central_angle_deg(altitude_km: float, phi_deg: float) -> float:
    """Inverse of `central_angle_deg`: elevation seen at geocentric angle phi."""
    phi = math.radians(abs(phi_deg))
    r, rs = EARTH_RADIUS_KM, EARTH_RADIUS_KM + altitude_km
    d = math.sqrt(r ** 2 + rs ** 2 - 2 * r * rs * math.cos(phi))
    if d <= 1e-9:
        return 90.0
    cos_el = min(1.0, max(-1.0, rs * math.sin(phi) / d))
    return math.degrees(math.acos(cos_el))


def free_space_loss_db(freq_ghz: float, distance_km: float) -> float:
    return 92.45 + 20 * math.log10(freq_ghz) + 20 * math.log10(max(1e-6, distance_km))


# Zenith gaseous attenuation, dB, standard atmosphere at sea level (ITU-R P.676,
# as tabulated for NTN in TR 38.811 Section 6.6.4). The 22.2 GHz water-vapour
# line is the reason this is not monotonic in frequency.
_ZENITH_GAS_DB: Tuple[Tuple[float, float], ...] = (
    (1.0, 0.0330), (2.0, 0.0350), (4.0, 0.0400), (6.0, 0.0450), (8.0, 0.0530),
    (10.0, 0.0620), (12.0, 0.0750), (14.0, 0.0950), (18.0, 0.1600),
    (20.0, 0.2100), (22.0, 0.2900), (24.0, 0.2400), (28.0, 0.1900),
    (30.0, 0.1900), (40.0, 0.3300),
)


def _interp(table: Sequence[Tuple[float, float]], x: float) -> float:
    """Linear interpolation in log-frequency, clamped at the table edges."""
    if x <= table[0][0]:
        return table[0][1]
    if x >= table[-1][0]:
        return table[-1][1]
    for (x0, y0), (x1, y1) in zip(table, table[1:]):
        if x0 <= x <= x1:
            t = (math.log10(x) - math.log10(x0)) / (math.log10(x1) - math.log10(x0))
            return y0 + t * (y1 - y0)
    return table[-1][1]


def gaseous_absorption_db(freq_ghz: float, elevation_deg: float) -> float:
    """TR 38.811 6.6.4: zenith attenuation scaled by the secant of the zenith angle."""
    el = math.radians(max(5.0, elevation_deg))
    return _interp(_ZENITH_GAS_DB, freq_ghz) / math.sin(el)


def scintillation_db(freq_ghz: float, elevation_deg: float) -> float:
    """TR 38.811 6.6.6: ionospheric below ~6 GHz, tropospheric above.

    Ionospheric scintillation scales roughly as f^-1.5 from the 1.1 dB / 4 GHz
    reference; tropospheric scintillation grows as elevation falls.
    """
    el = math.radians(max(5.0, elevation_deg))
    if freq_ghz < 6.0:
        return min(4.0, 1.1 * (4.0 / freq_ghz) ** 1.5)
    return 0.5 * (freq_ghz / 20.0) ** (7 / 12) / math.sqrt(math.sin(el))


# TR 38.811 Table 6.6.2-1, dense-urban S-band LOS shadow fading sigma by elevation.
_SHADOW_SIGMA_DB: Tuple[Tuple[float, float], ...] = (
    (10.0, 3.5), (20.0, 3.4), (30.0, 2.9), (40.0, 3.0), (50.0, 3.1),
    (60.0, 2.7), (70.0, 2.5), (80.0, 2.3), (90.0, 1.2),
)


def shadow_fading_std_db(elevation_deg: float, los: bool = True) -> float:
    """Elevation-dependent shadow fading sigma; NLOS is far deeper than LOS."""
    el = min(90.0, max(10.0, elevation_deg))
    sigma = _SHADOW_SIGMA_DB[-1][1]
    for (e0, s0), (e1, s1) in zip(_SHADOW_SIGMA_DB, _SHADOW_SIGMA_DB[1:]):
        if e0 <= el <= e1:
            sigma = s0 + (el - e0) / (e1 - e0) * (s1 - s0)
            break
    return sigma if los else sigma * 4.5


# ITU-R P.838-3 specific attenuation coefficients (horizontal polarization).
_RAIN_K_ALPHA: Tuple[Tuple[float, float, float], ...] = (
    (1.0, 0.0000259, 0.9691), (2.0, 0.0000847, 1.0664), (4.0, 0.0001071, 1.6009),
    (6.0, 0.000650, 1.1216), (8.0, 0.001915, 1.0440), (10.0, 0.01217, 1.2571),
    (12.0, 0.02386, 1.1825), (15.0, 0.04481, 1.1233), (20.0, 0.09164, 1.0568),
    (25.0, 0.1571, 1.0060), (30.0, 0.2403, 0.9485), (35.0, 0.3374, 0.8953),
    (40.0, 0.4431, 0.8516),
)


def rain_coefficients(freq_ghz: float, polarization_tilt_deg: float = 45.0) -> Tuple[float, float]:
    """ITU-R P.838 k and alpha, tilt-adjusted for the polarization angle."""
    freqs = [row[0] for row in _RAIN_K_ALPHA]
    k_h = _interp(tuple((f, r[1]) for f, r in zip(freqs, _RAIN_K_ALPHA)), freq_ghz)
    a_h = _interp(tuple((f, r[2]) for f, r in zip(freqs, _RAIN_K_ALPHA)), freq_ghz)
    # Vertical polarization runs a little lower; a 45-degree/circular tilt sits
    # between the two, which is the usual NTN case.
    k_v, a_v = k_h * 0.92, a_h * 0.98
    tau = math.radians(polarization_tilt_deg)
    k = (k_h + k_v + (k_h - k_v) * math.cos(tau) ** 2) / 2
    a = (k_h * a_h + k_v * a_v + (k_h * a_h - k_v * a_v) * math.cos(tau) ** 2) / (2 * k)
    return k, a


def rain_height_km(latitude_deg: float = 45.0) -> float:
    """ITU-R P.839 rain height: the 0-degree isotherm plus 0.36 km."""
    if abs(latitude_deg) < 23:
        h0 = 5.0
    else:
        h0 = 5.0 - 0.075 * (abs(latitude_deg) - 23)
    return max(0.5, h0 + 0.36)


def rain_attenuation_db(freq_ghz: float, elevation_deg: float,
                        rain_rate_mm_h: float, station_height_km: float = 0.05,
                        latitude_deg: float = 45.0,
                        polarization_tilt_deg: float = 45.0) -> float:
    """ITU-R P.618 slant-path rain attenuation exceeded for 0.01% of a year.

    S-band NTN barely notices rain; Ka-band is dominated by it, which is exactly
    the trade the twin needs to represent when it chooses a satellite band.
    """
    if rain_rate_mm_h <= 0:
        return 0.0
    h_r = rain_height_km(latitude_deg)
    if h_r <= station_height_km:
        return 0.0
    el = math.radians(max(5.0, elevation_deg))
    l_s = (h_r - station_height_km) / math.sin(el)      # slant path in rain, km
    l_g = l_s * math.cos(el)                            # horizontal projection
    k, alpha = rain_coefficients(freq_ghz, polarization_tilt_deg)
    gamma_r = k * rain_rate_mm_h ** alpha               # dB/km
    # P.618 horizontal reduction factor.
    r_001 = 1 / (1 + 0.78 * math.sqrt(l_g * gamma_r / freq_ghz)
                 - 0.38 * (1 - math.exp(-2 * l_g)))
    return gamma_r * l_s * max(0.0, r_001)


def orbital_velocity_km_s(altitude_km: float) -> float:
    """Circular orbital speed; 7.56 km/s at the 600 km LEO reference of TR 38.821."""
    mu = 398_600.4418  # Earth gravitational parameter, km^3/s^2
    return math.sqrt(mu / (EARTH_RADIUS_KM + altitude_km))


def doppler_shift_at_phi_hz(freq_ghz: float, altitude_km: float,
                            phi_deg: float) -> float:
    """Doppler as a signed function of the geocentric angle along the pass.

    `phi_deg` is positive while the satellite is still approaching and negative
    after it passes overhead, so the shift is odd in phi and crosses zero at
    zenith. Working in phi rather than elevation is what keeps the derivative
    well behaved through the zenith crossing.
    """
    phi = math.radians(phi_deg)
    r, rs = EARTH_RADIUS_KM, EARTH_RADIUS_KM + altitude_km
    d = math.sqrt(r ** 2 + rs ** 2 - 2 * r * rs * math.cos(phi))
    omega = orbital_velocity_km_s(altitude_km) / rs                # rad/s
    closing_rate = (r * rs * math.sin(phi) / max(1e-9, d)) * omega  # km/s
    return freq_ghz * 1e9 * closing_rate / (SPEED_OF_LIGHT / 1000.0)


def doppler_shift_hz(freq_ghz: float, altitude_km: float, elevation_deg: float,
                     approaching: bool = True) -> float:
    """TR 38.821 6.1.3: Doppler from the range rate of a circular orbit.

    Maximum magnitude at the lowest elevation, zero at zenith. At 2 GHz and
    600 km this peaks near +/-45 kHz, against the ~+/-48 kHz TR 38.821 quotes
    for S-band LEO (the balance is Earth rotation, which this model omits).
    """
    phi = central_angle_deg(altitude_km, elevation_deg)
    shift = doppler_shift_at_phi_hz(freq_ghz, altitude_km, phi)
    return shift if approaching else -shift


def doppler_rate_hz_s(freq_ghz: float, altitude_km: float,
                      elevation_deg: float) -> float:
    """Rate of change of the Doppler shift, differentiated along the pass.

    Peaks at zenith — where the shift itself is zero but sweeping fastest — which
    is the case that sets the frequency-tracking requirement.
    """
    phi = central_angle_deg(altitude_km, elevation_deg)
    omega_deg_s = math.degrees(orbital_velocity_km_s(altitude_km)
                               / (EARTH_RADIUS_KM + altitude_km))
    dt = 0.25
    dphi = omega_deg_s * dt
    # phi decreases at omega while the satellite approaches, hence the sign.
    ahead = doppler_shift_at_phi_hz(freq_ghz, altitude_km, phi - dphi)
    behind = doppler_shift_at_phi_hz(freq_ghz, altitude_km, phi + dphi)
    return (ahead - behind) / (2 * dt)


def doppler_from_velocity_hz(freq_ghz: float, speed_kmh: float,
                             angle_deg: float = 0.0) -> float:
    """Doppler from UE motion: f_c * v * cos(theta) / c.

    `angle_deg` is between the velocity vector and the line of sight to the
    serving point, so a UE crossing a cell broadside sees near-zero shift while
    one driving straight at it sees the maximum.
    """
    v_ms = speed_kmh / 3.6
    return freq_ghz * 1e9 * v_ms * math.cos(math.radians(angle_deg)) / SPEED_OF_LIGHT


def coherence_time_s(freq_ghz: float, speed_kmh: float) -> float:
    """Clarke's coherence time, 0.423/f_d — how long a channel estimate lasts."""
    f_d = abs(doppler_from_velocity_hz(freq_ghz, max(0.1, speed_kmh), 0.0))
    return 0.423 / max(1e-9, f_d)


def ici_ratio(residual_doppler_hz: float, subcarrier_spacing_hz: float) -> float:
    """Inter-carrier interference power ratio from a residual frequency offset.

    Standard small-offset OFDM result: ICI/S ~ (pi*eps)^2/3 for eps = df/SCS.
    This is why NTN pushes to wider subcarrier spacing.
    """
    eps = abs(residual_doppler_hz) / max(1.0, subcarrier_spacing_hz)
    return (math.pi * eps) ** 2 / 3


def spectral_efficiency_with_ici(peak_se_bps_hz: float, ici: float) -> float:
    """Derate a peak spectral efficiency by ICI, through the implied SINR."""
    if peak_se_bps_hz <= 0:
        return 0.0
    snr = 2 ** peak_se_bps_hz - 1
    sinr = snr / (1 + snr * max(0.0, ici))
    return max(0.0, math.log2(1 + sinr))


@dataclass
class NTNLink:
    """Everything the twin needs to know about the satellite link this step."""

    elevation_deg: float
    slant_range_km: float
    free_space_loss_db: float
    gaseous_absorption_db: float
    scintillation_db: float
    rain_attenuation_db: float
    shadow_fading_db: float
    total_loss_db: float
    link_margin_db: float
    doppler_shift_hz: float
    doppler_rate_hz_s: float
    residual_doppler_hz: float
    ici_ratio: float
    one_way_delay_ms: float

    @property
    def closes(self) -> bool:
        return self.link_margin_db > 0.0


def ntn_link_budget(freq_ghz: float, altitude_km: float, elevation_deg: float,
                    eirp_dbm: float, rx_gain_dbi: float, bandwidth_hz: float,
                    noise_figure_db: float, required_sinr_db: float,
                    rain_rate_mm_h: float = 0.0,
                    subcarrier_spacing_hz: float = 30_000.0,
                    doppler_precompensation: float = 0.95,
                    shadow_fading_db: float = 0.0,
                    implementation_margin_db: float = 3.0) -> NTNLink:
    """Full TR 38.811/38.821 screening link budget for one satellite pass point."""
    d_km = slant_range_km(altitude_km, elevation_deg)
    fspl = free_space_loss_db(freq_ghz, d_km)
    gas = gaseous_absorption_db(freq_ghz, elevation_deg)
    scint = scintillation_db(freq_ghz, elevation_deg)
    rain = rain_attenuation_db(freq_ghz, elevation_deg, rain_rate_mm_h)
    total = fspl + gas + scint + rain + shadow_fading_db + implementation_margin_db

    noise = thermal_noise_dbm(bandwidth_hz, noise_figure_db)
    margin = eirp_dbm + rx_gain_dbi - total - noise - required_sinr_db

    shift = doppler_shift_hz(freq_ghz, altitude_km, elevation_deg)
    rate = doppler_rate_hz_s(freq_ghz, altitude_km, elevation_deg)
    residual = shift * (1 - min(1.0, max(0.0, doppler_precompensation)))
    return NTNLink(
        elevation_deg=elevation_deg,
        slant_range_km=d_km,
        free_space_loss_db=fspl,
        gaseous_absorption_db=gas,
        scintillation_db=scint,
        rain_attenuation_db=rain,
        shadow_fading_db=shadow_fading_db,
        total_loss_db=total,
        link_margin_db=margin,
        doppler_shift_hz=shift,
        doppler_rate_hz_s=rate,
        residual_doppler_hz=residual,
        ici_ratio=ici_ratio(residual, subcarrier_spacing_hz),
        one_way_delay_ms=d_km / (SPEED_OF_LIGHT / 1000.0) * 1000.0,
    )
