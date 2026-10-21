#!/usr/bin/env python3
"""budget_check.py — DL/UL user-throughput check on twin5GNR's OWN stochastic
3GPP layer (planner_v4 machinery), closed with the budget terms of the user's
LTE link budget workbook (LTE_Link_budgetsNP_5MHz_20W_PA12.xlsm,
Qualcomm 80-W2559-5 Rev A).

WHY THIS SCRIPT EXISTS
  The RT layer (single isolated tower, rooftop RX at 6 m, noise-only) gives
  DL/UL map ratios of 1.94/1.76/1.71/1.71/1.18 (UL v3) — far from the field's
  3-5 and from the workbook's verdict (UL limiting in every band/morphology).
  This script asks: what does the twin's OTHER layer say — the stochastic
  3GPP UMa layer in planner_v4.py, evaluated at street level (RX 1.5 m),
  sectorized, with reuse-1 multi-sector interference — when DL and UL are
  each closed as a per-direction budget exactly the way the workbook does?

METHOD (per band; sites = the 4-site Rancho Bernardo layout, street level)
  - Per-site omni street PL from 3GPP UMa (planner_v4.uma_pl) on the real
    terrain/building LOS grids; RSRP_omni_street = 47 - PL; sectorize to
    12 sectors (TR 38.901 pattern, attenuation-only, as in planner_v4).
  - DL: best-server SINR with co-channel interference from ALL other
    sectors (reuse-1, fully loaded; this explicit interference REPLACES the
    workbook's analytic geometry term Ioc/Ior +3.94 dB — not added on top).
    Noise = thermal + 10log10(BW) + UE NF 7 dB (workbook).
    TP = BW * min(7, log2(1+SINR)) * (1 - 0.3039)   [workbook DL overhead]
    Unserved (SINR < -10.3 dB, workbook PDSCH 256K required SNR) -> TP 0.
  - UL: received grid per sector = UE_EIRP - PL + sector pattern;
    UE_EIRP = 23 + g_ue(f), g_ue(f) = 6.7292*ln(f_MHz) - 51.61 (workbook
    handset antenna fit; -7.53 dBi @ 700 MHz), clamped above 3600 MHz to
    the 3600 MHz value (+3.49 dBi) — documented adaptation, the LTE-era fit
    is not credible at FR3/mmWave. Serving sector = DL best server
    (reciprocity). Noise = thermal + 10log10(BW) + eNB NF 3 dB + target
    RoT 4 dB (workbook UL interference treatment).
    TP = BW * min(5, log2(1+SINR)) * (1 - 0.2234)   [workbook UL overhead]
    Unserved (SINR < -1.8 dB, workbook PUSCH required SNR) -> TP 0.
  - Averages are over ALL tile cells with unserved = 0 (edge-weighted, the
    way a user population spread over the cell area experiences it).

planner_v4.py is NOT imported: importing it executes its full pipeline
(scene build, site selection, JSON dumps, FC-specific asserts). The needed
functions (los_grid_rxsurf, uma_pl, pattern_gain, sectorize, and the street
branch of compute_omni) are copied verbatim below with attribution, and the
terrain/building rasters are loaded from planner_v4's own saved outputs
(data_terrain_500.npy, data_bldh_500.npy, data_sites_v4.json).

Known planner_v4 convention kept faithfully: uma_pl is called with
h_bs = TX_MAST_M (3 m) — that is what compute_omni does (the breakpoint
term), while the 3D distance uses the true rooftop TX height.
At 28 GHz the UMa formulas are extrapolated beyond their defined range
(38.901 UMa is a sub-6/FR3 model) and the tile is NLOS-dominated; the
28 GHz row is flagged wherever it appears.
"""
import json, math, time
import numpy as np
from numba import njit

# ============ constants copied from planner_v4.py (verbatim values) ============
S, W, N, E = 32.9925, -117.1120, 33.0375, -117.0580
GRID = 500
C = 3e8
TX_EIRP_DBM = 47.0
TX_MAST_M = 3.0              # rooftop mast above roof
RX_STREET_M = 1.5            # facade reference height above terrain
SECTOR_AZIMUTHS = [0.0, 120.0, 240.0]
HPBW_DEG = 65.0
A_MAX_DB = 30.0

lat0 = (S + N) / 2
step_m = (N - S) / GRID * 111320.0
rr, cc = np.mgrid[0:GRID, 0:GRID]

# ============ workbook budget terms (LTE_Link_budgetsNP_5MHz_20W_PA12.xlsm) ====
UE_P_DBM = 23.0              # user-specified (workbook uses 24 — noted)
UE_NF_DB = 7.0               # workbook DL budgets: Receiver Noise Figure UE
ENB_NF_DB = 3.0              # workbook UL: Receiver Noise Figure eNB
ROT_DB = 4.0                 # workbook UL: Target RoT
DL_OVERHEAD = 0.3039         # workbook DL total overhead (2 ant, 3 PDCCH sym)
UL_OVERHEAD = 0.2234         # workbook UL total overhead
DL_SE_CAP = 7.0              # workbook-class DL cap (twin convention)
UL_SE_CAP = 5.0              # single-layer UE cap (UL v3, retained)
DL_REQ_SNR_DB = -10.3        # workbook PDSCH 256K required SNR
UL_REQ_SNR_DB = -1.8         # workbook PUSCH required SNR (UL_Combined)

BANDS = [(700, 10e6), (2500, 10e6), (3600, 10e6), (7125, 10e6), (28000, 200e6)]


def g_ue(f_mhz):
    """Workbook handset antenna fit, clamped above 3600 MHz (adaptation)."""
    g = 6.7292 * math.log(f_mhz) - 51.61
    return min(g, 6.7292 * math.log(3600.0) - 51.61)


# ============ functions copied from planner_v4.py (verbatim, attribution) ====
@njit
def los_grid_rxsurf(surf, rx_surf, sr, sc, tx_h, rx_h, n):
    # copied verbatim from planner_v4.py
    out = np.zeros((n, n), dtype=np.bool_)
    txz = surf[sr, sc] + tx_h
    for r in range(n):
        for c in range(n):
            if r == sr and c == sc:
                out[r, c] = True; continue
            dr = r - sr; dc = c - sc
            steps = max(abs(dr), abs(dc))
            if steps == 0:
                out[r, c] = True; continue
            rxz = rx_surf[r, c] + rx_h
            blocked = False
            for k in range(1, steps):
                f = k / steps
                rr_ = int(sr + dr * f); cc_ = int(sc + dc * f)
                linez = txz + (rxz - txz) * f
                if surf[rr_, cc_] + 0.6 > linez:
                    blocked = True; break
            out[r, c] = not blocked
    return out


def uma_pl(d3d, los, h_bs, h_ut, fc_mhz):
    # copied verbatim from planner_v4.py except FC_MHZ/FC_HZ globals -> fc_mhz arg
    d2d = np.maximum(d3d, 1.0)
    fc_ghz = fc_mhz / 1000.0
    pl_los_seg1 = 28.0 + 22 * np.log10(d3d) + 20 * np.log10(fc_ghz)
    d_bp = 4 * h_bs * h_ut * (fc_mhz * 1e6) / C
    pl_los_seg2 = (28.0 + 40 * np.log10(d3d) + 20 * np.log10(fc_ghz)
                   - 9 * np.log10(d_bp ** 2 + (h_bs - h_ut) ** 2))
    pl_los = np.where(d2d < d_bp, pl_los_seg1, pl_los_seg2)
    pl_nlos = 13.54 + 39.08 * np.log10(d3d) + 20 * np.log10(fc_ghz) - 0.6 * (h_ut - 1.5)
    pl = np.where(los, pl_los, np.maximum(pl_los, pl_nlos))
    return pl


def pattern_gain(az_rel_deg, theta_deg):
    # copied verbatim from planner_v4.py
    a_h = -np.minimum(12.0 * (az_rel_deg / HPBW_DEG) ** 2, A_MAX_DB)
    a_v = -np.minimum(12.0 * ((theta_deg - 90.0) / HPBW_DEG) ** 2, A_MAX_DB)
    return -np.minimum(-(a_h + a_v), A_MAX_DB)


def sectorize(rsrp_omni, sites, dh):
    # copied verbatim from planner_v4.py
    nsites = len(sites); nsec = len(SECTOR_AZIMUTHS)
    step_x = (E - W) / GRID * 111320.0 * math.cos(math.radians(lat0))
    step_y = (N - S) / GRID * 110540.0
    rsrp_sec = np.zeros((nsites * nsec, GRID, GRID))
    for i, st in enumerate(sites):
        sr, sc = st['grid']
        dx = (cc - sc) * step_x
        dy = (sr - rr) * step_y
        az = (np.degrees(np.arctan2(dx, dy)) % 360.0)
        d2d = np.sqrt(dx ** 2 + dy ** 2)
        theta = 90.0 + np.degrees(np.arctan2(dh, np.maximum(d2d, 1.0)))
        for s, bore in enumerate(SECTOR_AZIMUTHS):
            rel = ((az - bore + 180.0) % 360.0) - 180.0
            rsrp_sec[i * nsec + s] = rsrp_omni[i] + pattern_gain(rel, theta)
    return rsrp_sec


# ============ scene (planner_v4's saved rasters) + street geometry cache =====
terrain = np.load('data_terrain_500.npy')
bld_h = np.load('data_bldh_500.npy')
surface = terrain + bld_h
sites = json.load(open('data_sites_v4.json'))
for st in sites:
    st['grid'] = tuple(st['grid'])
print('sites:', [(st['name'], st['grid']) for st in sites])

t0 = time.time()
street = []   # per site: (los_s, d3d_s) — frequency-independent
for st in sites:
    sr, sc = st['grid']
    txz = surface[sr, sc] + TX_MAST_M
    # street branch of planner_v4.compute_omni, verbatim geometry
    los_s = los_grid_rxsurf(surface, terrain, sr, sc, TX_MAST_M, RX_STREET_M, GRID)
    d3d_s = np.sqrt(((rr - sr) * step_m) ** 2 + ((cc - sc) * step_m) ** 2 +
                    (terrain + RX_STREET_M - txz) ** 2)
    d3d_s = np.maximum(d3d_s, 3.0)
    street.append((los_s, d3d_s))
    print(f"  {st['name']}: street LOS frac {los_s.mean():.3f}  ({time.time()-t0:.0f}s)")

DH_STREET = TX_MAST_M - RX_STREET_M   # sectorize dh used by planner for street
NCELL = GRID * GRID

results = []

# ============ SINGLE TOWER evaluation (each site alone; workbook constants) ====
# Isolated tower: no neighbour cells, so DL interference enters through the
# workbook's own geometry constant (Ioc/Ior +3.94 dB @ 90% load) and UL through
# its target RoT (+4 dB) — the workbook is itself a single-cell budget.
GEOM_DL_DB = 3.94
BANDS_ST = [(700, 10e6), (2500, 10e6), (3600, 10e6), (7125, 10e6)]
out = []
for f_mhz, bw_hz in BANDS_ST:
    ue_eirp = UE_P_DBM + g_ue(float(f_mhz))
    noise_dl_dbm = -174.0 + 10 * np.log10(bw_hz) + UE_NF_DB
    noise_ul_dbm = -174.0 + 10 * np.log10(bw_hz) + ENB_NF_DB + ROT_DB
    per_site = []
    for i, st in enumerate(sites):
        los_s, d3d_s = street[i]
        pl_i = uma_pl(d3d_s, los_s, TX_MAST_M, RX_STREET_M, float(f_mhz))
        rsrp_sec = sectorize((TX_EIRP_DBM - pl_i)[None, :, :], [st], DH_STREET)
        serving = np.argmax(rsrp_sec, axis=0)
        best_db = np.max(rsrp_sec, axis=0)
        sinr_dl = best_db - noise_dl_dbm - GEOM_DL_DB
        served_dl = sinr_dl >= DL_REQ_SNR_DB
        tp_dl = np.where(served_dl, bw_hz / 1e6 * np.minimum(DL_SE_CAP, np.log2(1 + 10 ** (sinr_dl / 10.0))) * (1 - DL_OVERHEAD), 0.0)
        ul_sec = sectorize((ue_eirp - pl_i)[None, :, :], [st], DH_STREET)
        sig_db = np.take_along_axis(ul_sec, serving[None, :, :], axis=0)[0]
        sinr_ul = sig_db - noise_ul_dbm
        served_ul = sinr_ul >= UL_REQ_SNR_DB
        tp_ul = np.where(served_ul, bw_hz / 1e6 * np.minimum(UL_SE_CAP, np.log2(1 + 10 ** (sinr_ul / 10.0))) * (1 - UL_OVERHEAD), 0.0)
        fp = served_dl
        per_site.append(dict(
            dl_fp=float(tp_dl[fp].mean()), ul_fp=float(tp_ul[fp].mean()),
            dl_tile=float(tp_dl.mean()), ul_tile=float(tp_ul.mean()),
            fp_frac=float(fp.mean()), ul_svc_in_fp=float(served_ul[fp].mean())))
    agg = {k: float(np.mean([s[k] for s in per_site])) for k in per_site[0]}
    out.append((f_mhz, agg))
    print(f"{f_mhz:>5} MHz single tower: DL {agg['dl_fp']:6.2f}  UL {agg['ul_fp']:6.2f}  "
          f"ratio {agg['dl_fp']/agg['ul_fp']:.2f}   [footprint {agg['fp_frac']*100:.0f}% of tile, "
          f"UL serves {agg['ul_svc_in_fp']*100:.0f}% of it; tile-mean ratio {agg['dl_tile']/agg['ul_tile']:.2f}]")
json.dump({str(f): a for f, a in out}, open('budget_single_results.json', 'w'), indent=1)
