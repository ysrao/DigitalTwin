#!/usr/bin/env python3
"""budget_common.py — shared setup for budget_check.py and budget_single.py.

Workbook budget terms (LTE_Link_budgetsNP_5MHz_20W_PA12.xlsm, Qualcomm
80-W2559-5 Rev A), the planner_v4.py functions both checks use (copied
verbatim, see budget_check.py), and the street-level geometry cache for the
4 Rancho Bernardo sites. Importing it loads the rasters and builds the
cache once (a few seconds); both scripts previously carried this code twice.
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

