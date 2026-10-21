#!/usr/bin/env python3
"""budget_single.py — single-tower DL/UL user-throughput check on the twin's
stochastic 3GPP UMa layer, closed with the workbook budget terms.

Each of the 4 sites is evaluated alone (no neighbour cells): DL interference
enters through the workbook's geometry constant (Ioc/Ior +3.94 dB at 90%
load), UL through its target RoT (+4 dB) -- the workbook is itself a
single-cell budget. Shared setup lives in budget_common.py; the method and
all workbook terms are documented in budget_check.py.

Output: budget_single_results.json (per band: DL/UL mean TP over the DL
footprint and over the whole tile, footprint fraction, UL service in it).
"""
from budget_common import *   # constants, planner functions, sites, street cache
from budget_common import (rr, cc, terrain, bld_h, surface, sites, street, t0,
                           DH_STREET, NCELL, g_ue, uma_pl, sectorize)

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
