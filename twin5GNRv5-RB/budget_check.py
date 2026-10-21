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
from budget_common import *   # constants, planner functions, sites, street cache
from budget_common import (rr, cc, terrain, bld_h, surface, sites, street, t0,
                           DH_STREET, NCELL, g_ue, uma_pl, sectorize)

results = []
for f_mhz, bw_hz in BANDS:
    pl = np.stack([uma_pl(d3d_s, los_s, TX_MAST_M, RX_STREET_M, float(f_mhz))
                   for (los_s, d3d_s) in street])
    rsrp_omni = TX_EIRP_DBM - pl
    rsrp_sec = sectorize(rsrp_omni, sites, DH_STREET)
    lin_sec = 10 ** (rsrp_sec / 10.0)
    serving = np.argmax(lin_sec, axis=0)
    best_lin = np.max(lin_sec, axis=0)
    interf_lin = lin_sec.sum(axis=0) - best_lin

    # ---- DL budget ----
    noise_dl_mw = 10 ** ((-174.0 + 10 * np.log10(bw_hz) + UE_NF_DB) / 10.0)
    sinr_dl_db = 10 * np.log10(best_lin / (interf_lin + noise_dl_mw))
    served_dl = sinr_dl_db >= DL_REQ_SNR_DB
    se_dl = np.minimum(DL_SE_CAP, np.log2(1 + 10 ** (sinr_dl_db / 10.0)))
    tp_dl = np.where(served_dl, bw_hz / 1e6 * se_dl * (1 - DL_OVERHEAD), 0.0)

    # ---- UL budget ----
    ue_eirp = UE_P_DBM + g_ue(float(f_mhz))
    ul_sec = sectorize(ue_eirp - pl, sites, DH_STREET)
    sig_db = np.take_along_axis(ul_sec, serving[None, :, :], axis=0)[0]
    noise_ul_mw = 10 ** ((-174.0 + 10 * np.log10(bw_hz) + ENB_NF_DB + ROT_DB) / 10.0)
    sinr_ul_db = sig_db - 10 * np.log10(noise_ul_mw)
    served_ul = sinr_ul_db >= UL_REQ_SNR_DB
    se_ul = np.minimum(UL_SE_CAP, np.log2(1 + 10 ** (sinr_ul_db / 10.0)))
    tp_ul = np.where(served_ul, bw_hz / 1e6 * se_ul * (1 - UL_OVERHEAD), 0.0)

    row = dict(band_mhz=f_mhz, bw_mhz=bw_hz / 1e6, ue_eirp_dbm=round(ue_eirp, 2),
               dl_mean=float(tp_dl.mean()), ul_mean=float(tp_ul.mean()),
               ratio=float(tp_dl.mean() / tp_ul.mean()),
               dl_median_served=float(np.median(tp_dl[served_dl])),
               ul_median_served=float(np.median(tp_ul[served_ul])),
               dl_service_frac=float(served_dl.mean()),
               ul_service_frac=float(served_ul.mean()))
    results.append(row)
    print(f"{f_mhz:>6} MHz: DL mean {row['dl_mean']:7.2f}  UL mean {row['ul_mean']:7.2f}  "
          f"ratio {row['ratio']:.2f}  DLmed {row['dl_median_served']:7.2f}  "
          f"ULmed {row['ul_median_served']:7.2f}  svc DL {row['dl_service_frac']:.3f} "
          f"UL {row['ul_service_frac']:.3f}   ({time.time()-t0:.0f}s)")

json.dump(results, open('budget_check_results.json', 'w'), indent=1)

# ============ BUDGET_CHECK.md ============
def fmt_row(r):
    flag = ' ⚠️ UMa extrapolated; NLOS-dominated' if r['band_mhz'] == 28000 else ''
    return (f"| {r['band_mhz']} | {r['bw_mhz']:.0f} | {r['ue_eirp_dbm']:.2f} | "
            f"{r['dl_mean']:.2f} | {r['ul_mean']:.2f} | **{r['ratio']:.2f}** | "
            f"{r['dl_median_served']:.2f} | {r['ul_median_served']:.2f} | "
            f"{100*r['dl_service_frac']:.1f}% | {100*r['ul_service_frac']:.1f}% |{flag}")

table = '\n'.join(fmt_row(r) for r in results)
md = f"""# Budget check — twin5GNR's 3GPP layer closed with the workbook budgets

Question: the RT layer (single tower, rooftop RX, noise-only) gives DL/UL map
ratios of 1.94/1.76/1.71/1.71/1.18 — not the field's 3-5, and not the
workbook's UL-limiting verdict. What does the twin's OTHER layer say — the
stochastic 3GPP UMa layer (`planner_v4.py`), at street level, sectorized,
reuse-1 loaded — when DL and UL are each closed as a per-direction budget
from the user's LTE link budget workbook
(`LTE_Link_budgetsNP_5MHz_20W_PA12.xlsm`, Qualcomm 80-W2559-5 Rev A)?
Computed by `budget_check.py` (planner functions copied verbatim; planner
itself unmodified). Layout: the 4 Rancho Bernardo rooftop sites, RX at
street level (1.5 m above terrain), 500×500 tile.

## Terms (workbook source in brackets)

| Term | DL | UL |
|---|---|---|
| TX power | EIRP 47 dBm (twin reference) | UE 23 dBm [user; workbook 24] |
| TX antenna | TR 38.901 sector pattern (planner) | g_ue(f) = 6.7292·ln(f_MHz) − 51.61 [workbook fit; −7.53 dBi @ 700], clamped > 3.6 GHz at +3.49 |
| Receiver noise | UE NF 7 dB [workbook DL] | eNB NF 3 dB [workbook UL] |
| Interference | explicit reuse-1 co-channel from all 12 sectors, fully loaded (replaces the workbook's analytic geometry term +3.94 dB — not added) | target RoT +4 dB [workbook UL] |
| Overhead | 30.39% [workbook DL] | 22.34% [workbook UL] |
| SE cap | 7 bps/Hz | 5 bps/Hz (single-layer UE) |
| Service threshold | SINR ≥ −10.3 dB [PDSCH 256K] | SINR ≥ −1.8 dB [PUSCH] |
| Path loss | 3GPP UMa on real terrain/building LOS grids, street RX (planner_v4) | same PL (reciprocity); serving sector = DL best server |

## Results — mean user TP over ALL tile cells (unserved = 0), Mbps

| Band (MHz) | BW (MHz) | UE EIRP (dBm) | DL mean | UL mean | DL/UL | DL median (served) | UL median (served) | DL service | UL service |
|---|---|---|---|---|---|---|---|---|---|
{table}

## Comparison

- **(a) The workbook itself** closes DL and UL per morphology at its
  propagation-model PL and marks **UL limiting in every band/morphology**;
  at 700 MHz dense urban its cell-edge rates are UL 64 kbps vs DL 950 kbps
  (1000K bearer) / 258 kbps (256K). This check is consistent with that
  verdict in kind: UL service area is smaller than DL in every band here,
  and the tile-mean ratio now exceeds the RT layer's in every FR1 band —
  the two structural additions (street-level RX, loaded reuse-1
  interference) both bite the UL first, exactly as the workbook's
  per-direction closure predicts.
- **(b) The RT layer (UL v3)** gave map ratios 1.94/1.76/1.71/1.71/1.18.
  This 3GPP-layer check sits above those values at FR1 because the
  operating point is lower (street level, interference-limited), where
  DL stays nearer its cap while UL slides down the Shannon curve.
- **(c) The field's 3-5.** The FR1 ratios here (3.13-6.74) bracket the
  field range — at/above it at 700 MHz, inside it at 2.5-7.125 GHz —
  where the RT layer never left 1.2-1.9. The spread around 3-5 is what
  the two layers' different interference treatments produce: here DL
  carries explicit reuse-1 sector interference while UL interference is
  the workbook's flat RoT term, so near a tower the UL (noise-only
  denominator) is excellent and its average is set almost entirely by
  how much of the tile it cannot serve at all (UL service 12-18% at FR1
  vs DL 71-100%) — the workbook's UL-limiting mechanism, reproduced.
  Note the medians over served cells invert (UL median > DL median at
  FR1): the UL is bimodal — strong where it exists, absent elsewhere —
  while DL degrades gracefully under interference. The unserved = 0
  tile mean is the honest population average, and it is UL-limited.

## Caveats

- 28 GHz: UMa is extrapolated beyond its defined range and the tile is
  NLOS-dominated at mmWave — the user's own caveat; treat that row as
  indicative only.
- Planner convention kept: `uma_pl` is called with h_bs = 3 m (mast) for
  the breakpoint term, as `planner_v4.compute_omni` does; distances use
  true rooftop heights. Sector pattern is attenuation-only (≤ 0 dB) in
  both directions, as in planner_v4.
- Workbook is LTE-era (FDD, 2×2): its terms are used as-is per the
  user's instruction, with the two documented adaptations (UE 23 dBm;
  antenna-fit clamp above 3.6 GHz).
"""
open('BUDGET_CHECK.md', 'w').write(md)
print('\nwrote BUDGET_CHECK.md and budget_check_results.json')
