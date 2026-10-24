#!/usr/bin/env python3
"""Uplink (UL) post-processing for twin5GNR single-tower grids -- v3,
the LINK-BUDGET method.

History (why this file looks the way it does):

  v1 (reciprocal shortcut): ul_rsrp = DL composition with P_COND swapped
  for 23 dBm; UE antenna an idealised 0 dBi at all bands; noise = BS
  NF 4 dB + 3 dB interference rise; capacity = Shannon cap 7 (the DL
  cap). Result: DL/UL mean throughput ratio ~1.0 in every band.
  REJECTED by the user: no per-direction budget was ever closed -- the
  uplink simply inherited the downlink's budget terms.

  v2 (tuned constants): interference rise 3 -> 10 dB, UL Shannon cap
  7 -> 5. Result: ratio ~1.5 at FR1, ~2.8 at 28 GHz. REJECTED by the
  user: tuning two constants until the output looks field-like is not
  a link budget ("I am not asking to fake anything"). The ray-traced
  path loss itself was audited and cleared (reciprocal; tracks FSPL /
  3GPP UMa LOS within ~1-2 dB on LOS cells, harsher than UMa in NLOS),
  so the fix had to be a real per-direction budget.

  v3 (this file): mirrors the structure of the user's own LTE link
  budget workbook (Qualcomm 80-W2559-5 Rev A,
  LTE_Link_budgetsNP_5MHz_20W_PA12.xlsm). In that workbook every
  direction/service gets its own budget closed at the path loss:
  per-subcarrier PSD at the transmitter, the receiver's own noise
  floor, its own margins -- and its "Summary table" sheet marks the
  UPLINK as the limiting direction for every band and morphology it
  covers. v3 keeps the twin's audited ray-traced path loss (in place
  of the workbook's morphology propagation model) and closes the UL
  budget per cell, per direction, on top of it:

    PL_cell     = EIRP0 (47) + G_UE (0) - RSRP_DL_cell      [dB]
                  (effective PL: the DL RSRP already contains the
                  pattern/array composition, so this removes exactly
                  the known DL budget terms; cross-checked against
                  -10*log10(sum pk_pw), see PL_CHECK below)
    ul_rsrp     = UE_EIRP(f) - PL_cell + G_ELEM_MAX + G_ARRAY
                = UE_EIRP(f) + 10*log10(P_dir) + G_ARRAY
                  (same directional composition as v1/v2: the BS
                  receive antenna is the twin's TR 38.901 pattern +
                  4x4 array by reciprocity. The workbook's separate
                  18 dBi eNB antenna line is NOT added on top -- in
                  twin terms that panel+combining gain already lives
                  inside this composition.)
    ul_snr      = ul_rsrp - (-174 + 10*log10(BW_Hz) + NF_BS + RoT)
    ul_cap      = BW * min(5, log2(1 + 10^(ul_snr/10))) * (1 - 0.2234)

  Workbook terms adopted (700_DU / UL_Combined / Tables sheets):
    - UE antenna gain g_ue(f) = 6.7292*ln(f_MHz) - 51.61 dBi -- the
      workbook's handset-antenna fit (-7.53 dBi at 700 MHz, so the
      workbook's UE EIRP is 24 - 7.53 = 16.47 dBm). The twin uses the
      user-set UE power 23 dBm (workbook: 24 dBm). The workbook's own
      formula caps the fit: g_ue = MIN(6.7292*ln(f_MHz) - 51.61, 2.14)
      dBi (verified in the workbook's sheets 2026-10-21; the cap binds
      above ~2.9 GHz). v3 originally clamped at the 3600 MHz value
      (+3.49 dBi) instead, overstating UE EIRP by 1.35 dB at >= 3.6 GHz;
      corrected to the workbook's own cap.
    - eNB receiver noise figure 3 dB (workbook 700 sheets; its AWS
      sheets use 2.5 dB).
    - Target RoT (rise over thermal) +4 dB, added to the noise floor
      (replaces v1's NF 4 + IoT 3 and v2's NF 4 + IoT 10).
    - UL overhead 22.34% (UL_Combined "Total overhead" 22.3413%), so
      throughput carries the workbook's effective-payload factor
      0.7766. SE is the Shannon form capped at 5 bps/Hz (v2's
      single-layer UE cap, retained) -- a labelled hybrid: the
      workbook maps SINR->efficiency through CQI tables, the twin
      keeps its Shannon convention and applies the workbook's
      overhead and cap on top.
    - Edge statistics with the workbook's margin stack: BPL 11 dB
      mean + log-normal fading. Main variant (700_DU): combined
      sigma sqrt(10^2 + 8^2) = 12.81 dB at 95% cell-edge reliability
      -> margin 32.06 dB. Variant (UL_Combined): combined sigma
      10 dB at 90% -> margin 23.82 dB.

  Edge model (statistics only, not a map): at the cell-border PL --
  proxied by the 90th percentile of PL over served cells -- the UL
  edge user gets a 1-RB (180 kHz) PUSCH allocation with the full UE
  power in it (workbook semantics: its noise floor and its budgets
  are per-allocation), margins applied to the SINR, throughput from
  the same capped-Shannon x 0.7766 form over 180 kHz. The comparison
  DL edge uses the workbook's DL structure at the same border PL:
  per-subcarrier EIRP from 43 dBm over the channel (occupied BW =
  0.903 x nominal, the workbook's 4.515/5 MHz ratio), UE NF 7 dB,
  other-cell interference at Ioc/Ior = +3.94 dB (geometry -4.4 dB at
  90% load), DL overhead 30.39%, 2x2 MIMO with SU-MIMO gain 1 (the
  workbook's own rate blocks use gain 1), Shannon-form SE capped at
  the workbook's top CQI efficiency 5.5547 bps/Hz. Full detail in
  UL_METHOD.md.

FDD n28 (f700) still reuses the DL path gains for the paired uplink
band; no frequency correction is invented (stated approximation).

Outputs: data_twin_tower_<site>_<band>_ul.npz next to each DL file
(xs, ys, ul_rsrp, ul_snr, ul_cap_mbps at the viewer-default BW, los,
meta), ul_stats.json (per-site-band rows + per-band pooled + the PL
cross-check), and a printed stats table. MODELLED, not measured.
"""
import json
import numpy as np

# --- UL v3 link-budget constants (workbook 80-W2559-5 unless noted) ---
UE_P_DBM = 23.0          # UE transmit power (user-set; workbook: 24)
BS_NF_DB = 3.0           # eNB receiver noise figure (workbook 700 sheets)
ROT_DB = 4.0             # target rise over thermal (workbook)
UL_OVERHEAD = 0.2234     # UL total overhead (UL_Combined: 22.3413%)
UL_PAYLOAD = 1.0 - UL_OVERHEAD          # 0.7766 effective payload
UL_SE_CAP_BPS_HZ = 5.0   # single-layer UE cap (retained from v2)
THERMAL_DBM_HZ = -174.0  # twin convention (workbook: -173.98)
# Workbook margin stack: BPL mean 11 dB + log-normal at cell edge
MARGIN_MAIN_DB = 32.0644     # 95% reliability, combined sigma 12.81 (700_DU)
MARGIN_VARIANT_DB = 23.8155  # 90% reliability, combined sigma 10 (UL_Combined)
EDGE_RB_BW_KHZ = 180.0       # 1 RB PUSCH allocation at the edge
# --- DL edge comparison (workbook DL structure; documentation only) ---
DL_BS_P_DBM = 43.0           # total power per cell (workbook)
DL_BS_LOSS_DB = -0.5         # cell-associated losses (workbook)
DL_BS_ANT_DBI = 18.0         # eNB transmit antenna gain (workbook)
DL_UE_NF_DB = 7.0            # UE receiver noise figure (workbook)
DL_IOC_IOR_DB = 3.9424       # geometry -4.4 dB at 90% load (workbook)
DL_PAYLOAD = 1.0 - 0.3039    # DL overhead 30.39% (Tables, 2-ant, 3 PDCCH)
DL_OCC_RATIO = 4.515 / 5.0   # effective / nominal BW (workbook)
DL_SE_CAP_BPS_HZ = 5.5547    # top 36.942 CQI efficiency (workbook Tables)
DL_SU_MIMO_GAIN = 1.0        # workbook rate blocks (2x2 system, gain 1)
SCS_KHZ = 15.0               # subcarrier spacing (workbook / twin bands)
# --- shared with the DL composition (single_tower_twin.py) ---
G_ELEM_MAX_DB = 8.0      # TR 38.901 element max gain (inside pattern fn)
ARRAY_ROWS, ARRAY_COLS = 4, 4
G_ARRAY_DB = 10 * np.log10(ARRAY_ROWS * ARRAY_COLS)   # 12.04 dB, assumption
EIRP0_DBM = 47.0
P_COND_DBM = EIRP0_DBM - G_ELEM_MAX_DB - G_ARRAY_DB   # 26.96 dBm (DL ref)
SECTORS_DEG = (90.0, 210.0, 330.0)
TILT_DEG = 4.0
SE_CAP_BPS_HZ = 7.0      # DL Shannon cap (existing DL map convention)
DL_NF_DB = 9.0           # DL noise figure used by the existing grids/viewer

SITES = ['nw', 'ne', 'sw', 'se']
# viewer-default bandwidth per band (build_viewer.py BANDCFG bwdef)
BANDS = [('f700', 10.0), ('f2500', 10.0), ('f3600', 10.0),
         ('f7125', 10.0), ('f28000', 200.0)]
PL_CHECK_KEY = 'se_f3600'   # designated file for the PL cross-check


def ue_ant_gain_dbi(f_mhz):
    """Workbook handset antenna fit, exactly as the workbook writes it:
    g = MIN(6.7292*ln(f_MHz) - 51.61, 2.14) dBi."""
    return min(6.7292 * np.log(float(f_mhz)) - 51.61, 2.14)


def tr38901_gain_db(theta, phi, tilt_deg, az_deg):
    """Copied verbatim from single_tower_twin.py (3GPP TR 38.901 element
    power pattern, dB, incl. 8 dBi max, with downtilt + boresight azimuth).
    theta: zenith [rad] from +z; phi: azimuth [rad] from +x."""
    t = np.deg2rad(tilt_deg)
    a = np.deg2rad(az_deg)
    th = theta - t
    ph = (phi - a + np.pi) % (2 * np.pi) - np.pi
    th3 = ph3 = np.deg2rad(65.0)
    a_max = sla_v = 30.0
    a_v = -np.minimum(12.0 * ((th - np.pi / 2) / th3) ** 2, sla_v)
    a_h = -np.minimum(12.0 * (ph / ph3) ** 2, a_max)
    return -np.minimum(-(a_v + a_h), a_max) + G_ELEM_MAX_DB


def directional_channel_power(pk_pw, pk_th, pk_ph):
    """Per-sector directional channel power (linear), best over sectors.
    Returns (best_pw, ) flat (n_rx,) -- identical weighting to DL post."""
    best = np.zeros(pk_pw.shape[0])
    for az in SECTORS_DEG:
        g_db = tr38901_gain_db(pk_th, pk_ph, TILT_DEG, az)
        g_lin = np.where(pk_pw > 0, 10.0 ** (g_db / 10.0), 0.0)
        pw = (pk_pw * g_lin).sum(axis=1)
        best = np.maximum(best, pw)
    return best


def ul_noise_dbm(bw_mhz):
    return THERMAL_DBM_HZ + 10 * np.log10(bw_mhz * 1e6) + BS_NF_DB + ROT_DB


def dl_noise_dbm(bw_mhz):
    return THERMAL_DBM_HZ + 10 * np.log10(bw_mhz * 1e6) + DL_NF_DB


def se_from_snr(snr_db, cap):
    return np.minimum(cap, np.log2(1 + 10 ** (np.asarray(snr_db) / 10)))


def ul_edge(pl90_db, ue_eirp_dbm, margin_db):
    """UL edge: 1-RB PUSCH at the border PL, margin applied to SINR.
    Returns (sinr_db, tp_kbps)."""
    p_rx = ue_eirp_dbm - pl90_db + G_ELEM_MAX_DB + G_ARRAY_DB
    noise = (THERMAL_DBM_HZ + 10 * np.log10(EDGE_RB_BW_KHZ * 1e3)
             + BS_NF_DB + ROT_DB)
    sinr = p_rx - noise - margin_db
    tp = EDGE_RB_BW_KHZ * float(se_from_snr(sinr, UL_SE_CAP_BPS_HZ)) * UL_PAYLOAD
    return float(sinr), float(tp)


def dl_edge(pl90_db, g_ue_dbi, bw_mhz, margin_db):
    """Workbook-structure DL edge at the same border PL (docs only).
    Per-subcarrier budget; interference tied to the margined own
    signal via the workbook geometry. Returns (sinr_db, tp_kbps)."""
    bw_occ_mhz = DL_OCC_RATIO * bw_mhz
    sc_eirp = (DL_BS_P_DBM + 10 * np.log10(SCS_KHZ * 1e3 / (bw_occ_mhz * 1e6))
               + DL_BS_LOSS_DB + DL_BS_ANT_DBI)
    s_m = sc_eirp - pl90_db + g_ue_dbi - margin_db
    ioc = s_m + DL_IOC_IOR_DB
    n_sc = THERMAL_DBM_HZ + 10 * np.log10(SCS_KHZ * 1e3) + DL_UE_NF_DB
    denom = 10 * np.log10(10 ** (n_sc / 10) + 10 ** (ioc / 10))
    sinr = s_m - denom
    se = float(se_from_snr(sinr, DL_SE_CAP_BPS_HZ))
    tp = se * bw_occ_mhz * 1e3 * DL_PAYLOAD * DL_SU_MIMO_GAIN
    return float(sinr), float(tp)


def main():
    rows = []
    pl_check = None
    print(f"{'site-band':16s} {'BW':>5s} {'gUE':>6s} {'DLcov%':>7s} "
          f"{'ULcov%':>7s} {'DLcap':>7s} {'ULcap':>7s} {'DL/UL':>6s} "
          f"{'PL90':>6s} {'ULedge95':>9s} {'DLedge95':>9s} {'ratio95':>8s}")
    for s in SITES:
        for fstr, bw in BANDS:
            src = f'data_twin_tower_{s}_{fstr}.npz'
            d = np.load(src)
            n = len(d['xs'])
            pk_pw, pk_th, pk_ph = d['pk_pw'], d['pk_th'], d['pk_ph']
            dl_rsrp = d['rsrp']
            meta_dl = json.loads(str(d['meta']))
            assert float(meta_dl['p_cond_dbm']) == float(P_COND_DBM), \
                (src, meta_dl['p_cond_dbm'], P_COND_DBM)
            assert float(meta_dl['g_array_db']) == float(G_ARRAY_DB)
            f_mhz = int(round(float(meta_dl['freq_ghz']) * 1000))
            g_ue = float(ue_ant_gain_dbi(f_mhz))
            ue_eirp = UE_P_DBM + g_ue

            pw = directional_channel_power(pk_pw, pk_th, pk_ph)
            ul_flat = np.where(pw > 0, ue_eirp
                               + 10 * np.log10(np.maximum(pw, 1e-30))
                               + G_ARRAY_DB, np.nan)
            ul_rsrp = ul_flat.reshape(n, n)

            # NaN pattern must match DL exactly (same paths)
            nan_match = bool((np.isnan(ul_rsrp) == np.isnan(dl_rsrp)).all())

            # effective per-cell PL from the DL grid (budget terms removed)
            pl_cell = EIRP0_DBM - dl_rsrp
            served = ~np.isnan(dl_rsrp)
            pl90 = float(np.nanpercentile(pl_cell, 90))

            ul_snr = ul_rsrp - ul_noise_dbm(bw)
            ul_cap = np.where(np.isnan(ul_rsrp), np.nan,
                              bw * se_from_snr(ul_snr, UL_SE_CAP_BPS_HZ)
                              * UL_PAYLOAD)
            dl_snr = dl_rsrp - dl_noise_dbm(bw)
            dl_cap = np.where(np.isnan(dl_rsrp), np.nan,
                              bw * se_from_snr(dl_snr, SE_CAP_BPS_HZ))

            u_sinr95, u_tp95 = ul_edge(pl90, ue_eirp, MARGIN_MAIN_DB)
            u_sinr90, u_tp90 = ul_edge(pl90, ue_eirp, MARGIN_VARIANT_DB)
            d_sinr95, d_tp95 = dl_edge(pl90, g_ue, bw, MARGIN_MAIN_DB)
            d_sinr90, d_tp90 = dl_edge(pl90, g_ue, bw, MARGIN_VARIANT_DB)

            shift = (ul_rsrp - dl_rsrp)[served]
            row = dict(
                key=f'{s}_{fstr}', bw_mhz=bw, freq_mhz=f_mhz,
                g_ue_dbi=g_ue, ue_eirp_dbm=ue_eirp,
                dl_cov90=float(np.mean(dl_rsrp >= -90)) * 100,
                ul_cov90=float(np.mean(ul_rsrp >= -90)) * 100,
                dl_cap_mean=float(np.nanmean(dl_cap)),
                dl_cap_median=float(np.nanmedian(dl_cap)),
                ul_cap_mean=float(np.nanmean(ul_cap)),
                ul_cap_median=float(np.nanmedian(ul_cap)),
                dl_ul_ratio=float(np.nanmean(dl_cap) / np.nanmean(ul_cap)),
                dl_ul_ratio_median=float(np.nanmedian(dl_cap)
                                         / np.nanmedian(ul_cap)),
                shift_median=float(np.median(shift)),
                pl90_db=pl90,
                ul_edge_sinr_95=u_sinr95, ul_edge_tp_kbps_95=u_tp95,
                ul_edge_sinr_90=u_sinr90, ul_edge_tp_kbps_90=u_tp90,
                dl_edge_sinr_95=d_sinr95, dl_edge_tp_kbps_95=d_tp95,
                dl_edge_sinr_90=d_sinr90, dl_edge_tp_kbps_90=d_tp90,
                dl_ul_edge_ratio_95=float(d_tp95 / u_tp95),
                dl_ul_edge_ratio_90=float(d_tp90 / u_tp90),
                nan_match=nan_match,
                served_frac=float(served.mean()) * 100)
            rows.append(row)
            print(f"{row['key']:16s} {bw:5.0f} {g_ue:6.2f} "
                  f"{row['dl_cov90']:7.1f} {row['ul_cov90']:7.1f} "
                  f"{row['dl_cap_mean']:7.0f} {row['ul_cap_mean']:7.0f} "
                  f"{row['dl_ul_ratio']:6.2f} {pl90:6.1f} "
                  f"{u_tp95:9.2f} {d_tp95:9.0f} "
                  f"{row['dl_ul_edge_ratio_95']:8.1f}k")

            if row['key'] == PL_CHECK_KEY:
                iso = pk_pw.astype(np.float64).sum(axis=1)
                ok = iso > 0
                pl_iso = -10 * np.log10(np.maximum(iso[ok], 1e-30))
                pl_der = (EIRP0_DBM - dl_rsrp.reshape(-1))[ok]
                diff = pl_der - pl_iso
                # internal identity: PL_derived == 8 - 10log10(P_dir)
                ident = pl_der - (G_ELEM_MAX_DB
                                  - 10 * np.log10(np.maximum(pw[ok], 1e-30)))
                pl_check = dict(
                    file=src, n_cells=int(ok.sum()),
                    pl_iso_median_db=float(np.median(pl_iso)),
                    pl_derived_median_db=float(np.median(pl_der)),
                    diff_median_db=float(np.median(diff)),
                    diff_mean_db=float(np.mean(diff)),
                    diff_p10_db=float(np.percentile(diff, 10)),
                    diff_p90_db=float(np.percentile(diff, 90)),
                    identity_max_abs_db=float(np.max(np.abs(ident))))
                print(f"  PL check ({src}): derived-vs-isotropic PL diff "
                      f"median {float(np.median(diff)):.2f} dB, "
                      f"mean {float(np.mean(diff)):.2f}, "
                      f"p10 {float(np.percentile(diff, 10)):.2f}, "
                      f"p90 {float(np.percentile(diff, 90)):.2f}; "
                      f"identity residual {float(np.max(np.abs(ident))):.2e} dB")

            meta = dict(
                link='uplink', version='v3-link-budget', source=src,
                bw_mhz=bw, freq_mhz=f_mhz,
                ue_p_dbm=UE_P_DBM, ue_ant_dbi=g_ue, ue_eirp_dbm=ue_eirp,
                bs_nf_db=BS_NF_DB, rot_db=ROT_DB,
                ul_noise_dbm=ul_noise_dbm(bw),
                ul_overhead=UL_OVERHEAD, ul_se_cap_bps_hz=UL_SE_CAP_BPS_HZ,
                g_array_db=float(G_ARRAY_DB),
                ant_cfg=meta_dl.get('ant_cfg', '3sec-4x4-tr38901-tilt4'),
                freq_ghz=meta_dl.get('freq_ghz'), site=meta_dl.get('site'),
                assumptions=(
                    'MODELLED uplink v3 (link-budget method), not measured. '
                    'Mirrors the user-supplied LTE link-budget workbook '
                    '(Qualcomm 80-W2559-5 Rev A): per-direction budget '
                    'closed at the ray-traced path loss. UE TX 23 dBm '
                    '(workbook 24) + workbook UE antenna fit '
                    'MIN(6.7292*ln(f_MHz)-51.61, 2.14) dBi (the '
                    'workbook\'s own cap); BS receive = TR 38.901 '
                    'sector pattern + 12.04 dB 4x4 array by reciprocity '
                    '(the workbook 18 dBi eNB antenna lives inside this '
                    'composition, not added again); noise = -174 dBm/Hz '
                    '+ 10log10(BW) + eNB NF 3 dB + target RoT 4 dB; '
                    'throughput = BW*min(5, log2(1+SNR))*0.7766 '
                    '(workbook UL overhead 22.34%); FDD n28 reuses DL '
                    'path gains for UL (no frequency correction).'))
            np.savez(f'data_twin_tower_{s}_{fstr}_ul.npz',
                     xs=d['xs'], ys=d['ys'], ul_rsrp=ul_rsrp,
                     ul_snr=ul_snr, ul_cap_mbps=ul_cap, los=d['los'],
                     meta=json.dumps(meta))

    pooled = []
    for fstr, bw in BANDS:
        rs = [r for r in rows if r['key'].endswith(fstr)]
        p = dict(
            band=fstr, bw_mhz=bw,
            freq_mhz=rs[0]['freq_mhz'], g_ue_dbi=rs[0]['g_ue_dbi'],
            ue_eirp_dbm=rs[0]['ue_eirp_dbm'],
            dl_cap_mean=float(np.mean([r['dl_cap_mean'] for r in rs])),
            ul_cap_mean=float(np.mean([r['ul_cap_mean'] for r in rs])),
            dl_cap_median=float(np.mean([r['dl_cap_median'] for r in rs])),
            ul_cap_median=float(np.mean([r['ul_cap_median'] for r in rs])),
            pl90_db=float(np.mean([r['pl90_db'] for r in rs])),
            ul_edge_tp_kbps_95=float(np.mean(
                [r['ul_edge_tp_kbps_95'] for r in rs])),
            dl_edge_tp_kbps_95=float(np.mean(
                [r['dl_edge_tp_kbps_95'] for r in rs])),
            ul_edge_tp_kbps_90=float(np.mean(
                [r['ul_edge_tp_kbps_90'] for r in rs])),
            dl_edge_tp_kbps_90=float(np.mean(
                [r['dl_edge_tp_kbps_90'] for r in rs])))
        p['dl_ul_ratio'] = p['dl_cap_mean'] / p['ul_cap_mean']
        p['dl_ul_ratio_median'] = p['dl_cap_median'] / p['ul_cap_median']
        p['dl_ul_edge_ratio_95'] = (p['dl_edge_tp_kbps_95']
                                    / p['ul_edge_tp_kbps_95'])
        p['dl_ul_edge_ratio_90'] = (p['dl_edge_tp_kbps_90']
                                    / p['ul_edge_tp_kbps_90'])
        pooled.append(p)
        print(f"POOLED {fstr:8s} map DL/UL {p['dl_ul_ratio']:.2f} "
              f"(median {p['dl_ul_ratio_median']:.2f}); "
              f"edge95 UL {p['ul_edge_tp_kbps_95']:.2f} kbps vs DL "
              f"{p['dl_edge_tp_kbps_95']:.0f} kbps -> "
              f"{p['dl_ul_edge_ratio_95']:.1f}x; "
              f"edge90 UL {p['ul_edge_tp_kbps_90']:.2f} vs DL "
              f"{p['dl_edge_tp_kbps_90']:.0f} -> "
              f"{p['dl_ul_edge_ratio_90']:.1f}x")

    with open('ul_stats.json', 'w') as f:
        json.dump(dict(version='v3-link-budget', rows=rows, pooled=pooled,
                       pl_check=pl_check), f, indent=1)
    print('\nwrote 20 *_ul.npz + ul_stats.json (v3)')


if __name__ == '__main__':
    main()
