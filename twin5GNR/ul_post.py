#!/usr/bin/env python3
"""Uplink (UL) post-processing for twin5GNR single-tower grids.

Mirrors the downlink composition in single_tower_twin.py (apply_antenna)
EXACTLY, changing only the link direction:

  DL (existing): rsrp_sector = P_COND_DBM + 10*log10(sum_paths pk_pw *
                 10^(g_db/10)) + G_ARRAY_DB, best over 3 sectors, where
                 g_db is the TR 38.901 element pattern (incl. +8 dBi max)
                 on the stored per-path departure angles (pk_th, pk_ph).
  UL (this file): ul_rsrp_sector = UE_P_DBM + 10*log10(same directional
                 channel power) + G_ARRAY_DB, best over the same sectors.
                 By reciprocity the BS receive pattern weights each path
                 with the same g_db on the same stored angles (the angle of
                 arrival at the BS for the reversed path equals the DL
                 departure angle -- same first/last segment, opposite
                 traversal). The UE transmits omni, 0 dBi, so no UE-side
                 pattern term appears.

Because both directions are the same constant plus 10*log10 of the SAME
per-sector directional channel power, the best sector is the same and
ul_rsrp - dl_rsrp = UE_P_DBM - P_COND_DBM = 23 - 26.96 = -3.96 dB on every
served cell. This script measures that shift and reports any deviation.

Differences vs DL (link budget only, all stated assumptions -- MODELLED,
not measured):
  - UE transmit power 23 dBm, UE antenna omni 0 dBi.
  - UL noise floor: -174 dBm/Hz + 10*log10(BW_Hz) + BS noise figure 4 dB
    + interference rise 3 dB.  (DL used UE NF 9 dB, no rise term.)
  - UL capacity = BW * min(7, log2(1 + SNR)) -- same Shannon cap as DL.
  - FDD n28 (f700): UL reuses the DL path gains (paired UL band is close
    in frequency); no frequency correction is invented. Approximation.

Outputs: data_twin_tower_<site>_<band>_ul.npz next to each DL file, with
xs, ys, ul_rsrp, ul_snr (at the viewer-default BW), ul_cap_mbps (same BW),
los, meta. Also writes ul_stats.json and prints a stats table.
"""
import json
import numpy as np

# --- UL link-budget constants (the only new assumptions) ---
UE_P_DBM = 23.0        # UE transmit power
BS_NF_DB = 4.0         # base-station receiver noise figure
IOT_RISE_DB = 3.0      # interference-over-thermal rise allowance
# --- shared with DL composition (single_tower_twin.py) ---
UE_ANT_DBI = 0.0       # UE antenna: omni
G_ELEM_MAX_DB = 8.0    # TR 38.901 element max gain (inside pattern fn)
ARRAY_ROWS, ARRAY_COLS = 4, 4
G_ARRAY_DB = 10 * np.log10(ARRAY_ROWS * ARRAY_COLS)   # 12.04 dB, assumption
EIRP0_DBM = 47.0
P_COND_DBM = EIRP0_DBM - G_ELEM_MAX_DB - G_ARRAY_DB   # 26.96 dBm (DL ref)
SECTORS_DEG = (90.0, 210.0, 330.0)
TILT_DEG = 4.0
SE_CAP_BPS_HZ = 7.0
DL_NF_DB = 9.0         # DL noise figure used by the existing grids/viewer

SITES = ['nw', 'ne', 'sw', 'se']
# viewer-default bandwidth per band (build_viewer.py BANDCFG bwdef)
BANDS = [('f700', 10.0), ('f2500', 10.0), ('f3600', 10.0),
         ('f7125', 10.0), ('f28000', 200.0)]


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
    return -174.0 + 10 * np.log10(bw_mhz * 1e6) + BS_NF_DB + IOT_RISE_DB


def dl_noise_dbm(bw_mhz):
    return -174.0 + 10 * np.log10(bw_mhz * 1e6) + DL_NF_DB


def capacity_mbps(rsrp, noise_dbm, bw_mhz):
    snr = rsrp - noise_dbm
    se = np.minimum(SE_CAP_BPS_HZ, np.log2(1 + 10 ** (snr / 10)))
    return np.where(np.isnan(rsrp), np.nan, bw_mhz * se), snr


def main():
    rows = []
    print(f"{'site-band':16s} {'BW':>5s} {'DLcov%':>7s} {'ULcov%':>7s} "
          f"{'DLcap':>7s} {'ULcap':>7s} {'medShift':>9s} {'shiftRange':>16s}")
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

            pw = directional_channel_power(pk_pw, pk_th, pk_ph)
            ul_flat = np.where(pw > 0, UE_P_DBM + UE_ANT_DBI
                               + 10 * np.log10(np.maximum(pw, 1e-30))
                               + G_ARRAY_DB, np.nan)
            ul_rsrp = ul_flat.reshape(n, n)

            # NaN pattern must match DL exactly (same paths)
            nan_match = bool((np.isnan(ul_rsrp) == np.isnan(dl_rsrp)).all())

            ul_cap, ul_snr = capacity_mbps(ul_rsrp, ul_noise_dbm(bw), bw)
            dl_cap, _ = capacity_mbps(dl_rsrp, dl_noise_dbm(bw), bw)

            served = ~np.isnan(dl_rsrp)
            shift = (ul_rsrp - dl_rsrp)[served]
            row = dict(
                key=f'{s}_{fstr}', bw_mhz=bw,
                dl_cov90=float(np.mean(dl_rsrp >= -90)) * 100,
                ul_cov90=float(np.mean(ul_rsrp >= -90)) * 100,
                dl_cap_mean=float(np.nanmean(dl_cap)),
                ul_cap_mean=float(np.nanmean(ul_cap)),
                shift_median=float(np.median(shift)),
                shift_min=float(np.min(shift)), shift_max=float(np.max(shift)),
                nan_match=nan_match,
                served_frac=float(served.mean()) * 100)
            rows.append(row)
            print(f"{row['key']:16s} {bw:5.0f} {row['dl_cov90']:7.1f} "
                  f"{row['ul_cov90']:7.1f} {row['dl_cap_mean']:7.0f} "
                  f"{row['ul_cap_mean']:7.0f} {row['shift_median']:9.2f} "
                  f"[{row['shift_min']:.2f},{row['shift_max']:.2f}] "
                  f"nan_match={nan_match}")

            meta = dict(
                link='uplink', source=src, bw_mhz=bw,
                ue_p_dbm=UE_P_DBM, ue_ant_dbi=UE_ANT_DBI,
                bs_nf_db=BS_NF_DB, iot_rise_db=IOT_RISE_DB,
                ul_noise_dbm=ul_noise_dbm(bw), g_array_db=float(G_ARRAY_DB),
                ant_cfg=meta_dl.get('ant_cfg', '3sec-4x4-tr38901-tilt4'),
                freq_ghz=meta_dl.get('freq_ghz'), site=meta_dl.get('site'),
                assumptions=('MODELLED uplink, not measured: UE TX 23 dBm '
                             'omni 0 dBi; BS receive = TR 38.901 sector '
                             'pattern by reciprocity on the DL stored path '
                             'angles + 12.04 dB 4x4 array gain; noise = '
                             '-174 dBm/Hz + 10log10(BW) + NF 4 dB + 3 dB '
                             'interference rise; capacity = BW*min(7, '
                             'log2(1+SNR)); FDD n28 reuses DL path gains '
                             'for UL (no frequency correction).'))
            np.savez(f'data_twin_tower_{s}_{fstr}_ul.npz',
                     xs=d['xs'], ys=d['ys'], ul_rsrp=ul_rsrp,
                     ul_snr=ul_snr, ul_cap_mbps=ul_cap, los=d['los'],
                     meta=json.dumps(meta))
    with open('ul_stats.json', 'w') as f:
        json.dump(rows, f, indent=1)
    print(f"\nexpected constant shift UE_P - P_COND = "
          f"{UE_P_DBM - P_COND_DBM:.2f} dB")
    print('wrote 20 *_ul.npz + ul_stats.json')


if __name__ == '__main__':
    main()
