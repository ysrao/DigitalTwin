#!/usr/bin/env python3
"""Single-tower digital twin: full-area Sionna RT coverage map for one site.

Fills the gap between the v4 sampled calibration (192 points) and a true
digital-twin coverage surface: every grid cell gets a ray-traced path gain.

Scene   : v4/scene_v4/scene.xml (USGS 3DEP terrain + SanGIS/LiDAR buildings,
          ITU concrete / medium-dry-ground materials) - same as sionna_cal_v4
Solver  : isotropic TX/RX in the ray tracer; the TOP-K paths per RX cell
          (power + departure angles) are stored, so the TX antenna is applied
          in POST-PROCESSING - pattern, tilt, azimuth, sectors are free.
TX (post): 3 sectors (az 90/210/330 deg), 3GPP TR 38.901 65-deg element
          pattern, 4x4 array (12.0 dB ideal-beamforming array gain, stated
          assumption), electrical downtilt --tilt-deg (default 4.0).
          Conducted power is set so boresight EIRP = 47 dBm (CBRS Cat-B),
          i.e. directly comparable with the isotropic twin.
RX      : regular grid over 2x2 km centred on the TX, 6 m above surface
          (rooftop-CPE convention, same as the v4 calibration)
RT      : PathSolver max_depth=3, LOS + specular + diffraction + edge
          diffraction; 1M samples/src (same as the v4 calibration)
Post    : per-sector RSRP -> best-server composite -> SNR (NF 9 dB, --bw-mhz)
          -> Shannon capacity; heatmap PNG.

Usage: single_tower_twin.py --chunk K   (one chunk per process; results appended
         to v4/data_twin_tower_nw_f3600.npz, safe against OOM in later chunks)
       single_tower_twin.py --smoke      (3x3 grid pipeline test)
       single_tower_twin.py --post [--tilt-deg D]
         (build directional RSRP/SNR grids + heatmap from stored paths)
"""
import argparse, json, math, os, sys, time
import numpy as np

sys.path.insert(0, 'v4')
from build_scene_v4 import DEM, to_local, Z_REF

S, W, N, E = 32.9925, -117.1120, 33.0375, -117.0580
GRID = 500
FREQ_GHZ = 3.6            # overridden by --freq-ghz
MAT_MODE = 'itu'          # overridden by --freq-ghz
BW_MHZ = 10.0            # overridden by --bw-mhz (noise floor only)
TILT_DEG = 4.0           # overridden by --tilt-deg (post only)
TX_MAST_M = 3.0
RX_H_M = 6.0
# --- directional TX (post-processing) ---
SECTORS_DEG = (90.0, 210.0, 330.0)   # boresight azimuths, deg from +x (East)
ARRAY_ROWS, ARRAY_COLS = 4, 4
G_ARRAY_DB = 10 * np.log10(ARRAY_ROWS * ARRAY_COLS)   # ideal BF, assumption
G_ELEM_MAX_DB = 8.0                                   # tr38901 element max
P_COND_DBM = 47.0 - G_ELEM_MAX_DB - G_ARRAY_DB  # boresight EIRP = 47 dBm
TOPK = 32                # paths stored per RX cell
def _noise_dbm(): return -174 + 10 * np.log10(BW_MHZ * 1e6) + 9.0  # NF 9 dB
HALF = 1000.0            # 2x2 km box
STEP = 40.0              # RX grid resolution (m)
CHUNK = 300              # RX points per solver invocation
SITE = 'Site-NW'         # overridden by --site
SUF = 'nw'
FSTR = 'f3600'
def _npz(): return f'v4/data_twin_tower_{SUF}_{FSTR}.npz'
def _png(): return f'v4/twin_tower_{SUF}_{FSTR}_heatmap.png'


def build_rx_grid():
    sites = json.load(open('v4/data_sites_v4.json'))
    st = next(s for s in sites if s['name'] == SITE)
    x0, y0 = to_local(st['lon'], st['lat'])
    z_tx = float(st['terr_m'] + st['roof_m'] + TX_MAST_M - Z_REF)
    dem = DEM('v4/sd_data/dem_3dep.tif')
    terr = np.load('v4/data_terrain_500.npy')
    bldh = np.load('v4/data_bldh_500.npy')
    surf = terr + bldh
    n = int(2 * HALF / STEP) + 1
    xs = x0 + (np.arange(n) - n // 2) * STEP
    ys = y0 + (np.arange(n) - n // 2) * STEP
    # bilinear surface height at each grid point
    lat = N - (np.arange(GRID) + 0.5) / GRID * (N - S)
    lon = W + (np.arange(GRID) + 0.5) / GRID * (E - W)
    # local x of grid column centres / y of row centres
    xcent = (lon - (-117.085)) * 111320.0 * math.cos(math.radians(33.015))
    ycent = (lat - 33.015) * 110574.0
    pts = []
    for yy in ys:
        for xx in xs:
            c = int(np.clip(round((xx - xcent[0]) / (xcent[-1] - xcent[0]) * (GRID - 1)), 0, GRID - 1))
            r2 = int(np.clip(round((ycent[0] - yy) / (ycent[0] - ycent[-1]) * (GRID - 1)), 0, GRID - 1))
            z = float(surf[r2, c] + RX_H_M - Z_REF)
            pts.append((xx, yy, z))
    return st, (x0, y0, z_tx), xs, ys, np.array(pts)


def tr38901_gain_db(theta, phi, tilt_deg, az_deg):
    """3GPP TR 38.901 element power pattern (dB, incl. 8 dBi max), with
    electrical downtilt and boresight azimuth applied. theta: zenith [rad]
    from +z; phi: azimuth [rad] from +x. Matches Sionna's v_tr38901_pattern."""
    t = np.deg2rad(tilt_deg)
    a = np.deg2rad(az_deg)
    th = theta - t                      # downtilt shifts boresight down
    ph = (phi - a + np.pi) % (2 * np.pi) - np.pi
    th3 = ph3 = np.deg2rad(65.0)
    a_max = sla_v = 30.0
    a_v = -np.minimum(12.0 * ((th - np.pi / 2) / th3) ** 2, sla_v)
    a_h = -np.minimum(12.0 * (ph / ph3) ** 2, a_max)
    return -np.minimum(-(a_v + a_h), a_max) + G_ELEM_MAX_DB


def apply_antenna(pk_pw, pk_th, pk_ph, tilt_deg):
    """pk_*: (n_rx, TOPK) path power / departure zenith / departure azimuth.
    Returns best-server RSRP (dBm) over the 3 sectors."""
    n = pk_pw.shape[0]
    rsrp_best = np.full(n, -np.inf)
    for az in SECTORS_DEG:
        g_db = tr38901_gain_db(pk_th, pk_ph, tilt_deg, az)   # (n,K)
        g_lin = np.where(pk_pw > 0, 10.0 ** (g_db / 10.0), 0.0)
        pw = (pk_pw * g_lin).sum(axis=1)                      # directional ch. power
        rsrp = np.where(pw > 0, P_COND_DBM + 10 * np.log10(np.maximum(pw, 1e-30))
                        + G_ARRAY_DB, np.nan)
        rsrp_best = np.maximum(rsrp_best, np.where(np.isnan(rsrp), -np.inf, rsrp))
    return np.where(np.isneginf(rsrp_best), np.nan, rsrp_best)


def solve_chunk(pts_chunk):
    import mitsuba as mi
    mi.set_variant('llvm_ad_mono_polarized')
    import sionna.rt as rt
    s = rt.load_scene('v4/scene_v4/scene.xml', merge_shapes=True)
    mat_mode = 'itu'
    if FREQ_GHZ < 1.0:
        # ITU-R P.2040 materials are only defined >= 1 GHz and their
        # frequency callback RAISES below that, so clamp the callback to
        # the 1-GHz boundary values (documented standard practice).
        import drjit as dr
        from sionna.rt.radio_materials.itu import itu_material
        mat_mode = 'itu-clamped-1ghz'
        for m in s.radio_materials.values():
            if type(m).__name__ == 'ITURadioMaterial':
                t = m.itu_type
                m.frequency_update_callback = (
                    lambda f, _t=t: itu_material(_t, dr.maximum(f, 1e9)))
    s.frequency = FREQ_GHZ * 1e9
    # isotropic in the solver: antenna pattern applied in post-processing
    iso = rt.PlanarArray(num_rows=1, num_cols=1, pattern='iso', polarization='V')
    s.tx_array = iso
    s.rx_array = iso
    _, (x0, y0, z_tx), _, _, _ = build_rx_grid()
    s.add(rt.Transmitter('tx0', position=[float(x0), float(y0), float(z_tx)]))
    for i, (x, y, z) in enumerate(pts_chunk):
        s.add(rt.Receiver(f'r{i}', position=[float(x), float(y), float(z)]))
    t0 = time.time()
    paths = rt.PathSolver()(s, max_depth=3, los=True, specular_reflection=True,
                            refraction=False, diffraction=True,
                            edge_diffraction=True,
                            samples_per_src=1_000_000,
                            max_num_paths_per_src=1_000_000)
    dt = time.time() - t0
    ar, ai = paths.a
    v = paths.valid
    th = paths.theta_t
    ph = paths.phi_t
    inter = paths.interactions
    n_rx = len(pts_chunk)
    pk_pw = np.zeros((n_rx, TOPK), np.float32)
    pk_th = np.full((n_rx, TOPK), np.nan, np.float32)
    pk_ph = np.full((n_rx, TOPK), np.nan, np.float32)
    los = np.zeros(n_rx, bool)
    for i in range(n_rx):
        vi = np.asarray(v[i, 0]).astype(bool)
        if not vi.any():
            continue
        ari = np.asarray(ar[i, 0, 0, 0, :])[vi]
        aii = np.asarray(ai[i, 0, 0, 0, :])[vi]
        pwi = (ari ** 2 + aii ** 2).astype(np.float32)
        thi = np.asarray(th[i, 0, :]).astype(np.float32)[vi]
        phi = np.asarray(ph[i, 0, :]).astype(np.float32)[vi]
        iti = np.asarray(inter[:, i, 0, :])[:, vi]
        los[i] = bool(((iti == 0).all(axis=0)).any())
        order = np.argsort(-pwi)[:TOPK]
        m = len(order)
        pk_pw[i, :m] = pwi[order]
        pk_th[i, :m] = thi[order]
        pk_ph[i, :m] = phi[order]
    return pk_pw, pk_th, pk_ph, los, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chunk', type=int, default=None)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--post', action='store_true')
    ap.add_argument('--site', default='Site-NW', choices=['Site-NW','Site-NE','Site-SW','Site-SE'])
    ap.add_argument('--freq-ghz', type=float, default=3.6)
    ap.add_argument('--bw-mhz', type=float, default=10.0)
    ap.add_argument('--tilt-deg', type=float, default=4.0)
    args = ap.parse_args()
    global SITE, SUF, FREQ_GHZ, BW_MHZ, FSTR, MAT_MODE, TILT_DEG
    SITE = args.site; SUF = SITE.split('-')[1].lower()
    FREQ_GHZ = args.freq_ghz; BW_MHZ = args.bw_mhz; TILT_DEG = args.tilt_deg
    FSTR = f'f{int(round(FREQ_GHZ*1000))}'
    MAT_MODE = 'itu-clamped-1ghz' if FREQ_GHZ < 1.0 else 'itu'

    st, tx, xs, ys, pts = build_rx_grid()
    n = len(xs)
    print(f'TX {SITE} @ local ({tx[0]:.0f},{tx[1]:.0f},{tx[2]:.1f}); RX grid {n}x{n} = {len(pts)} pts')

    if args.smoke:
        sel = pts[::max(1, len(pts) // 9)][:9]
        pk_pw, pk_th, pk_ph, los, dt = solve_chunk(sel)
        rsrp = apply_antenna(pk_pw, pk_th, pk_ph, TILT_DEG)
        print(f'smoke: {len(sel)} pts in {dt:.0f}s; RSRP sample {np.round(rsrp[:5],1)}; '
              f'LOS {los.sum()}/{len(los)}; paths/cell {np.isfinite(pk_th).sum(1)[:5]}')
        return

    if args.post:
        d = np.load(_npz())
        n = len(d['xs'])
        rsrp = apply_antenna(d['pk_pw'], d['pk_th'], d['pk_ph'], TILT_DEG).reshape(n, n)
        los = d['los'].reshape(n, n)
        meta = json.loads(str(d['meta']))
        noise = meta['noise_dbm']
        snr = rsrp - noise
        cov90 = float(np.nanmean(rsrp >= -90)) * 100
        cov100 = float(np.nanmean(rsrp >= -100)) * 100
        med_snr = float(np.nanmedian(snr))
        print(f'RSRP>=-90 dBm: {cov90:.1f}%   RSRP>=-100 dBm: {cov100:.1f}%   '
              f'median SNR: {med_snr:.1f} dB   LOS frac: {los.mean()*100:.1f}%   '
              f'no-path: {np.isnan(rsrp).mean()*100:.1f}%')
        ant = f'3sec-4x4-tr38901-tilt{TILT_DEG:g}'
        np.savez(_npz(), xs=d['xs'], ys=d['ys'], pl=np.full_like(rsrp, np.nan),
                 los=los, rsrp=rsrp, snr=snr,
                 pk_pw=d['pk_pw'], pk_th=d['pk_th'], pk_ph=d['pk_ph'],
                 meta=json.dumps(dict(noise_dbm=noise, bw_mhz=meta['bw_mhz'],
                                      rx_h_m=RX_H_M, freq_ghz=FREQ_GHZ, site=SITE,
                                      mat_mode=MAT_MODE, ant_cfg=ant,
                                      p_cond_dbm=P_COND_DBM, g_array_db=G_ARRAY_DB,
                                      tilt_deg=TILT_DEG)))
        # heatmap
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(rsrp, origin='lower',
                       extent=[xs[0]/1000, xs[-1]/1000, ys[0]/1000, ys[-1]/1000],
                       vmin=-110, vmax=-50, cmap='viridis')
        ax.plot([tx[0]/1000], [tx[1]/1000], 'r^', ms=10, label=SITE)
        ax.set_xlabel('km E'); ax.set_ylabel('km N')
        ax.set_title(f'Single-tower digital twin: {SITE} RSRP ({FREQ_GHZ:g} GHz, {ant})')
        fig.colorbar(im, label='RSRP dBm')
        ax.legend()
        fig.savefig(_png(), dpi=110, bbox_inches='tight')
        print(f'wrote {_png()}')
        return

    k = args.chunk
    chunks = [pts[i:i + CHUNK] for i in range(0, len(pts), CHUNK)]
    if k is None or k >= len(chunks):
        print(f'{len(chunks)} chunks; pass --chunk K (0..{len(chunks)-1})')
        return
    pk_pw_c, pk_th_c, pk_ph_c, los_c, dt = solve_chunk(chunks[k])
    print(f'chunk {k}/{len(chunks)-1}: {len(chunks[k])} pts in {dt:.0f}s')
    if os.path.exists(_npz()):
        d = np.load(_npz())
        if 'pk_pw' in d:
            pk_pw_a, pk_th_a, pk_ph_a = d['pk_pw'].copy(), d['pk_th'].copy(), d['pk_ph'].copy()
            los_a, done = d['los'].copy(), set(d['done'].tolist())
        else:
            pk_pw_a = pk_th_a = pk_ph_a = los_a = done = None
    else:
        pk_pw_a = pk_th_a = pk_ph_a = los_a = done = None
    if pk_pw_a is None:
        pk_pw_a = np.zeros((len(pts), TOPK), np.float32)
        pk_th_a = np.full((len(pts), TOPK), np.nan, np.float32)
        pk_ph_a = np.full((len(pts), TOPK), np.nan, np.float32)
        los_a = np.zeros(len(pts), bool)
        done = set()
    sl = slice(k * CHUNK, k * CHUNK + len(chunks[k]))
    pk_pw_a[sl] = pk_pw_c
    pk_th_a[sl] = pk_th_c
    pk_ph_a[sl] = pk_ph_c
    los_a[sl] = los_c
    done.add(k)
    np.savez(_npz(), xs=xs, ys=ys, pk_pw=pk_pw_a, pk_th=pk_th_a, pk_ph=pk_ph_a,
             los=los_a, done=np.array(sorted(done)),
             meta=json.dumps(dict(noise_dbm=_noise_dbm(), bw_mhz=BW_MHZ,
                                  rx_h_m=RX_H_M, freq_ghz=FREQ_GHZ, site=SITE,
                                  mat_mode=MAT_MODE)))
    print(f'saved; chunks done: {sorted(done)}/{len(chunks)}')


if __name__ == '__main__':
    main()
