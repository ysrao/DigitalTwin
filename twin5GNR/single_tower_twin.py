#!/usr/bin/env python3
"""Single-tower digital twin: full-area Sionna RT coverage map for one site.

Fills the gap between the v4 sampled calibration (192 points) and a true
digital-twin coverage surface: every grid cell gets a ray-traced path gain.

Scene   : v4/scene_v4/scene.xml (USGS 3DEP terrain + SanGIS/LiDAR buildings,
          ITU concrete / medium-dry-ground materials) - same as sionna_cal_v4
TX      : per --site, rooftop + 3 m mast, per --freq-ghz, isotropic
RX      : regular grid over 2x2 km centred on the TX, 6 m above surface
          (rooftop-CPE convention, same as the v4 calibration)
Solver  : PathSolver max_depth=3, LOS + specular + diffraction + edge
          diffraction; 1M samples/src (same as the v4 calibration)
Post    : path gain -> PL -> RSRP (47 dBm Cat-B EIRP) -> SNR (NF 9 dB, 10 MHz)

Usage: single_tower_twin.py --chunk K   (one chunk per process; results appended
         to v4/data_twin_tower_nw.npz, safe against OOM in later chunks)
       single_tower_twin.py --smoke      (3x3 grid pipeline test)
       single_tower_twin.py --post       (build RSRP/SNR grids + heatmap from npz)
"""
import argparse, json, math, os, sys, time
import numpy as np

sys.path.insert(0, '.')
from build_scene_v4 import DEM, to_local, Z_REF

S, W, N, E = 32.9925, -117.1120, 33.0375, -117.0580
GRID = 500
FREQ_GHZ = 3.6            # overridden by --freq-ghz
BW_MHZ = 10.0            # overridden by --bw-mhz (noise floor only)
TX_MAST_M = 3.0
RX_H_M = 6.0
EIRP_DBM = 47.0          # CBRS Cat-B, same as planner_v4 TX_EIRP_DBM
def _noise_dbm(): return -174 + 10 * np.log10(BW_MHZ * 1e6) + 9.0  # NF 9 dB
HALF = 1000.0            # 2x2 km box
STEP = 40.0              # RX grid resolution (m)
CHUNK = 300              # RX points per solver invocation
SITE = 'Site-NW'         # overridden by --site
SUF = 'nw'
FSTR = 'f3600'
def _npz(): return f'data_twin_tower_{SUF}_{FSTR}.npz'
def _png(): return f'twin_tower_{SUF}_{FSTR}_heatmap.png'


def build_rx_grid():
    sites = json.load(open('data_sites_v4.json'))
    st = next(s for s in sites if s['name'] == SITE)
    x0, y0 = to_local(st['lon'], st['lat'])
    z_tx = float(st['terr_m'] + st['roof_m'] + TX_MAST_M - Z_REF)
    dem = DEM('sd_data/dem_3dep.tif')
    terr = np.load('data_terrain_500.npy')
    bldh = np.load('data_bldh_500.npy')
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


def solve_chunk(pts_chunk):
    import mitsuba as mi
    mi.set_variant('llvm_ad_mono_polarized')
    import sionna.rt as rt
    s = rt.load_scene('scene_v4/scene.xml', merge_shapes=True)
    s.frequency = FREQ_GHZ * 1e9
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
    ar, ai = paths.a
    v = np.asarray(paths.valid.numpy()).astype(bool)          # (rx, tx, paths)
    a = ar.numpy()                                            # (rx,1,tx,1,paths)
    pw = ((a ** 2 + ai.numpy() ** 2) * v[:, None, :, None, :]).sum(axis=(1, 3, 4))[:, 0]
    it = np.asarray(paths.interactions.numpy())               # (depth, rx, tx, paths)
    los = ((it == 0).all(axis=0) & v).any(axis=-1)[:, 0]
    dt = time.time() - t0
    pl = np.where(pw > 0, -10 * np.log10(np.maximum(pw, 1e-30)), np.nan)
    return pl, los.astype(bool), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chunk', type=int, default=None)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--post', action='store_true')
    ap.add_argument('--site', default='Site-NW', choices=['Site-NW','Site-NE','Site-SW','Site-SE'])
    ap.add_argument('--freq-ghz', type=float, default=3.6)
    ap.add_argument('--bw-mhz', type=float, default=10.0)
    args = ap.parse_args()
    global SITE, SUF, FREQ_GHZ, BW_MHZ, FSTR
    SITE = args.site; SUF = SITE.split('-')[1].lower()
    FREQ_GHZ = args.freq_ghz; BW_MHZ = args.bw_mhz
    FSTR = f'f{int(round(FREQ_GHZ*1000))}'

    st, tx, xs, ys, pts = build_rx_grid()
    n = len(xs)
    print(f'TX {SITE} @ local ({tx[0]:.0f},{tx[1]:.0f},{tx[2]:.1f}); RX grid {n}x{n} = {len(pts)} pts')

    if args.smoke:
        sel = pts[::max(1, len(pts) // 9)][:9]
        pl, los, dt = solve_chunk(sel)
        print(f'smoke: {len(sel)} pts in {dt:.0f}s; PL sample {np.round(pl[:5],1)}; LOS {los.sum()}/{len(los)}')
        return

    if args.post:
        d = np.load(_npz())
        n = len(d['xs'])
        pl = d['pl'].reshape(n, n)
        los = d['los'].reshape(n, n)
        meta = json.loads(str(d['meta']))
        rsrp = EIRP_DBM - pl
        snr = rsrp - meta['noise_dbm']
        cov90 = float(np.nanmean(rsrp >= -90)) * 100
        cov100 = float(np.nanmean(rsrp >= -100)) * 100
        med_snr = float(np.nanmedian(snr))
        print(f'RSRP>=-90 dBm: {cov90:.1f}%   RSRP>=-100 dBm: {cov100:.1f}%   '
              f'median SNR: {med_snr:.1f} dB   LOS frac: {los.mean()*100:.1f}%   '
              f'no-path: {np.isnan(pl).mean()*100:.1f}%')
        np.savez(_npz(), xs=d['xs'], ys=d['ys'], pl=pl, los=los,
                 rsrp=rsrp, snr=snr,
                 meta=json.dumps(dict(eirp_dbm=EIRP_DBM, noise_dbm=meta['noise_dbm'],
                                      bw_mhz=meta['bw_mhz'], rx_h_m=RX_H_M,
                                      freq_ghz=FREQ_GHZ, site=SITE)))
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
        ax.set_title(f'Single-tower digital twin: {SITE} RSRP (Sionna RT, 3.6 GHz)')
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
    pl_c, los_c, dt = solve_chunk(chunks[k])
    print(f'chunk {k}/{len(chunks)-1}: {len(chunks[k])} pts in {dt:.0f}s')
    if os.path.exists(_npz()):
        d = np.load(_npz())
        pl_all, los_all, done = d['pl'].copy(), d['los'].copy(), set(d['done'].tolist())
    else:
        pl_all = np.full(len(pts), np.nan)
        los_all = np.zeros(len(pts), bool)
        done = set()
    sl = slice(k * CHUNK, k * CHUNK + len(chunks[k]))
    pl_all[sl] = pl_c
    los_all[sl] = los_c
    done.add(k)
    np.savez(_npz(), xs=xs, ys=ys, pl=pl_all, los=los_all, done=np.array(sorted(done)),
             meta=json.dumps(dict(eirp_dbm=EIRP_DBM, noise_dbm=_noise_dbm(),
                                  bw_mhz=BW_MHZ, rx_h_m=RX_H_M,
                                  freq_ghz=FREQ_GHZ, site=SITE)))
    print(f'saved; chunks done: {sorted(done)}/{len(chunks)}')


if __name__ == '__main__':
    main()
