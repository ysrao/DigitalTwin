#!/usr/bin/env python3
"""rt_v5.py - single-tower ray tracing, v5: diffuse scattering + street-level grid.

What changed vs single_tower_twin.py (v4, kept unchanged for the shipped grids):
  - Diffuse scattering ON (Lambertian). Scattering coefficients are stated
    assumptions until calibrated against field data: buildings S = 0.3,
    ground S = 0.2 (--s-bld / --s-ground). Specular power is reduced by
    (1 - S^2) as Sionna does, so energy is conserved.
  - Two receiver grids per site and band (--rx):
        roof   : 6 m above the local surface (terrain + building) - rooftop CPE,
                 same points as v4
        street : 1.5 m above terrain - pedestrian / vehicular / indoor handsets.
                 Points that fall inside a building footprint are moved to the
                 nearest outdoor 10 m raster cell within 30 m (the facade side,
                 where 3GPP's O2I model measures the outdoor signal); points
                 with no outdoor cell that close keep their place and are
                 flagged (moved = -1).
  - The antenna is applied INSIDE the solver to all paths (diffuse scattering
    spreads power over hundreds of weak paths; a top-K cut would lose it):
    3 sectors (az 90/210/330 from +x), TR 38.901 element, 4x4 array, for every
    tilt in TILTS. The top-32 paths are still stored for later what-ifs.

Usage (one chunk per process keeps memory bounded on small machines):
    python rt_v5.py --site Site-NW --freq-ghz 3.6 --rx street --chunk K
    python rt_v5.py --site Site-NW --freq-ghz 3.6 --rx street --post
Output: data_v5_<site>_<band>_<rx>.npz
"""
import argparse, json, math, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from build_scene_v4 import to_local, Z_REF

S, W, N, E = 32.9925, -117.1120, 33.0375, -117.0580
GRID = 500
TX_MAST_M = 3.0
RX_H = {'roof': 6.0, 'street': 1.5}
HALF, STEP = 1000.0, 40.0
CHUNK = 300
TOPK = 32
SECTORS_DEG = (90.0, 210.0, 330.0)
TILTS = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0)
G_ELEM_MAX_DB = 8.0
G_ARRAY_DB = 10 * np.log10(16)
P_COND_DBM = 47.0 - G_ELEM_MAX_DB - G_ARRAY_DB      # boresight EIRP 47 dBm
NUDGE_CELLS = 3                                      # 30 m search radius


def tr38901_gain_db(theta, phi, tilt_deg, az_deg):
    """TR 38.901 element pattern incl. 8 dBi max (same as single_tower_twin.py)."""
    th = theta - np.deg2rad(tilt_deg)
    ph = (phi - np.deg2rad(az_deg) + np.pi) % (2 * np.pi) - np.pi
    th3 = ph3 = np.deg2rad(65.0)
    a_v = -np.minimum(12.0 * ((th - np.pi / 2) / th3) ** 2, 30.0)
    a_h = -np.minimum(12.0 * (ph / ph3) ** 2, 30.0)
    return -np.minimum(-(a_v + a_h), 30.0) + G_ELEM_MAX_DB


def raster_rc(x, y):
    """local metres -> (row, col) in the 500x500 rasters."""
    lon = x / (111320.0 * math.cos(math.radians(33.015))) - 117.085
    lat = y / 110574.0 + 33.015
    r = int(np.clip((N - lat) / (N - S) * GRID, 0, GRID - 1))
    c = int(np.clip((lon - W) / (E - W) * GRID, 0, GRID - 1))
    return r, c


def rc_to_xy(r, c):
    lat = N - (r + 0.5) / GRID * (N - S)
    lon = W + (c + 0.5) / GRID * (E - W)
    return to_local(lon, lat)


def build_rx_grid(site, rx):
    sites = json.load(open(os.path.join(ROOT, 'data_sites_v4.json')))
    st = next(s for s in sites if s['name'] == site)
    x0, y0 = to_local(st['lon'], st['lat'])
    z_tx = float(st['terr_m'] + st['roof_m'] + TX_MAST_M - Z_REF)
    terr = np.load(os.path.join(ROOT, 'data_terrain_500.npy'))
    bldh = np.load(os.path.join(ROOT, 'data_bldh_500.npy'))
    n = int(2 * HALF / STEP) + 1
    xs = x0 + (np.arange(n) - n // 2) * STEP
    ys = y0 + (np.arange(n) - n // 2) * STEP
    pts, moved, indoor = [], [], []
    for yy in ys:
        for xx in xs:
            r, c = raster_rc(xx, yy)
            inside = bldh[r, c] > 0.5
            indoor.append(inside)
            px, py, mv = xx, yy, 0
            if rx == 'street' and inside:
                best = None
                for dr in range(-NUDGE_CELLS, NUDGE_CELLS + 1):
                    for dc in range(-NUDGE_CELLS, NUDGE_CELLS + 1):
                        rr_, cc_ = r + dr, c + dc
                        if 0 <= rr_ < GRID and 0 <= cc_ < GRID and bldh[rr_, cc_] <= 0.5:
                            d = dr * dr + dc * dc
                            if best is None or d < best[0]:
                                best = (d, rr_, cc_)
                if best is not None:
                    px, py = rc_to_xy(best[1], best[2]); mv = 1
                    r, c = best[1], best[2]
                else:
                    mv = -1
            surf = terr[r, c] + (bldh[r, c] if rx == 'roof' else 0.0)
            pts.append((float(px), float(py), float(surf + RX_H[rx] - Z_REF)))
            moved.append(mv)
    return st, (float(x0), float(y0), z_tx), xs, ys, np.array(pts), np.array(moved), np.array(indoor)


def load_scene(freq_ghz, s_bld, s_ground):
    import mitsuba as mi
    mi.set_variant('llvm_ad_mono_polarized')
    import sionna.rt as rt
    import drjit as dr
    from sionna.rt.radio_materials.itu import itu_material
    s = rt.load_scene(os.path.join(ROOT, 'scene_v4', 'scene.xml'), merge_shapes=True)
    grounds = {'very_dry_ground', 'medium_dry_ground', 'wet_ground'}
    for m in s.radio_materials.values():
        if type(m).__name__ != 'ITURadioMaterial':
            continue
        t = m.itu_type
        m.scattering_coefficient = s_ground if t in grounds else s_bld
        if freq_ghz < 1.0 or freq_ghz > 10.0:     # P.2040 range clamp (as in v4)
            lo, hi = (1e9, 10e9) if t in grounds else (1e9, 1e12)
            m.frequency_update_callback = (
                lambda f, _t=t, _lo=lo, _hi=hi: itu_material(_t, dr.minimum(dr.maximum(f, _lo), _hi)))
    s.frequency = freq_ghz * 1e9
    iso = rt.PlanarArray(num_rows=1, num_cols=1, pattern='iso', polarization='V')
    s.tx_array = iso; s.rx_array = iso
    return s, rt


def solve_chunk(pts, tx, freq_ghz, s_bld, s_ground):
    s, rt = load_scene(freq_ghz, s_bld, s_ground)
    s.add(rt.Transmitter('tx0', position=[float(v) for v in tx]))
    for i, (x, y, z) in enumerate(pts):
        s.add(rt.Receiver(f'r{i}', position=[float(x), float(y), float(z)]))
    t0 = time.time()
    p = rt.PathSolver()(s, max_depth=3, los=True, specular_reflection=True,
                        diffuse_reflection=True, refraction=False, diffraction=True,
                        edge_diffraction=True, samples_per_src=1_000_000,
                        max_num_paths_per_src=1_000_000)
    dt = time.time() - t0
    ar, ai = (np.asarray(a) for a in p.a)
    v = np.asarray(p.valid)[:, 0, :].astype(bool)              # (rx, paths)
    pw = np.where(v, (ar ** 2 + ai ** 2)[:, 0, 0, 0, :], 0.0)   # (rx, paths)
    th = np.asarray(p.theta_t)[:, 0, :]
    ph = np.asarray(p.phi_t)[:, 0, :]
    inter = np.asarray(p.interactions)[:, :, 0, :]              # (depth, rx, paths)
    is_los = (inter == 0).all(axis=0) & v
    n = len(pts)
    dirpw = np.zeros((n, len(TILTS), len(SECTORS_DEG)))
    for ti, tilt in enumerate(TILTS):
        for si, az in enumerate(SECTORS_DEG):
            g = np.where(v, 10 ** (tr38901_gain_db(th, ph, tilt, az) / 10), 0.0)
            dirpw[:, ti, si] = (pw * g).sum(axis=1)
    iso = pw.sum(axis=1)
    los_pw = np.where(is_los, pw, 0.0).sum(axis=1)
    diff_types = np.asarray(inter)
    has_diffuse = ((diff_types == 2) | (diff_types == 2.0)).any(axis=0) & v
    diffuse_pw = np.where(has_diffuse, pw, 0.0).sum(axis=1)
    order = np.argsort(-pw, axis=1)[:, :TOPK]
    take = lambda a: np.take_along_axis(a, order, axis=1).astype(np.float32)
    return dict(dirpw=dirpw, iso=iso, los=is_los.any(axis=1), los_pw=los_pw,
                diffuse_pw=diffuse_pw, npath=v.sum(axis=1),
                pk_pw=take(pw), pk_th=np.where(take(pw) > 0, take(th), np.nan),
                pk_ph=np.where(take(pw) > 0, take(ph), np.nan)), dt


def fname(site, fstr, rx):
    return os.path.join(ROOT, f"data_v5_{site.split('-')[1].lower()}_{fstr}_{rx}.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--site', required=True, choices=['Site-NW', 'Site-NE', 'Site-SW', 'Site-SE'])
    ap.add_argument('--freq-ghz', type=float, required=True)
    ap.add_argument('--rx', required=True, choices=['roof', 'street'])
    ap.add_argument('--chunk', type=int)
    ap.add_argument('--post', action='store_true')
    ap.add_argument('--s-bld', type=float, default=0.3)
    ap.add_argument('--s-ground', type=float, default=0.2)
    ap.add_argument('--npts', type=int, default=None, help='debug: only the first N points')
    a = ap.parse_args()
    fstr = f'f{int(round(a.freq_ghz * 1000))}'
    out = fname(a.site, fstr, a.rx)
    st, tx, xs, ys, pts, moved, indoor = build_rx_grid(a.site, a.rx)
    if a.npts:
        pts = pts[:a.npts]
    chunks = [(i, min(i + CHUNK, len(pts))) for i in range(0, len(pts), CHUNK)]
    if a.post:
        d = dict(np.load(out))
        n = len(xs)
        assert len(d['done']) == len(chunks), f"chunks done {sorted(d['done'])} of {len(chunks)}"
        rsrp = np.where(d['dirpw'].max(axis=2) > 0,
                        P_COND_DBM + G_ARRAY_DB + 10 * np.log10(np.maximum(d['dirpw'].max(axis=2), 1e-30)),
                        np.nan)                                     # (n_rx, n_tilt)
        d['rsrp_tilt'] = rsrp.astype(np.float32)
        d['pl'] = np.where(d['iso'] > 0, -10 * np.log10(np.maximum(d['iso'], 1e-30)), np.nan).astype(np.float32)
        np.savez_compressed(out, **d)
        t4 = TILTS.index(4.0)
        r4 = rsrp[:, t4]
        print(f"{a.site} {fstr} {a.rx}: no-path {np.isnan(r4).mean()*100:.1f}%  LOS {d['los'].mean()*100:.1f}%  "
              f"RSRP>=-90 (tilt 4) {np.mean(r4 >= -90)*100:.1f}% of all cells  "
              f"diffuse share of power median {np.nanmedian(d['diffuse_pw']/np.where(d['iso']>0,d['iso'],np.nan))*100:.1f}%")
        return
    if a.chunk is None or a.chunk >= len(chunks):
        print(f'{len(chunks)} chunks; pass --chunk 0..{len(chunks)-1}'); return
    i0, i1 = chunks[a.chunk]
    res, dt = solve_chunk(pts[i0:i1], tx, a.freq_ghz, a.s_bld, a.s_ground)
    n = len(pts)
    if os.path.exists(out):
        d = dict(np.load(out)); done = set(d['done'].tolist())
    else:
        d = dict(dirpw=np.zeros((n, len(TILTS), len(SECTORS_DEG))), iso=np.zeros(n),
                 los=np.zeros(n, bool), los_pw=np.zeros(n), diffuse_pw=np.zeros(n),
                 npath=np.zeros(n, np.int32),
                 pk_pw=np.zeros((n, TOPK), np.float32), pk_th=np.full((n, TOPK), np.nan, np.float32),
                 pk_ph=np.full((n, TOPK), np.nan, np.float32))
        done = set()
    for k, val in res.items():
        d[k][i0:i1] = val
    done.add(a.chunk)
    d.update(xs=xs, ys=ys, rx_xyz=pts, moved=moved, indoor=indoor, tilts=np.array(TILTS),
             done=np.array(sorted(done)),
             meta=json.dumps(dict(site=a.site, freq_ghz=a.freq_ghz, rx=a.rx, rx_h_m=RX_H[a.rx],
                                  s_bld=a.s_bld, s_ground=a.s_ground, max_depth=3,
                                  samples_per_src=1_000_000, diffuse=True, sectors=SECTORS_DEG,
                                  p_cond_dbm=P_COND_DBM, g_array_db=float(G_ARRAY_DB),
                                  version='v5-diffuse')))
    np.savez_compressed(out, **d)
    print(f"{a.site} {fstr} {a.rx} chunk {a.chunk}/{len(chunks)-1}: {i1-i0} pts in {dt:.0f}s; "
          f"paths/RX median {np.median(res['npath']):.0f}")


if __name__ == '__main__':
    main()
