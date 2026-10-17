#!/usr/bin/env python3
"""v4 Sionna calibration on REAL terrain — multi-site.

TX: all 4 v4 rooftop sites (roof + 3 m mast), isotropic, 3.6 GHz.
RX: 200 points (50/site) stratified by distance and by our v4 LOS flag,
    at 6 m above surface (rooftop-CPE convention).
Compares Sionna RT path loss vs our planner_v4 uma_pl (same h_bs/h_ut/d3d).
"""
import json, math, time
import numpy as np
import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")
import sionna.rt as rt
from numba import njit

import sys
sys.path.insert(0, '.')
from build_scene_v4 import DEM, to_local, Z_REF

S, W, N, E = 32.9925, -117.1120, 33.0375, -117.0580
GRID = 500
FC = 3.6e9
C = 3e8
TX_MAST_M = 3.0
RX_H_M = 6.0
LAT0 = 33.015
KX = 111320.0 * math.cos(math.radians(LAT0)); KY = 110540.0
t0 = time.time()

dem = DEM("sd_data/dem_3dep.tif")
sites = json.load(open('data_sites_v4.json'))
terrain = np.load('data_terrain_500.npy')
bld_h = np.load('data_bldh_500.npy')
surface = terrain + bld_h
step_m = (N - S) / GRID * 111320.0

@njit
def los_grid_rxsurf(surf, rx_surf, sr, sc, tx_h, rx_h, n):
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
                rr = int(sr + dr * f); cc = int(sc + dc * f)
                linez = txz + (rxz - txz) * f
                if surf[rr, cc] + 0.6 > linez:
                    blocked = True; break
            out[r, c] = not blocked
    return out

def uma_pl(d3d, los, h_bs, h_ut):
    d2d = np.maximum(d3d, 1.0)
    fc_ghz = 3.6
    pl_los_seg1 = 28.0 + 22 * np.log10(d3d) + 20 * np.log10(fc_ghz)
    d_bp = 4 * h_bs * h_ut * FC / C
    pl_los_seg2 = (28.0 + 40 * np.log10(d3d) + 20 * np.log10(fc_ghz)
                   - 9 * np.log10(d_bp ** 2 + (h_bs - h_ut) ** 2))
    pl_los = np.where(d2d < d_bp, pl_los_seg1, pl_los_seg2)
    pl_nlos = 13.54 + 39.08 * np.log10(d3d) + 20 * np.log10(fc_ghz) - 0.6 * (h_ut - 1.5)
    return np.where(los, pl_los, np.maximum(pl_los, pl_nlos))

def ll2local(lat, lon):
    x, y = to_local(lon, lat)
    return float(x), float(y)

# ---- sample points: 50/site, stratified by distance band and v4 LOS flag ----
rng = np.random.default_rng(42)
pts = []  # (site_idx, grid_r, grid_c, x, y, z_rx, d3d, v4_los)
rr, cc = np.mgrid[0:GRID, 0:GRID]
for si, st in enumerate(sites):
    sr, sc = st['grid']
    los = los_grid_rxsurf(surface, surface, sr, sc, TX_MAST_M, RX_H_M, GRID)
    txz = surface[sr, sc] + TX_MAST_M
    d2d = np.sqrt(((rr - sr) * step_m) ** 2 + ((cc - sc) * step_m) ** 2)
    bands = [(100, 400), (400, 900), (900, 2500)]
    per_band = 50 // 3
    for (d1, d2) in bands:
        band = (d2d >= d1) & (d2d < d2)
        los_c = np.argwhere(band & los); nlos_c = np.argwhere(band & ~los)
        take_l = min(len(los_c), per_band // 2); take_n = min(len(nlos_c), per_band - take_l)
        sel = []
        if take_l: sel += [tuple(los_c[i]) for i in rng.choice(len(los_c), take_l, replace=False)]
        if take_n: sel += [tuple(nlos_c[i]) for i in rng.choice(len(nlos_c), take_n, replace=False)]
        for (gr, gc) in sel:
            lat = N - (gr + 0.5) / GRID * (N - S); lon = W + (gc + 0.5) / GRID * (E - W)
            x, y = ll2local(lat, lon)
            z_rx = float(surface[gr, gc] + RX_H_M - Z_REF)
            d3d = math.sqrt(((gr - sr) * step_m) ** 2 + ((gc - sc) * step_m) ** 2 +
                              (surface[gr, gc] + RX_H_M - txz) ** 2)
            pts.append(dict(site=si, site_name=st['name'], gr=int(gr), gc=int(gc),
                            x=x, y=y, z_rx=z_rx, d3d=max(d3d, 3.0),
                            v4_los=bool(los[gr, gc])))
print(f"sampled {len(pts)} points across {len(sites)} sites")

# ---- Sionna scene ----
s = rt.load_scene("scene_v4/scene.xml", merge_shapes=True)
s.frequency = FC
iso = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
s.tx_array = iso; s.rx_array = iso
tx_z = []
for si, st in enumerate(sites):
    x, y = ll2local(st['lat'], st['lon'])
    z = float(st['terr_m'] + st['roof_m'] + TX_MAST_M - Z_REF)
    s.add(rt.Transmitter(f"tx{si}", position=[x, y, z]))
    tx_z.append(z)
    print(f"  tx{si} ({st['name']}): z_AGL {st['roof_m']+TX_MAST_M:.1f} m")
for i, p in enumerate(pts):
    s.add(rt.Receiver(f"r{i}", position=[p['x'], p['y'], p['z_rx']]))

print("solving paths...")
paths = rt.PathSolver()(s, max_depth=3, los=True, specular_reflection=True,
                        refraction=False, diffraction=True, edge_diffraction=True,
                        samples_per_src=1_000_000, max_num_paths_per_src=1_000_000)
ar, ai = paths.a
v = np.asarray(paths.valid.numpy()).astype(bool)   # (rx, tx, paths)
a = ar.numpy()                                      # (rx, 1, tx, 1, paths)
pw = ((a**2 + ai.numpy()**2) * v[:, None, :, None, :]).sum(axis=(1, 3, 4))  # (rx, tx)
it = np.asarray(paths.interactions.numpy())        # (depth, rx, tx, paths)
los_rt = ((it == 0).all(axis=0) & v).any(axis=-1)   # (rx, tx)
print(f"solve done in {time.time()-t0:.0f}s")

# ---- compare ----
rows = []
for i, p in enumerate(pts):
    si = p['site']
    rt_pl = float(-10 * np.log10(pw[i, si])) if pw[i, si] > 0 else None
    uma = float(uma_pl(p['d3d'], p['v4_los'], TX_MAST_M, RX_H_M))
    uma_los_only = float(uma_pl(p['d3d'], True, TX_MAST_M, RX_H_M))
    rows.append(dict(site=p['site_name'], d3d=round(p['d3d'], 1),
                     v4_los=p['v4_los'], los_rt=bool(los_rt[i, si]),
                     rt_pl=round(rt_pl, 2) if rt_pl else None,
                     uma_pl=round(uma, 2), uma_los_pl=round(uma_los_only, 2)))

def stats(rs):
    e = np.array([r['uma_pl'] - r['rt_pl'] for r in rs if r['rt_pl'] is not None])
    if not len(e):
        return dict(n=len(rs), usable=0, bias_uma_minus_rt=None, mae=None, rmse=None)
    return dict(n=len(rs), usable=int(len(e)),
                bias_uma_minus_rt=round(float(e.mean()), 2),
                mae=round(float(np.abs(e).mean()), 2),
                rmse=round(float(np.sqrt((e ** 2).mean())), 2))

out = {
    'note': 'v4 Sionna calibration on REAL terrain (3DEP + County/LiDAR scene), '
            '4 rooftop TX sites, RX 6 m (rooftop-CPE convention). '
            'bias>0 means UMa more pessimistic than RT.',
    'n_points': len(rows),
    'overall': stats(rows),
    'by_rt_los': stats([r for r in rows if r['los_rt']]),
    'by_rt_nlos': stats([r for r in rows if not r['los_rt']]),
    'by_v4_los': stats([r for r in rows if r['v4_los']]),
    'by_v4_nlos': stats([r for r in rows if not r['v4_los']]),
    'los_flag_agreement': float(np.mean([r['los_rt'] == r['v4_los'] for r in rows])),
    'rt_los_frac': float(np.mean([r['los_rt'] for r in rows])),
    'v4_los_frac': float(np.mean([r['v4_los'] for r in rows])),
    'no_path_frac': float(np.mean([r['rt_pl'] is None for r in rows])),
    'per_site': {s['name']: stats([r for r in rows if r['site'] == s['name']]) for s in sites},
}
json.dump({'summary': out, 'points': rows}, open('data_sionna_cal_v4.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'per_site'}, indent=1))
print("per-site:", json.dumps(out['per_site'], indent=1))
