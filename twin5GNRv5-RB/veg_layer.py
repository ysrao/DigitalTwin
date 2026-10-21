#!/usr/bin/env python3
"""veg_layer.py - foliage depth on the direct path, per site and receiver grid.

Input : data_tcc_2021.tif - NLCD 2021 Tree Canopy Cover (USFS/MRLC, public
        domain), % cover, ~25 m pixels, WGS84, fetched for the tile bbox
        (fetch_tcc() below).
Model : the straight TX->RX line is sampled every 5 m; wherever it runs
        between the ground and TREE_H_M above terrain it accumulates
        canopy_fraction x 5 m. The result, an effective foliage depth d (m),
        is frequency independent and stored per cell.
        Loss per band is applied in post (viewer / twin_model.py) with
        Weissberger's modified exponential decay model, capped at
        FOLIAGE_CAP_DB (beyond that, diffraction over and around the canopy
        dominates):
            L = 0.45 f^0.284 d          for d < 14 m
            L = 1.33 f^0.284 d^0.588    for 14 <= d <= 400 m   (f in GHz)
        It is applied to the line-of-sight component only. Reflected and
        diffuse paths usually leave the canopy layer near facades, so this
        is a lower bound on foliage loss until calibrated against field data.
Output: data_veg_<site>.npz with depth_roof, depth_street (m, per grid cell).
"""
import json, math, os, sys, urllib.request
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import rt_v5 as R

S, W, N, E = R.S, R.W, R.N, R.E
TREE_H_M = 8.0          # typical suburban street-tree height (assumption)
STEP_M = 5.0
TCC = os.path.join(ROOT, 'data_tcc_2021.tif')
WCS = ("https://www.mrlc.gov/geoserver/ows?service=WCS&version=2.0.1&request=GetCoverage"
       "&coverageId=mrlc_download__nlcd_tcc_conus_2021_v2021-4&format=image/geotiff"
       "&subsettingCrs=http://www.opengis.net/def/crs/EPSG/0/4326"
       "&outputCrs=http://www.opengis.net/def/crs/EPSG/0/4326"
       f"&subset=Lat({S},{N})&subset=Long({W},{E})")


def fetch_tcc():
    urllib.request.urlretrieve(WCS, TCC)


def weissberger_db(f_ghz, d_m, cap_db=20.0):
    d = np.asarray(d_m, float)
    k = f_ghz ** 0.284
    L = np.where(d < 14.0, 0.45 * k * d, 1.33 * k * np.maximum(d, 1e-9) ** 0.588)
    return np.minimum(L, cap_db)


def main():
    import tifffile
    if not os.path.exists(TCC):
        fetch_tcc()
    tcc = tifffile.imread(TCC).astype(float)
    tcc = np.where(tcc > 100, 0, tcc) / 100.0           # 0..1, nodata -> 0
    H, Wd = tcc.shape
    terr = np.load(os.path.join(ROOT, 'data_terrain_500.npy'))
    kx = 111320.0 * math.cos(math.radians(33.015)); ky = 110574.0

    def sample(x, y):
        lon = x / kx - 117.085; lat = y / ky + 33.015
        r = np.clip(((N - lat) / (N - S) * 500).astype(int), 0, 499)
        c = np.clip(((lon - W) / (E - W) * 500).astype(int), 0, 499)
        rt_ = np.clip(((N - lat) / (N - S) * H).astype(int), 0, H - 1)
        ct = np.clip(((lon - W) / (E - W) * Wd).astype(int), 0, Wd - 1)
        return terr[r, c] - R.Z_REF, tcc[rt_, ct]

    for site in ['Site-NW', 'Site-NE', 'Site-SW', 'Site-SE']:
        out = {}
        for rx in ['roof', 'street']:
            st, tx, xs, ys, pts, moved, indoor = R.build_rx_grid(site, rx)
            depth = np.zeros(len(pts))
            for i, (x, y, z) in enumerate(pts):
                L = math.hypot(x - tx[0], y - tx[1])
                n = max(2, int(L / STEP_M))
                t = (np.arange(n) + 0.5) / n
                px = tx[0] + (x - tx[0]) * t; py = tx[1] + (y - tx[1]) * t
                pz = tx[2] + (z - tx[2]) * t
                g, cov = sample(px, py)
                hag = pz - g
                inside = (hag > 0) & (hag < TREE_H_M)
                depth[i] = float((cov * inside).sum() * (L / n))
            out[f'depth_{rx}'] = depth.astype(np.float32)
            print(f"{site} {rx}: foliage depth on direct path: median {np.median(depth):.1f} m, "
                  f"p90 {np.percentile(depth, 90):.1f} m, max {depth.max():.0f} m; "
                  f"3.6 GHz loss median {np.median(weissberger_db(3.6, depth)):.1f} dB, p90 {np.percentile(weissberger_db(3.6, depth), 90):.1f} dB")
        np.savez_compressed(os.path.join(ROOT, f"data_veg_{site.split('-')[1].lower()}.npz"),
                            meta=json.dumps(dict(tree_h_m=TREE_H_M, step_m=STEP_M,
                                                 source='NLCD 2021 Tree Canopy Cover v2021-4 (USFS/MRLC)',
                                                 model='Weissberger MED, LOS component only, cap 20 dB')),
                            **out)


if __name__ == '__main__':
    main()
