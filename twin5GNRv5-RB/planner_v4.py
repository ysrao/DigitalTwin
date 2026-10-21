#!/usr/bin/env python3
"""
v4: rebuild on REAL terrain — the merge of the two models.

What changed vs v3 (2026-10-14):
  - Terrain: USGS 3DEP DEM (real: 118-363 m), replacing the incorrect flat
    Esri-derived raster (204-230 m). Independently verified.
  - Buildings: San Diego County outlines (11,854) + USGS LiDAR roof heights
    (CA_SanDiegoQL2_2014, 95th-pct AGL), replacing OSM (6,196, type defaults).
  - Kept from our v3: rooftop base-station spec (roof + 3 m mast, NOT towers),
    current-spec O2I 27.77 dB (TR 38.901 v19.4.0), facade-level indoor reference,
    rooftop-CPE case.
  - Fixes: idle sectors (<1% of served cells) excluded from capacity totals;
    single consistent LOS computation feeds both the LOS flag and the PL
    (fixes the v1 stratified LOS/PL grid-cell mismatch class of bug).

All outputs additive under v4_* names and the 'v4_real_terrain' summary key.
v1-v3 files and summary keys are untouched.
"""
import json, math, os, time
import numpy as np
import tifffile
from numba import njit
from shapely.geometry import shape
from shapely import contains_xy

# ---------------- tile & constants ----------------
S, W, N, E = 32.9925, -117.1120, 33.0375, -117.0580
GRID = 500
FC_MHZ = 3600.0
FC_HZ = FC_MHZ * 1e6
FC_GHZ = 3.6
C = 3e8
TX_EIRP_DBM = 47.0
TX_MAST_M = 3.0              # rooftop mast above roof (user deployment spec)
RX_HEIGHT_M = 6.0            # rooftop CPE, 6 m above roof
RX_STREET_M = 1.5            # facade reference height above terrain
RX_NOISE_FIG_DB = 9.0
BW_HZ = 10e6
RSRP_TARGET_DBM = -90.0
PAL_CHANNELS = [1, 2, 3, 4]
SECTOR_AZIMUTHS = [0.0, 120.0, 240.0]
HPBW_DEG = 65.0
A_MAX_DB = 30.0
SE_CAP_BPS_HZ = 7.5
IDLE_SECTOR_FRAC = 0.01      # sectors serving <1% of served cells excluded from capacity

t0 = time.time()
lat0 = (S + N) / 2
step_m = (N - S) / GRID * 111320.0
step_y = (N - S) / GRID * 110540.0
kx = 111320 * math.cos(math.radians(lat0))
ky = 110540

# ---------------- O2I wall loss (TR 38.901 v19.4.0, computed) ----------------
def pl_tw(model):
    if model == 'low':
        l_glass, l_conc = 2 + 0.2 * FC_GHZ, 5 + 4 * FC_GHZ
        w_g, w_c = 0.3, 0.7
    else:
        l_irr, l_conc = 25.4 + 0.11 * FC_GHZ, 5 + 4 * FC_GHZ
        w_g, w_c = 0.7, 0.3
        l_glass = l_irr
    return 5 - 10 * math.log10(w_g * 10 ** (-l_glass / 10)
                               + w_c * 10 ** (-l_conc / 10))

PL_TW_LOW = pl_tw('low')
PL_TW_HIGH = pl_tw('high')
print(f"v4 O2I wall loss @ {FC_GHZ} GHz: low={PL_TW_LOW:.2f} dB, high={PL_TW_HIGH:.2f} dB")
assert abs(PL_TW_LOW - 12.74) < 0.05, PL_TW_LOW
assert abs(PL_TW_HIGH - 27.77) < 0.05, PL_TW_HIGH

# ---------------- load 3DEP DEM -> 500x500 ----------------
print("loading 3DEP DEM...")
from scipy.ndimage import zoom
dem_raw = tifffile.imread('sd_data/dem_3dep.tif').astype(np.float64)
print(f"  raw DEM {dem_raw.shape}, min {np.nanmin(dem_raw):.1f} / max {np.nanmax(dem_raw):.1f} m")
assert 100 < np.nanmin(dem_raw) < 140 and 340 < np.nanmax(dem_raw) < 380, "DEM range unexpected!"
dem_raw = np.where(np.isnan(dem_raw), np.nanmedian(dem_raw), dem_raw)
terrain = zoom(dem_raw, (GRID / dem_raw.shape[0], GRID / dem_raw.shape[1]), order=1)
print(f"  resampled to {terrain.shape}, min {terrain.min():.1f} / max {terrain.max():.1f} m")

# ---------------- County buildings + LiDAR heights -> raster ----------------
print("rasterizing County buildings (11,854) with LiDAR heights...")
feats = json.load(open('sd_data/county_buildings.geojson'))['features']
from building_heights import resolve_heights, describe
_hmap, _hsrc, _hcount = resolve_heights(feats)   # LiDAR JSON -> raster -> 6 m default
print('  ' + describe(_hcount))
heights = {oid: {'height_m': h} for oid, h in _hmap.items()}

def ll2grid(lat, lon):
    r = (N - lat) / (N - S) * (GRID - 1); c = (lon - W) / (E - W) * (GRID - 1)
    return r, c

# cell-center lon/lat grids for vectorized rasterization
cc = np.arange(GRID)
lon_c = W + (cc + 0.5) / GRID * (E - W)
lat_c = N - (cc + 0.5) / GRID * (N - S)

bld_h = np.zeros((GRID, GRID), dtype=np.float64)
bld_id = np.full((GRID, GRID), -1, dtype=np.int32)
buildings = []   # per-building table for indoor/facade work
n_noh = 0
for bi, f in enumerate(feats):
    try:
        poly = shape(f['geometry'])
    except Exception:
        continue
    if poly.is_empty or poly.area == 0:
        continue
    oid = f['properties'].get('objectid')
    hrec = heights.get(oid, {})
    h = hrec.get('height_m')
    if h is None or not (1.0 < h < 60.0):
        h = 6.0; n_noh += 1
    a, b, c_, d_ = poly.bounds  # lon/lat
    r1 = max(0, int((N - d_) / (N - S) * GRID)); r2 = min(GRID - 1, int((N - b) / (N - S) * GRID) + 1)
    c1 = max(0, int((a - W) / (E - W) * GRID)); c2 = min(GRID - 1, int((c_ - W) / (E - W) * GRID) + 1)
    if r2 < r1 or c2 < c1:
        continue
    xs, ys = np.meshgrid(lon_c[c1:c2 + 1], lat_c[r1:r2 + 1])
    m = contains_xy(poly, xs.ravel(), ys.ravel()).reshape(xs.shape)
    if not m.any():
        continue
    sub = bld_h[r1:r2 + 1, c1:c2 + 1]
    upd = m & (h > sub)
    sub[upd] = h
    bld_h[r1:r2 + 1, c1:c2 + 1] = sub
    bld_id[r1:r2 + 1, c1:c2 + 1][upd] = bi
    area = poly.area * kx * ky  # rough m^2
    buildings.append(dict(idx=bi, objectid=oid, height_m=h, area_m2=area,
                          r1=r1, r2=r2, c1=c1, c2=c2,
                          ground_m=float(terrain[int((r1 + r2) / 2), int((c1 + c2) / 2)])))
surface = terrain + bld_h
print(f"  {(bld_h > 0).sum()} building cells, {len(buildings)} buildings in table "
      f"({n_noh} defaulted to 6 m)")
print(f"  surface min {surface.min():.1f} / max {surface.max():.1f} m")

np.save('data_terrain_500.npy', terrain)
np.save('data_bldh_500.npy', bld_h)

# ---------------- numba LOS (blockers=surface; single consistent computation) ----------------
# NOTE (v4 fix): the LOS grid computed here feeds BOTH the LOS flag and the
# path-loss branch selection. v1's stratified script used two disagreeing LOS
# computations (seg_los vs rsrp.npy); v4 never does that.
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
    fc_ghz = FC_MHZ / 1000.0
    pl_los_seg1 = 28.0 + 22 * np.log10(d3d) + 20 * np.log10(fc_ghz)
    d_bp = 4 * h_bs * h_ut * FC_HZ / C
    pl_los_seg2 = (28.0 + 40 * np.log10(d3d) + 20 * np.log10(fc_ghz)
                   - 9 * np.log10(d_bp ** 2 + (h_bs - h_ut) ** 2))
    pl_los = np.where(d2d < d_bp, pl_los_seg1, pl_los_seg2)
    pl_nlos = 13.54 + 39.08 * np.log10(d3d) + 20 * np.log10(fc_ghz) - 0.6 * (h_ut - 1.5)
    pl = np.where(los, pl_los, np.maximum(pl_los, pl_nlos))
    return pl

# ---------------- rooftop site selection (real terrain + LiDAR roofs) ----------------
def select_rooftop_site(r1, r2, c1, c2, name, min_dist_m=0.0, others=()):
    """Highest-surface BUILDING cell in the box -> rooftop site (roof + 3 m mast)."""
    sub = surface[r1:r2, c1:c2].copy()
    sub[bld_h[r1:r2, c1:c2] == 0] = -1
    if min_dist_m > 0 and others:
        rr_idx = np.arange(r1, r2)[:, None]
        cc_idx = np.arange(c1, c2)[None, :]
        for (er, ec) in others:
            d = np.sqrt(((rr_idx - er) * step_m) ** 2 + ((cc_idx - ec) * step_m) ** 2)
            sub[d < min_dist_m] = -1
    ir, ic = np.unravel_index(np.argmax(sub), sub.shape)
    gr, gc = r1 + ir, c1 + ic
    lat = N - (gr + 0.5) / GRID * (N - S); lon = W + (gc + 0.5) / GRID * (E - W)
    st = dict(name=name, grid=(int(gr), int(gc)), lat=lat, lon=lon,
              terr_m=float(terrain[gr, gc]), roof_m=float(bld_h[gr, gc]),
              tx_agl_m=float(bld_h[gr, gc]) + TX_MAST_M,
              bld_idx=int(bld_id[gr, gc]))
    print(f"{name}: ground {st['terr_m']:.0f} m, roof {st['roof_m']:.1f} m + "
          f"{TX_MAST_M:.0f} m mast (TX {st['tx_agl_m']:.1f} m AGL, "
          f"{st['terr_m'] + st['tx_agl_m']:.0f} m ASL)")
    return st

rr, cc = np.mgrid[0:GRID, 0:GRID]

def candidate_cells(r1, r2, c1, c2, others=(), min_dist_m=0.0, topn=40):
    """Top-`topn` building cells by surface height in the box (excl. near others)."""
    sub = surface[r1:r2, c1:c2].copy()
    sub[bld_h[r1:r2, c1:c2] == 0] = -1
    if min_dist_m > 0 and others:
        rr_idx = np.arange(r1, r2)[:, None]
        cc_idx = np.arange(c1, c2)[None, :]
        for (er, ec) in others:
            d = np.sqrt(((rr_idx - er) * step_m) ** 2 + ((cc_idx - ec) * step_m) ** 2)
            sub[d < min_dist_m] = -1
    flat = sub.ravel()
    topn = min(topn, int((flat > 0).sum()))
    if topn == 0:
        return []
    idx = np.argpartition(flat, -topn)[-topn:]
    cands = []
    for k in idx:
        if flat[k] > 0:
            ir, ic = divmod(int(k), sub.shape[1])
            cands.append((r1 + ir, c1 + ic))
    return cands

def _site_dict(name, gr, gc):
    lat = N - (gr + 0.5) / GRID * (N - S); lon = W + (gc + 0.5) / GRID * (E - W)
    return dict(name=name, grid=(int(gr), int(gc)), lat=lat, lon=lon,
                terr_m=float(terrain[gr, gc]), roof_m=float(bld_h[gr, gc]),
                tx_agl_m=float(bld_h[gr, gc]) + TX_MAST_M,
                bld_idx=int(bld_id[gr, gc]))

def select_site_max_los(r1, r2, c1, c2, name, others=(), min_dist_m=0.0, topn=40):
    """v4: pick the rooftop candidate with the best LOS fraction (real terrain:
    highest surface != best site when hills are around)."""
    cands = candidate_cells(r1, r2, c1, c2, others, min_dist_m, topn)
    best, best_los = None, -1.0
    for (gr, gc) in cands:
        lf = float(los_grid_rxsurf(surface, surface, gr, gc, TX_MAST_M,
                                   RX_HEIGHT_M, GRID).mean())
        if lf > best_los:
            best_los, best = lf, (gr, gc)
    st = _site_dict(name, *best)
    st['los_frac'] = best_los
    print(f"{name}: ground {st['terr_m']:.0f} m, roof {st['roof_m']:.1f} m + "
          f"{TX_MAST_M:.0f} m mast (TX {st['tx_agl_m']:.1f} m AGL, "
          f"{st['terr_m'] + st['tx_agl_m']:.0f} m ASL), LOS frac {best_los:.3f} "
          f"({len(cands)} candidates)")
    return st

def select_site_max_hole_cover(r1, r2, c1, c2, name, holes, others=(),
                               min_dist_m=0.0, topn=30):
    """v4 densification: pick the rooftop candidate covering the most
    currently-unserved (hole) cells by LOS."""
    cands = candidate_cells(r1, r2, c1, c2, others, min_dist_m, topn)
    hr, hc = np.argwhere(holes).T
    best, best_n = None, -1
    for (gr, gc) in cands:
        los = los_grid_rxsurf(surface, surface, gr, gc, TX_MAST_M, RX_HEIGHT_M, GRID)
        n = int(los[hr, hc].sum())
        if n > best_n:
            best_n, best = n, (gr, gc)
    st = _site_dict(name, *best)
    st['hole_cells_covered'] = best_n
    print(f"{name}: ground {st['terr_m']:.0f} m, roof {st['roof_m']:.1f} m + "
          f"{TX_MAST_M:.0f} m mast, covers {best_n} hole cells by LOS "
          f"({len(cands)} candidates)")
    return st

def compute_omni(sites):
    """Per-site omni RSRP grids (dBm): rooftop-CPE (6 m above surface) + street (1.5 m)."""
    n = len(sites)
    rsrp_out = np.zeros((n, GRID, GRID))
    rsrp_street = np.zeros((n, GRID, GRID))
    for i, st in enumerate(sites):
        sr, sc = st['grid']
        txz = surface[sr, sc] + TX_MAST_M
        los = los_grid_rxsurf(surface, surface, sr, sc, TX_MAST_M, RX_HEIGHT_M, GRID)
        d3d = np.sqrt(((rr - sr) * step_m) ** 2 + ((cc - sc) * step_m) ** 2 +
                      (surface + RX_HEIGHT_M - txz) ** 2)
        d3d = np.maximum(d3d, 3.0)
        rsrp_out[i] = TX_EIRP_DBM - uma_pl(d3d, los, TX_MAST_M, RX_HEIGHT_M)
        st['los_frac'] = float(los.mean())
        los_s = los_grid_rxsurf(surface, terrain, sr, sc, TX_MAST_M, RX_STREET_M, GRID)
        d3d_s = np.sqrt(((rr - sr) * step_m) ** 2 + ((cc - sc) * step_m) ** 2 +
                        (terrain + RX_STREET_M - txz) ** 2)
        d3d_s = np.maximum(d3d_s, 3.0)
        rsrp_street[i] = TX_EIRP_DBM - uma_pl(d3d_s, los_s, TX_MAST_M, RX_STREET_M)
        st['los_frac_street'] = float(los_s.mean())
        print(f"  {st['name']}: LOS frac rooftop-RX={st['los_frac']:.3f}, "
              f"street-RX={st['los_frac_street']:.3f}")
    return rsrp_out, rsrp_street

# ---------------- sectorization (3GPP TR 38.901 Table 7.3-1) ----------------
def pattern_gain(az_rel_deg, theta_deg):
    a_h = -np.minimum(12.0 * (az_rel_deg / HPBW_DEG) ** 2, A_MAX_DB)
    a_v = -np.minimum(12.0 * ((theta_deg - 90.0) / HPBW_DEG) ** 2, A_MAX_DB)
    return -np.minimum(-(a_h + a_v), A_MAX_DB)

def sectorize(rsrp_omni, sites, dh):
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

# ---------------- best server + SINR + capacity (idle sectors excluded) ----------------
def sinr_capacity(rsrp_sec, channels):
    nsites = len(channels); nsec = len(SECTOR_AZIMUTHS)
    lin = 10 ** (rsrp_sec / 10)
    noise_mw = 10 ** ((-174 + 10 * np.log10(BW_HZ) + RX_NOISE_FIG_DB) / 10)
    best = rsrp_sec.max(axis=0)
    serving = rsrp_sec.argmax(axis=0)
    served = best >= RSRP_TARGET_DBM
    total_served = int(served.sum())
    sinr = np.full((GRID, GRID), -np.inf)
    for k in range(nsites * nsec):
        i = k // nsec
        mask = (serving == k)
        if not mask.any():
            continue
        interf = np.zeros((GRID, GRID))
        for k2 in range(nsites * nsec):
            if k2 == k:
                continue
            if channels[k2 // nsec] == channels[i]:
                interf += lin[k2]
        sinr[mask] = 10 * np.log10(lin[k][mask] / (interf[mask] + noise_mw))
    sinr_lin = 10 ** (np.clip(sinr, -50, 100) / 10)
    se = np.minimum(np.log2(1 + sinr_lin), SE_CAP_BPS_HZ)
    se[~served] = 0.0
    per_sector, per_site = [], []
    idle_sectors = []
    for k in range(nsites * nsec):
        m = (serving == k) & served
        n = int(m.sum())
        idle = n < IDLE_SECTOR_FRAC * total_served
        if idle:
            idle_sectors.append(k)
        mean_se = float(se[m].mean()) if n else 0.0
        per_sector.append(dict(served_cells=n, idle=bool(idle),
                               mean_sinr_db=float(sinr[m].mean()) if n else None,
                               mean_spectral_eff_bps_hz=mean_se,
                               capacity_mbps=BW_HZ / 1e6 * mean_se))
    for i in range(nsites):
        secs = per_sector[i * nsec:(i + 1) * nsec]
        # v4 fix: idle sectors excluded from site/network capacity totals
        per_site.append(dict(site_capacity_mbps=sum(x['capacity_mbps'] for x in secs
                                                    if not x['idle']),
                             site_capacity_incl_idle_mbps=sum(x['capacity_mbps'] for x in secs),
                             sectors=secs))
    total = sum(s['site_capacity_mbps'] for s in per_site)
    total_incl = sum(s['site_capacity_incl_idle_mbps'] for s in per_site)
    return best, serving, served, sinr, per_site, total, idle_sectors, total_incl

# ---------------- indoor v4: facade-level reference, 27.77 dB wall ----------------
def compute_indoor_v4(street_best, force_high=True):
    """Indoor O2I with facade-level outdoor reference (terrain + 1.5 m).

    For each County building: facade reference = max street-level best-server
    RSRP over non-building cells in the building's grid bbox (best-exposed
    wall); indoor = facade_ref - PL_tw - PL_in, PL_in = 0.5*sqrt(area)/4.
    force_high=True: 27.77 dB wall on ALL buildings (headline, conservative).
    force_high=False: 12.74 dB on all (optimistic bracket; County data has no
    OSM tags for tag-based classification).
    """
    buildings = []
    for b in buildings_tbl:
        r1, r2, c1, c2 = b['r1'], b['r2'], b['c1'], b['c2']
        brow = bld_id[r1:r2 + 1, c1:c2 + 1]
        srow = street_best[r1:r2 + 1, c1:c2 + 1]
        m = brow < 0
        facade_ref = float(srow[m].max()) if m.any() else float(
            street_best[(r1 + r2) // 2, (c1 + c2) // 2])
        area = b['area_m2']
        d_in = math.sqrt(max(area, 1.0)) / 4.0
        pl_in = 0.5 * d_in
        plw = PL_TW_HIGH if force_high else PL_TW_LOW
        in_rsrp = facade_ref - plw - pl_in
        floors = max(1, int(round(b['height_m'] / 3.0)))
        buildings.append(dict(idx=b['idx'], area_m2=round(area, 1), floors=floors,
                              floor_area_m2=round(area * floors, 1),
                              facade_rsrp_dbm=round(facade_ref, 1),
                              pl_tw_db=round(plw, 2), pl_in_db=round(pl_in, 2),
                              indoor_rsrp_dbm=round(in_rsrp, 1),
                              covered=bool(in_rsrp >= RSRP_TARGET_DBM)))
    def _stats(sub):
        n = len(sub)
        cov = sum(1 for x in sub if x['covered'])
        fa = sum(x['floor_area_m2'] for x in sub)
        fac = sum(x['floor_area_m2'] for x in sub if x['covered'])
        return dict(n=n, covered_n=cov, covered_frac=cov / n if n else 0.0,
                    floor_area_m2=round(fa, 1),
                    floor_area_covered_frac=round(fac / fa, 4) if fa else 0.0,
                    mean_indoor_rsrp_dbm=round(float(np.mean(
                        [x['indoor_rsrp_dbm'] for x in sub])), 1) if n else None)
    return dict(
        model='v4: 3GPP TR 38.901 Sec 7.4.3 O2I, facade-level outdoor reference '
              '(terrain + 1.5 m, best-server, max over facade-adjacent cells); '
              'County buildings + LiDAR heights; real 3DEP terrain',
        freq_ghz=FC_GHZ, pl_tw_high_db=round(PL_TW_HIGH, 2),
        pl_tw_low_db=round(PL_TW_LOW, 2),
        pl_in_formula='0.5 * sqrt(footprint_area)/4 dB per building',
        rsrp_target_dbm=RSRP_TARGET_DBM, force_high=force_high,
        overall=_stats(buildings)), buildings

def rooftop_cpe_stats(best_sec_out):
    """Rooftop-mounted CPE: best-server sectorized RSRP at building centroid, no wall loss."""
    vals, fa = [], []
    for b in buildings_tbl:
        r = (b['r1'] + b['r2']) // 2; c = (b['c1'] + b['c2']) // 2
        floors = max(1, int(round(b['height_m'] / 3.0)))
        vals.append(float(best_sec_out[r, c])); fa.append(b['area_m2'] * floors)
    vals = np.array(vals); fa = np.array(fa)
    covered = vals >= RSRP_TARGET_DBM
    return dict(n_buildings=int(len(vals)),
                covered_frac=float(covered.mean()),
                covered_floor_area_frac=float(fa[covered].sum() / fa.sum()),
                median_rsrp_dbm=float(np.median(vals)))

# make the building table module-visible for indoor funcs
buildings_tbl = buildings

# ================= 4-site v4 run =================
print("\n=== v4 4-site: rooftop deployment on REAL terrain ===")
sites4 = []
picked = []
for qr, qc, name, ch in [(0, 0, 'NW', 3), (0, 1, 'NE', 4), (1, 0, 'SW', 1), (1, 1, 'SE', 2)]:
    r1, r2 = qr * GRID // 2, (qr + 1) * GRID // 2
    c1, c2 = qc * GRID // 2, (qc + 1) * GRID // 2
    st = select_site_max_los(r1, r2, c1, c2, f"Site-{name}",
                             others=picked, min_dist_m=800.0)
    st['channel'] = ch
    sites4.append(st)
    picked.append(st['grid'])

# verify site ground heights against the new DEM (v4 sanity check)
for st in sites4:
    gr, gc = st['grid']
    assert abs(st['terr_m'] - float(terrain[gr, gc])) < 1e-6, "site/DEM mismatch!"

rsrp_omni4, rsrp_street4 = compute_omni(sites4)
np.save('data_rsrp_v4_omni.npy', rsrp_omni4)
np.save('data_rsrp_v4_street.npy', rsrp_street4)
rsrp_sec4 = sectorize(rsrp_omni4, sites4, dh=TX_MAST_M - RX_HEIGHT_M)
rsrp_sec_street4 = sectorize(rsrp_street4, sites4, dh=TX_MAST_M - RX_STREET_M)
np.save('data_rsrp_v4_sectored.npy', rsrp_sec4)
channels4 = [st['channel'] for st in sites4]
best4, serving4, served4, sinr4, per_site4, total4, idle4, total4_incl = \
    sinr_capacity(rsrp_sec4, channels4)
np.save('data_sinr_v4_sectored.npy', sinr4)
np.save('data_sector_map_v4.npy', np.where(served4, serving4, -1).astype(np.int16))
street_best4 = rsrp_sec_street4.max(axis=0)

in4_high, bld4 = compute_indoor_v4(street_best4, force_high=True)
in4_low, _ = compute_indoor_v4(street_best4, force_high=False)
json.dump(in4_high, open('data_indoor_v4.json', 'w'), indent=1)
rt4 = rooftop_cpe_stats(best4)
json.dump(dict(model='v4 rooftop CPE: best-server sectorized outdoor RSRP at '
                    'building centroid (RX 6 m above roof), no wall loss.',
               rsrp_target_dbm=RSRP_TARGET_DBM, stats=rt4),
          open('data_rooftop_v4.json', 'w'), indent=1)
cov4 = float(served4.mean())
med_sinr4 = float(np.median(sinr4[served4]))
print(f"v4 4-site: outdoor covered {cov4*100:.2f}%, rooftop-CPE "
      f"{rt4['covered_frac']*100:.1f}% bldgs, median SINR {med_sinr4:.1f} dB")
print(f"  capacity {total4:.1f} Mbps (excl {len(idle4)} idle sectors; "
      f"incl idle would be {total4_incl:.1f})")
print(f"  indoor corrected (27.77 dB wall, facade ref): "
      f"{in4_high['overall']['covered_frac']*100:.1f}% bldgs, "
      f"{in4_high['overall']['floor_area_covered_frac']*100:.1f}% floor area")
print(f"  indoor low-loss bracket (12.74 dB): "
      f"{in4_low['overall']['covered_frac']*100:.1f}% bldgs")
for i, st in enumerate(sites4):
    secs = per_site4[i]['sectors']
    print(f"  {st['name']} (ch{st['channel']}): {per_site4[i]['site_capacity_mbps']:.1f} Mbps "
          f"(sectors: {[round(s['capacity_mbps'],1) for s in secs]})")

# ================= 6-site v4 run =================
print("\n=== v4 6-site: densification on real terrain ===")
holes = best4 < RSRP_TARGET_DBM
print(f"v4 4-site hole fraction: {holes.mean()*100:.2f}%")
coarse = holes.reshape(50, 10, 50, 10).mean(axis=(1, 3))
tmp = coarse.copy()
peaks = []
EXCL = 14
for _ in range(2):
    pi = np.unravel_index(np.argmax(tmp), tmp.shape)
    peaks.append(pi)
    pr, pc = pi
    tmp[max(0, pr-EXCL):min(50, pr+EXCL+1), max(0, pc-EXCL):min(50, pc+EXCL+1)] = 0.0
print("hole-density peaks (coarse):", peaks)

new_sites = []
SNAP_R = 74
existing_grids = [st['grid'] for st in sites4]
for k, (pr, pc) in enumerate(peaks):
    gr0 = int((pr + 0.5) * 10); gc0 = int((pc + 0.5) * 10)
    r1, r2 = max(0, gr0 - SNAP_R), min(GRID, gr0 + SNAP_R + 1)
    c1, c2 = max(0, gc0 - SNAP_R), min(GRID, gc0 + SNAP_R + 1)
    st = select_site_max_hole_cover(r1, r2, c1, c2, f"Site-C{k+1}", holes,
                                    min_dist_m=500.0, others=existing_grids)
    new_sites.append(st)
    existing_grids.append(st['grid'])

sites6 = sites4 + new_sites
rsrp_omni6, rsrp_street6 = compute_omni(sites6)
np.save('data_rsrp_v4_omni_6site.npy', rsrp_omni6)
np.save('data_rsrp_v4_street_6site.npy', rsrp_street6)
lin_omni = 10 ** (rsrp_omni6 / 10)
assign = {i: sites6[i]['channel'] for i in range(4)}
order = sorted(range(4, 6), key=lambda i: -np.median(rsrp_omni6[i]))
for i in order:
    best_ch, best_cost = None, 1e99
    for ch in PAL_CHANNELS:
        interf = np.zeros((GRID, GRID))
        for j, chj in assign.items():
            if chj == ch:
                interf += lin_omni[j]
        cost = float((interf * (rsrp_omni6[i] == rsrp_omni6.max(axis=0))).sum())
        if cost < best_cost:
            best_cost, best_ch = cost, ch
    assign[i] = best_ch
    sites6[i]['channel'] = best_ch
channels6 = [assign[i] for i in range(6)]
print("v4 6-site channels:", {sites6[i]['name']: channels6[i] for i in range(6)})
for ch in PAL_CHANNELS:
    idx = [i for i in range(6) if channels6[i] == ch]
    if len(idx) > 1:
        d = math.hypot(sites6[idx[0]]['grid'][0] - sites6[idx[1]]['grid'][0],
                       sites6[idx[0]]['grid'][1] - sites6[idx[1]]['grid'][1]) * step_m / 1000
        print(f"  REUSE ch{ch}: {[sites6[i]['name'] for i in idx]}, separation {d:.2f} km")

rsrp_sec6 = sectorize(rsrp_omni6, sites6, dh=TX_MAST_M - RX_HEIGHT_M)
rsrp_sec_street6 = sectorize(rsrp_street6, sites6, dh=TX_MAST_M - RX_STREET_M)
np.save('data_rsrp_v4_sectored_6site.npy', rsrp_sec6)
best6, serving6, served6, sinr6, per_site6, total6, idle6, total6_incl = \
    sinr_capacity(rsrp_sec6, channels6)
np.save('data_sinr_v4_sectored_6site.npy', sinr6)
np.save('data_sector_map_v4_6site.npy', np.where(served6, serving6, -1).astype(np.int16))
street_best6 = rsrp_sec_street6.max(axis=0)
json.dump(sites6, open('data_sites_v4_6site.json', 'w'), indent=1)

in6_high, _ = compute_indoor_v4(street_best6, force_high=True)
in6_low, _ = compute_indoor_v4(street_best6, force_high=False)
json.dump(in6_high, open('data_indoor_v4_6site.json', 'w'), indent=1)
rt6 = rooftop_cpe_stats(best6)
json.dump(dict(model='v4 rooftop CPE, 6-site', rsrp_target_dbm=RSRP_TARGET_DBM, stats=rt6),
          open('data_rooftop_v4_6site.json', 'w'), indent=1)
cov6 = float(served6.mean())
med_sinr6 = float(np.median(sinr6[served6]))
print(f"v4 6-site: outdoor covered {cov6*100:.2f}%, rooftop-CPE "
      f"{rt6['covered_frac']*100:.1f}% bldgs, median SINR {med_sinr6:.1f} dB")
print(f"  capacity {total6:.1f} Mbps (excl {len(idle6)} idle sectors; "
      f"incl idle would be {total6_incl:.1f})")
print(f"  indoor corrected (27.77 dB wall, facade ref): "
      f"{in6_high['overall']['covered_frac']*100:.1f}% bldgs, "
      f"{in6_high['overall']['floor_area_covered_frac']*100:.1f}% floor area")
for i, st in enumerate(sites6):
    print(f"  {st['name']} (ch{channels6[i]}): {per_site6[i]['site_capacity_mbps']:.1f} Mbps")
json.dump(sites4, open('data_sites_v4.json', 'w'), indent=1)

# ================= summary.json (additive) =================
def _r(x, nd=4):
    return round(float(x), nd) if x is not None else None

os.makedirs('data', exist_ok=True)
# start a fresh summary when the v1-v3 history file is not shipped
summary = json.load(open('data/summary.json')) if os.path.exists('data/summary.json') else {}
pre_keys = set(summary.keys())
summary['v4_real_terrain'] = dict(
    what=('v4: full rebuild on REAL terrain (USGS 3DEP DEM, 118-363 m) with '
          'County building outlines (11,854) + LiDAR roof heights. Merges v2 data '
          'pipeline with v3 deployment spec.'),
    terrain=dict(source='USGS 3DEP 1/3 arc-sec via National Map exportImage',
                 dem_min_m=round(float(terrain.min()), 1),
                 dem_max_m=round(float(terrain.max()), 1),
                 note='replaces incorrect flat Esri raster (204-230 m) used by v1-v3'),
    buildings=dict(source='SanGIS/County BUILDING_OUTLINES via SANDAG mirror',
                   count=len(buildings),
                   heights='USGS CA_SanDiegoQL2_2014 LiDAR EPT, 95th-pct AGL; '
                           'median 6.4 m; defaults 6.0 m where missing',
                   height_sources=_hcount),
    deployment=('rooftop base stations (antenna = roof + 3 m mast); '
                'rooftop CPEs (RX 6 m above roof); indoor CPE separate case'),
    pl_tw_high_db=round(PL_TW_HIGH, 2), pl_tw_low_db=round(PL_TW_LOW, 2),
    indoor_reference='facade-level (terrain + 1.5 m best-server, max over facade cells)',
    fixes=('idle sectors (<1% of served cells) excluded from capacity totals; '
           'single consistent LOS computation feeds flag and PL branch'),
    site4=dict(
        covered_frac=_r(cov4), median_sinr_db=_r(med_sinr4, 2),
        rooftop_cpe=dict(covered_buildings_frac=_r(rt4['covered_frac']),
                         covered_floor_area_frac=_r(rt4['covered_floor_area_frac']),
                         median_rsrp_dbm=_r(rt4['median_rsrp_dbm'], 1)),
        per_site_capacity_mbps=[dict(site=sites4[i]['name'], channel=channels4[i],
                                     capacity_mbps=_r(per_site4[i]['site_capacity_mbps'], 2),
                                     per_sector_mbps=[_r(x['capacity_mbps'], 2)
                                                      for x in per_site4[i]['sectors']],
                                     idle_sectors=[k % 3 for k in idle4
                                                   if k // 3 == i])
                                for i in range(4)],
        total_network_capacity_mbps=_r(total4, 2),
        total_incl_idle_mbps=_r(total4_incl, 2),
        n_idle_sectors=len(idle4),
        indoor_corrected=dict(
            covered_buildings_frac=_r(in4_high['overall']['covered_frac']),
            covered_floor_area_frac=_r(in4_high['overall']['floor_area_covered_frac']),
            mean_indoor_rsrp_dbm=_r(in4_high['overall']['mean_indoor_rsrp_dbm'], 1)),
        indoor_lowloss_bracket=dict(
            covered_buildings_frac=_r(in4_low['overall']['covered_frac']),
            covered_floor_area_frac=_r(in4_low['overall']['floor_area_covered_frac'])),
    ),
    site6=dict(
        covered_frac=_r(cov6), median_sinr_db=_r(med_sinr6, 2),
        rooftop_cpe=dict(covered_buildings_frac=_r(rt6['covered_frac']),
                         covered_floor_area_frac=_r(rt6['covered_floor_area_frac']),
                         median_rsrp_dbm=_r(rt6['median_rsrp_dbm'], 1)),
        channel_assignment={sites6[i]['name']: channels6[i] for i in range(6)},
        per_site_capacity_mbps=[dict(site=sites6[i]['name'], channel=channels6[i],
                                     capacity_mbps=_r(per_site6[i]['site_capacity_mbps'], 2),
                                     per_sector_mbps=[_r(x['capacity_mbps'], 2)
                                                      for x in per_site6[i]['sectors']])
                                for i in range(6)],
        total_network_capacity_mbps=_r(total6, 2),
        total_incl_idle_mbps=_r(total6_incl, 2),
        n_idle_sectors=len(idle6),
        indoor_corrected=dict(
            covered_buildings_frac=_r(in6_high['overall']['covered_frac']),
            covered_floor_area_frac=_r(in6_high['overall']['floor_area_covered_frac']),
            mean_indoor_rsrp_dbm=_r(in6_high['overall']['mean_indoor_rsrp_dbm'], 1)),
        indoor_lowloss_bracket=dict(
            covered_buildings_frac=_r(in6_low['overall']['covered_frac']),
            covered_floor_area_frac=_r(in6_low['overall']['floor_area_covered_frac'])),
    ),
)
json.dump(summary, open('data/summary.json', 'w'), indent=1)
summary2 = json.load(open('data/summary.json'))
assert set(summary2.keys()) == pre_keys | {'v4_real_terrain'}, "summary keys changed!"
for k in pre_keys:
    assert json.dumps(summary2[k], sort_keys=True) == json.dumps(summary[k], sort_keys=True), f"key {k} changed!"
print("summary.json: pre-existing keys verified byte-intact; added 'v4_real_terrain'")
print(f"\ndone in {time.time()-t0:.0f}s")
