"""Building heights for the Rancho Bernardo tile, with fallbacks.

Lookup order per footprint (keyed by County objectid):
  1. sd_data/building_heights.json  - per-building LiDAR heights (USGS
     CA_SanDiegoQL2_2014, 95th-pct AGL). Used when present, exactly as before.
  2. data_bldh_500.npy              - the 500x500 (~10 m) building-height
     raster planner_v4 saved from those same LiDAR heights; the footprint
     gets the median of the raster cells whose centres fall inside it, or
     the cell under its centroid for footprints smaller than a cell.
  3. DEFAULT_HEIGHT_M (6.0 m)       - planner_v4's existing default for
     buildings with no LiDAR height (two-storey residential).

Fallback heights are clipped to [MIN_HEIGHT_M, MAX_HEIGHT_M]; LiDAR values pass through. The source of each
height is returned so callers can report how many came from each tier.
"""
import json, os
import numpy as np
from shapely import contains_xy

S, W, N, E = 32.9925, -117.112, 33.0375, -117.058
GRID = 500
DEFAULT_HEIGHT_M = 6.0
MIN_HEIGHT_M, MAX_HEIGHT_M = 2.5, 60.0
LIDAR_JSON = os.path.join('sd_data', 'building_heights.json')
RASTER_NPY = 'data_bldh_500.npy'


def _raster_height(poly, rast):
    a, b, c, d = poly.bounds                     # lon/lat bounds
    r1 = max(0, int((N - d) / (N - S) * GRID)); r2 = min(GRID - 1, int((N - b) / (N - S) * GRID) + 1)
    c1 = max(0, int((a - W) / (E - W) * GRID)); c2 = min(GRID - 1, int((c - W) / (E - W) * GRID) + 1)
    if r2 >= r1 and c2 >= c1:
        lon = W + (np.arange(c1, c2 + 1) + 0.5) / GRID * (E - W)
        lat = N - (np.arange(r1, r2 + 1) + 0.5) / GRID * (N - S)
        X, Y = np.meshgrid(lon, lat)
        m = contains_xy(poly, X.ravel(), Y.ravel()).reshape(X.shape)
        vals = rast[r1:r2 + 1, c1:c2 + 1][m]
        vals = vals[vals > 0]
        if vals.size:
            return float(np.median(vals))
    pt = poly.representative_point()
    r = int(np.clip((N - pt.y) / (N - S) * GRID, 0, GRID - 1))
    cc = int(np.clip((pt.x - W) / (E - W) * GRID, 0, GRID - 1))
    v = float(rast[r, cc])
    return v if v > 0 else None


def resolve_heights(features, root='.'):
    """features: GeoJSON features (County outlines). Returns
    (heights dict objectid -> metres, source dict objectid -> tier, counts)."""
    from shapely.geometry import shape
    lidar = {}
    p = os.path.join(root, LIDAR_JSON)
    if os.path.exists(p):
        lidar = {b['objectid']: b.get('height_m') for b in json.load(open(p))['buildings']}
    rast_p = os.path.join(root, RASTER_NPY)
    rast = np.load(rast_p) if os.path.exists(rast_p) else None

    heights, source = {}, {}
    counts = {'lidar': 0, 'raster': 0, 'default': 0}
    for f in features:
        oid = f['properties'].get('objectid')
        h, tier = lidar.get(oid), 'lidar'
        if h is None or not (1.0 < h < MAX_HEIGHT_M):
            h, tier = None, None
            if rast is not None:
                try:
                    g = shape(f['geometry'])
                    if not g.is_empty and g.area > 0:
                        h = _raster_height(g, rast)
                except Exception:
                    h = None
                tier = 'raster' if h is not None else None
            if h is None:
                h, tier = DEFAULT_HEIGHT_M, 'default'
        # LiDAR values pass through unchanged (callers apply their own limits,
        # as before); fallback values are clipped to a physical range.
        heights[oid] = float(h) if tier == 'lidar' else float(np.clip(h, MIN_HEIGHT_M, MAX_HEIGHT_M))
        source[oid] = tier
        counts[tier] += 1
    return heights, source, counts


def describe(counts):
    n = sum(counts.values())
    return (f"building heights: {counts['lidar']} LiDAR, {counts['raster']} from "
            f"{RASTER_NPY}, {counts['default']} default {DEFAULT_HEIGHT_M:g} m (of {n})")
