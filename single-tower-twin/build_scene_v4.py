"""Sionna RT scene from San Diego County open data.

  terrain   : USGS 3DEP DEM (v4/sd_data/dem_3dep.tif), resampled to TERRAIN_RES m, ITU medium-dry ground
  buildings : SanGIS/County BUILDING_OUTLINES (11,854 in tile), extruded to LiDAR-measured height
              (USGS CA_SanDiegoQL2_2014), base set on the DEM, ITU concrete
  rx surface: terrain-following mesh at +HRX m (for RadioMapSolver), NOT part of the occluding scene

Local frame: x East, y North (m) about tile centre (33.015, -117.085); z = NAVD88 elevation - Z_REF.
"""
import json, os, sys
import numpy as np, tifffile, mapbox_earcut as earcut
from shapely.geometry import shape

S, W, N, E = 32.9925, -117.112, 33.0375, -117.058
LAT0, LON0 = 33.015, -117.085
Z_REF = 200.0
KX = 111320.0 * np.cos(np.radians(LAT0)); KY = 110574.0


def to_local(lon, lat):
    return (np.asarray(lon) - LON0) * KX, (np.asarray(lat) - LAT0) * KY


class DEM:
    def __init__(self, path="sd_data/dem_3dep.tif"):
        self.z = tifffile.imread(path).astype(np.float64)
        self.H, self.W = self.z.shape
        self.xmin, self.ymax = to_local(W, N); self.xmax, self.ymin = to_local(E, S)

    def __call__(self, x, y):   # bilinear, local metres -> NAVD88 m
        c = (np.asarray(x) - self.xmin) / (self.xmax - self.xmin) * (self.W - 1)
        r = (self.ymax - np.asarray(y)) / (self.ymax - self.ymin) * (self.H - 1)
        c = np.clip(c, 0, self.W - 1.001); r = np.clip(r, 0, self.H - 1.001)
        c0, r0 = c.astype(int), r.astype(int); fc, fr = c - c0, r - r0
        z = self.z
        return ((1 - fr) * ((1 - fc) * z[r0, c0] + fc * z[r0, c0 + 1]) +
                fr * ((1 - fc) * z[r0 + 1, c0] + fc * z[r0 + 1, c0 + 1]))


def write_ply(path, V, F):
    V = np.asarray(V, np.float32); F = np.asarray(F, np.int32)
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(V)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 f"element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n").encode())
        f.write(V.tobytes())
        rec = np.zeros(len(F), dtype=[("n", "u1"), ("i", "<i4", 3)]); rec["n"] = 3; rec["i"] = F
        f.write(rec.tobytes())


def grid_mesh(dem, res, dz=0.0):
    xs = np.arange(dem.xmin, dem.xmax + 1e-6, res); ys = np.arange(dem.ymin, dem.ymax + 1e-6, res)
    X, Y = np.meshgrid(xs, ys)
    Z = dem(X, Y) - Z_REF + dz
    V = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    nx = len(xs); idx = np.arange(len(V)).reshape(len(ys), nx)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    F = np.r_[np.c_[a, b, c], np.c_[a, c, d]]          # CCW seen from above -> normals +z
    return V, F


def signed_area(r):
    return 0.5 * np.sum(r[:, 0] * np.roll(r[:, 1], -1) - np.roll(r[:, 0], -1) * r[:, 1])


def build(out_dir="scene_v4", terrain_res=10.0, rx_res=20.0, hrx=1.5):
    os.makedirs(f"{out_dir}/meshes", exist_ok=True)
    dem = DEM()
    V, F = grid_mesh(dem, terrain_res); write_ply(f"{out_dir}/meshes/terrain.ply", V, F)
    Vr, Fr = grid_mesh(dem, rx_res, dz=hrx); write_ply(f"{out_dir}/meshes/rx_surface.ply", Vr, Fr)

    feats = json.load(open("sd_data/county_buildings.geojson"))["features"]
    hts = json.load(open("sd_data/building_heights.json"))["buildings"]
    med = float(np.median([h["height_m"] for h in hts if h["height_m"]]))
    BV, BF, meta, nv = [], [], [], 0
    for f, ht in zip(feats, hts):
        g = shape(f["geometry"])
        if g.geom_type != "Polygon" or g.area <= 0:
            continue
        lon, lat = np.array(g.exterior.coords).T
        x, y = to_local(lon, lat); ring = np.c_[x, y][:-1]
        if len(ring) < 3 or abs(signed_area(ring)) < 4.0:
            continue
        if signed_area(ring) < 0:
            ring = ring[::-1]
        h = ht["height_m"] if ht["height_m"] is not None else med
        h = float(np.clip(h, 2.5, 60.0))
        zg = dem(ring[:, 0], ring[:, 1])
        base = zg.min() - 0.5 - Z_REF            # embed slightly so no gap on slopes
        top = zg.max() + h - Z_REF if h > 0 else base + 3
        top = max(top, base + 2.5)
        n = len(ring)
        BV += [np.c_[ring, np.full(n, base)], np.c_[ring, np.full(n, top)]]
        i = np.arange(n); j = (i + 1) % n
        BF += [np.c_[nv + i, nv + j, nv + n + j], np.c_[nv + i, nv + n + j, nv + n + i]]
        tri = earcut.triangulate_float64(ring.astype(np.float64), np.array([n], np.uint32)).reshape(-1, 3)
        BF.append(nv + n + tri)
        area = abs(signed_area(ring))
        meta.append({"objectid": f["properties"].get("objectid"), "cx": float(ring[:, 0].mean()),
                     "cy": float(ring[:, 1].mean()), "ground_m": float(zg.mean()), "height_m": h,
                     "area_m2": area, "ring": ring.round(2).tolist()})
        nv += 2 * n
    write_ply(f"{out_dir}/meshes/buildings.ply", np.concatenate(BV), np.concatenate(BF))
    xml = """<scene version="2.1.0">
  <bsdf type="itu-radio-material" id="mat-concrete"><string name="type" value="concrete"/><float name="thickness" value="0.2"/></bsdf>
  <bsdf type="itu-radio-material" id="mat-ground"><string name="type" value="medium_dry_ground"/><float name="thickness" value="1.0"/></bsdf>
  <shape type="ply" id="obj-buildings"><string name="filename" value="meshes/buildings.ply"/><ref id="mat-concrete" name="bsdf"/></shape>
  <shape type="ply" id="obj-terrain"><string name="filename" value="meshes/terrain.ply"/><ref id="mat-ground" name="bsdf"/></shape>
</scene>"""
    open(f"{out_dir}/scene.xml", "w").write(xml)
    json.dump({"z_ref": Z_REF, "lat0": LAT0, "lon0": LON0, "median_h": med, "buildings": meta},
              open(f"{out_dir}/buildings_meta.json", "w"))
    print(f"terrain tris {len(F)}, rx-surface tris {len(Fr)}, buildings {len(meta)} (median h {med:.1f} m)")


if __name__ == "__main__":
    build()
