"""Download San Diego County building outlines and USGS 3DEP terrain for the 5x5 km Rancho Bernardo tile.

  sd_data/county_buildings.geojson : SanGIS / County BUILDING_OUTLINES (SANDAG mirror), all footprints in the tile
  sd_data/dem_3dep.tif             : USGS 3DEP bare-earth DEM, float32, 2700 x 2250 px (~2 m) in EPSG:4326
The County server returns fewer features per page than its advertised maxRecordCount, so features are fetched
by object ID in chunks of 500.
"""
import json, os, urllib.request, urllib.parse

S, W, N, E = 32.9925, -117.112, 33.0375, -117.058
BLD = "https://geo.sandag.org/server/rest/services/Hosted/Building_Outlines/FeatureServer/0/query"
DEM = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"


def post(url, q, timeout=180):
    return json.load(urllib.request.urlopen(url, data=urllib.parse.urlencode(q).encode(), timeout=timeout))


def buildings(out="sd_data/county_buildings.geojson"):
    ids = post(BLD, dict(where="1=1", geometry=f"{W},{S},{E},{N}", geometryType="esriGeometryEnvelope", inSR=4326,
                         spatialRel="esriSpatialRelIntersects", returnIdsOnly="true", f="json"))["objectIds"]
    feats = []
    for i in range(0, len(ids), 500):
        feats += post(BLD, dict(objectIds=",".join(map(str, ids[i:i + 500])), outFields="objectid,bldgid,area",
                                outSR=4326, f="geojson"))["features"]
    json.dump({"type": "FeatureCollection", "features": feats}, open(out, "w"))
    print(f"buildings: {len(feats)} -> {out}")


def dem(out="sd_data/dem_3dep.tif"):
    q = dict(bbox=f"{W},{S},{E},{N}", bboxSR=4326, imageSR=4326, size="2700,2250", format="tiff", pixelType="F32",
             noDataInterpretation="esriNoDataMatchAny", interpolation="RSP_BilinearInterpolation", f="json")
    d = json.load(urllib.request.urlopen(DEM + "?" + urllib.parse.urlencode(q), timeout=300))
    urllib.request.urlretrieve(d["href"], out)
    print(f"DEM {d['width']}x{d['height']} -> {out}")


if __name__ == "__main__":
    os.makedirs("sd_data", exist_ok=True)
    buildings(); dem()
