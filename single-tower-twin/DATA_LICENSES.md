# Data licenses

The code in this package is proposed MIT (see LICENSE). The *input data*
used for the Rancho Bernardo demo tile carries its own terms:

| Dataset | Source | License | Notes |
|---|---|---|---|
| Elevation (DEM) | USGS 3DEP | Public domain (US Government work) | free to redistribute |
| Building heights | USGS CA_SanDiegoQL2_2014 LiDAR | Public domain (US Government work) | free to redistribute |
| Building footprints | SanGIS / County of San Diego | **Check before redistributing** | verify SanGIS terms; replace with OpenStreetMap extracts if unsure |
| (alternative) Buildings | OpenStreetMap | ODbL 1.0 | attribution required; share-alike on the database |

To run the twin on your own area, replace `v4/sd_data/` with local DEM +
footprint files and re-run `build_scene_v4.py`. The pipeline only assumes a
GeoTIFF DEM and a GeoJSON of building polygons with heights.
