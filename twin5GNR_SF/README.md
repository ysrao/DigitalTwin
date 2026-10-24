# twin5GNR_SF — twin5GNR on the Vexcel San Francisco mesh

First twin5GNR run on a commercial high-resolution mesh instead of the usual
3DEP-derived geometry: the Vexcel 1 km² sample of downtown San Francisco
(Chinatown / Nob Hill; centre 37.801105, −122.404175), supplied free by Vexcel
via Sameer Lalwani for this demo.

- `SF_mesh_twin_report.html` — the report: configuration, results table and all
  20 heat maps (open `maps/` alongside it).
- `maps/` — RSRP and DL throughput maps, 5 bands (0.7 / 2.5 / 3.6 / 7.125 /
  28 GHz) × rooftop and street grids. Throughput maps use per-band scales.
- `SF_mesh_twin_results.zip` — the same package as a single download.
- `summary_sf.json` — the numbers behind the table.

Method in brief: L20 Cesium tiles converted to a Sionna scene (1,024 tiles,
~1.95M vertices, single ITU concrete material, S = 0.3); one tower at the tile
centre, 30 m AGL (assumption); Sionna RT 2.2.0 ray tracing; RSRP and DL
throughput derived with the twin5GNR v5 workbook model. Path gains are
**simulated**, throughput **modelled** — nothing here is measured.
Uplink (modelled, summary level): `UL_NOTES.md` and `UL_summary.json`.
