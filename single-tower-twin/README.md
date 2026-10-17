# Single-tower digital twin (CBRS, 3.6 GHz)

An open, batch-oriented digital twin for CBRS network planning: ray-traced
coverage maps from real terrain and buildings, built with NVIDIA Sionna RT.
In the spirit of Sionna itself — a research instrument, not a real-time
product: no GPU required, no RAN stack, just the physics plus an honest
calibration report.

## What it does

For each of four rooftop CBRS sites in a 5x5 km Rancho Bernardo (San Diego)
tile, it computes a full-area ray-traced coverage map:

- 2x2 km per site, 51x51 grid at 40 m (2,601 RX points), 6 m above surface
  (rooftop-CPE convention)
- Sionna RT 2.2.0, max_depth=3, LOS + specular + diffraction + edge
  diffraction, 1M samples/source, 3.6 GHz, isotropic
- Scene: USGS 3DEP terrain + SanGIS/LiDAR building heights, ITU concrete /
  medium-dry-ground materials
- Post: path gain -> RSRP (47 dBm Cat-B EIRP) -> SNR (NF 9 dB, 10 MHz)

Plus an interactive browser viewer (`twin_viewer.html`): site switcher,
EIRP slider (analytic, instant), click-to-probe RSRP/SNR/LOS readouts.

## Provenance (read this first)

- Ray-tracing output only: path gain per grid cell. RSRP/SNR are derived
  with stated EIRP/noise assumptions, not measured.
- No-path cells (typically ~20%) are genuine dead zones at depth 3, not
  missing data. They are shown, not interpolated over.
- The twin was cross-checked against an independent 192-point Sionna
  calibration (`sionna_cal_v4.py`, `data_sionna_cal_v4.json`): same bias
  pattern (RT LOS slightly stronger than 3GPP UMa, RT NLOS much weaker).
- Single tower = no inter-cell interference. Multi-site SINR/capacity
  comes from the statistical planner (`planner_v4.py`), not from ray tracing.

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. build the scene (terrain + buildings -> scene.xml), ~minutes
python build_scene_v4.py

# 2. run one tower, chunked for modest RAM (9 chunks x ~30 s on CPU)
for k in 0 1 2 3 4 5 6 7 8; do
  python single_tower_twin.py --site Site-NW --chunk $k
done
python single_tower_twin.py --site Site-NW --post   # stats + heatmap

# 3. build the interactive viewer (needs all four sites' .npz)
python build_viewer.py   # -> twin_viewer.html, open in any browser
```

`--site` choices: Site-NW, Site-NE, Site-SW, Site-SE.
`--smoke` runs a 9-point pipeline test first (recommended on a new machine).

## Files

| File | Role |
|---|---|
| `build_scene_v4.py` | DEM + building footprints -> Mitsuba scene XML/PLY |
| `single_tower_twin.py` | chunked full-area RT + RSRP/SNR post |
| `sionna_cal_v4.py` | 192-point independent calibration vs 3GPP UMa |
| `planner_v4.py` | statistical multi-site SINR/capacity layer |
| `build_viewer.py` | generates the standalone `twin_viewer.html` |
| `data_twin_tower_<site>.npz` | per-site grids: pl, los, rsrp, snr |

## Data licenses

See `DATA_LICENSES.md`. USGS 3DEP elevation is public domain; OpenStreetMap
data is ODbL; SanGIS building footprints require a license check before
redistribution.

## License

MIT (proposed) - see `LICENSE`. Final license choice is the author's.
