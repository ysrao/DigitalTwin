# twin5GNR — single-tower digital twin for 5G NR (0.7–7.125 GHz)

An open, batch-oriented digital twin for 5G NR network planning: ray-traced
coverage maps from real terrain and buildings, built with NVIDIA Sionna RT.
In the spirit of Sionna itself — a research instrument, not a real-time
product: no GPU required, no RAN stack, just the physics plus an honest
calibration report.

## What it does

For each of four rooftop 5G sites in a 5x5 km Rancho Bernardo (San Diego)
tile, it computes full-area ray-traced coverage maps in four 5G NR bands:

- **0.7 GHz** (low-band, n28) — ITU-R P.2040 materials clamped to their
  1 GHz boundary values (below the recommendation's defined range; documented)
- **2.5 GHz** (mid-band, n41), **3.6 GHz** (CBRS/C-band, n48/n77),
  **7.125 GHz** (FR3)
- 2x2 km per site, 51x51 grid at 40 m (2,601 RX points), 6 m above surface
  (rooftop-CPE convention)
- Sionna RT 2.2.0, max_depth=3, LOS + specular + diffraction + edge
  diffraction, 1M samples/source, isotropic
- Scene: USGS 3DEP terrain + SanGIS/LiDAR building heights, ITU concrete /
  medium-dry-ground materials
- Post: path gain -> RSRP (47 dBm reference EIRP) -> SNR (NF 9 dB;
  bandwidth-selectable noise floor)

Plus an interactive browser viewer (`twin_viewer.html`): site switcher,
band switcher, EIRP slider (analytic, instant), bandwidth slider 5-100 MHz
(analytic noise floor, instant), click-to-probe RSRP/SNR/LOS readouts.

## Uplink

The viewer has a **Downlink / Uplink toggle**. The uplink layer
(`ul_post.py`, method in `UL_METHOD.md`) reuses the same stored ray-traced
paths and mirrors the DL post-processing with only the link direction
changed — all of it **modelled, not measured**:

- UE transmit 23 dBm, omni 0 dBi; base-station receive uses the same
  TR 38.901 sector pattern **by reciprocity** on the stored path angles,
  plus the same 12.04 dB 4x4 array gain
- UL noise floor: −174 dBm/Hz + 10·log10(BW) + BS noise figure 4 dB +
  3 dB interference rise; capacity = Shannon bound, cap 7 bps/Hz (same
  convention as DL)
- FDD n28 reuses the DL path gains for its paired uplink band (stated
  approximation, no invented frequency correction)
- Because both directions share the same per-sector channel powers, UL
  RSRP = DL RSRP − 3.96 dB exactly on every served cell (verified on all
  20 site-bands); UL coverage ≥ −90 dBm runs 1–4 points below DL, and
  mean UL capacity is within ~1–2 Mbps of DL at FR1, ~15–20% below at
  28 GHz / 200 MHz
- In UL mode the EIRP slider is disabled (base-station EIRP does not
  affect the uplink); the map, stats and click-probe show UL RSRP / UL SNR

Per-band UL grids live in `data_twin_tower_<site>_<band>_ul.npz`
(`ul_rsrp`, `ul_snr`, `ul_cap_mbps` at the viewer-default bandwidth).

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

# 2. run one tower + band, chunked for modest RAM (9 chunks x ~30 s on CPU)
for k in 0 1 2 3 4 5 6 7 8; do
  python single_tower_twin.py --site Site-NW --freq-ghz 3.6 --chunk $k
done
python single_tower_twin.py --site Site-NW --freq-ghz 3.6 --post   # stats + heatmap

# 3. uplink post (needs all sites x bands solver .npz) + build the viewer
python ul_post.py        # -> data_twin_tower_<site>_<band>_ul.npz
python build_viewer.py   # -> twin_viewer.html, open in any browser
```

`--site`: Site-NW, Site-NE, Site-SW, Site-SE. `--freq-ghz`: 0.7, 2.5, 3.6,
7.125 (default 3.6). `--bw-mhz` sets the noise floor for the stored SNR
(default 10); the viewer's bandwidth slider re-derives SNR analytically.
`--smoke` runs a 9-point pipeline test first (recommended on a new machine).

## Files

| File | Role |
|---|---|
| `build_scene_v4.py` | DEM + building footprints -> Mitsuba scene XML/PLY |
| `single_tower_twin.py` | chunked full-area RT + RSRP/SNR post |
| `sionna_cal_v4.py` | 192-point independent calibration vs 3GPP UMa |
| `planner_v4.py` | statistical multi-site SINR/capacity layer |
| `ul_post.py` | uplink post: UL RSRP/SNR/capacity grids from the stored paths |
| `UL_METHOD.md` | uplink model, assumptions and results table |
| `build_viewer.py` | generates the standalone `twin_viewer.html` |
| `data_twin_tower_<site>.npz` | per-site grids: pl, los, rsrp, snr |
| `data_twin_tower_<site>_<band>_ul.npz` | per-site-band uplink grids: ul_rsrp, ul_snr, ul_cap_mbps |

## Data licenses

See `DATA_LICENSES.md`. USGS 3DEP elevation is public domain; OpenStreetMap
data is ODbL; SanGIS building footprints require a license check before
redistribution.

## License

MIT (proposed) - see `LICENSE`. Final license choice is the author's.
