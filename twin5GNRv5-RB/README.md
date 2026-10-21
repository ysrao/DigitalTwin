# twin5GNR — single-tower digital twin for 5G NR (0.7–28 GHz)

An open, batch-oriented digital twin for 5G NR network planning: ray-traced
coverage maps from real terrain and buildings, built with NVIDIA Sionna RT.
In the spirit of Sionna itself — a research instrument, not a real-time
product: no GPU required, no RAN stack, just the physics plus an honest
calibration report.

## What it does

For each of four rooftop 5G sites in a 5x5 km Rancho Bernardo (San Diego)
tile, it computes full-area ray-traced coverage maps in five 5G NR bands:

- **0.7 GHz** (low-band, n28) — ITU-R P.2040 materials clamped to their
  1 GHz boundary values (below the recommendation's defined range; documented)
- **2.5 GHz** (mid-band, n41), **3.6 GHz** (CBRS/C-band, n48/n77),
  **7.125 GHz** (FR3)
- **28 GHz** (mmWave, n257) — ground materials clamped to their 10 GHz
  boundary values; NLOS-dominated tile, treat as indicative; stored SNR and
  the viewer default use 200 MHz
- 2x2 km per site, 51x51 grid at 40 m (2,601 RX points), 6 m above surface
  (rooftop-CPE convention)
- Sionna RT 2.2.0, max_depth=3, LOS + specular + diffraction + edge
  diffraction, 1M samples/source, isotropic
- Scene: USGS 3DEP terrain + SanGIS/LiDAR building heights, ITU concrete /
  medium-dry-ground materials
- Post: path gain -> RSRP (47 dBm reference EIRP) -> SNR (NF 9 dB;
  bandwidth-selectable noise floor; default 10 MHz, 200 MHz at 28 GHz)
- Coverage is reported two ways: of all 2,601 cells (dead zones count as
  uncovered) and of served cells only (the viewer's headline). `ul_stats.json`
  carries both (`dl_cov90` / `dl_cov90_served`)

Plus an interactive browser viewer (`twin_viewer.html`): site switcher,
band switcher, EIRP slider (analytic, instant), bandwidth slider 5-100 MHz
(analytic noise floor, instant), click-to-probe RSRP/SNR/LOS readouts.

## Uplink

The viewer has a **Downlink / Uplink toggle**. The uplink layer
(`ul_post.py`, method in `UL_METHOD.md`) is **UL v3 — the link-budget
method**: it closes a per-direction budget, mirroring the user's own
LTE link-budget workbook (Qualcomm 80-W2559-5 Rev A), on top of the
same audited ray-traced path loss — all of it **modelled, not
measured**:

- UE transmit 23 dBm (workbook: 24) plus the workbook's handset
  antenna fit 6.7292·ln(f_MHz) − 51.61 dBi: −7.53 dBi at 0.7 GHz,
  +1.04 at 2.5 GHz, +3.49 at 3.6 GHz, clamped above 3.6 GHz (the
  LTE-era fit runs away at FR3/mmWave; the clamp is a stated
  adaptation). Base-station receive uses the same TR 38.901 sector
  pattern **by reciprocity** on the stored path angles plus the same
  12.04 dB 4x4 array gain — the workbook's 18 dBi eNB antenna lives
  inside that composition and is not added twice
- UL noise floor: −174 dBm/Hz + 10·log10(BW) + eNB noise figure 3 dB
  + target RoT 4 dB; capacity = Shannon bound, cap 5 bps/Hz
  (single-layer UE), × (1 − 22.34%) workbook UL overhead; DL keeps
  its cap-7 convention
- v1 (reciprocal shortcut, ratio ~1.0) and v2 (two tuned constants,
  ratio ~1.5) were both rejected — v3 adopts the workbook's budget
  terms instead of fitting the answer. Pooled DL/UL mean-capacity
  ratios are now **1.94 / 1.76 / 1.71 / 1.71** at 0.7 / 2.5 / 3.6 /
  7.125 GHz and **1.18** at 28 GHz (median 0.99 — at mmWave the
  workbook budget leaves the median UL cell level with DL; reported
  as computed). Still short of the field's 3–5: that ratio is a
  loaded multi-cell effect no single-tower budget contains, and the
  workbook reaches its own "uplink limiting" verdict (all its
  band/morphology columns) through MAPL comparisons at its borders.
  `UL_METHOD.md` has the full method, the workbook-vs-twin line
  items, workbook-style edge budgets with its margin stack, and the
  honest reading
- The UL−DL RSRP shift is now band-dependent (UE EIRP − 26.96 dBm):
  −11.49 dB at 0.7 GHz, −2.92 dB at 2.5 GHz, −0.47 dB at 3.6 GHz and
  above; UL coverage ≥ −90 dBm runs below DL in every site-band —
  by 8.5–10.4 points at 0.7 GHz, within ~2 points elsewhere
- FDD n28 reuses the DL path gains for its paired uplink band (stated
  approximation, no invented frequency correction)
- In UL mode the EIRP slider is disabled (base-station EIRP does not
  affect the uplink); the map, stats and click-probe show UL RSRP / UL SNR

Per-band UL grids live in `data_twin_tower_<site>_<band>_ul.npz`
(`ul_rsrp`, `ul_snr`, `ul_cap_mbps` at the viewer-default bandwidth).

**Comparison — ray-traced layer vs link-budget layer** (same workbook
budget terms; average user TP in Mbps, mean of the 4 sites). The RT
layer receives at a 6 m rooftop CPE; the budget layer
(`budget_check.py`, `budget_single.py`, details in `BUDGET_CHECK.md`
and `UL_METHOD.md`) is the twin's stochastic 3GPP UMa planner at a
1.5 m street-level handset, unserved counted as zero:

| Band | RT DL | RT UL | RT ratio | Budget (single tower) DL | UL | ratio | Budget (loaded) DL | UL | ratio |
|---|---|---|---|---|---|---|---|---|---|
| 0.7 GHz | 52.4 | 26.9 | 1.94 | 13.3 | 0.8 | 17.0 | 16.9 | 2.5 | 6.74 |
| 2.5 GHz | 43.8 | 24.9 | 1.76 | 13.9 | 2.4 | 5.9 | 14.3 | 3.6 | 4.00 |
| 3.6 GHz | 41.7 | 24.3 | 1.71 | 15.0 | 3.4 | 4.4 | 13.2 | 4.0 | 3.25 |
| 7.125 GHz | 38.4 | 22.4 | 1.71 | 17.8 | 4.4 | 4.0 | 11.3 | 3.6 | 3.13 |

The workbook marks the uplink limiting in every band/morphology and
field experience puts DL/UL at 3–5; the budget layer brackets that
range, the rooftop RT layer does not — the difference is the
operating point (receiver height and clutter), not the budget.
28 GHz is excluded: NLOS-dominated tile, UMa extrapolated (RT 1.18;
budget-layer 22.6, not credible).

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

# 1. fetch County building outlines + USGS 3DEP DEM -> sd_data/ (~1 min)
python fetch_sd_data.py
# 2. build the scene (terrain + buildings -> scene_v4/scene.xml), ~10 s
python build_scene_v4.py

# 3. run one tower + band, chunked for modest RAM (9 chunks x ~30 s on CPU)
for k in 0 1 2 3 4 5 6 7 8; do
  python single_tower_twin.py --site Site-NW --freq-ghz 3.6 --chunk $k
done
python single_tower_twin.py --site Site-NW --freq-ghz 3.6 --post   # stats + heatmap

# 4. uplink post (needs all sites x bands solver .npz) + build the viewer
python ul_post.py        # -> data_twin_tower_<site>_<band>_ul.npz
python build_viewer.py   # -> twin_viewer.html, open in any browser
```

`--site`: Site-NW, Site-NE, Site-SW, Site-SE. `--freq-ghz`: 0.7, 2.5, 3.6,
7.125 (default 3.6). `--bw-mhz` sets the noise floor for the stored SNR
(default 10); the viewer's bandwidth slider re-derives SNR analytically.
`--smoke` runs a 9-point pipeline test first (recommended on a new machine).

## Building heights

`sd_data/building_heights.json` (per-building LiDAR heights, USGS
CA_SanDiegoQL2_2014, 95th-pct AGL) is not produced by any script in this
package. `building_heights.py` resolves each footprint's height in order:

1. `sd_data/building_heights.json`, when present (used unchanged)
2. `data_bldh_500.npy`, the ~10 m height raster `planner_v4.py` saved from
   those LiDAR heights: median of the raster cells inside the footprint
   (centroid cell for footprints smaller than a cell)
3. 6.0 m default (planner_v4's existing default), for footprints the
   raster misses

Fallback heights are clipped to 2.5-60 m. `build_scene_v4.py` and
`planner_v4.py` both print how many heights came from each tier and record
it (`buildings_meta.json`, `data/summary.json`). Without the LiDAR file, a
fresh build takes 10,949 of 11,854 heights from the raster and 905 from the
default. Re-running the planner on those heights reproduces the shipped
raster within 0.5 m on 99.9% of building cells, with the same four sites.
A 300-cell ray-trace check (Site-NW, 3.6 GHz) differs from the shipped grid
by the same amount as two solver seeds on one scene (median 0.3 dB), so the
fallback is within the solver's own sampling noise.

## Files

| File | Role |
|---|---|
| `fetch_sd_data.py` | downloads County building outlines and the 3DEP DEM into `sd_data/` |
| `building_heights.py` | per-footprint heights: LiDAR JSON, then height raster, then 6 m default |
| `budget_common.py` | shared setup for the two budget checks (workbook terms, planner functions, street geometry) |
| `build_scene_v4.py` | DEM + building footprints -> Mitsuba scene XML/PLY |
| `single_tower_twin.py` | chunked full-area RT + RSRP/SNR post |
| `sionna_cal_v4.py` | 192-point independent calibration vs 3GPP UMa |
| `planner_v4.py` | statistical multi-site SINR/capacity layer |
| `ul_post.py` | uplink post: UL RSRP/SNR/capacity grids from the stored paths |
| `UL_METHOD.md` | uplink model, assumptions and results table |
| `build_viewer.py` | generates the standalone `twin_viewer.html` |
| `data_twin_tower_<site>_<band>.npz` | per-site-band DL grids: pl (isotropic path loss, dB), los, rsrp, snr, stored top-32 paths |
| `data_twin_tower_<site>_<band>_ul.npz` | per-site-band uplink grids: ul_rsrp, ul_snr, ul_cap_mbps |

## v5: street level, diffuse scattering, vegetation, user mix (2026-10-20)

v5 adds a second ray-traced receiver grid and a throughput model built on the
LTE link-budget workbook's terms, for single-tower **upgrade what-ifs** with a
mix of rooftop CPEs and handsets. The v4 grids and scripts above are unchanged.

```bash
python fetch_sd_data.py && python build_scene_v4.py   # scene (once)
./run_v5_all.sh            # 4 sites x 5 bands x {roof, street}; resumable, ~6 min per grid on 2 CPUs
python veg_layer.py        # foliage depth on the direct path (NLCD 2021 canopy)
python build_viewer_v5.py  # -> twin_viewer_v5.html (standalone) + twin_v5_data.json
python twin_model.py --site nw \
    --before band=f700,bw=10,eirp=47,tilt=4,ports=2 \
    --after  band=f3600,bw=40,eirp=47,tilt=4,ports=4,tdd=DDDSU
```

| File | Role |
|---|---|
| `rt_v5.py` | ray tracing with diffuse scattering (S = 0.3 buildings, 0.2 ground); grids `roof` (6 m above roof) and `street` (1.5 m, points moved out of building footprints); antenna applied to all paths for tilts 0–10° |
| `run_v5_all.sh` | runs every grid, one chunk per process, resumable |
| `veg_layer.py` | NLCD 2021 tree canopy cover -> foliage depth on the direct TX-RX line (8 m trees) |
| `twin_model.js` / `twin_model.py` | identical link-budget and throughput model (browser / Python); cross-checked to 2e-16 |
| `build_viewer_v5.py`, `viewer_v5_template.html` | upgrade-planner viewer |
| `data_v5_<site>_<band>_<roof|street>.npz` | per-tilt DL received power at the 47 dBm reference, LOS and diffuse power, top-32 paths |
| `data_veg_<site>.npz`, `data_tcc_2021.tif` | foliage depth per cell; canopy raster (public domain) |

**Device classes** (mobility decides grid, extra loss and MIMO rank): rooftop
CPE (fixed, roof grid, 9 dBi, 26 dBm, up to 4 layers); indoor handset (nomadic,
street grid at the facade + TR 38.901 O2I, low- or high-loss); outdoor handset
(pedestrian); in-vehicle handset (car body 9 dB, at most 2 layers). Handset
antenna: the workbook fit capped at 2.14 dBi.

**Throughput**: workbook 36.942 CQI curves (DL, UL), 2x2 and Vendor-B 4x4 MIMO
gain, workbook DL overhead (normal CP, 3 PDCCH symbols, 2/4 ports) and UL 22.34%,
TDD patterns DDDSU / DDDDDDDSUU / DDSUU (special slot 10:2:2), per-user =
single-user x (1 - load). DL needs SINR >= -8.7 dB, UL needs a 1-RB PUSCH at
-0.7 dB. **RSRP is true per-RE RSRP** (v4's "RSRP" was total channel power).

**Assumptions to calibrate**: scattering coefficients (street-level power is
about 90% diffuse in the median cell), CPE gain and power, foliage model. The
Vodacom KPI file (2 sites, daily PRB / throughput) supports a load sanity check
but not path-loss calibration, which needs geolocated RSRP (MDT / drive test)
or per-cell TA and RSRP/CQI distributions.

## Data licenses

See `DATA_LICENSES.md`. USGS 3DEP elevation is public domain; OpenStreetMap
data is ODbL; SanGIS building footprints require a license check before
redistribution.

## License

MIT (proposed) - see `LICENSE`. Final license choice is the author's.

## Original package

`twin5GNR_v5.zip` is the original Claude v5 package exactly as delivered (28 MB; NW/NE fully traced, SW/SE partial). The surrounding tree is the completed version: all four sites traced, the DL efficiency-cap fix applied, and the viewer rebuilt (Current/Change labels).
