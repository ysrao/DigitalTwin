# Input Formats — Carrier Topology & RAN Configuration

The twin runs on synthetic hexagonal topologies out of the box. To study
**your** network, feed it your real topology: the formats below are
standard, documented, and inputtable — plain CSV/JSON, no proprietary
tooling.

## 1. Topology CSV (`topology.csv`)

One row per cell. Each macro sector is its own row (sectors of one site
share a `site_id`). Coordinates are **WGS84 lat/lon** — the carrier
standard — projected to local meters inside the twin.

| Column | Required | Description |
|---|---|---|
| `cell_id` | yes | Unique cell identifier (string or number) |
| `site_id` | yes | Site identifier; macro sectors of one site share it |
| `cell_type` | yes | `macro` \| `pico` \| `femto` |
| `lat` | yes | WGS84 latitude, degrees (-90..90) |
| `lon` | yes | WGS84 longitude, degrees (-180..180) |
| `height_m` | no | Antenna height above ground (m). Defaults: 25 macro, 10 small |
| `azimuth_deg` | no | Sector azimuth, clockwise from North (0..360). Empty = omni |
| `tx_power_dbm` | no | Per-cell TX power (dBm). Empty = scenario default for the type |

Rules enforced on load (invalid files are rejected *before* execution):
- required columns present; `cell_id` unique and non-empty
- `cell_type` in {macro, pico, femto}; lat/lon in range; azimuth 0..360
- extra columns are ignored (your export can carry more fields)

Example: `examples/carrier_topology_example.csv`.

## 2. Scenario JSON — pointing at your topology

In the scenario file, under `network`:

```json
"network": {
  "topology_csv": "path/to/topology.csv",
  "geo_reference": {"lat": 37.7749, "lon": -122.4194}
}
```

- `topology_csv`: path to the file above (relative to the working dir).
- `geo_reference`: a WGS84 point near your area (e.g. the city centre).
  Used as the projection origin; any point within ~10 km of your cells
  gives centimetre-accurate local coordinates.
- When `topology_csv` is set, the synthetic hex-grid/small-cell
  generation is **skipped entirely** — every cell comes from your file.
  `macro_sites` / `small_cells` are ignored in this mode.

Everything else in the scenario stays the same: carriers (bands,
bandwidths, numerologies), `tx_power_dbm` defaults per type, slices and
their SLA targets, mobility, handover, control. That is the roadmap for
"other input configs of RAN at a later date": each block is already a
separate, documented section of the schema (`configs/scenario_schema_v1.yaml`).

## 3. What the twin does with it

`build_layout` → per-cell carrier/PRB inventory → Sionna 3GPP TR 38.901
coupling loss on your geometry → the full slicing/dimensioning workflow
runs unchanged (deterministic per seed, matched realizations across
policies).
