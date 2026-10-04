"""Standard carrier topology import (CSV).

A carrier feeds their REAL network: one row per cell (each macro sector
is its own row, sharing a site_id). Coordinates are WGS84 lat/lon — the
carrier-standard — converted to local meters (equirectangular around a
reference point; centimetre-accurate over urban extents).

Standard columns (case-insensitive, extra columns ignored):
  cell_id       unique cell identifier (string or int)
  site_id       site identifier; macro sectors share one site_id
  cell_type     macro | pico | femto
  lat, lon      WGS84 degrees
  height_m      antenna height above ground (m)
  azimuth_deg   sector azimuth clockwise from North; empty = omni
  tx_power_dbm  per-cell TX power; empty = scenario default for the type
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

from .loader import ScenarioError

REQUIRED = ["cell_id", "site_id", "cell_type", "lat", "lon"]
OPTIONAL_DEFAULTS = {
    "height_m": None,      # -> type default (macro 25, small 10)
    "azimuth_deg": None,   # -> None (omni)
    "tx_power_dbm": None,  # -> scenario default for the type
}
CELL_TYPES = ("macro", "pico", "femto")
_EARTH_R = 6371000.0


def latlon_to_enu(lat: float, lon: float,
                  ref_lat: float, ref_lon: float) -> tuple[float, float]:
    """Equirectangular projection to local meters around reference."""
    x = math.radians(lon - ref_lon) * _EARTH_R * math.cos(math.radians(ref_lat))
    y = math.radians(lat - ref_lat) * _EARTH_R
    return x, y


def load_topology_csv(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise ScenarioError(f"topology_csv not found: {p}")
    with p.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ScenarioError(f"{p}: empty CSV or missing header")
        cols = {c.strip().lower(): c for c in reader.fieldnames}
        missing = [c for c in REQUIRED if c not in cols]
        if missing:
            raise ScenarioError(
                f"{p}: missing required columns {missing}; "
                f"required: {REQUIRED}")
        cells = []
        seen = set()
        for ln, row in enumerate(reader, start=2):
            g = lambda c: (row[cols[c]].strip()
                           if cols.get(c) and row[cols[c]] is not None else "")
            cell_id = g("cell_id")
            if not cell_id:
                raise ScenarioError(f"{p}:{ln}: empty cell_id")
            if cell_id in seen:
                raise ScenarioError(f"{p}:{ln}: duplicate cell_id '{cell_id}'")
            seen.add(cell_id)
            ctype = g("cell_type").lower()
            if ctype not in CELL_TYPES:
                raise ScenarioError(
                    f"{p}:{ln}: cell_type '{ctype}' must be one of {CELL_TYPES}")
            try:
                lat, lon = float(g("lat")), float(g("lon"))
            except ValueError:
                raise ScenarioError(f"{p}:{ln}: lat/lon must be numeric")
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ScenarioError(f"{p}:{ln}: lat/lon out of range")
            cell = {"cell_id": cell_id, "site_id": g("site_id") or cell_id,
                    "type": ctype, "lat": lat, "lon": lon}
            for col, _ in OPTIONAL_DEFAULTS.items():
                v = g(col)
                cell[col] = float(v) if v else None
            if cell["azimuth_deg"] is not None and not (
                    0 <= cell["azimuth_deg"] < 360):
                raise ScenarioError(f"{p}:{ln}: azimuth_deg must be 0..360")
            cells.append(cell)
    if not cells:
        raise ScenarioError(f"{p}: no cell rows found")
    return cells
