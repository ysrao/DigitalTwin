"""Heterogeneous layout builder.

Macro: hexagonal grid of 3-sector sites. Small cells: pico/femto underlay,
uniform / hotspot / explicit placement. Every cell carries its FDD+TDD
component carriers with usable DL PRB budgets (see numerology.py).

Cell dict fields: cell_id, type ("macro"|"pico"|"femto"), site_id, sector,
x, y, height_m, azimuth_deg (None for omni small cells), tx_power_dbm,
carriers: [{cc_id, duplex, freq_mhz, bw_mhz, numerology, n_prb, dl_prbs}].
"""
from __future__ import annotations

import math
import numpy as np

from ..scenarios.loader import Scenario
from .numerology import n_prb, usable_dl_prbs, dl_prb_units_per_grid_slot

_MACRO_HEIGHT = 25.0
_SMALL_HEIGHT = 10.0


def _build_from_topology_csv(scenario: Scenario) -> list:
    """Carrier mode: every cell comes from the standard topology CSV."""
    from ..scenarios.topology import load_topology_csv, latlon_to_enu
    from ..scenarios.loader import ScenarioError
    ref = getattr(scenario, "geo_reference", {}) or {}
    if "lat" not in ref or "lon" not in ref:
        raise ScenarioError(
            "network.topology_csv requires network.geo_reference: {lat, lon} "
            "(WGS84 reference point for local projection)")
    rows = load_topology_csv(scenario.topology_csv)
    cells = []
    for cid, r in enumerate(rows):
        x, y = latlon_to_enu(r["lat"], r["lon"], ref["lat"], ref["lon"])
        ctype = r["type"]
        h = r["height_m"] if r["height_m"] is not None else (
            _MACRO_HEIGHT if ctype == "macro" else _SMALL_HEIGHT)
        pwr = (r["tx_power_dbm"] if r["tx_power_dbm"] is not None
               else scenario.tx_power_dbm.get(ctype, 46.0))
        cells.append(_make_cell(
            scenario, r["cell_id"], ctype, r["site_id"], 0,
            x, y, h, r["azimuth_deg"], pwr))
    return cells


def build_layout(scenario: Scenario, rng: np.random.Generator) -> list:
    # --- carrier mode: full topology from standard CSV, no synthetic grid ---
    if getattr(scenario, "topology_csv", ""):
        return _build_from_topology_csv(scenario)
    cells = []
    cid = 0
    isd = scenario.site_spacing_m

    # --- macro sites on a hex grid (pointy-top axial -> cartesian) ---
    n_sites = scenario.macro_sites
    # lay out as compact hex cluster: rings around center
    positions = _hex_cluster(n_sites, isd)
    for site_id, (sx, sy) in enumerate(positions):
        for sector in range(scenario.sectors_per_site):
            az = (sector * 360.0 / scenario.sectors_per_site + 30.0) % 360.0
            cells.append(_make_cell(
                scenario, cid, "macro", site_id, sector,
                sx, sy, _MACRO_HEIGHT, az,
                scenario.tx_power_dbm.get("macro", 46.0)))
            cid += 1

    # --- small-cell underlay ---
    sc = scenario.small_cells
    if sc.placement == "explicit" and sc.explicit:
        spots = [(float(p["x"]), float(p["y"]),
                  float(p.get("height_m", _SMALL_HEIGHT))) for p in sc.explicit]
    else:
        span = isd * 1.2
        if sc.placement == "hotspot":
            # cluster around 2 hotspots inside the macro area
            hubs = [(span * 0.3, span * 0.2), (-span * 0.35, -span * 0.25)]
            spots = []
            for _ in range(sc.count):
                hx, hy = hubs[rng.integers(len(hubs))]
                spots.append((hx + rng.normal(0, 60), hy + rng.normal(0, 60),
                              _SMALL_HEIGHT))
        else:  # uniform
            spots = [(rng.uniform(-span, span), rng.uniform(-span, span),
                      _SMALL_HEIGHT) for _ in range(sc.count)]
    for i, (x, y, h) in enumerate(spots):
        ctype = "pico" if i % 2 == 0 else "femto"
        cells.append(_make_cell(
            scenario, cid, ctype, f"sc{i}", 0, x, y, h, None,
            scenario.tx_power_dbm.get(ctype, 30.0 if ctype == "pico" else 20.0)))
        cid += 1

    # --- explicit per-cell overrides (location inputs) ---
    for override in scenario.cells:
        for cell in cells:
            if cell["cell_id"] == override.get("cell_id"):
                for k in ("x", "y", "height_m", "azimuth_deg"):
                    if k in override:
                        cell[k] = override[k]
    return cells


def _make_cell(scenario, cid, ctype, site_id, sector, x, y, h, az, pwr):
    grid_mu = scenario.carriers[0].numerology  # sim slot grid = carriers[0]
    carriers = []
    for c in scenario.carriers:
        carriers.append({
            "cc_id": c.cc_id, "duplex": c.duplex, "freq_mhz": c.freq_mhz,
            "bw_mhz": c.bw_mhz, "numerology": c.numerology,
            "n_prb": n_prb(c), "dl_prbs": usable_dl_prbs(c),
            # pooled-scheduler budget in uniform PRB units per grid slot
            "dl_prb_units": dl_prb_units_per_grid_slot(c, grid_mu),
        })
    return {
        "cell_id": cid, "type": ctype, "site_id": site_id, "sector": sector,
        "x": float(x), "y": float(y), "height_m": float(h),
        "azimuth_deg": az, "tx_power_dbm": float(pwr),
        "carriers": carriers,
        # pooled DL budget per grid slot in uniform PRB units
        "total_dl_prbs": sum(cc["dl_prb_units"] for cc in carriers),
    }


def _hex_cluster(n_sites: int, isd: float) -> list:
    """Compact hex cluster of site positions (center + rings)."""
    pts = [(0.0, 0.0)]
    ring = 1
    while len(pts) < n_sites:
        for k in range(6):
            for j in range(ring):
                if len(pts) >= n_sites:
                    break
                ang = math.radians(60 * k + 60 * j / max(ring, 1))
                pts.append((isd * ring * math.cos(ang),
                            isd * ring * math.sin(ang)))
            if len(pts) >= n_sites:
                break
        ring += 1
    return pts[:n_sites]
