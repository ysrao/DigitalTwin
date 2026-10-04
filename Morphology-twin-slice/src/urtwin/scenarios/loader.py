"""Scenario loading + validation for schema v1.

Reads the JSON exported by the RAN Twin Scenario Inputs artifact (or the YAML
example in configs/) into typed dataclasses. Validation mirrors the brief's
"show invalid resource totals or missing fields before execution" rule.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ScenarioError(ValueError):
    pass


@dataclass
class Carrier:
    cc_id: int
    duplex: str          # "FDD" | "TDD"
    freq_mhz: float
    bw_mhz: float
    numerology: int      # 0..4
    tdd_pattern: str = ""  # e.g. "DDDSU", required for TDD


@dataclass
class SmallCellCfg:
    count: int
    placement: str       # "uniform" | "hotspot" | "explicit"
    explicit: list = field(default_factory=list)  # [{x, y, height_m}]


@dataclass
class MobilityClass:
    name: str
    share: float
    speed_mps: float


@dataclass
class SliceCfg:
    name: str
    latency_ms: float
    reliability: float
    min_share: float


@dataclass
class Scenario:
    name: str
    seed: int
    macro_sites: int
    sectors_per_site: int
    site_spacing_m: float
    small_cells: SmallCellCfg
    wraparound: bool
    carriers: list      # list[Carrier]
    propagation: str
    tx_power_dbm: dict
    mobility_classes: list  # list[MobilityClass]
    handover: dict       # {hysteresis_db, time_to_trigger_ms}
    slices: list        # list[SliceCfg]
    budget_decision_ms: float
    forecast_horizon: int
    # optional explicit per-cell locations: [{cell_id, x, y, height_m, azimuth_deg, type}]
    cells: list = field(default_factory=list)
    # carrier topology import: standard CSV (see topology.py) + WGS84 reference
    topology_csv: str = ""
    geo_reference: dict = field(default_factory=dict)  # {lat, lon}


def _req(d: dict, key: str, where: str) -> Any:
    if key not in d:
        raise ScenarioError(f"{where}: missing required field '{key}'")
    return d[key]


def from_dict(d: dict) -> Scenario:
    s = _req(d, "scenario", "root")
    net = _req(d, "network", "root")
    radio = _req(d, "radio", "root")
    mob = _req(d, "mobility", "root")
    svc = _req(d, "services", "root")
    ctl = _req(d, "control", "root")

    carriers = []
    for c in _req(radio, "carriers", "radio"):
        cc = Carrier(
            cc_id=int(_req(c, "cc_id", "carrier")),
            duplex=str(_req(c, "duplex", "carrier")).upper(),
            freq_mhz=float(_req(c, "freq_mhz", "carrier")),
            bw_mhz=float(_req(c, "bw_mhz", "carrier")),
            numerology=int(_req(c, "numerology", "carrier")),
            tdd_pattern=str(c.get("tdd_pattern", "")),
        )
        if cc.duplex not in ("FDD", "TDD"):
            raise ScenarioError(f"carrier {cc.cc_id}: duplex must be FDD or TDD")
        if cc.duplex == "TDD" and not cc.tdd_pattern:
            raise ScenarioError(f"carrier {cc.cc_id}: TDD needs tdd_pattern")
        if not (0 <= cc.numerology <= 4):
            raise ScenarioError(f"carrier {cc.cc_id}: numerology 0..4")
        carriers.append(cc)
    if not carriers:
        raise ScenarioError("radio: at least one carrier required")

    sc = net.get("small_cells", {})
    classes = [
        MobilityClass(name=str(_req(m, "name", "mobility class")),
                      share=float(_req(m, "share", "mobility class")),
                      speed_mps=float(_req(m, "speed_mps", "mobility class")))
        for m in _req(mob, "classes", "mobility")
    ]
    share_sum = sum(c.share for c in classes)
    if abs(share_sum - 1.0) > 1e-6:
        raise ScenarioError(f"mobility class shares sum to {share_sum}, must be 1.0")

    slices = [
        SliceCfg(name=str(_req(x, "name", "slice")),
                 latency_ms=float(_req(x, "latency_ms", "slice")),
                 reliability=float(_req(x, "reliability", "slice")),
                 min_share=float(_req(x, "min_share", "slice")))
        for x in _req(svc, "slices", "services")
    ]
    if sum(x.min_share for x in slices) > 1.0 + 1e-9:
        raise ScenarioError("slice min_shares exceed 1.0 — infeasible by construction")

    return Scenario(
        name=str(_req(s, "name", "scenario")),
        seed=int(s.get("seed", 0)),
        macro_sites=int(_req(net, "macro_sites", "network")),
        sectors_per_site=int(net.get("sectors_per_site", 3)),
        site_spacing_m=float(_req(net, "site_spacing_m", "network")),
        small_cells=SmallCellCfg(
            count=int(sc.get("count", 0)),
            placement=str(sc.get("placement", "uniform")),
            explicit=list(sc.get("explicit", [])),
        ),
        wraparound=bool(net.get("wraparound", True)),
        carriers=carriers,
        propagation=str(radio.get("propagation", "UMa")),
        tx_power_dbm=dict(radio.get("tx_power_dbm",
                                    {"macro": 46, "pico": 30, "femto": 20})),
        mobility_classes=classes,
        handover=dict(mob.get("handover",
                              {"hysteresis_db": 3, "time_to_trigger_ms": 160})),
        slices=slices,
        budget_decision_ms=float(_req(ctl, "budget_decision_ms", "control")),
        forecast_horizon=int(ctl.get("forecast_horizon", 6)),
        cells=list(net.get("cells", []) or []),
        topology_csv=str(net.get("topology_csv", "") or ""),
        geo_reference=dict(net.get("geo_reference", {}) or {}),
    )


def load(path: str | Path) -> Scenario:
    p = Path(path)
    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise ScenarioError("pyyaml needed for YAML scenarios") from e
        return from_dict(yaml.safe_load(text))
    return from_dict(json.loads(text))
