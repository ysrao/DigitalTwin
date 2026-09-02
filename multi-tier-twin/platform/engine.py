"""Dependency-light two-timescale RAN slicing decision engine (research MVP)."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "synthetic_profiles" / "all_profiles.csv"

SLICE_NAMES = ("eMBB", "URLLC", "IoT")
TEMPLATES = np.array([
    [0.55, 0.30, 0.15],  # carrier hard reference
    [0.45, 0.40, 0.15],
    [0.65, 0.22, 0.13],
    [0.42, 0.25, 0.33],
    [0.34, 0.48, 0.18],
    [0.72, 0.18, 0.10],
    [0.34, 0.22, 0.44],
    [0.25, 0.65, 0.10],
    [0.20, 0.15, 0.65],
], dtype=np.float64)


@dataclass
class Scenario:
    profile: str = "P1_balanced_busy_hour"
    area_km2: float = 1.0
    configured_cells: int = 12
    cell_area_km2: float = 0.083333
    prbs_per_cell: int = 273
    iot_density_km2: float = 1_000_000
    registered_capacity_cell: int = 120_000
    active_capacity_cell: int = 3_000
    iot_active_fraction: float = 0.001
    morphology: str = "urban"
    carrier_frequency_ghz: float = 3.5
    limiting_tx_power_dbm: float = 23.0
    receive_antenna_gain_dbi: float = 17.0
    receiver_noise_figure_db: float = 5.0
    reference_bandwidth_khz: float = 360.0
    required_sinr_db: float = 0.0
    propagation_margin_db: float = 20.0
    planning_efficiency: float = 0.65
    cost_per_active_cell: float = 1.0
    material_gain_pct: float = 2.0
    noninferiority_margin_pct: float = 0.5
    train_steps: int = 4096
    seed: int = 3080


MORPHOLOGY_EXPONENT = {
    "dense_urban": 3.8,
    "urban": 3.5,
    "suburban": 3.0,
    "rural": 2.7,
    "indoor": 3.2,
}


def coverage_plan(s: Scenario) -> Dict[str, float]:
    """Screening-level link-budget coverage estimate using a log-distance model.

    The limiting link should normally be uplink, hence the 23 dBm default. A calibrated
    3GPP/vendor propagation model must replace this screening model for carrier decisions.
    """
    exponent = MORPHOLOGY_EXPONENT.get(s.morphology)
    if exponent is None:
        raise ValueError(f"Unsupported morphology: {s.morphology}")
    if not (0 < s.planning_efficiency <= 1):
        raise ValueError("planning_efficiency must be in (0, 1]")
    noise_dbm = -174 + 10 * math.log10(max(1.0, s.reference_bandwidth_khz * 1000)) + s.receiver_noise_figure_db
    sensitivity_dbm = noise_dbm + s.required_sinr_db
    max_path_loss_db = s.limiting_tx_power_dbm + s.receive_antenna_gain_dbi - sensitivity_dbm - s.propagation_margin_db
    intercept_db = 32.4 + 20 * math.log10(max(0.1, s.carrier_frequency_ghz))
    radius_m = 10 ** ((max_path_loss_db - intercept_db) / (10 * exponent))
    radius_km = min(20.0, max(0.02, radius_m / 1000))
    ideal_area_km2 = (3 * math.sqrt(3) / 2) * radius_km ** 2
    effective_area_km2 = ideal_area_km2 * s.planning_efficiency
    cells = math.ceil(s.area_km2 / max(1e-9, effective_area_km2))
    return {
        "morphology": s.morphology,
        "path_loss_exponent": exponent,
        "receiver_sensitivity_dbm": sensitivity_dbm,
        "maximum_path_loss_db": max_path_loss_db,
        "estimated_cell_radius_km": radius_km,
        "ideal_hex_area_km2": ideal_area_km2,
        "effective_cell_area_km2": effective_area_km2,
        "coverage_cells": max(1, cells),
        "model": "screening_log_distance",
    }


def cell_plan(s: Scenario) -> Dict[str, float]:
    coverage = coverage_plan(s)
    total_registered = s.iot_density_km2 * s.area_km2
    total_active = total_registered * s.iot_active_fraction
    by_registered = math.ceil(total_registered / max(1, s.registered_capacity_cell))
    by_active = math.ceil(total_active / max(1, s.active_capacity_cell))
    required = max(coverage["coverage_cells"], by_registered, by_active)
    return {
        "coverage_floor": coverage["coverage_cells"],
        "coverage": coverage,
        "registered_iot_cells": by_registered,
        "active_iot_cells": by_active,
        "required_cells": required,
        "configured_cells": s.configured_cells,
        "capacity_gap_cells": max(0, required - s.configured_cells),
        "estimated_active_cell_cost": required * s.cost_per_active_cell,
    }


def load_rows(profile: str) -> List[dict]:
    with DATA.open(newline="") as f:
        all_rows = list(csv.DictReader(f))
    if profile == "P6_ppo_advantage_demo":
        # Demonstration-only trace: feasible single-slice-dominant phases expose the
        # opportunity cost of protected floors in the reactive soft baseline.
        rows = [dict(r) for r in all_rows if r["profile"] == "P1_balanced_busy_hour"]
        for idx, r in enumerate(rows):
            phase = (int(float(r["timestamp_ms"])) // 300) % 3
            r["profile"] = profile
            if phase == 0:  # eMBB-dominant
                r["embb_offered_dl_mbps"] = "160"
                r["embb_active_ues"] = "28"
                r["urllc_packets"] = "0"
                r["urllc_active_ues"] = "0"
                r["iot_active_devices"] = "0"
            elif phase == 1:  # URLLC-dominant
                r["embb_offered_dl_mbps"] = "12"
                r["embb_active_ues"] = "2"
                r["urllc_packets"] = "2200"
                r["urllc_active_ues"] = "250"
                r["iot_active_devices"] = "0"
            else:  # IoT alarm/reporting wave
                r["embb_offered_dl_mbps"] = "12"
                r["embb_active_ues"] = "2"
                r["urllc_packets"] = "0"
                r["urllc_active_ues"] = "0"
                r["iot_active_devices"] = "5000"
        return rows
    else:
        rows = [r for r in all_rows if r["profile"] == profile]
    if not rows:
        raise ValueError(f"Unknown or empty profile: {profile}")
    return rows


def f(row: dict, key: str, default=0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)


def state_vector(row: dict, previous: np.ndarray, scenario: Scenario) -> np.ndarray:
    prbs = max(1.0, f(row, "available_prbs", scenario.prbs_per_cell))
    embb_demand = f(row, "embb_offered_dl_mbps")
    urllc_demand = f(row, "urllc_packets") * f(row, "urllc_payload_bytes", 32) * 8 / max(1, f(row, "interval_ms", 10)) / 1000
    iot_demand = f(row, "iot_active_devices") * f(row, "iot_min_rate_kbps_per_device", 0.2048) / 1000
    embb_se = f(row, "embb_spectral_eff_bps_hz", 1)
    urllc_se = f(row, "urllc_spectral_eff_bps_hz", 1)
    iot_se = f(row, "iot_spectral_eff_bps_hz", 0.3)
    return np.array([
        min(2.0, prbs / max(1, scenario.prbs_per_cell)),
        min(4.0, embb_demand / max(1, scenario.prbs_per_cell * 0.36 * embb_se)),
        min(4.0, urllc_demand / max(0.1, scenario.prbs_per_cell * 0.36 * urllc_se)),
        min(4.0, iot_demand / max(0.01, scenario.prbs_per_cell * 0.36 * iot_se)),
        min(2.0, f(row, "embb_active_ues") / 50),
        min(2.0, f(row, "urllc_packets") / 30),
        min(2.0, f(row, "iot_active_devices") / 3000),
        min(2.0, embb_se / 4),
        min(2.0, urllc_se / 2),
        min(2.0, iot_se / 0.8),
        min(1.0, f(row, "neighbor_load_ratio")),
        float(row.get("cell_available", "1") == "0"),
        float(np.argmax(previous)) / max(1, len(TEMPLATES) - 1),
    ], dtype=np.float64)


def demand_and_capacity(row: dict, shares: np.ndarray, scenario: Scenario):
    available = int(f(row, "available_prbs", scenario.prbs_per_cell))
    available = min(available, scenario.prbs_per_cell)
    allocations = np.floor(shares * available).astype(int)
    allocations[0] += available - int(allocations.sum())
    demand = np.array([
        f(row, "embb_offered_dl_mbps"),
        f(row, "urllc_packets") * f(row, "urllc_payload_bytes", 32) * 8 / max(1, f(row, "interval_ms", 10)) / 1000,
        f(row, "iot_active_devices") * f(row, "iot_min_rate_kbps_per_device", 0.2048) / 1000,
    ])
    se = np.array([
        f(row, "embb_spectral_eff_bps_hz", 1),
        f(row, "urllc_spectral_eff_bps_hz", 1),
        f(row, "iot_spectral_eff_bps_hz", 0.3),
    ])
    capacity = allocations * 0.36 * se
    served = np.minimum(demand, capacity)
    satisfaction = np.divide(served, np.maximum(demand, 1e-9))
    return allocations, demand, capacity, served, satisfaction


def outcome(row: dict, shares: np.ndarray, scenario: Scenario) -> Dict:
    allocations, demand, capacity, served, sat = demand_and_capacity(row, shares, scenario)
    weights = np.array([0.27, 0.48, 0.15])
    utilization = served.sum() / max(1e-9, capacity.sum())
    fairness = float((sat.sum() ** 2) / (3 * np.square(sat).sum() + 1e-9))
    thresholds = np.array([0.95, 0.999, 0.99])
    # No offered demand means no SLA opportunity in that interval; it is not a failure.
    sla = np.logical_or(demand <= 1e-9, sat >= thresholds).astype(float)
    violation_penalty = float(((1 - sla) * np.array([1.0, 4.0, 1.5])).sum())
    reward = float(weights @ sat + 0.06 * utilization + 0.04 * fairness - 0.12 * violation_penalty)
    return {
        "allocations": allocations,
        "demand": demand,
        "served": served,
        "satisfaction": sat,
        "sla": sla,
        "utilization": utilization,
        "fairness": fairness,
        "reward": reward,
    }


class ActorCritic:
    """Custom `n_inputs` -> 64 -> 64 categorical PPO actor and matching critic.

    Defaults reproduce the original 13 -> 64 -> 64 -> 9 network exactly. The
    dimensions are parameters so the multi-tier twin in `twin.py` can train the
    same architecture on its wider observation and action spaces.
    """
    def __init__(self, seed=3080, n_inputs=13, n_actions=None, hidden=64):
        n_actions = len(TEMPLATES) if n_actions is None else int(n_actions)
        self.n_inputs, self.n_actions, self.hidden = int(n_inputs), n_actions, int(hidden)
        rng = np.random.default_rng(seed)
        h = self.hidden
        self.aw = [rng.normal(0, 0.12, (self.n_inputs, h)), rng.normal(0, 0.12, (h, h)), rng.normal(0, 0.08, (h, n_actions))]
        self.ab = [np.zeros(h), np.zeros(h), np.zeros(n_actions)]
        self.cw = [rng.normal(0, 0.12, (self.n_inputs, h)), rng.normal(0, 0.12, (h, h)), rng.normal(0, 0.08, (h, 1))]
        self.cb = [np.zeros(h), np.zeros(h), np.zeros(1)]

    @staticmethod
    def _forward(x, w, b, softmax=False):
        z1 = x @ w[0] + b[0]; h1 = np.maximum(z1, 0)
        z2 = h1 @ w[1] + b[1]; h2 = np.maximum(z2, 0)
        out = h2 @ w[2] + b[2]
        if softmax:
            out = out - out.max(axis=1, keepdims=True)
            e = np.exp(out); out = e / e.sum(axis=1, keepdims=True)
        return out, (x, z1, h1, z2, h2)

    def probs(self, x): return self._forward(x, self.aw, self.ab, True)[0]
    def values(self, x): return self._forward(x, self.cw, self.cb, False)[0][:, 0]

    @staticmethod
    def _backprop(grad_out, cache, w, lr):
        x, z1, h1, z2, h2 = cache
        gw2 = h2.T @ grad_out; gb2 = grad_out.sum(0)
        gh2 = grad_out @ w[2].T; gz2 = gh2 * (z2 > 0)
        gw1 = h1.T @ gz2; gb1 = gz2.sum(0)
        gh1 = gz2 @ w[1].T; gz1 = gh1 * (z1 > 0)
        gw0 = x.T @ gz1; gb0 = gz1.sum(0)
        for i, (gw, gb) in enumerate(((gw0, gb0), (gw1, gb1), (gw2, gb2))):
            np.clip(gw, -2, 2, out=gw); np.clip(gb, -2, 2, out=gb)
            w[i] -= lr * gw
            yield i, gb

    def update(self, x, actions, oldp, advantages, returns, epochs=8, lr=0.012, clip=0.2):
        n = len(x)
        for _ in range(epochs):
            probs, acache = self._forward(x, self.aw, self.ab, True)
            chosen = probs[np.arange(n), actions]
            ratio = chosen / np.maximum(oldp, 1e-10)
            active = ~(((advantages >= 0) & (ratio > 1 + clip)) | ((advantages < 0) & (ratio < 1 - clip)))
            coeff = -(advantages * ratio * active) / n
            grad_logits = -probs
            grad_logits[np.arange(n), actions] += 1
            grad_logits *= coeff[:, None]
            for i, gb in self._backprop(grad_logits, acache, self.aw, lr): self.ab[i] -= lr * gb
            values, ccache = self._forward(x, self.cw, self.cb, False)
            grad_v = (2 * (values[:, 0] - returns) / n)[:, None]
            for i, gb in self._backprop(grad_v, ccache, self.cw, lr): self.cb[i] -= lr * gb


def train_ppo(rows: List[dict], scenario: Scenario) -> Tuple[ActorCritic, Dict]:
    rng = np.random.default_rng(scenario.seed)
    model = ActorCritic(scenario.seed)
    start = time.perf_counter()
    history = []
    previous = np.array([1.0, 0, 0, 0, 0, 0, 0])
    batch_size = min(512, max(128, scenario.train_steps // 4))
    updates = max(4, scenario.train_steps // batch_size)
    for update in range(updates):
        idx = rng.integers(0, len(rows), size=batch_size)
        states = np.stack([state_vector(rows[i], previous, scenario) for i in idx])
        probs = model.probs(states)
        actions = np.array([rng.choice(len(TEMPLATES), p=p) for p in probs])
        rewards = np.array([outcome(rows[i], TEMPLATES[a], scenario)["reward"] for i, a in zip(idx, actions)])
        values = model.values(states)
        adv = rewards - values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        model.update(states, actions, probs[np.arange(batch_size), actions], adv, rewards)
        previous = probs.mean(0)
        history.append(float(rewards.mean()))
    return model, {"updates": updates, "samples": updates * batch_size, "reward_history": history, "train_seconds": time.perf_counter() - start}


def soft_shares(row: dict, scenario: Scenario) -> np.ndarray:
    # Protected floors with deterministic demand/urgency borrowing.
    floors = np.array([0.34, 0.25, 0.12])
    dummy = np.array([1/3, 1/3, 1/3])
    _, demand, capacity, _, _ = demand_and_capacity(row, dummy, scenario)
    pressure = demand / np.maximum(capacity, 1e-6)
    pressure[1] *= 1.7
    remaining = 1 - floors.sum()
    return floors + remaining * pressure / max(1e-9, pressure.sum())


def greedy_shares(row: dict, scenario: Scenario) -> np.ndarray:
    _, demand, capacity, _, _ = demand_and_capacity(row, np.ones(3)/3, scenario)
    p = demand / np.maximum(capacity, 1e-6)
    return p / max(1e-9, p.sum())


def evaluate_policy(rows, scenario, name, model=None):
    sums = {"reward": 0.0, "utilization": 0.0, "fairness": 0.0}
    sat = np.zeros(3); sla = np.zeros(3); served = np.zeros(3); demand = np.zeros(3)
    previous = np.array([1.0, 0, 0, 0, 0, 0, 0])
    series = []
    start = time.perf_counter()
    for k, row in enumerate(rows):
        if name == "hard": shares = TEMPLATES[0]
        elif name == "soft": shares = soft_shares(row, scenario)
        elif name == "greedy": shares = greedy_shares(row, scenario)
        else:
            state = state_vector(row, previous, scenario)[None, :]
            probs = model.probs(state)[0]
            action = int(np.argmax(probs)); shares = TEMPLATES[action]; previous = probs
        out = outcome(row, shares, scenario)
        for key in sums: sums[key] += out[key]
        sat += out["satisfaction"]; sla += out["sla"]; served += out["served"]; demand += out["demand"]
        if k % max(1, len(rows)//80) == 0:
            series.append({"t": k, "embb": round(float(shares[0]), 4), "urllc": round(float(shares[1]), 4), "iot": round(float(shares[2]), 4), "reward": round(out["reward"], 4)})
    n = len(rows)
    return {
        "policy": name,
        "mean_reward": sums["reward"] / n,
        "prb_utilization": sums["utilization"] / n,
        "jain_fairness": sums["fairness"] / n,
        "mean_satisfaction": dict(zip(SLICE_NAMES, (sat/n).tolist())),
        "sla_compliance": dict(zip(SLICE_NAMES, (sla/n).tolist())),
        "offered_mbps": dict(zip(SLICE_NAMES, demand.tolist())),
        "served_mbps": dict(zip(SLICE_NAMES, served.tolist())),
        "inference_ms_per_row": (time.perf_counter() - start) * 1000 / n,
        "series": series,
    }


def decision_status(results: Dict[str, dict], scenario: Scenario, plan: Dict) -> Dict:
    ppo, soft = results["ppo"], results["soft"]
    primary_gain = 100 * (ppo["mean_reward"] - soft["mean_reward"]) / max(abs(soft["mean_reward"]), 1e-9)
    compliance_delta = {s: 100 * (ppo["sla_compliance"][s] - soft["sla_compliance"][s]) for s in SLICE_NAMES}
    protected_delta = min(compliance_delta.values())
    if plan["capacity_gap_cells"] > 0:
        status = "SLA/FEASIBILITY FAIL"; reason = "Configured cells are below the IoT/coverage capacity envelope."
    elif protected_delta < -scenario.noninferiority_margin_pct:
        status = "UNDERPERFORMING"; reason = "PPO is inferior to tuned soft provisioning on at least one protected slice SLA."
    elif primary_gain < 0:
        status = "UNDERPERFORMING"; reason = "PPO has negative aggregate gain versus tuned soft provisioning."
    elif primary_gain < scenario.material_gain_pct:
        status = "NO MATERIAL BENEFIT"; reason = "PPO gain is below the configured materiality threshold."
    else:
        status = "PASS — PPO preferred"; reason = "PPO is non-inferior on protected SLA and exceeds the material-gain threshold."
    return {"status": status, "reason": reason, "ppo_gain_vs_soft_pct": primary_gain, "minimum_sla_delta_pp": protected_delta, "sla_compliance_delta_pp": compliance_delta, "recommended_policy": "ppo" if status.startswith("PASS") else "soft"}


def run_evaluation(payload: dict) -> Dict:
    allowed = Scenario.__dataclass_fields__
    scenario = Scenario(**{k: v for k, v in payload.items() if k in allowed})
    if scenario.profile == "P6_ppo_advantage_demo" and scenario.train_steps < 32768:
        scenario.train_steps = 32768
    rows = load_rows(scenario.profile)
    # Use a deterministic subset for responsive MVP evaluation.
    rows = rows[: min(len(rows), 2400)]
    model, training = train_ppo(rows, scenario)
    results = {name: evaluate_policy(rows, scenario, name, model if name == "ppo" else None) for name in ("hard", "soft", "greedy", "ppo")}
    plan = cell_plan(scenario)
    decision = decision_status(results, scenario, plan)
    is_demo = scenario.profile == "P6_ppo_advantage_demo"
    return {
        "implementation": "research_mvp_categorical_ppo",
        "evidence_class": "functional_demonstration_only" if is_demo else "synthetic_evaluation",
        "scenario": asdict(scenario),
        "cell_plan": plan,
        "training": training,
        "results": results,
        "decision": decision,
        "limitations": [
            *( ["P6 is a deliberately constructed functional demonstration of a PPO allocation opportunity, uses at least 32,768 training samples, and is not research evidence."] if is_demo else [] ),
            "Synthetic trace replay is not carrier evidence.",
            "Categorical PRB templates are the MVP action space; continuous 3N-cell allocation is future work.",
            "Reported SLA compliance is modeled interval compliance, not proof of five- or seven-nines reliability.",
            "No direct RIC control is performed.",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(run_evaluation({}), indent=2))
