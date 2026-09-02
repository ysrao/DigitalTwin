"""Matched comparison of rule-based, PPO and DQN policies, and threshold tuning.

Two entry points:

    run_comparison(payload)      rule-based baselines vs PPO vs Double DQN on
                                 one scenario, evaluated on identical seeds,
                                 reported as a KPI table the UI charts directly.

    optimize_thresholds(payload) coordinate search over the A3/A5 and cell
                                 individual offset parameters for a scenario,
                                 scored on handover success and session
                                 continuity rather than throughput alone.

Named scenarios exist because the right thresholds are not universal: the
offsets that keep a dense indoor deployment stable will ping-pong a highway,
and the ones that hold a highway together will strand indoor users on a macro.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from agents import DQNAgent, PPOAgent, train_dqn, train_ppo
from engine import TEMPLATES
from handover import SteeringPolicy
from mobility import SESSION_TYPES
from twin import (DEFAULT_CELL_COUNTS, LIMITATIONS, MultiTierTwin,
                  STEERING_POSTURES, TwinConfig, branch_sizes, default_tiers)

N_SLICES = len(SESSION_TYPES)


# ==========================================================================
# Scenarios
# ==========================================================================

SCENARIOS: Dict[str, Dict] = {
    "urban_dense": {
        "label": "Urban dense — full 6G stack",
        "description": "Mixed indoor/outdoor city centre with every tier available.",
        "config": {},
    },
    "highway_mobility": {
        "label": "Highway — high mobility",
        "description": "Fast outdoor traffic. Beam-tracking tiers derate and "
                       "shed sessions to the macro layers; only the fastest "
                       "few become eligible for the satellite.",
        "config": {"indoor_fraction": 0.05, "n_sessions": 900,
                   "session_mix": {"eMBB": 0.6, "URLLC": 0.3, "mIoT": 0.1},
                   "area_km": (3.0, 0.6)},
    },
    "indoor_enterprise": {
        "label": "Indoor enterprise — Wi-Fi offload",
        "description": "Mostly indoor, high-loss buildings; Wi-Fi 7 carries the load.",
        "config": {"indoor_fraction": 0.92, "high_loss_building_fraction": 0.7,
                   "n_sessions": 1000},
    },
    "coverage_hole": {
        "label": "Rural coverage hole — NTN fallback",
        "description": "One macro site over 64 km2. Most of the area is below "
                       "the usable terrestrial level, so the satellite is the "
                       "only tier that reaches it.",
        "config": {"area_km": (8.0, 8.0), "indoor_fraction": 0.15,
                   "cell_counts": {"macro_low": 1, "macro_mid": 0, "umb_6g": 0,
                                   "mmwave": 0, "wifi7_indoor": 0, "ntn_leo": 1}},
    },
    "rain_fade": {
        "label": "Heavy rain — NTN fade",
        "description": "25 mm/h rain; tests whether the policy pulls sessions "
                       "off the satellite before the link degrades.",
        "config": {"rain_rate_mm_h": 25.0, "indoor_fraction": 0.3},
    },
    "cellular_only": {
        "label": "Cellular only — no Wi-Fi, no satellite",
        "description": "The stack with both offload tiers removed, to size what "
                       "they were actually contributing.",
        "config": {"cell_counts": {"macro_low": 3, "macro_mid": 7, "umb_6g": 12,
                                   "mmwave": 16}},
    },
}


def build_config(payload: Optional[dict] = None) -> TwinConfig:
    """Build a TwinConfig from a named scenario plus explicit overrides."""
    payload = dict(payload or {})
    scenario = payload.pop("scenario", "urban_dense")
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    fields = dict(SCENARIOS[scenario]["config"])
    allowed = set(TwinConfig.__dataclass_fields__) - {"tiers", "cells", "steering"}
    fields.update({k: v for k, v in payload.items() if k in allowed})
    # Merge scenario-level tier overrides with any from the request.
    tier_overrides = dict(fields.get("tier_overrides", {}))
    for name, over in (payload.get("tier_overrides") or {}).items():
        tier_overrides[name] = {**tier_overrides.get(name, {}), **over}
    fields["tier_overrides"] = tier_overrides

    if "area_km" in fields:
        fields["area_km"] = tuple(fields["area_km"])
    counts = dict(DEFAULT_CELL_COUNTS)
    counts.update(fields.get("cell_counts", {}))
    # A tier with no cells is simply absent; the twin runs without it.
    fields["cell_counts"] = counts
    tiers = [t for t in default_tiers() if counts.get(t.name, 0) > 0]
    if not tiers:
        raise ValueError("Scenario removes every tier; at least one is required")
    return TwinConfig(tiers=tiers, **fields)


# ==========================================================================
# Rule-based baselines
# ==========================================================================

def _template_for_demand(obs: np.ndarray) -> int:
    demand = np.maximum(obs[:N_SLICES], 0.0)
    total = demand.sum()
    mix = demand / total if total > 1e-9 else np.full(N_SLICES, 1 / N_SLICES)
    return int(np.argmin(np.abs(TEMPLATES - mix).sum(axis=1)))


# Baselines return a 2-vector (global template, posture); the twin broadcasts
# the template to every tier. Only the learned policies exercise per-tier
# freedom, which is what the comparison is meant to isolate.

def policy_static(obs: np.ndarray) -> List[int]:
    """Fixed carrier reference split, balanced steering. The 'hard' baseline."""
    return [0, 0]


def policy_demand_follow(obs: np.ndarray) -> List[int]:
    """Track the offered demand mix; leave steering alone."""
    return [_template_for_demand(obs), 0]


def policy_rule_based(obs: np.ndarray) -> List[int]:
    """Tuned rule-based controller: demand-following mix plus a steering rule.

    Reads continuity and handover success out of the observation and reacts:
    poor continuity means sessions are being stranded, so hold them on the
    terrestrial tiers; heavy load with good continuity means push indoor
    traffic onto Wi-Fi.
    """
    template = _template_for_demand(obs)
    utilization, continuity, ho_success, ping_pong = obs[4], obs[5], obs[6], obs[7]
    if continuity < 0.9 or ho_success < 0.85:
        posture = 3          # terrestrial_hold: stop losing sessions first
    elif ping_pong > 0.15:
        posture = 3          # thrashing between tiers; settle down
    elif utilization > 0.6:
        posture = 1          # wifi_lean: offload indoor demand
    else:
        posture = 0
    return [template, posture]


RULE_BASED = {
    "static": policy_static,
    "demand_follow": policy_demand_follow,
    "rule_based": policy_rule_based,
}


# ==========================================================================
# Evaluation
# ==========================================================================

def evaluate(twin: MultiTierTwin, policy: Callable[[np.ndarray], int],
             seed: int, trace_points: int = 60) -> Dict:
    """Run one full episode and collect the KPI set the UI charts."""
    obs = twin.reset(seed)
    rewards: List[float] = []
    sat = np.zeros(N_SLICES)
    sla = np.zeros(N_SLICES)
    offered = np.zeros(N_SLICES)
    served = np.zeros(N_SLICES)
    scalars = {"network_utilization": 0.0, "jain_fairness": 0.0,
               "mean_access_latency_ms": 0.0, "energy_fraction": 0.0,
               "mean_spectral_efficiency": 0.0, "unattached_sessions": 0.0}
    coverage_gap = np.zeros(N_SLICES)
    capacity_gap = np.zeros(N_SLICES)
    tier_headroom: Dict[str, float] = {}
    tier_sessions: Dict[str, float] = {}
    tier_intent: Dict[str, np.ndarray] = {}
    postures: Dict[str, int] = {}
    trace: List[Dict] = []

    start = time.perf_counter()
    steps = twin.cfg.episode_steps
    every = max(1, steps // max(1, trace_points))
    done = False
    k = 0
    while not done:
        action = policy(obs)
        obs, reward, done, info = twin.step(action)
        rewards.append(reward)
        sat += [info["satisfaction"][s] for s in SESSION_TYPES]
        sla += [info["sla"][s] for s in SESSION_TYPES]
        offered += [info["offered_mbps"][s] for s in SESSION_TYPES]
        served += [info["served_mbps"][s] for s in SESSION_TYPES]
        for key in scalars:
            scalars[key] += info[key]
        coverage_gap += [info["coverage_gap_mbps"][s] for s in SESSION_TYPES]
        capacity_gap += [info["capacity_gap_mbps"][s] for s in SESSION_TYPES]
        for name, count in info["tier_sessions"].items():
            tier_sessions[name] = tier_sessions.get(name, 0.0) + count
        for name, value in info["tier_headroom_mbps"].items():
            tier_headroom[name] = tier_headroom.get(name, 0.0) + value
        for name, vec in info["tier_intent"].items():
            tier_intent[name] = tier_intent.get(name, np.zeros(N_SLICES)) + np.array(vec)
        postures[info["posture"]] = postures.get(info["posture"], 0) + 1
        if k % every == 0:
            trace.append({
                "t": k, "reward": round(reward, 4),
                "continuity": round(info["session_continuity"]["overall"], 4),
                "handover_success": round(info["handover_success_rate"], 4),
                "utilization": info["network_utilization"],
                "posture": info["posture"],
            })
        k += 1

    n = max(1, k)
    report = twin.ho.report()
    overall = report["overall"]
    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_satisfaction": dict(zip(SESSION_TYPES, (sat / n).round(4).tolist())),
        "sla_compliance": dict(zip(SESSION_TYPES, (sla / n).round(4).tolist())),
        "offered_mbps": dict(zip(SESSION_TYPES, offered.round(1).tolist())),
        "served_mbps": dict(zip(SESSION_TYPES, served.round(1).tolist())),
        **{key: round(value / n, 4) for key, value in scalars.items()},
        "coverage_gap_mbps": dict(zip(SESSION_TYPES, (coverage_gap / n).round(2).tolist())),
        "capacity_gap_mbps": dict(zip(SESSION_TYPES, (capacity_gap / n).round(2).tolist())),
        "tier_headroom_mbps": {k: round(v / n, 1) for k, v in tier_headroom.items()},
        "tier_intent": {k: (v / n).round(3).tolist() for k, v in tier_intent.items()},
        "tier_session_share": {
            name: round(total / max(1e-9, sum(tier_sessions.values())), 4)
            for name, total in tier_sessions.items()},
        "handover": {
            "attempts": overall["attempts"],
            "success_rate": round(overall["success_rate"], 4),
            "failure_rate": round(overall["failure_rate"], 4),
            "ping_pong_rate": round(overall["ping_pong_rate"], 4),
            "radio_link_failures": overall["radio_link_failures"],
            "mean_interruption_ms": round(overall["mean_interruption_ms"], 2),
        },
        "handover_by_session_type": {
            name: {"attempts": v["attempts"],
                   "success_rate": round(v["success_rate"], 4),
                   "ping_pong_rate": round(v["ping_pong_rate"], 4),
                   "radio_link_failures": v["radio_link_failures"]}
            for name, v in report["by_session_type"].items()},
        "session_continuity": {k: round(v, 4)
                               for k, v in report["session_continuity"].items()},
        "posture_mix": postures,
        "evaluation_seconds": round(time.perf_counter() - start, 3),
        "trace": trace,
    }


def preview(payload: Optional[dict] = None) -> Dict:
    """Build the deployment and return its capacity / coverage, without training.

    This is the fast read the UI uses to answer "what throughput and coverage
    does this config give me" before anything is run. It reflects every scenario
    setting, cell-count change and per-tier override in the request.
    """
    cfg = build_config(dict(payload or {}))
    twin = MultiTierTwin(cfg)
    plan = twin.plan_summary()
    return {
        "implementation": "deployment_preview",
        "scenario": (payload or {}).get("scenario", "urban_dense"),
        "config": {k: v for k, v in asdict(cfg).items()
                   if k not in ("tiers", "cells", "steering")},
        "action_space": {"branches": twin.n_branches,
                         "branch_sizes": twin.branch_sizes,
                         "equivalent_joint_actions": twin.joint_action_count},
        "observation_dim": twin.obs_dim,
        "plan": plan,
        "capacity": plan["capacity"],
        "coverage": plan["coverage"],
    }


def run_comparison(payload: Optional[dict] = None) -> Dict:
    """Rule-based vs PPO vs Double DQN on one scenario, matched seeds."""
    payload = dict(payload or {})
    scenario_name = payload.get("scenario", "urban_dense")
    episodes = int(payload.pop("train_episodes", 12))
    eval_seed = int(payload.get("seed", TwinConfig.seed)) + 500

    cfg = build_config(payload)
    twin = MultiTierTwin(cfg)

    results: Dict[str, Dict] = {}
    for name, policy in RULE_BASED.items():
        results[name] = evaluate(twin, policy, eval_seed)

    ppo, ppo_info = train_ppo(twin, episodes=episodes, seed=cfg.seed)
    results["ppo"] = evaluate(twin, lambda o: ppo.act(o, greedy=True)[0], eval_seed)

    dqn, dqn_info = train_dqn(twin, episodes=episodes, seed=cfg.seed)
    results["dqn"] = evaluate(twin, lambda o: dqn.act(o, greedy=True)[0], eval_seed)

    baseline = results["rule_based"]["mean_reward"]
    for name, result in results.items():
        result["gain_vs_rule_based_pct"] = round(
            100 * (result["mean_reward"] - baseline) / max(abs(baseline), 1e-9), 2)

    best = max(results, key=lambda k: results[k]["mean_reward"])
    return {
        "implementation": "multi_tier_6g_twin_ppo_dqn",
        "evidence_class": "synthetic_model",
        "architecture": "single-agent centralized, per-tier action branching",
        "scenario": scenario_name,
        "scenario_label": SCENARIOS[scenario_name]["label"],
        "scenario_description": SCENARIOS[scenario_name]["description"],
        "config": {k: v for k, v in asdict(cfg).items()
                   if k not in ("tiers", "cells", "steering")},
        "plan": twin.plan_summary(),
        "training": {"ppo": ppo_info, "dqn": dqn_info},
        "action_space": {
            "type": "per_tier_branching",
            "branches": twin.n_branches,
            "branch_sizes": twin.branch_sizes,
            "per_tier_templates": len(TEMPLATES),
            "postures": [p[0] for p in STEERING_POSTURES],
            "equivalent_joint_actions": twin.joint_action_count},
        "observation_dim": twin.obs_dim,
        "results": results,
        "best_policy": best,
        "decision": _decision(results, best),
        "feasibility": feasibility_verdict(results["rule_based"], twin),
        "limitations": LIMITATIONS,
    }


def _decision(results: Dict[str, Dict], best: str,
              material_gain_pct: float = 2.0) -> Dict:
    """Guarded decision, in the same spirit as `engine.decision_status`.

    A learned policy has to beat the tuned rule-based controller on reward AND
    not regress session continuity, because a policy that raises throughput by
    dropping sessions has not improved anything.
    """
    rule = results["rule_based"]
    learned = max(("ppo", "dqn"), key=lambda k: results[k]["mean_reward"])
    candidate = results[learned]
    gain = candidate["gain_vs_rule_based_pct"]
    continuity_delta = 100 * (candidate["session_continuity"]["overall"]
                              - rule["session_continuity"]["overall"])
    ho_delta = 100 * (candidate["handover"]["success_rate"]
                      - rule["handover"]["success_rate"])

    if continuity_delta < -0.5:
        status, reason = "UNDERPERFORMING", (
            "The learned policy regresses session continuity against the tuned "
            "rule-based controller.")
    elif gain < 0:
        status, reason = "UNDERPERFORMING", (
            "The learned policy has negative aggregate gain versus the tuned "
            "rule-based controller.")
    elif gain < material_gain_pct:
        status, reason = "NO MATERIAL BENEFIT", (
            "Gain is below the materiality threshold; the rule-based "
            "controller is sufficient for this scenario.")
    else:
        status, reason = f"PASS — {learned.upper()} preferred", (
            "The learned policy exceeds the material-gain threshold without "
            "regressing session continuity.")
    return {
        "status": status, "reason": reason,
        "best_learned": learned,
        "gain_vs_rule_based_pct": gain,
        "continuity_delta_pp": round(continuity_delta, 3),
        "handover_success_delta_pp": round(ho_delta, 3),
        "recommended_policy": learned if status.startswith("PASS") else "rule_based",
    }


# ==========================================================================
# Feasibility
# ==========================================================================

def feasibility_verdict(result: Dict, twin: MultiTierTwin) -> Dict:
    """Say whether the deployment can meet demand at all, and if not, why.

    The point the user raised: if no tier has the capacity or the coverage, no
    amount of handover-threshold tuning will fix it. This separates the two
    failure modes so the UI can say "add spectrum/sites" rather than
    "re-run the optimizer".
    """
    offered = {s: result["offered_mbps"][s] for s in SESSION_TYPES}
    coverage_gap = result["coverage_gap_mbps"]
    capacity_gap = result["capacity_gap_mbps"]
    total_offered = max(1e-9, sum(offered.values()))
    coverage_frac = sum(coverage_gap.values()) / total_offered
    capacity_frac = sum(capacity_gap.values()) / total_offered
    headroom = result.get("tier_headroom_mbps", {})
    total_headroom = sum(headroom.values())

    limited_by = []
    if coverage_frac > 0.05:
        limited_by.append("coverage")
    if capacity_frac > 0.10 and total_headroom < 0.1 * total_offered:
        limited_by.append("capacity")

    if not limited_by:
        status = "FEASIBLE"
        note = ("Demand is served within the deployed tiers; threshold tuning "
                "is the right lever for the remaining gap.")
        remedy = "tune"
    elif limited_by == ["coverage"]:
        status = "COVERAGE-LIMITED"
        note = (f"{coverage_frac:.0%} of offered demand is from sessions no "
                "tier can reach. Handover thresholds cannot recover this.")
        remedy = "add_sites_or_lowband_or_ntn"
    elif limited_by == ["capacity"]:
        status = "CAPACITY-LIMITED"
        note = (f"{capacity_frac:.0%} of offered demand exceeds the PRB capacity "
                "of the reachable tiers, which have no headroom left.")
        remedy = "add_spectrum_or_cells"
    else:
        status = "COVERAGE-AND-CAPACITY-LIMITED"
        note = ("Both a coverage hole and a capacity shortfall are present; "
                "the deployment is under-provisioned for this demand.")
        remedy = "add_sites_and_spectrum"

    return {
        "status": status,
        "note": note,
        "remedy": remedy,
        "coverage_gap_fraction": round(coverage_frac, 4),
        "capacity_gap_fraction": round(capacity_frac, 4),
        "coverage_gap_mbps": coverage_gap,
        "capacity_gap_mbps": capacity_gap,
        "tier_headroom_mbps": headroom,
        "thresholds_can_help": status == "FEASIBLE",
    }


# ==========================================================================
# Multi-objective optimization
# ==========================================================================

# A "knob" is one tunable parameter: where it applies, and the grid to try.
#   ("a3",  session_type, param)   -> twin.set_threshold_override
#   ("cio", tier_name,   None)     -> twin.set_cell_offsets
#   ("steer", steering_field, None)-> twin.set_steering_override
#   ("minrsrp", session_type, None)-> twin.set_min_rsrp_override
KNOB_GRIDS: Dict[str, Sequence[float]] = {
    ("a3", "a3_offset_db"): (1.0, 2.0, 3.0, 4.5, 6.0),
    ("a3", "a3_hysteresis_db"): (0.5, 1.0, 2.0, 3.0),
    ("a3", "a3_time_to_trigger_ms"): (40.0, 100.0, 160.0, 320.0, 640.0),
    ("cio", None): (-6.0, 0.0, 4.0, 8.0, 12.0, 16.0),
    ("steer", "ntn_speed_threshold_kmh"): (60.0, 90.0, 120.0, 160.0, 220.0),
    ("steer", "ntn_coverage_hole_dbm"): (-126.0, -120.0, -114.0, -108.0),
    ("steer", "wifi_indoor_bias_db"): (2.0, 6.0, 10.0, 14.0),
    ("minrsrp", None): (-135.0, -130.0, -125.0, -120.0),
}


def _objective_score(name: str, r: Dict) -> float:
    """Scalar to maximize, per optimize target.

    Every objective still subtracts a continuity penalty, so none of them can
    'win' by stranding sessions.
    """
    cont = r["session_continuity"]["overall"]
    ho = r["handover"]
    served = sum(r["served_mbps"].values())
    guard = 2.0 * (1 - cont) + 1.0 * ho["ping_pong_rate"]
    if name == "handover":
        return (1.5 * cont + 1.0 * ho["success_rate"]
                - 0.8 * ho["ping_pong_rate"] - 0.5 * ho["radio_link_failures"] / 1000.0
                + 0.5 * r["mean_reward"])
    if name == "throughput":
        return served / 1000.0 + 0.5 * r["mean_satisfaction"]["eMBB"] - guard
    if name == "latency":
        # Lower access latency is the headline; URLLC delivery is the guard-rail.
        return (-r["mean_access_latency_ms"] / 5.0
                + 0.5 * r["mean_satisfaction"]["URLLC"] - guard)
    if name == "coverage":
        total = max(1e-9, sum(r["offered_mbps"].values()))
        covered = 1 - sum(r["coverage_gap_mbps"].values()) / total
        return 2.0 * covered + 1.0 * cont - guard
    raise ValueError(f"Unknown objective: {name}")


OBJECTIVES: Dict[str, Dict] = {
    "handover": {
        "label": "Handover thresholds",
        "knobs": [("a3", s, p) for s in SESSION_TYPES
                  for p in ("a3_offset_db", "a3_hysteresis_db", "a3_time_to_trigger_ms")],
    },
    "throughput": {
        "label": "Throughput (eMBB)",
        "knobs": [("cio", t, None) for t in
                  ("macro_mid", "umb_6g", "mmwave", "macro_low")]
                 + [("steer", "wifi_indoor_bias_db", None)],
    },
    "latency": {
        "label": "Latency (URLLC)",
        "knobs": [("a3", "URLLC", "a3_time_to_trigger_ms"),
                  ("a3", "URLLC", "a3_offset_db"),
                  ("steer", "ntn_speed_threshold_kmh", None),
                  ("cio", "umb_6g", None), ("cio", "mmwave", None)],
    },
    "coverage": {
        "label": "Coverage (mIoT / edge)",
        "knobs": [("steer", "ntn_coverage_hole_dbm", None),
                  ("steer", "ntn_speed_threshold_kmh", None),
                  ("cio", "macro_low", None),
                  ("minrsrp", "mIoT", None), ("minrsrp", "eMBB", None)],
    },
}


def _grid_key(knob: Tuple[str, str, Optional[str]]) -> Tuple[str, Optional[str]]:
    """Map a knob to its KNOB_GRIDS key: A3 by param, steer by field, others flat."""
    kind, target, param = knob
    if kind == "a3":
        return (kind, param)
    if kind == "steer":
        return (kind, target)
    return (kind, None)


def _apply_knobs(twin: MultiTierTwin, values: Dict) -> None:
    """Push the current knob assignment into the twin's override hooks."""
    a3: Dict[str, Dict[str, float]] = {}
    cio: Dict[str, float] = {}
    steer: Dict[str, float] = {}
    minrsrp: Dict[str, float] = {}
    for (kind, target, param), value in values.items():
        if kind == "a3":
            a3.setdefault(target, {})[param] = value
        elif kind == "cio":
            cio[target] = value
        elif kind == "steer":
            steer[target] = value
        elif kind == "minrsrp":
            minrsrp[target] = value
    if a3:
        twin.set_threshold_override(a3)
    if cio:
        twin.set_cell_offsets(cio)
    if steer:
        twin.set_steering_override(**steer)
    if minrsrp:
        twin.set_min_rsrp_override(minrsrp)


def _knob_default(twin: MultiTierTwin, kind: str, target: str, param: Optional[str]):
    if kind == "a3":
        stype = next(t for t in twin.session_types if t.name == target)
        return getattr(stype, param)
    if kind == "cio":
        ti = twin.tier_index.get(target)
        return float(twin.cell_cio_db[twin.cell_tier == ti][0]) if ti is not None else 0.0
    if kind == "steer":
        return getattr(twin.cfg.steering, target)
    if kind == "minrsrp":
        stype = next(t for t in twin.session_types if t.name == target)
        return stype.min_rsrp_dbm
    return 0.0


def optimize(payload: Optional[dict] = None) -> Dict:
    """Coordinate search for one objective: handover | throughput | latency | coverage.

    Each objective tunes a different set of knobs (A3/A5 thresholds, per-tier
    cell individual offsets, steering thresholds, per-class minimum RSRP) and is
    scored on that KPI, always net of a session-continuity guard. Coordinate
    descent, one pass by default.
    """
    payload = dict(payload or {})
    scenario_name = payload.get("scenario", "urban_dense")
    objective = payload.pop("objective", "handover")
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {sorted(OBJECTIVES)}")
    passes = int(payload.pop("passes", 1))
    eval_seed = int(payload.get("seed", TwinConfig.seed)) + 500

    twin = MultiTierTwin(build_config(payload))
    policy = RULE_BASED["rule_based"]
    knobs = [k for k in OBJECTIVES[objective]["knobs"]
             if k[0] != "cio" or k[1] in twin.tier_index]

    def run() -> Dict:
        obs = twin.reset(eval_seed)
        result = evaluate(twin, policy, eval_seed)
        return result

    current = {k: _knob_default(twin, *k) for k in knobs}
    initial = dict(current)
    _apply_knobs(twin, current)
    baseline = run()
    baseline_score = _objective_score(objective, baseline)

    feas = feasibility_verdict(baseline, twin)
    history: List[Dict] = []
    best_score = baseline_score
    evaluations = 1
    start = time.perf_counter()
    for _ in range(passes):
        for knob in knobs:
            kind, target, param = knob
            grid = KNOB_GRIDS[_grid_key(knob)]
            best_value = current[knob]
            for value in grid:
                if value == best_value:
                    continue
                trial = dict(current)
                trial[knob] = value
                _apply_knobs(twin, trial)
                result = run()
                evaluations += 1
                score = _objective_score(objective, result)
                history.append({
                    "knob": f"{kind}:{target}:{param or ''}".rstrip(":"),
                    "value": value, "score": round(score, 5),
                    "objective_kpi": _objective_kpi(objective, result),
                    "continuity": round(result["session_continuity"]["overall"], 4),
                })
                if score > best_score:
                    best_score, best_value = score, value
            current[knob] = best_value
            _apply_knobs(twin, current)

    tuned = run()
    return {
        "implementation": "multi_objective_coordinate_search",
        "objective": objective,
        "objective_label": OBJECTIVES[objective]["label"],
        "scenario": scenario_name,
        "scenario_label": SCENARIOS[scenario_name]["label"],
        "feasibility": feas,
        "thresholds_can_help": feas["thresholds_can_help"],
        "evaluations": evaluations,
        "search_seconds": round(time.perf_counter() - start, 2),
        "knobs": [f"{k[0]}:{k[1]}:{k[2] or ''}".rstrip(":") for k in knobs],
        "baseline": {
            "values": {f"{k[0]}:{k[1]}:{k[2] or ''}".rstrip(":"): v
                       for k, v in initial.items()},
            "score": round(baseline_score, 5),
            "objective_kpi": _objective_kpi(objective, baseline),
            "continuity": round(baseline["session_continuity"]["overall"], 4),
            "handover_success": round(baseline["handover"]["success_rate"], 4),
            "mean_access_latency_ms": baseline["mean_access_latency_ms"],
            "served_mbps": baseline["served_mbps"],
        },
        "optimized": {
            "values": {f"{k[0]}:{k[1]}:{k[2] or ''}".rstrip(":"): v
                       for k, v in current.items()},
            "score": round(best_score, 5),
            "objective_kpi": _objective_kpi(objective, tuned),
            "continuity": round(tuned["session_continuity"]["overall"], 4),
            "handover_success": round(tuned["handover"]["success_rate"], 4),
            "mean_access_latency_ms": tuned["mean_access_latency_ms"],
            "served_mbps": tuned["served_mbps"],
        },
        "improvement": {
            "score": round(best_score - baseline_score, 5),
            "objective_kpi_delta": round(
                _objective_kpi(objective, tuned) - _objective_kpi(objective, baseline), 4),
            "continuity_pp": round(
                100 * (tuned["session_continuity"]["overall"]
                       - baseline["session_continuity"]["overall"]), 3),
        },
        "history": history,
        "limitations": LIMITATIONS + [
            "Coordinate search over one pass is not a global optimum.",
            "Tuned against a single evaluation seed, so fitted to that traffic.",
            "When feasibility is not FEASIBLE, threshold tuning cannot close the "
            "gap and the reported improvement will be small by construction.",
        ],
    }


def _objective_kpi(objective: str, r: Dict) -> float:
    """The single headline number the objective is trying to move."""
    if objective == "handover":
        return round(r["handover"]["success_rate"], 4)
    if objective == "throughput":
        return round(sum(r["served_mbps"].values()), 1)
    if objective == "latency":
        return round(r["mean_access_latency_ms"], 3)
    if objective == "coverage":
        total = max(1e-9, sum(r["offered_mbps"].values()))
        return round(1 - sum(r["coverage_gap_mbps"].values()) / total, 4)
    return 0.0


# Back-compatible name.
def optimize_thresholds(payload: Optional[dict] = None) -> Dict:
    payload = dict(payload or {})
    payload.setdefault("objective", "handover")
    return optimize(payload)


if __name__ == "__main__":
    print(json.dumps(run_comparison({"train_episodes": 6}), indent=2)[:4000])
