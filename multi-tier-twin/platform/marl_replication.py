"""Independent re-implementation of a centralized-PPO vs cooperative-MARL comparison,
matched to a written protocol, run against the EXISTING `engine.py` decision engine.

This module does not modify engine.py, agents.py, or twin.py. It imports and reuses:
  - engine.ActorCritic         (the 13/7-input -> 64 -> 64 -> N categorical PPO net+critic)
  - engine.TEMPLATES           (the 9 discrete joint PRB-share templates)
  - engine.state_vector        (the 13-D centralized observation)
  - engine.demand_and_capacity (allocation / demand / capacity / served / satisfaction)
  - engine.load_rows, engine.Scenario, engine.SLICE_NAMES, engine.f

It implements a NEW reward function (does not use engine.outcome()'s reward), a NEW
episodic PPO training loop (does not use engine.train_ppo, which is a random-batch
sampler), a NEW 3-agent cooperative MARL scheme with a deterministic coordinator, and
a NEW paired-evaluation + bootstrap statistics harness.

See the module docstring sections below and the final report for every place a
judgment call was made where the source paper's protocol was underspecified.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from engine import ActorCritic, TEMPLATES, state_vector, demand_and_capacity, load_rows, Scenario, SLICE_NAMES, f

# --------------------------------------------------------------------------------------
# Judgment-call constants (all documented in the final report)
# --------------------------------------------------------------------------------------
PROFILE = "P1_balanced_busy_hour"          # engine.Scenario's own default profile
EPISODES = 260
STEPS_PER_EPISODE = 48
GAMMA = 0.97
GAE_LAMBDA = 0.95
PPO_CLIP = 0.2                             # matches ActorCritic.update's own default
TRAIN_SEEDS = list(range(1301, 1311))      # 10 training seeds, as specified
EVAL_SEEDS_FULL = list(range(5001, 5041))  # 40 held-out traffic seeds, disjoint from TRAIN_SEEDS
EVAL_SEEDS_SMOKE = list(range(5001, 5006)) # 5 held-out seeds for the smoke test

SLA_THRESHOLDS = np.array([0.95, 0.999, 0.99])   # eMBB, URLLC, mIoT
REWARD_WEIGHTS = np.array([0.35, 0.45, 0.20])
UTIL_WEIGHT = 0.05
PENALTY_WEIGHTS = np.array([0.70, 1.30, 0.80])
CHURN_WEIGHT = 0.025

K_LEVELS = 5
BID_LEVELS = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
FLOORS = np.array([0.25, 0.20, 0.10])            # eMBB, URLLC, mIoT deterministic-coordinator floors
RESIDUAL_BUDGET = 1.0 - FLOORS.sum()             # 0.45

N_INPUTS_CENTRAL = 13
N_ACTIONS_CENTRAL = len(TEMPLATES)               # 9
N_INPUTS_MARL = 7


# --------------------------------------------------------------------------------------
# Shared reward (NEW - does not reuse engine.outcome())
# --------------------------------------------------------------------------------------
def per_slice_satisfaction(row: dict, shares: np.ndarray, scenario: Scenario):
    """Wraps engine.demand_and_capacity, but forces satisfaction=1.0 when there is no
    offered demand for a slice in this interval (per the reward spec), since
    demand_and_capacity's own division-by-eps would otherwise yield ~0 there."""
    allocations, demand, capacity, served, sat = demand_and_capacity(row, shares, scenario)
    sat = sat.copy()
    no_demand = demand <= 1e-9
    sat[no_demand] = 1.0
    return sat, demand, capacity, served


def new_reward(sat: np.ndarray, util: float, shares: np.ndarray, prev_shares: np.ndarray) -> float:
    base = float(REWARD_WEIGHTS @ sat) + UTIL_WEIGHT * util
    deficits = np.maximum(0.0, SLA_THRESHOLDS - sat)
    penalty = float(PENALTY_WEIGHTS @ deficits)
    churn = CHURN_WEIGHT * float(np.abs(shares - prev_shares).sum())
    return base - penalty - churn


# --------------------------------------------------------------------------------------
# GAE(lambda) - shared by both controllers' episodic training loops
# --------------------------------------------------------------------------------------
def compute_gae(rewards: np.ndarray, values: np.ndarray, gamma=GAMMA, lam=GAE_LAMBDA):
    """Rollout is treated as a complete episode: bootstrap value after the final
    (48th) step is 0 (done=True at episode end). Standard backward GAE recursion."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float64)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * lam * last_gae
        adv[t] = last_gae
    returns = adv + values
    return adv, returns


def _normalize(adv: np.ndarray) -> np.ndarray:
    return (adv - adv.mean()) / (adv.std() + 1e-8)


# --------------------------------------------------------------------------------------
# Centralized controller: reuses engine.ActorCritic(13 -> 64 -> 64 -> 9) and
# engine.state_vector / engine.TEMPLATES directly.
# --------------------------------------------------------------------------------------
def train_centralized(training_seed: int, profile: str = PROFILE, episodes: int = EPISODES,
                       steps: int = STEPS_PER_EPISODE) -> Tuple[ActorCritic, List[float]]:
    rows = load_rows(profile)
    scenario = Scenario(profile=profile)
    n = len(rows)
    model = ActorCritic(training_seed, n_inputs=N_INPUTS_CENTRAL, n_actions=N_ACTIONS_CENTRAL)
    act_rng = np.random.default_rng(training_seed * 10 + 1)  # independent stream from net init
    cursor = 0
    reward_history = []
    for _ in range(episodes):
        prev_onehot = np.zeros(N_ACTIONS_CENTRAL); prev_onehot[0] = 1.0   # reset each episode
        prev_shares = TEMPLATES[0].copy()                                  # reset each episode
        states, actions, oldp, rewards, values = [], [], [], [], []
        for t in range(steps):
            row = rows[(cursor + t) % n]
            state = state_vector(row, prev_onehot, scenario)
            probs = model.probs(state[None, :])[0]
            a = int(act_rng.choice(N_ACTIONS_CENTRAL, p=probs))
            shares = TEMPLATES[a]
            sat, demand, capacity, served = per_slice_satisfaction(row, shares, scenario)
            util = served.sum() / max(1e-9, capacity.sum())
            r = new_reward(sat, util, shares, prev_shares)
            v = float(model.values(state[None, :])[0])
            states.append(state); actions.append(a); oldp.append(probs[a])
            rewards.append(r); values.append(v)
            prev_onehot = np.zeros(N_ACTIONS_CENTRAL); prev_onehot[a] = 1.0
            prev_shares = shares
        cursor = (cursor + steps) % n
        rewards_arr = np.array(rewards); values_arr = np.array(values)
        adv, returns = compute_gae(rewards_arr, values_arr)
        adv = _normalize(adv)
        model.update(np.array(states), np.array(actions), np.array(oldp), adv, returns)
        reward_history.append(float(rewards_arr.mean()))
    return model, reward_history


def eval_centralized(model: ActorCritic, eval_seed: int, profile: str = PROFILE,
                      steps: int = STEPS_PER_EPISODE) -> Dict:
    rows = load_rows(profile)
    scenario = Scenario(profile=profile)
    n = len(rows)
    rng = np.random.default_rng(eval_seed)
    start = int(rng.integers(0, n))
    prev_onehot = np.zeros(N_ACTIONS_CENTRAL); prev_onehot[0] = 1.0
    prev_shares = TEMPLATES[0].copy()
    rewards, sats, utils, churns = [], [], [], []
    for t in range(steps):
        row = rows[(start + t) % n]
        state = state_vector(row, prev_onehot, scenario)
        probs = model.probs(state[None, :])[0]
        a = int(np.argmax(probs))  # greedy, no exploration
        shares = TEMPLATES[a]
        sat, demand, capacity, served = per_slice_satisfaction(row, shares, scenario)
        util = served.sum() / max(1e-9, capacity.sum())
        r = new_reward(sat, util, shares, prev_shares)
        churns.append(float(np.abs(shares - prev_shares).sum()))
        rewards.append(r); sats.append(sat); utils.append(util)
        prev_onehot = np.zeros(N_ACTIONS_CENTRAL); prev_onehot[a] = 1.0
        prev_shares = shares
    sats = np.array(sats)
    return {
        "reward": float(np.mean(rewards)),
        "sat_embb": float(sats[:, 0].mean()),
        "sat_urllc": float(sats[:, 1].mean()),
        "sat_miot": float(sats[:, 2].mean()),
        "utilization": float(np.mean(utils)),
        "churn": float(np.mean(churns)),
    }


# --------------------------------------------------------------------------------------
# Cooperative MARL: 3 independent engine.ActorCritic(7 -> 64 -> 64 -> 5) agents + a
# deterministic coordinator. Discrete bid-intensity levels approximate the source
# paper's continuous bids (disclosed difference; see report).
# --------------------------------------------------------------------------------------
def marl_obs(row: dict, slice_idx: int, prev_shares: np.ndarray, prev_sat: np.ndarray,
             scenario: Scenario, step_idx: int, steps: int) -> np.ndarray:
    """7-D per-slice observation. Features 1-6 mirror engine.state_vector's own
    normalization conventions for continuity with the centralized controller's inputs.
    Feature 7 (own SLA deficit indicator) is this implementation's disclosed choice."""
    prbs = max(1.0, f(row, "available_prbs", scenario.prbs_per_cell))
    embb_demand = f(row, "embb_offered_dl_mbps")
    urllc_demand = f(row, "urllc_packets") * f(row, "urllc_payload_bytes", 32) * 8 / max(1, f(row, "interval_ms", 10)) / 1000
    iot_demand = f(row, "iot_active_devices") * f(row, "iot_min_rate_kbps_per_device", 0.2048) / 1000
    demands = np.array([embb_demand, urllc_demand, iot_demand])
    embb_se = f(row, "embb_spectral_eff_bps_hz", 1)
    urllc_se = f(row, "urllc_spectral_eff_bps_hz", 1)
    iot_se = f(row, "iot_spectral_eff_bps_hz", 0.3)
    ses = np.array([embb_se, urllc_se, iot_se])
    se_ref = np.array([4.0, 2.0, 0.8])
    full_cap = max(1, scenario.prbs_per_cell) * 0.36 * ses   # capacity if slice had 100% of PRBs
    demand_cap_ratio = np.minimum(4.0, demands / np.maximum(full_cap, 1e-6))

    own_ratio = float(demand_cap_ratio[slice_idx])
    own_se_ind = float(min(2.0, ses[slice_idx] / se_ref[slice_idx]))
    own_prev_share = float(prev_shares[slice_idx])
    agg_load = float(min(4.0, demand_cap_ratio.sum()) / 3.0)
    resource_avail = float(min(2.0, prbs / max(1, scenario.prbs_per_cell)))
    phase = float(step_idx / max(1, steps))
    own_deficit = float(max(0.0, SLA_THRESHOLDS[slice_idx] - prev_sat[slice_idx]))
    return np.array([own_ratio, own_se_ind, own_prev_share, agg_load, resource_avail, phase, own_deficit])


def coordinate(levels: np.ndarray) -> np.ndarray:
    """Deterministic coordinator: clip each slice's bid to (at least) its floor, then
    renormalize the residual budget (1 - sum(floors)) in proportion to the
    bids-above-floor. Falls back to splitting the residual by floor-proportion when
    every bid sits at/below its own floor (disclosed edge case)."""
    bids = np.asarray(levels, dtype=np.float64)
    clipped = np.maximum(bids, FLOORS)
    above = clipped - FLOORS
    total_above = above.sum()
    if total_above < 1e-12:
        extra = RESIDUAL_BUDGET * (FLOORS / FLOORS.sum())
    else:
        extra = RESIDUAL_BUDGET * (above / total_above)
    shares = FLOORS + extra
    return shares / shares.sum()


def train_marl(training_seed: int, profile: str = PROFILE, episodes: int = EPISODES,
               steps: int = STEPS_PER_EPISODE) -> Tuple[List[ActorCritic], List[float]]:
    rows = load_rows(profile)
    scenario = Scenario(profile=profile)
    n = len(rows)
    # Each slice-agent gets its own network-init seed (seed, seed+1000, seed+2000) and
    # its own independent action-sampling RNG stream, so the three agents are truly
    # independent, not sharing any random source.
    agents = [ActorCritic(training_seed + off, n_inputs=N_INPUTS_MARL, n_actions=K_LEVELS)
              for off in (0, 1000, 2000)]
    act_rngs = [np.random.default_rng(training_seed * 10 + 2 + i) for i in range(3)]
    cursor = 0
    reward_history = []
    init_shares = coordinate(np.zeros(3))  # floors + equal split of residual, used as ep-0 "previous"
    for _ in range(episodes):
        prev_shares = init_shares.copy()
        prev_sat = np.array([1.0, 1.0, 1.0])
        bufs = [{"states": [], "actions": [], "oldp": [], "rewards": [], "values": []} for _ in range(3)]
        for t in range(steps):
            row = rows[(cursor + t) % n]
            obs_list, probs_list, levels = [], [], []
            for i in range(3):
                obs = marl_obs(row, i, prev_shares, prev_sat, scenario, t, steps)
                probs = agents[i].probs(obs[None, :])[0]
                a = int(act_rngs[i].choice(K_LEVELS, p=probs))
                obs_list.append(obs); probs_list.append(probs); levels.append(BID_LEVELS[a])
                bufs[i]["states"].append(obs); bufs[i]["actions"].append(a); bufs[i]["oldp"].append(probs[a])
            shares = coordinate(np.array(levels))
            sat, demand, capacity, served = per_slice_satisfaction(row, shares, scenario)
            util = served.sum() / max(1e-9, capacity.sum())
            r = new_reward(sat, util, shares, prev_shares)  # shared reward, identical for all 3 agents
            for i in range(3):
                v = float(agents[i].values(obs_list[i][None, :])[0])
                bufs[i]["values"].append(v); bufs[i]["rewards"].append(r)
            prev_shares = shares
            prev_sat = sat
        cursor = (cursor + steps) % n
        for i in range(3):
            rewards_arr = np.array(bufs[i]["rewards"]); values_arr = np.array(bufs[i]["values"])
            adv, returns = compute_gae(rewards_arr, values_arr)
            adv = _normalize(adv)
            agents[i].update(np.array(bufs[i]["states"]), np.array(bufs[i]["actions"]),
                              np.array(bufs[i]["oldp"]), adv, returns)
        reward_history.append(float(np.mean([np.mean(b["rewards"]) for b in bufs])))
    return agents, reward_history


def eval_marl(agents: List[ActorCritic], eval_seed: int, profile: str = PROFILE,
              steps: int = STEPS_PER_EPISODE) -> Dict:
    rows = load_rows(profile)
    scenario = Scenario(profile=profile)
    n = len(rows)
    rng = np.random.default_rng(eval_seed)
    start = int(rng.integers(0, n))  # identical draw sequence to eval_centralized -> paired window
    init_shares = coordinate(np.zeros(3))
    prev_shares = init_shares.copy()
    prev_sat = np.array([1.0, 1.0, 1.0])
    rewards, sats, utils, churns = [], [], [], []
    for t in range(steps):
        row = rows[(start + t) % n]
        levels = []
        for i in range(3):
            obs = marl_obs(row, i, prev_shares, prev_sat, scenario, t, steps)
            probs = agents[i].probs(obs[None, :])[0]
            a = int(np.argmax(probs))  # greedy, no exploration
            levels.append(BID_LEVELS[a])
        shares = coordinate(np.array(levels))
        sat, demand, capacity, served = per_slice_satisfaction(row, shares, scenario)
        util = served.sum() / max(1e-9, capacity.sum())
        r = new_reward(sat, util, shares, prev_shares)
        churns.append(float(np.abs(shares - prev_shares).sum()))
        rewards.append(r); sats.append(sat); utils.append(util)
        prev_shares = shares; prev_sat = sat
    sats = np.array(sats)
    return {
        "reward": float(np.mean(rewards)),
        "sat_embb": float(sats[:, 0].mean()),
        "sat_urllc": float(sats[:, 1].mean()),
        "sat_miot": float(sats[:, 2].mean()),
        "utilization": float(np.mean(utils)),
        "churn": float(np.mean(churns)),
    }


# --------------------------------------------------------------------------------------
# Sweep + statistics
# --------------------------------------------------------------------------------------
def run_sweep(train_seeds: List[int], eval_seeds: List[int], profile: str = PROFILE) -> Dict:
    per_seed_rows = []
    all_central, all_marl = [], []
    t_train_total = 0.0
    t_eval_total = 0.0
    for ts in train_seeds:
        t0 = time.perf_counter()
        central_model, central_hist = train_centralized(ts, profile=profile)
        marl_agents, marl_hist = train_marl(ts, profile=profile)
        t_train_total += time.perf_counter() - t0

        t1 = time.perf_counter()
        central_evals = [eval_centralized(central_model, es, profile=profile) for es in eval_seeds]
        marl_evals = [eval_marl(marl_agents, es, profile=profile) for es in eval_seeds]
        t_eval_total += time.perf_counter() - t1

        for e in central_evals: e["training_seed"] = ts
        for e in marl_evals: e["training_seed"] = ts
        all_central.extend(central_evals)
        all_marl.extend(marl_evals)
        per_seed_rows.append({
            "training_seed": ts,
            "centralized_mean_reward": float(np.mean([e["reward"] for e in central_evals])),
            "marl_mean_reward": float(np.mean([e["reward"] for e in marl_evals])),
            "centralized_final_train_reward": central_hist[-1],
            "marl_final_train_reward": marl_hist[-1],
        })

    def agg(evals):
        return {
            "mean_reward": float(np.mean([e["reward"] for e in evals])),
            "mean_sat_embb": float(np.mean([e["sat_embb"] for e in evals])),
            "mean_sat_urllc": float(np.mean([e["sat_urllc"] for e in evals])),
            "mean_sat_miot": float(np.mean([e["sat_miot"] for e in evals])),
            "mean_utilization": float(np.mean([e["utilization"] for e in evals])),
            "mean_churn": float(np.mean([e["churn"] for e in evals])),
            "n": len(evals),
        }

    central_agg = agg(all_central)
    marl_agg = agg(all_marl)

    # Training-seed-clustered bootstrap on mean-reward difference (MARL - centralized)
    rng = np.random.default_rng(999)
    central_by_seed = {ts: [e["reward"] for e in all_central if e["training_seed"] == ts] for ts in train_seeds}
    marl_by_seed = {ts: [e["reward"] for e in all_marl if e["training_seed"] == ts] for ts in train_seeds}
    diffs = []
    seeds_arr = np.array(train_seeds)
    n_boot = 10000
    for _ in range(n_boot):
        draw = rng.choice(seeds_arr, size=len(seeds_arr), replace=True)
        c_vals, m_vals = [], []
        for ts in draw:
            c_vals.extend(central_by_seed[ts])
            m_vals.extend(marl_by_seed[ts])
        diffs.append(np.mean(m_vals) - np.mean(c_vals))
    diffs = np.array(diffs)
    ci_lo, ci_hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

    seeds_marl_wins = sum(1 for r in per_seed_rows if r["marl_mean_reward"] > r["centralized_mean_reward"])
    pct_gain = 100.0 * (marl_agg["mean_reward"] - central_agg["mean_reward"]) / abs(central_agg["mean_reward"])

    return {
        "profile": profile,
        "train_seeds": train_seeds,
        "eval_seeds": eval_seeds,
        "n_paired_evals": len(train_seeds) * len(eval_seeds),
        "aggregate": {"centralized": central_agg, "marl": marl_agg},
        "pct_gain_marl_vs_centralized": pct_gain,
        "per_seed": per_seed_rows,
        "seeds_marl_wins_of": [seeds_marl_wins, len(train_seeds)],
        "bootstrap_ci_95_marl_minus_centralized": [ci_lo, ci_hi],
        "bootstrap_mean_diff": float(diffs.mean()),
        "n_bootstrap": n_boot,
        "wall_clock_seconds": {"train_total": t_train_total, "eval_total": t_eval_total,
                                "total": t_train_total + t_eval_total},
    }


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        result = run_sweep([1301], EVAL_SEEDS_SMOKE)
    else:
        result = run_sweep(TRAIN_SEEDS, EVAL_SEEDS_FULL)
    print(json.dumps(result, indent=2))
