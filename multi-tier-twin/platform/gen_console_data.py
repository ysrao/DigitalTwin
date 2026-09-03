"""Precompute preview / compare / optimize responses for every built-in scenario,
emitted as a JS object literal for the standalone browser console."""

import argparse
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
import re

from compare import OBJECTIVES, SCENARIOS, optimize, preview, run_comparison

FAST = dict(train_episodes=8, episode_steps=12, control_interval_ticks=25)
VALIDATION_SEEDS = [3080, 3081, 3082, 3083, 3084]
MODEL_VERSION = "multi-tier-twin-v2"


def _mean_ci95(values):
    """Mean and two-sided 95% Student-t CI (df=4 for five seeds)."""
    mean = statistics.fmean(values)
    if len(values) < 2:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean}
    t_critical = 2.776 if len(values) == 5 else 1.96
    half = t_critical * statistics.stdev(values) / math.sqrt(len(values))
    return {"mean": mean, "ci95_low": mean - half, "ci95_high": mean + half}


def _compact_run(data):
    return {policy: {
        "mean_reward": result["mean_reward"],
        "handover_success": result["handover"]["success_rate"],
        "session_continuity": result["session_continuity"]["overall"],
        "mean_access_latency_ms": result["mean_access_latency_ms"],
    } for policy, result in data["results"].items()}


def _validation(runs):
    compact = [{"seed": seed, "policies": _compact_run(data)}
               for seed, data in runs]
    summary = {}
    for policy in compact[0]["policies"]:
        summary[policy] = {}
        for metric in compact[0]["policies"][policy]:
            summary[policy][metric] = _mean_ci95([
                run["policies"][policy][metric] for run in compact])
    ppo_wins = sum(run["policies"]["ppo"]["mean_reward"] >
                   run["policies"]["dqn"]["mean_reward"] for run in compact)
    dqn_wins = sum(run["policies"]["dqn"]["mean_reward"] >
                   run["policies"]["ppo"]["mean_reward"] for run in compact)
    return {
        "seeds": VALIDATION_SEEDS,
        "training_episodes_per_seed": FAST["train_episodes"],
        "scope": "exploratory synthetic sensitivity; not convergence evidence",
        "runs": compact,
        "summary": summary,
        "ppo_reward_wins": ppo_wins,
        "dqn_reward_wins": dqn_wins,
        "ties": len(compact) - ppo_wins - dqn_wins,
    }


def build() -> dict:
    config_id = hashlib.sha256(json.dumps({"model_version": MODEL_VERSION,
                                          "fast": FAST, "seeds": VALIDATION_SEEDS,
                                          "nr_grid_revision": "explicit-scs-v2"},
                                          sort_keys=True).encode()).hexdigest()[:12]
    out = {"metadata": {
               "model_version": MODEL_VERSION,
               "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
               "config_id": config_id,
               "generator": "platform/gen_console_data.py",
               "execution": "precomputed",
           },
           "scenarios": [{"id": k, "label": v["label"], "description": v["description"]}
                         for k, v in SCENARIOS.items()],
           "preview": {}, "compare": {}, "validation": {}, "optimize": {}}
    for name in SCENARIOS:
        sys.stderr.write(f"[{name}] preview\n")
        out["preview"][name] = preview({"scenario": name})
        runs = []
        for seed in VALIDATION_SEEDS:
            sys.stderr.write(f"[{name}] compare seed:{seed}\n")
            runs.append((seed, run_comparison({"scenario": name, "seed": seed, **FAST})))
        out["compare"][name] = runs[0][1]
        out["validation"][name] = _validation(runs)
        out["optimize"][name] = {}
        for obj in OBJECTIVES:
            sys.stderr.write(f"[{name}] optimize:{obj}\n")
            out["optimize"][name][obj] = optimize(
                {"scenario": name, "objective": obj,
                 "episode_steps": 8, "control_interval_ticks": 20})
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-release", action="store_true",
                        help="write the JSON dataset and refresh twin_console.html")
    parser.add_argument("--refresh-console", action="store_true",
                        help="refresh twin_console.html from the existing v2 JSON dataset")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dataset_path = root / "validation_dataset_v2.json"
    data = (json.loads(dataset_path.read_text(encoding="utf-8"))
            if args.refresh_console else build())
    literal = "window.PRECOMPUTED = " + json.dumps(data, separators=(",", ":")) + ";"
    if not args.write_release and not args.refresh_console:
        print(literal)
    else:
        if args.write_release:
            dataset_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        console_path = root / "twin_console.html"
        console = console_path.read_text(encoding="utf-8")
        updated, count = re.subn(
            r"<script>window\.PRECOMPUTED = .*?</script>",
            lambda _: "<script>" + literal + "</script>",
            console, count=1, flags=re.DOTALL)
        if count != 1:
            raise RuntimeError("expected one PRECOMPUTED block in twin_console.html")
        console_path.write_text(updated, encoding="utf-8")
        if args.write_release:
            sys.stderr.write(f"wrote {dataset_path}\n")
        sys.stderr.write(f"updated {console_path}\n")
