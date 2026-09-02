"""Precompute preview / compare / optimize responses for every built-in scenario,
emitted as a JS object literal for the standalone browser console."""

import json
import sys

from compare import OBJECTIVES, SCENARIOS, optimize, preview, run_comparison

FAST = dict(train_episodes=8, episode_steps=12, control_interval_ticks=25)


def build() -> dict:
    out = {"scenarios": [{"id": k, "label": v["label"], "description": v["description"]}
                         for k, v in SCENARIOS.items()],
           "preview": {}, "compare": {}, "optimize": {}}
    for name in SCENARIOS:
        sys.stderr.write(f"[{name}] preview\n")
        out["preview"][name] = preview({"scenario": name})
        sys.stderr.write(f"[{name}] compare\n")
        out["compare"][name] = run_comparison({"scenario": name, **FAST})
        out["optimize"][name] = {}
        for obj in OBJECTIVES:
            sys.stderr.write(f"[{name}] optimize:{obj}\n")
            out["optimize"][name][obj] = optimize(
                {"scenario": name, "objective": obj,
                 "episode_steps": 8, "control_interval_ticks": 20})
    return out


if __name__ == "__main__":
    print("window.PRECOMPUTED = " + json.dumps(build(), separators=(",", ":")) + ";")
