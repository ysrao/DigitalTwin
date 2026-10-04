"""Stage C corpus generation.

Runs the Stage-B baseline across a grid of layouts x traffic x seeds,
recording per-epoch per-cell per-slice KPIs. Output: versioned .npz files +
manifest. Held-out layouts (by seed) are reserved for the transfer test —
no area/time leakage by construction.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
from datetime import date
from pathlib import Path

import numpy as np
import yaml

from ..scenarios.loader import from_dict
from ..sim.baseline import BaselineSimulator

VERSION = "corpus_v1"


def _variants(base: dict, traffic_scales=(0.8, 1.2)):
    grid = {
        "sc_count": [6, 12],
        "sc_placement": ["uniform", "hotspot"],
        "n_ues": [60, 90],
        "traffic_scale": list(traffic_scales),
    }
    keys = list(grid)
    for i, combo in enumerate(itertools.product(*[grid[k] for k in keys])):
        cfg = dict(zip(keys, combo))
        d = copy.deepcopy(base)
        d["network"]["small_cells"]["count"] = cfg["sc_count"]
        d["network"]["small_cells"]["placement"] = cfg["sc_placement"]
        d["scenario"]["name"] = f"corpus-{i:02d}"
        yield i, cfg, d


def _static_features(cells: list) -> np.ndarray:
    feats = []
    for c in cells:
        t = [1.0 if c["type"] == k else 0.0 for k in ("macro", "pico", "femto")]
        feats.append([c["x"] / 1000.0, c["y"] / 1000.0, *t,
                      c["tx_power_dbm"] / 50.0, c["total_dl_prbs"] / 300.0,
                      len(c["carriers"]) / 4.0])
    return np.array(feats, dtype=np.float32)


def generate(out_dir: str | Path, n_slots: int = 1000,
             seeds_per_cfg: int = 2, base_yaml: str = "configs/scenario_schema_v1.yaml",
             quick: bool = False, budget_policy: str = "reactive",
             traffic_scales: tuple = (0.8, 1.2),
             version: str = VERSION) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(open(base_yaml))
    runs = []
    for i, cfg, d in _variants(base, traffic_scales):
        if quick and i >= 2:
            break
        for s in range(seeds_per_cfg):
            seed = 1000 + i * 10 + s
            scenario = from_dict(d)
            sim = BaselineSimulator(scenario, n_ues=cfg["n_ues"],
                                    n_slots=n_slots, seed=seed,
                                    budget_policy=budget_policy)
            # traffic scale: smaller inter-arrival = more load
            sim.traffic.rates = {k: v / cfg["traffic_scale"]
                                 for k, v in sim.traffic.rates.items()}
            sim.prach.p_access = min(1.0, 0.3 * cfg["traffic_scale"])
            snaps = []
            sim.recorder = lambda ep, snap: snaps.append(  # noqa: E731
                {k: v for k, v in snap.items()})
            rep = sim.run()
            E = len(snaps)
            C = len(sim.cells)
            S = len(sim._slice_names)
            arr = lambda k: np.stack([sn[k] for sn in snaps]).astype(np.float32)
            data = {
                "static": _static_features(sim.cells),
                "offered": arr("offered"), "served": arr("served"),
                "viol": arr("viol"), "prb_used": arr("prb_used"),
                "backlog": arr("backlog"), "budgets": arr("budgets"),
                "delay_mean": arr("delay_mean_slots"),
                "slice_names": np.array(sim._slice_names),
                "seed": np.int64(seed),
            }
            cfg_hash = hashlib.sha1(
                json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
            fname = f"run_{i:02d}_seed{seed}_{cfg_hash}.npz"
            np.savez_compressed(out / fname, **data)
            runs.append({"file": fname, "config_id": i, "seed": seed,
                         "cfg": cfg, "epochs": E, "cells": C,
                         "violations_total": sum(
                             rep[s]["violations"] for s in sim._slice_names)})
            print(f"run {i:02d} seed {seed}: {E} epochs, {C} cells, "
                  f"viol={runs[-1]['violations_total']}", flush=True)

    # held-out layouts: last quarter of config ids (unseen areas for transfer)
    cfg_ids = sorted({r["config_id"] for r in runs})
    n_hold = max(1, len(cfg_ids) // 4)
    held = set(cfg_ids[-n_hold:])
    manifest = {
        "version": version, "date": str(date.today()),
        "n_slots": n_slots, "epoch_slots": 20,
        "budget_policy": budget_policy,
        "runs": runs,
        "held_out_config_ids": sorted(held),
        "train_config_ids": sorted(set(cfg_ids) - held),
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"corpus {version}: {len(runs)} runs, "
          f"held-out layouts {sorted(held)}")
    return out
