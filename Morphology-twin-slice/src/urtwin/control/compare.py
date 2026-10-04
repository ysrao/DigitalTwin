"""Stage D: matched proactive-vs-reactive comparison.

Same layout + seed => identical traffic/mobility/channel realizations
across policies (budget decisions consume no simulator RNG). Policies:
fixed (equal thirds), reactive (Stage B), twin (MPC over predictor v2).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, "src")
from urtwin.learning.corpus import _variants
from urtwin.scenarios.loader import from_dict
from urtwin.sim.baseline import BaselineSimulator
from urtwin.control.mpc import TwinMPC


def build(config_id: int, seed: int, n_slots: int = 1000,
          traffic_scales=(0.8, 1.2)):
    base = yaml.safe_load(
        Path("configs/scenario_schema_v1.yaml").read_text())
    for i, cfg, d in _variants(base, traffic_scales):
        if i == config_id:
            scenario = from_dict(d)
            return scenario, cfg
    raise ValueError(config_id)


def apply_load(sim, cfg):
    sim.traffic.rates = {k: v / cfg["traffic_scale"]
                         for k, v in sim.traffic.rates.items()}
    sim.prach.p_access = min(1.0, 0.3 * cfg["traffic_scale"])


def metrics(rep: dict, sim=None) -> dict:
    sl = ["eMBB", "URLLC", "mIoT"]
    served = {x: rep[x]["served_kb"] for x in sl}
    viol = {x: rep[x]["violations"] for x in sl}
    prb_used = sum(rep[x]["prb_used"] for x in sl)
    avail = (sum(sim.cell_total_prbs) * rep["slots"] if sim is not None
             else 1.0)
    return {
        "served_kb_total": round(sum(served.values()), 1),
        "served_kb": {x: round(served[x], 1) for x in sl},
        "violations": viol,
        "viol_total": int(sum(viol.values())),
        "prb_used": int(prb_used),
        "prb_util": round(prb_used / avail, 4),
        "eff_kb_per_prb": round(sum(served.values()) / max(1, prb_used), 4),
        "ue_thr_p5_kbps": round(rep["ue_thr_kbps_p5"], 2),
        "ue_thr_mean_kbps": round(rep["ue_thr_kbps_mean"], 2),
        "handovers": rep["handovers"],
    }


def run_fixed(scenario, cfg, seed, n_slots):
    sim = BaselineSimulator(scenario, n_ues=cfg["n_ues"], n_slots=n_slots,
                            seed=seed, budget_policy="external")
    apply_load(sim, cfg)
    thirds = np.tile(sim.cell_total_prbs[:, None] / 3, (1, 3))
    sim.set_budgets(thirds)
    t0 = time.perf_counter()
    rep = sim.run()
    return metrics(rep, sim), time.perf_counter() - t0, {}


def run_reactive(scenario, cfg, seed, n_slots):
    sim = BaselineSimulator(scenario, n_ues=cfg["n_ues"], n_slots=n_slots,
                            seed=seed, budget_policy="reactive")
    apply_load(sim, cfg)
    t0 = time.perf_counter()
    rep = sim.run()
    return metrics(rep, sim), time.perf_counter() - t0, {}


def run_twin(scenario, cfg, seed, n_slots, burst=None,
             ckpt="checkpoints/predictor_v2.pt"):
    m = TwinMPC(scenario, n_ues=cfg["n_ues"], n_slots=n_slots, seed=seed,
                burst=burst, ckpt=ckpt)
    apply_load(m.sim, cfg)
    t0 = time.perf_counter()
    rep = m.run()
    return metrics(rep, m.sim), time.perf_counter() - t0, rep["mpc"]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/predictor_v2.pt")
    ap.add_argument("--scales", default="0.8,1.2")
    ap.add_argument("--out", default="data/stage_d_comparison.json")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    scales = tuple(float(x) for x in a.scales.split(","))
    layouts = [0, 1, 12, 13]  # 2 train + 2 held-out
    results = []
    for lid in layouts:
        for s in range(2):
            seed = 1000 + lid * 10 + s
            scenario, cfg = build(lid, seed, traffic_scales=scales)
            tag = f"layout{lid:02d}_seed{s}"
            for name, fn in (("fixed", run_fixed),
                             ("reactive", run_reactive),
                             ("twin", run_twin)):
                scenario2, _ = build(lid, seed, traffic_scales=scales)
                kw = {"ckpt": a.ckpt} if name == "twin" else {}
                m, wall, extra = fn(scenario2, cfg, seed, 1000, **kw)
                results.append({"tag": tag, "layout": lid, "seed": s,
                                "policy": name, "metrics": m,
                                "wall_s": round(wall, 1), **extra})
                print(f"{tag} {name}: served={m['served_kb_total']} "
                      f"viol={m['viol_total']} eff={m['eff_kb_per_prb']} "
                      f"p5={m['ue_thr_p5_kbps']} wall={wall:.0f}s",
                      flush=True)
    # burst test: eMBB x3 arrival surge, epochs 20-30, layout 12
    for s in range(2):
        seed = 1000 + 12 * 10 + s
        for name, fn in (("reactive", run_reactive), ("twin", run_twin)):
            scenario, cfg = build(12, seed, traffic_scales=scales)
            burst = (20, 30) if name == "twin" or True else None
            if name == "reactive":
                sim = BaselineSimulator(scenario, n_ues=cfg["n_ues"],
                                        n_slots=1000, seed=seed,
                                        budget_policy="reactive")
                apply_load(sim, cfg)
                t0 = time.perf_counter()
                dt_s = sim.slot_ms / 1e3
                epoch = 0
                for slot in range(sim.n_slots):
                    if slot % sim.budget_epoch == 0:
                        sim._update_budgets()
                        sim._epoch_snapshot(epoch)
                        epoch += 1
                    sim.traffic.scale["eMBB"] = (
                        3.0 if 20 <= epoch < 30 else 1.0)
                    sim.ues.step(dt_s, sim.slot_ms)
                    if slot % sim.coupling_refresh_slots == 0 and slot > 0:
                        sim.ues.refresh_coupling(
                            sim.provider.compute(sim.ues.xy, sim.cells))
                    for p in sim.traffic.arrivals(slot):
                        sim._enqueue(p)
                    for p in sim.prach.step(
                            slot, sim.slot_ms,
                            next(x.latency_ms for x in sim.scenario.slices
                                 if x.name == "mIoT")):
                        sim._enqueue(p)
                    sim._schedule(slot)
                    sim._expire(slot)
                rep = sim._report()
                m, wall, extra = metrics(rep, sim), time.perf_counter() - t0, {}
            else:
                m, wall, extra = fn(scenario, cfg, seed, 1000, burst=(20, 30),
                                    ckpt=a.ckpt)
            results.append({"tag": f"burst12_seed{s}", "layout": 12,
                            "seed": s, "policy": name + "_burst",
                            "metrics": m, "wall_s": round(wall, 1), **extra})
            print(f"burst12_seed{s} {name}_burst: served={m['served_kb_total']} "
                  f"viol={m['viol_total']} eff={m['eff_kb_per_prb']}",
                  flush=True)
    out = Path(a.out)
    out.write_text(json.dumps(results, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
