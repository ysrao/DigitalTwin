"""Suburban morphology dimensioning grid: same 9 structures x 2 layouts x 20
seeds as the urban FR3 grid, but macro ISD 1000 m, macro height 35 m and the
TR 38.901 RMa channel for the macro layer (small cells stay UMi-SC). Traffic,
UE counts, carriers and calibrated load scales are identical to the urban run.
Usage: python scripts/run_dimensioning_suburban.py <worker_idx> <n_workers>
Writes data/stage_e_dimensioning_suburban_w<k>.json (resumable)."""
import copy, json, os, sys
from pathlib import Path
import yaml
sys.path.insert(0, "src")
import urtwin.control.dimension as D
from urtwin.learning.corpus import _variants
from urtwin.scenarios.loader import from_dict
from urtwin.sim.baseline import BaselineSimulator

SUB_ISD_M = float(__import__("os").environ.get("SUB_ISD", "1000"))
SUB_SMALL_CELLS = __import__("os").environ.get("SUB_SC")  # None = keep layout default; "0" = rural
SUB_MACRO_H_M = 35.0
SCALES = tuple(json.loads(Path("data/calibration_choice.json").read_text()))
import os
TAG = os.environ.get("SUB_TAG", "")
if os.environ.get("SUB_SCALES"):
    SCALES = tuple(float(x) for x in os.environ["SUB_SCALES"].split(","))


def run_one_suburban(layout, seed, embb_min, urllc_min, scales=SCALES):
    base = yaml.safe_load(Path("configs/scenario_schema_v1.yaml").read_text())
    base["network"]["site_spacing_m"] = SUB_ISD_M
    base["radio"]["propagation"] = "RMa"
    for i, cfg, d in _variants(base, scales):
        if i == layout:
            break
    d = copy.deepcopy(d)
    if SUB_SMALL_CELLS is not None:
        d["network"]["small_cells"]["count"] = int(SUB_SMALL_CELLS)
    scenario = from_dict(d)
    scenario.macro_height_m = SUB_MACRO_H_M
    for s in scenario.slices:
        s.min_share = {"eMBB": embb_min, "URLLC": urllc_min,
                       "mIoT": D.MIOT_MIN}[s.name]
    # Rural traffic profile (RURAL_PROFILE=1): UE mix 30/20/50 (eMBB/URLLC/mIoT),
    # eMBB per-UE demand at half the layout load factor, URLLC unchanged.
    rural = os.environ.get("RURAL_PROFILE") == "1"
    mix = {"eMBB": 0.3, "URLLC": 0.2, "mIoT": 0.5} if rural else None
    slice_scale = {"eMBB": 0.5} if rural else {}
    sim = BaselineSimulator(scenario, n_ues=cfg["n_ues"], n_slots=1000,
                            seed=seed, budget_policy="reactive", borrowing=True,
                            slice_mix=mix)
    sim.traffic.rates = {k: v / (cfg["traffic_scale"] * slice_scale.get(k, 1.0))
                         for k, v in sim.traffic.rates.items()}
    rep = sim.run()
    sl = ["eMBB", "URLLC", "mIoT"]
    served = sum(rep[x]["served_kb"] for x in sl)
    prb_used = sum(rep[x]["prb_used"] for x in sl)
    avail = float(sum(sim.cell_total_prbs) * rep["slots"])
    return {"layout": layout, "seed": seed, "morphology": __import__("os").environ.get("SUB_MORPH", "suburban-RMa"), "isd_m": SUB_ISD_M,
            "embb_min": embb_min, "urllc_min": urllc_min, "miot_min": D.MIOT_MIN,
            "served_kb": round(served, 1),
            "violations": int(sum(rep[x]["violations"] for x in sl)),
            "prb_util": round(prb_used / avail, 4),
            "eff_kb_per_prb": round(served / max(1, prb_used), 4),
            "p5_kbps": round(rep["ue_thr_kbps_p5"], 2),
            **{f"viol_{x.lower()}": rep[x]["violations"] for x in sl},
            **{f"served_{x.lower()}": round(rep[x]["served_kb"], 1) for x in sl},
            **{f"prb_{x.lower()}": rep[x]["prb_used"] for x in sl},
            "traffic_scales": list(scales),
            "rural_profile": os.environ.get("RURAL_PROFILE") == "1"}


if __name__ == "__main__":
    w, nw = int(sys.argv[1]), int(sys.argv[2])
    jobs = [(L, 5000 + L * 100 + s, e, u) for L in (0, 1) for s in range(20)
            for e, u in D.GRID]
    jobs = jobs[w::nw]
    out = Path(f"data/stage_e_dimensioning_suburban{TAG}_w{w}.json")
    res = json.loads(out.read_text()) if out.exists() else []
    done = {(r["layout"], r["seed"], r["embb_min"], r["urllc_min"]) for r in res}
    for j in jobs:
        if j in done:
            continue
        res.append(run_one_suburban(*j))
        if len(res) % 5 == 0:
            out.write_text(json.dumps(res))
            print(f"w{w}: {len(res)}/{len(jobs)}", flush=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"w{w} done: {len(res)}")
