"""Stage E (part 2): slice dimensioning for a fixed morphology.

Carrier's question: given the network I already own (fixed topology and
propagation), what spectrum fractions should each slice be *guaranteed*?
Grid-search min-share structures with reactive borrowing enabled, on
deterministic matched realizations, heavy load. Output: KPI Pareto map.
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, "src")
from urtwin.learning.corpus import _variants
from urtwin.scenarios.loader import from_dict
from urtwin.sim.baseline import BaselineSimulator

# (embb_min, urllc_min); miot_min fixed at 0.05
GRID = [(0.30, 0.10), (0.30, 0.20), (0.30, 0.30),
        (0.45, 0.10), (0.45, 0.20), (0.45, 0.30),
        (0.60, 0.10), (0.60, 0.20), (0.60, 0.30)]
MIOT_MIN = 0.05


def run_one(layout, seed, embb_min, urllc_min, scales=(2.0, 3.0),
            borrowing=True, trace_csv=None):
    base = yaml.safe_load(Path("configs/scenario_schema_v1.yaml").read_text())
    for i, cfg, d in _variants(base, scales):
        if i == layout:
            break
    scenario = from_dict(copy.deepcopy(d))
    for s in scenario.slices:
        if s.name == "eMBB":
            s.min_share = embb_min
        elif s.name == "URLLC":
            s.min_share = urllc_min
        elif s.name == "mIoT":
            s.min_share = MIOT_MIN
    sim = BaselineSimulator(scenario, n_ues=cfg["n_ues"], n_slots=1000,
                            seed=seed, budget_policy="reactive",
                            borrowing=borrowing, trace_csv=trace_csv)
    sim.traffic.rates = {k: v / cfg["traffic_scale"]
                         for k, v in sim.traffic.rates.items()}
    rep = sim.run()
    sl = ["eMBB", "URLLC", "mIoT"]
    served = sum(rep[x]["served_kb"] for x in sl)
    viol = sum(rep[x]["violations"] for x in sl)
    prb_used = sum(rep[x]["prb_used"] for x in sl)
    avail = float(sum(sim.cell_total_prbs) * rep["slots"])
    return {
        "layout": layout, "seed": seed,
        "embb_min": embb_min, "urllc_min": urllc_min, "miot_min": MIOT_MIN,
        "served_kb": round(served, 1), "violations": int(viol),
        "prb_util": round(prb_used / avail, 4),
        "eff_kb_per_prb": round(served / max(1, prb_used), 4),
        "p5_kbps": round(rep["ue_thr_kbps_p5"], 2),
        "viol_embb": rep["eMBB"]["violations"],
        "viol_urllc": rep["URLLC"]["violations"],
        "viol_miot": rep["mIoT"]["violations"],
        "served_embb": round(rep["eMBB"]["served_kb"], 1),
        "served_urllc": round(rep["URLLC"]["served_kb"], 1),
        "served_miot": round(rep["mIoT"]["served_kb"], 1),
        "prb_embb": rep["eMBB"]["prb_used"],
        "prb_urllc": rep["URLLC"]["prb_used"],
        "prb_miot": rep["mIoT"]["prb_used"],
    }


def main():
    results = []
    for layout in [0, 12]:
        for s in range(5):
            seed = 1000 + layout * 10 + s
            for embb_min, urllc_min in GRID:
                r = run_one(layout, seed, embb_min, urllc_min)
                results.append(r)
                print(f"layout{layout:02d} seed{s} "
                      f"e={embb_min:.2f}/u={urllc_min:.2f}: "
                      f"served={r['served_kb']:.0f} viol={r['violations']} "
                      f"(e:{r['viol_embb']} u:{r['viol_urllc']} m:{r['viol_miot']}) "
                      f"util={r['prb_util']:.3f}", flush=True)
    out = Path("data/stage_e_dimensioning.json")
    out.write_text(json.dumps(results, indent=1))
    print(f"-> {out} ({len(results)} runs)")


if __name__ == "__main__":
    main()
