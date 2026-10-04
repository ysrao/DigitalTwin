"""PRACH flooding probe: collision rate + access delay vs offered mIoT load."""
import copy
import json
import yaml
import numpy as np
from pathlib import Path

from urtwin.scenarios.loader import from_dict
from urtwin.sim.baseline import BaselineSimulator

OUT = Path("/home/hatch/workspace/urban-ran-twin/data/prach_flood_probe.json")

base = yaml.safe_load(Path("/home/hatch/workspace/urban-ran-twin/configs/scenario_schema_v1.yaml").read_text())
out = []
for n_ues in [60, 120, 240, 480, 960]:
    for seed in range(2):
        d = copy.deepcopy(base)
        sc = from_dict(d)
        sim = BaselineSimulator(sc, n_ues=n_ues, n_slots=1000,
                                seed=100 + seed, budget_policy="reactive")
        rep = sim.run()
        pr = rep["prach"]
        delays = sim.prach.access_delays
        rec = {"n_ues": n_ues, "seed": seed,
               "attempts": pr["attempts"], "collisions": pr["collisions"],
               "successes": pr["successes"],
               "coll_rate": pr["collisions"] / max(1, pr["attempts"]),
               "mean_delay_slots": float(np.mean(delays)) if delays else 0.0,
               "p95_delay_slots": float(np.percentile(delays, 95)) if delays else 0.0,
               "miot_viol": rep["mIoT"]["violations"],
               "miot_served": round(rep["mIoT"]["served_kb"], 1)}
        out.append(rec)
        print(f"n_ues={n_ues} seed={seed}: coll={rec['coll_rate']:.1%} "
              f"mean_delay={rec['mean_delay_slots']:.1f}sl p95={rec['p95_delay_slots']:.0f}sl "
              f"miot_viol={rec['miot_viol']}", flush=True)
OUT.write_text(json.dumps(out, indent=1))
print("saved", OUT)
