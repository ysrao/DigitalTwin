"""v2.0 PRACH flooding probe (10 seeds per load up to 960 UEs, 3 seeds above) on the antenna-fixed channel (urban layout).
Usage: python scripts/run_prach_v2.py <worker> <n_workers>  -> data/v2_prach_w<k>.json"""
import copy, json, sys
from pathlib import Path
import numpy as np, yaml
sys.path.insert(0, "src")
from urtwin.scenarios.loader import from_dict
from urtwin.sim.baseline import BaselineSimulator
w, nw = int(sys.argv[1]), int(sys.argv[2])
base = yaml.safe_load(Path("configs/scenario_schema_v1.yaml").read_text())
J = ([(n, s) for n in [60, 120, 240, 480, 960] for s in range(10)] + [(n, s) for n in [1440, 1920, 2880, 5760, 7680] for s in range(3)])[w::nw]
out = Path(f"data/v2_prach_w{w}.json"); res = json.loads(out.read_text()) if out.exists() else []
for n, s in J[len(res):]:
    sim = BaselineSimulator(from_dict(copy.deepcopy(base)), n_ues=n, n_slots=1000, seed=5000 + s, budget_policy="reactive")
    rep = sim.run(); pr = rep["prach"]; d = sim.prach.access_delays
    res.append({"n_ues": n, "seed": s, "attempts": pr["attempts"], "collisions": pr["collisions"],
                "successes": pr["successes"], "coll_rate": pr["collisions"] / max(1, pr["attempts"]),
                "mean_delay_slots": float(np.mean(d)) if d else 0.0, "miot_viol": rep["mIoT"]["violations"]})
    out.write_text(json.dumps(res)); print(f"prach w{w}: n={n} s={s} coll={res[-1]['coll_rate']:.3f}", flush=True)
print(f"prach w{w} done")
