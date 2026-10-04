"""Resume extended PRACH sweep: skip completed (n_ues,seed)."""
import copy, json, sys
from pathlib import Path
import numpy as np, yaml
sys.path.insert(0, "src")
from urtwin.scenarios.loader import from_dict
from urtwin.sim.baseline import BaselineSimulator

OUT = Path("data/prach_extended.json")
out = json.loads(OUT.read_text())
done = {(r["n_ues"], r["seed"]) for r in out}
print(f"resuming with {len(done)} done")
base = yaml.safe_load(Path("configs/scenario_schema_v1.yaml").read_text())
for n_ues in [60, 120, 240, 480, 960, 1440, 1920]:
    for seed in range(10):
        if (n_ues, seed) in done:
            continue
        d = copy.deepcopy(base)
        sc = from_dict(d)
        sim = BaselineSimulator(sc, n_ues=n_ues, n_slots=1000, seed=5000+seed, budget_policy="reactive")
        rep = sim.run()
        pr = rep["prach"]
        delays = sim.prach.access_delays
        out.append({"n_ues": n_ues, "seed": seed,
            "attempts": pr["attempts"], "collisions": pr["collisions"],
            "successes": pr["successes"],
            "coll_rate": pr["collisions"]/max(1, pr["attempts"]),
            "mean_delay_slots": float(np.mean(delays)) if delays else 0.0,
            "p95_delay_slots": float(np.percentile(delays, 95)) if delays else 0.0,
            "miot_viol": rep["mIoT"]["violations"]})
        print(f"n_ues={n_ues} seed={seed}: coll={out[-1]['coll_rate']:.1%}", flush=True)
    OUT.write_text(json.dumps(out))
OUT.write_text(json.dumps(out, indent=1))
print("saved", OUT, f"({len(out)} total)")
