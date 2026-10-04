"""Lightweight PRACH-only sweep (no Sionna): collision rate vs mIoT load.
The PRACH model is MAC-only; preamble collisions don't depend on channels.
33 cells like the full sim; mIoT UEs assigned uniformly to cells."""
import json, sys
from pathlib import Path
import numpy as np, yaml
sys.path.insert(0, "src")
from urtwin.scenarios.loader import from_dict
from urtwin.sim.prach import Prach

OUT = Path("data/prach_extended.json")
out = json.loads(OUT.read_text()) if OUT.exists() else []
done = {(r["n_ues"], r["seed"]) for r in out}
print(f"resuming with {len(done)} done")
base = yaml.safe_load(Path("configs/scenario_schema_v1.yaml").read_text())
scenario = from_dict(base)
N_CELLS = 33
for n_ues in [60, 120, 240, 480, 960, 1440, 1920, 2880]:
    for seed in range(10):
        if (n_ues, seed) in done:
            continue
        rng = np.random.default_rng(5000 + seed)
        n_miot = int(0.3 * n_ues)
        slice_of_ue = (["mIoT"] * n_miot + ["eMBB"] * (n_ues - n_miot))
        serving = rng.integers(0, N_CELLS, size=n_ues)
        prach = Prach(scenario, slice_of_ue, serving, rng)
        for slot in range(1000):
            prach.step(slot, 0.5, 1000.0)
        rec = {"n_ues": n_ues, "seed": seed,
               "attempts": prach.attempts, "collisions": prach.collisions,
               "successes": prach.successes,
               "coll_rate": prach.collisions / max(1, prach.attempts),
               "mean_delay_slots": float(np.mean(prach.access_delays)) if prach.access_delays else 0.0,
               "miot_viol": 0}
        out.append(rec)
        print(f"n_ues={n_ues} seed={seed}: coll={rec['coll_rate']:.1%} "
              f"(attempts={rec['attempts']})", flush=True)
    OUT.write_text(json.dumps(out))
OUT.write_text(json.dumps(out, indent=1))
print("saved", OUT, f"({len(out)} total)")
