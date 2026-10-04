"""Final dimensioning re-run: 9 structures x 2 layouts x 20 seeds on the
new FR3 config, at the calibrated traffic scale. Writes
data/stage_e_dimensioning_fr3_20seed.json (pilot data untouched)."""
import json, sys
from pathlib import Path
sys.path.insert(0, "src")
from urtwin.control.dimension import run_one, GRID

# calibrated traffic scales (layout 0, layout 1); set from calibration run
SCALES = tuple(json.loads(Path("data/calibration_choice.json").read_text()))

out_path = Path("data/stage_e_dimensioning_fr3_20seed.json")
done = set()
if out_path.exists():
    for r in json.loads(out_path.read_text()):
        done.add((r["layout"], r["seed"], r["embb_min"], r["urllc_min"]))
results = json.loads(out_path.read_text()) if out_path.exists() else []

n_new = 0
for layout in (0, 1):
    for s in range(20):
        seed = 3000 + layout * 100 + s
        for embb_min, urllc_min in GRID:
            key = (layout, seed, embb_min, urllc_min)
            if key in done:
                continue
            r = run_one(layout, seed, embb_min, urllc_min, scales=SCALES)
            r["traffic_scales"] = list(SCALES)
            results.append(r)
            n_new += 1
            if n_new % 18 == 0:
                out_path.write_text(json.dumps(results))
                print(f"... {n_new} new runs, checkpointed", flush=True)
out_path.write_text(json.dumps(results, indent=1))
print(f"-> {out_path} ({len(results)} total, {n_new} new)")
