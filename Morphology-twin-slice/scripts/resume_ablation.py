"""Resume borrowing-disabled ablation: skip completed (layout,seed,structure)."""
import json, sys
from pathlib import Path
sys.path.insert(0, "src")
from urtwin.control.dimension import run_one, GRID

SCALES = tuple(json.loads(Path("data/calibration_choice.json").read_text()))
out_path = Path("data/stage_e_ablation_noborrow.json")
results = json.loads(out_path.read_text())
done = {(r["layout"], r["seed"], r["embb_min"], r["urllc_min"]) for r in results}
print(f"resuming with {len(done)} done")
n_new = 0
for layout in (0, 1):
    for s in range(10):
        seed = 3000 + layout * 100 + s
        for embb_min, urllc_min in GRID:
            if (layout, seed, embb_min, urllc_min) in done:
                continue
            r = run_one(layout, seed, embb_min, urllc_min, scales=SCALES, borrowing=False)
            r["traffic_scales"] = list(SCALES)
            results.append(r)
            n_new += 1
            print(f"layout{layout} seed{s} e={embb_min:.2f}/u={urllc_min:.2f}: viol={r['violations']}", flush=True)
    out_path.write_text(json.dumps(results))
out_path.write_text(json.dumps(results, indent=1))
print(f"-> {out_path} ({len(results)} total, {n_new} new)")
