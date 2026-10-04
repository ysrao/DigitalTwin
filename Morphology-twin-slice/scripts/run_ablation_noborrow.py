"""Borrowing-disabled ablation: 9 structures x 2 layouts x 10 seeds,
strict guaranteed shares (borrowing=False). Paired seeds with the main grid."""
import json, sys
from pathlib import Path
sys.path.insert(0, "src")
from urtwin.control.dimension import run_one, GRID

SCALES = tuple(json.loads(Path("data/calibration_choice.json").read_text()))
out_path = Path("data/stage_e_ablation_noborrow.json")
results = []
for layout in (0, 1):
    for s in range(10):
        seed = 3000 + layout * 100 + s  # paired with main grid seeds
        for embb_min, urllc_min in GRID:
            r = run_one(layout, seed, embb_min, urllc_min, scales=SCALES,
                        borrowing=False)
            r["traffic_scales"] = list(SCALES)
            results.append(r)
            print(f"layout{layout} seed{s} e={embb_min:.2f}/u={urllc_min:.2f}: "
                  f"viol={r['violations']} (e:{r['viol_embb']} u:{r['viol_urllc']})",
                  flush=True)
    out_path.write_text(json.dumps(results))
out_path.write_text(json.dumps(results, indent=1))
print(f"-> {out_path} ({len(results)} runs)")
