"""v2.0 generic grid runner (antenna-fixed channel). One script for every
dimensioning experiment, split across workers.
Usage: python scripts/run_grid.py <experiment> <worker> <n_workers>
Experiments: urban, ablation, trace, suburban, suburban_heavy, rural_uniform, rural_profile,
plus calib_urban / calib_suburban (load sweeps at the 30%/10% structure).
Load factors come from data/v2_calibration_choice.json."""
import json, os, sys
from pathlib import Path
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from urtwin.control.dimension import run_one, GRID

CHOICE = Path("data/v2_calibration_choice.json")
TRACE = "data/trace_synthetic_24h.csv"
CAL_TS = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]


def scales(key):
    c = json.loads(CHOICE.read_text())
    return tuple(c[key])


def set_morph(exp):
    env = {"suburban": dict(SUB_ISD="1000"), "suburban_heavy": dict(SUB_ISD="1000"),
           "calib_suburban": dict(SUB_ISD="1000"),
           "rural_uniform": dict(SUB_ISD="1732", SUB_SC="0", SUB_MORPH="rural-RMa"),
           "rural_profile": dict(SUB_ISD="1732", SUB_SC="0", SUB_MORPH="rural-RMa-profile",
                                 RURAL_PROFILE="1")}.get(exp, {})
    os.environ.update(env)


def jobs(exp):
    J = []
    if exp in ("urban", "ablation"):
        nseed = 20 if exp == "urban" else 10
        J = [(L, 3000 + 100 * L + s, e, u) for L in (0, 1) for s in range(nseed) for e, u in GRID]
    elif exp == "trace":
        J = [(L, 7000 + 100 * L + s, e, u) for L in (0, 1) for s in range(10) for e, u in GRID]
    elif exp.startswith("calib"):
        J = [(0, 2000 + s, 0.30, 0.10, ts) for ts in CAL_TS for s in range(5)]
    else:
        J = [(L, 5000 + 100 * L + s, e, u) for L in (0, 1) for s in range(20) for e, u in GRID]
    return J


def run(exp, j):
    if exp == "urban":
        return run_one(*j, scales=scales("urban"))
    if exp == "ablation":
        return run_one(*j, scales=scales("urban"), borrowing=False)
    if exp == "trace":
        return run_one(*j, scales=scales("urban"), trace_csv=TRACE)
    if exp == "calib_urban":
        L, seed, e, u, ts = j
        return {"traffic_scale": ts, **run_one(L, seed, e, u, scales=(ts, ts))}
    from run_dimensioning_suburban import run_one_suburban
    if exp == "calib_suburban":
        L, seed, e, u, ts = j
        return {"traffic_scale": ts, **run_one_suburban(L, seed, e, u, scales=(ts, ts))}
    key = {"suburban": "urban", "suburban_heavy": "suburban", "rural_uniform": "suburban",
           "rural_profile": "urban"}[exp]
    r = run_one_suburban(*j, scales=scales(key))
    r["experiment"] = exp
    return r


if __name__ == "__main__":
    exp, w, nw = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    set_morph(exp)
    import importlib, run_dimensioning_suburban as RS
    importlib.reload(RS)
    J = jobs(exp)[w::nw]
    out = Path(f"data/v2_{exp}_w{w}.json")
    res = json.loads(out.read_text()) if out.exists() else []
    done = len(res)
    for j in J[done:]:
        res.append(run(exp, j))
        if len(res) % 5 == 0:
            out.write_text(json.dumps(res)); print(f"{exp} w{w}: {len(res)}/{len(J)}", flush=True)
    out.write_text(json.dumps(res, indent=1)); print(f"{exp} w{w} done: {len(res)}", flush=True)
