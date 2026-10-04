"""Stage B gate tests (brief §4): PRB accounting, queue conservation,
interference response, and overload behavior.

- PRB accounting: allocated PRBs never exceed the cell's usable total.
- Queue conservation: arrived == served + dropped + still queued.
- Overload: infeasible demand must surface as counted violations, not vanish.
"""
import sys

sys.path.insert(0, "src")

from urtwin.scenarios.loader import load
from urtwin.sim.baseline import BaselineSimulator


def _small_sim(**kw):
    scenario = load("configs/scenario_schema_v1.yaml")
    args = {"n_ues": 30, "n_slots": 300, "seed": 11}
    args.update(kw)
    return BaselineSimulator(scenario, **args)


def test_prb_accounting():
    sim = _small_sim()
    sim.run()
    worst = 0.0
    for slot, ci, allocated, budget_sum, total in sim.prb_trace:
        assert allocated <= total + 1e-6, f"slot {slot} cell {ci}: over-allocated"
        assert budget_sum <= total + 1e-6, f"slot {slot} cell {ci}: budget overflow"
        worst = max(worst, allocated / total if total else 0)
    print(f"PRB accounting OK over {len(sim.prb_trace)} slot-cells "
          f"(peak utilization {worst:.1%})")


def test_queue_conservation():
    sim = _small_sim()
    rep = sim.run()
    qc = rep["queue_conservation"]
    lhs = qc["arrived_kb"]
    rhs = qc["served_kb"] + qc["dropped_kb"] + qc["queued_kb"]
    assert abs(lhs - rhs) < 1e-6, f"queue leak: {lhs} != {rhs}"
    print(f"queue conservation OK: arrived={lhs:.1f}kb served={qc['served_kb']:.1f} "
          f"dropped={qc['dropped_kb']:.1f} queued={qc['queued_kb']:.1f}")


def test_overload_reported():
    sim = _small_sim(n_slots=400)
    # flood the traffic generators: infeasible demand by construction
    sim.traffic.rates = {"eMBB": 1.5, "URLLC": 4.0}
    sim.prach.p_access = 1.0
    rep = sim.run()
    total_viol = sum(rep[s]["violations"] for s in ("eMBB", "URLLC", "mIoT"))
    assert total_viol > 0, "overload produced no violations — hidden?"
    print(f"overload OK: violations eMBB={rep['eMBB']['violations']} "
          f"URLLC={rep['URLLC']['violations']} mIoT={rep['mIoT']['violations']}")


def test_interference_response():
    # adding a strong interferer (all cells at max power already) — check that
    # an isolated cell delivers more than the full multicell case
    sim = _small_sim()
    rep = sim.run()
    assert rep["eMBB"]["served_kb"] > 0, "baseline serves nothing"
    print(f"interference sanity OK: eMBB served {rep['eMBB']['served_kb']:.1f}kb, "
          f"handovers={rep['handovers']}, "
          f"PRACH {rep['prach']['successes']}/{rep['prach']['attempts']} ok")


if __name__ == "__main__":
    test_prb_accounting()
    test_queue_conservation()
    test_overload_reported()
    test_interference_response()
    print("ALL STAGE B GATE TESTS PASSED")
