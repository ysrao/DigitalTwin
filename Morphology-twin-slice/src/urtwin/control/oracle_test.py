"""Oracle test: can the TRUE 6-epoch outcome rank MPC candidates?

Deep-copy the sim at a decision epoch, roll each candidate forward 6
epochs, and compare the true ranking with the v5 model's predicted ranking.
If even the truth can't separate candidates, there's no headroom (reactive
is optimal). If truth separates but the model doesn't, the model is at fault.
"""
import copy
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, "src")
from urtwin.learning.corpus import _variants
from urtwin.scenarios.loader import from_dict
from urtwin.control.mpc import TwinMPC

base = yaml.safe_load(Path("configs/scenario_schema_v1.yaml").read_text())
for i, cfg, d in _variants(base, (2.0, 3.0)):
    if i == 1:
        break
scenario = from_dict(d)

mpc = TwinMPC(scenario, n_ues=cfg["n_ues"], n_slots=1000, seed=1010,
              ckpt="checkpoints/predictor_v5.pt")
mpc.sim.traffic.rates = {k: v / cfg["traffic_scale"]
                         for k, v in mpc.sim.traffic.rates.items()}
sim = mpc.sim

# drive to epoch 12 with reactive warmup + 6 MPC epochs is overkill;
# just do reactive warmup to epoch 8, then oracle at epoch 8
dt_s = sim.slot_ms / 1e3
epoch = 0
for slot in range(sim.n_slots):
    if slot % sim.budget_epoch == 0:
        sim._epoch_snapshot(epoch)
        sim._update_budgets()
        epoch += 1
        if epoch == 8:
            break
    sim.ues.step(dt_s, sim.slot_ms)
    for p in sim.traffic.arrivals(slot):
        sim._enqueue(p)
    for p in sim.prach.step(slot, sim.slot_ms,
                            next(x.latency_ms for x in sim.scenario.slices
                                 if x.name == "mIoT")):
        sim._enqueue(p)
    sim._schedule(slot)
    sim._expire(slot)

# replicate candidate generation from _decide (grab reactive shares)
sim._update_budgets()
C = sim.n_cells
r_shares = np.array([[sim.budgets[ci].budgets[n] / sim.cell_total_prbs[ci]
                      for n in sim._slice_names] for ci in range(C)])
last = {"backlog": np.zeros((C, 3)), "offered": np.zeros((C, 3))}
# fake mpc internals for candidate gen
mpc.snaps = [{"backlog": np.zeros((C, 3)), "offered": np.ones((C, 3))}]
cands0 = mpc._candidates(0, r_shares[0], np.ones(3))
print(f"cell 0: {len(cands0)} candidates; reactive shares={r_shares[0].round(2)}")

# oracle rollout for cell 0's candidates: apply candidate to ALL cells
# (use candidate k for every cell, tilted from each cell's reactive)
true_scores = []
for k in range(len(cands0)):
    s2 = copy.deepcopy(sim)
    bud = np.zeros((C, 3))
    for ci in range(C):
        ck_ = mpc._candidates(ci, r_shares[ci], np.ones(3))
        kk = min(k, len(ck_) - 1)
        bud[ci] = ck_[kk] * s2.cell_total_prbs[ci]
    s2.set_budgets(bud)
    served0, viol0 = s2.served_bits, sum(s2.kpi[x]["violations"] for x in s2.kpi)
    for slot in range(6 * s2.budget_epoch):
        s2.ues.step(dt_s, s2.slot_ms)
        for p in s2.traffic.arrivals(slot):
            s2._enqueue(p)
        for p in s2.prach.step(slot, s2.slot_ms,
                                next(x.latency_ms for x in s2.scenario.slices
                                     if x.name == "mIoT")):
            s2._enqueue(p)
        s2._schedule(slot)
        s2._expire(slot)
    ds = (s2.served_bits - served0) / 1e3
    dv = sum(s2.kpi[x]["violations"] for x in s2.kpi) - viol0
    true_scores.append(ds - 4 * dv)
    print(f"cand {k}: true 6-epoch served={ds:.0f}kb viol={dv} score={ds-4*dv:.0f}")

print("true best candidate:", int(np.argmax(true_scores)),
      "true worst:", int(np.argmin(true_scores)))
print("true score spread (best-worst):", round(max(true_scores) - min(true_scores), 1))
