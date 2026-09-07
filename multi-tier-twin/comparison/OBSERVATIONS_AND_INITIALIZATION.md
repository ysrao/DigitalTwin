# Observation vectors and weight initialization

Background reference for `centralized-ppo-vs-marl-comparison-v8.pdf` and
`platform/marl_replication.py`. Written for reviewers and for our own future
reference — not a new claim, just the exact formulas and mechanism behind
Sections IV and VI-C of the paper.

## Centralized controller — 13 inputs

From `engine.py`'s `state_vector(row, previous, scenario)`, reused unmodified.
Every input is capped so it stays in a small, comparable numeric range.

| # | Input | Formula |
|---|---|---|
| 1 | PRB availability | `min(2.0, available_prbs / prbs_per_cell)` |
| 2 | eMBB demand/capacity | `min(4.0, embb_demand / (prbs_per_cell·0.36·embb_se))` |
| 3 | URLLC demand/capacity | `min(4.0, urllc_demand / (prbs_per_cell·0.36·urllc_se))` |
| 4 | mIoT demand/capacity | `min(4.0, iot_demand / (prbs_per_cell·0.36·iot_se))` |
| 5 | eMBB active UEs | `min(2.0, embb_active_ues / 50)` |
| 6 | URLLC packets | `min(2.0, urllc_packets / 30)` |
| 7 | mIoT active devices | `min(2.0, iot_active_devices / 3000)` |
| 8 | eMBB spectral efficiency | `min(2.0, embb_se / 4)` |
| 9 | URLLC spectral efficiency | `min(2.0, urllc_se / 2)` |
| 10 | mIoT spectral efficiency | `min(2.0, iot_se / 0.8)` |
| 11 | Neighbor-cell load | `min(1.0, neighbor_load_ratio)` |
| 12 | Cell outage flag | `1.0` if unavailable, else `0.0` |
| 13 | Previous action | index of the last template chosen, normalized to `[0,1]` |

Action: one of the 9 rows of `TEMPLATES` (a joint eMBB/URLLC/mIoT share vector).

## Each MARL slice-agent — 7 inputs

From `marl_replication.py`'s `marl_obs(row, slice_idx, prev_shares, prev_sat, ...)`.
One instance per slice (eMBB, URLLC, mIoT) — each agent sees only its own
version of the slice-specific features, plus one shared broadcast signal.

| # | Input | Scope |
|---|---|---|
| 1 | Demand/capacity ratio | own slice only |
| 2 | Spectral-efficiency indicator | own slice only |
| 3 | Previous share (last allocation received) | own slice only |
| 4 | Aggregate offered load, all 3 slices combined | shared/broadcast |
| 5 | PRB availability | shared (same signal as centralized input #1) |
| 6 | Episode phase: `step_index / 48` | shared |
| 7 | SLA deficit: `max(0, threshold − last-step satisfaction)`, reset to 0 at episode start | own slice only |

Action: one of `K=5` discrete bid levels, `[0.1, 0.3, 0.5, 0.7, 0.9]`. The
deterministic coordinator turns the three agents' chosen levels into a
feasible joint share vector (clip to floors `(0.25, 0.20, 0.10)`, renormalize
the remainder in proportion to bids above floor).

**The structural difference this creates:** the centralized controller sees
all three slices' demand/UE/spectral-efficiency numbers at once (inputs 2–10
above) in a single 13-D vector feeding one 9-way joint decision. Each MARL
agent sees only its *own* slice's version of those numbers, a 5-way decision,
and one shared broadcast value (aggregate load) as its only cross-slice
information. That narrower observation and smaller action space is what makes
each MARL sub-problem easier to converge on — see Section VI-C of the paper.

## Weight initialization and what the training seed actually controls

Both the centralized `ActorCritic` and each MARL agent's `ActorCritic` are
initialized the same way:

```python
rng = np.random.default_rng(seed)
h = hidden  # 64

# actor
aw = [rng.normal(0, 0.12, (n_inputs, h)),   # layer 1
      rng.normal(0, 0.12, (h, h)),          # layer 2
      rng.normal(0, 0.08, (h, n_actions))]  # output layer
ab = [zeros(h), zeros(h), zeros(n_actions)] # all biases start at 0

# critic — same shapes/std devs, drawn immediately after the actor's,
# from the same rng sequence
cw = [rng.normal(0, 0.12, (n_inputs, h)),
      rng.normal(0, 0.12, (h, h)),
      rng.normal(0, 0.08, (h, 1))]
cb = [zeros(h), zeros(h), zeros(1)]
```

So the seed does not change anything about the environment, the reward, or
the architecture — it only picks a different random starting point in
weight-space (a different draw from the same Gaussian distributions) for
gradient descent to start from. Two runs with different seeds are the same
network, same problem, different starting point.

For the MARL controller, the three slice-agents are never initialized
identically even within one training run: each derives its own seed by
offsetting the training seed (`seed`, `seed + 1000`, `seed + 2000`) before
constructing its `ActorCritic`.

**Why this matters for the paper's finding:** because the only thing a seed
changes is the starting point, a training run that "does worse" isn't
evidence the architecture is worse — it can simply mean that starting point
led gradient descent into a worse local optimum. That is exactly the
mechanism Section VI-C documents: centralized PPO reaches the same-or-better
solution as MARL on 6 of 10 seeds, and a visibly worse one on the other 4,
purely as a function of where each seed happened to start.
