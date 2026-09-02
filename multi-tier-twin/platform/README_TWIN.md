# Multi-Tier Digital Twin v1

`engine.py` (the original slice MVP) replays a single-cell CSV trace. This adds
the network model underneath it: real cell coordinates and antenna heights, a
population of UE sessions with velocity vectors, 3GPP propagation, and an A3/A5
handover state machine — so handover success, ping-pong and session continuity
become measurable rather than assumed.

## Modules

| File | Purpose |
|---|---|
| `radio.py` | TR 38.901 terrestrial path loss (UMa/UMi/RMa/InH), O2I penetration, TR 38.811/38.821 NTN link budget, ITU-R P.618/838/839 rain attenuation, Doppler shift and rate, ICI |
| `mobility.py` | Session types (eMBB/URLLC/mIoT), mobility classes, velocity vectors, per-tier mobility limits and beam-tracking derate |
| `handover.py` | A3/A5 event state machine with hysteresis and time-to-trigger, execution outcomes, 6G access steering |
| `twin.py` | Tiers, per-tier radio/MIMO/spectrum config, cells, the two-timescale `reset`/`step` model |
| `agents.py` | Branching PPO and Double DQN — one head per tier, on `engine.ActorCritic` primitives |
| `compare.py` | Scenarios, rule-based baselines, matched comparison, multi-objective optimizer, feasibility verdict |
| `web/twin.html` | Browser UI: KPI tables, comparison bar charts, per-tier config, optimizer |

## Ticks and control intervals

Two timescales, as in the paper:

- **Tick** (`tick_seconds`, default 0.1 s) — the fast loop. Every tick, UEs move
  along their velocity vectors, RSRP is recomputed, and the A3/A5 measurement
  state machine advances (entry conditions, time-to-trigger timers, handover
  execution). Mobility and handovers only exist at this timescale.
- **Control interval** (`control_interval_ticks`, default 100 ticks = 10 s) — the
  slow loop. Once per interval the RL policy (or a baseline) picks an action:
  a slice-mix template for each tier plus a global steering posture. PRBs are
  then allocated and the interval's KPIs aggregated.

`twin.step()` runs one control interval — i.e. `control_interval_ticks` mobility
ticks, then one allocation. `episode_steps` control intervals make an episode.
So the defaults are a 288-step day of 10 s control periods over 100 ms mobility.

## Architecture

**Single-agent, centralized, per-tier action branching.** One policy observes
the whole network (a 46-dim vector: demand, time, utilization, continuity,
handover stats, NTN elevation/rain, plus 6 features per tier) and emits one
action **per tier** — which of 9 slice-mix templates that tier runs — plus one
global steering branch (4 postures). That is `9^n_tiers × 4` joint actions
(~2.1M at six tiers), represented as `n_tiers + 1` softmax heads on a shared
trunk, not a flat table. It is **not** multi-agent: no per-cell/per-tier agents,
no inter-agent coordination.

- **PPO** — on-policy, clipped surrogate on the joint (summed-log-prob) ratio,
  GAE(λ).
- **Double DQN** — per-tier Q heads, shared bootstrap target (BDQN style),
  replay, target network, ε-greedy.

Both reuse `engine.ActorCritic`'s forward/backprop, generalized from
`13→64→64→9` to `46→64→64→[9×6, 4]`. `engine.py` defaults are unchanged and
`test_engine.py` still passes.

## Per-tier radio config

Each `Tier` carries its own morphology, spectrum and MIMO. `plan_summary()`
reports the resolved values; `TwinConfig.tier_overrides` changes them per tier
without redefining the list:

```python
tier_overrides={"macro_mid": {"bandwidth_mhz": 200, "carriers": 2,
                              "numerology_khz": 60, "mimo_layers": 8,
                              "mimo_mode": "MU", "scenario": "UMi"}}
```

- **morphology** (`scenario`: UMa/UMi/RMa/InH) selects the TR 38.901 path-loss
  model; NTN tiers use TR 38.811.
- **spectrum** — `band_ghz`, `bandwidth_mhz`, `carriers` (CA). The PRB grid
  (`prb_bandwidth_mhz`, `prbs_per_cell`) is **derived** from bandwidth +
  `numerology_khz` unless given explicitly, and re-derived when any of those is
  overridden.
- **MIMO** — `mimo_layers`, `mimo_mode` (SU/MU), `mimo_efficiency`. Multiplies
  delivered cell capacity; per-stream spectral efficiency stays Shannon-bounded.

## Tier stack (defaults)

| Tier | RAT | Band | BW | Morphology | Notes |
|---|---|---|---|---|---|
| `macro_low` | NR | 0.7 GHz | 10 MHz | UMa | Coverage anchor, 350 km/h |
| `macro_mid` | NR | 3.5 GHz | 100 MHz | UMa | Primary capacity layer |
| `umb_6g` | 6G-NR | 7.5 GHz | 200 MHz | UMi | Upper mid-band, beamformed |
| `mmwave` | NR | 28 GHz | 400 MHz | UMi | Hotspot; fails above 120 km/h |
| `wifi7_indoor` | Wi-Fi 7 | 6 GHz | 320 MHz | InH | Indoor only, no URLLC |
| `ntn_leo` | NR-NTN | 2 GHz | 30 MHz | — | LEO 600 km, outdoor only, no URLLC |

**Any subset runs** — no Wi-Fi + no satellite (`cellular_only`), or a single tier.

## Optimizer

`optimize(payload)` with `objective ∈ {handover, throughput, latency, coverage}`.
Each tunes a different knob set (A3/A5 thresholds, per-tier cell individual
offsets, steering thresholds, per-class minimum RSRP) by coordinate descent, and
is scored on that KPI **net of a session-continuity guard** — nothing can win by
stranding sessions.

## Feasibility

Every comparison and optimize run carries a `feasibility` verdict:

- `FEASIBLE` — demand is served within the tiers; threshold tuning is the right lever.
- `COVERAGE-LIMITED` — demand from sessions no tier can reach; needs sites / low
  band / NTN, not thresholds.
- `CAPACITY-LIMITED` — demand exceeds the PRB capacity of the reachable tiers
  and they have no headroom; needs spectrum / cells.
- `COVERAGE-AND-CAPACITY-LIMITED` — both.

`thresholds_can_help` is `true` only for `FEASIBLE`.

## Running

```sh
cd platform
python3 server.py            # http://127.0.0.1:8765/twin.html
python3 -m unittest -v test_twin.py test_engine.py
python3 compare.py           # comparison as JSON
```

API: `GET /api/twin/scenarios`, `POST /api/twin/compare`,
`POST /api/twin/optimize` (`{"objective": ...}`). The original slice MVP is
untouched at `/` and `POST /api/evaluate`.

## Not yet modelled

- **PRACH / random-access overload** — no contention or preamble-capacity limit;
  mIoT arrival storms do not yet cause access failures.
- **Explicit per-KPI targets as inputs** — eMBB target throughput, URLLC target
  latency and mIoT coverage per km² are baked into `SessionType` and the SLA
  thresholds, not exposed as scenario knobs.
- **Editable per-tier config and cell coordinates in the browser** — overrides
  are API-only; the UI shows them read-only.

## Limitations

- Screening-grade propagation: per-tier distance tables with a LOS-probability
  blend, no per-link shadow fading realization or ray tracing.
- **No explicit interference model** — SINR is against thermal noise, so
  utilization and spectral efficiency are optimistic.
- MIMO is a capacity multiplier, not a spatial channel model.
- NTN: single beam, circular orbit over a non-rotating Earth, no Earth-rotation
  Doppler term (hence ~45 kHz where TR 38.821 quotes 48 kHz).
- Wi-Fi is a scheduled tier, not CSMA/CA with contention.
- Handover failure probabilities are parametric, not from an RRC signalling sim.
- Synthetic sessions; not a carrier trace. SLA figures are modeled interval
  compliance, not proof of five- or seven-nines reliability.
- The optimizer is one-pass coordinate descent against a single evaluation seed.
