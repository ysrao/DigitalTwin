# Multi-Tier Digital Twin

A stateful digital twin of an AI-enabled 6G RAN: cellular + Wi-Fi + LEO satellite
tiers, a population of UE sessions with velocity vectors, 3GPP propagation, an
A3/A5 handover state machine, and a single centralized RL policy (branching
PPO / Double DQN) that sets **per-tier** slice allocation and access steering.

- **`twin_method.html`** / **`twin_method.pdf`** — the method note: the full
  computation pipeline of one control step, with the formula behind each stage.
- **`platform/`** — the implementation.

| File | Purpose |
|---|---|
| `platform/radio.py` | TR 38.901 path loss (UMa/UMi/RMa/InH) + O2I, TR 38.811/38.821 NTN link budget, ITU-R P.618/838/839 rain, Doppler shift/rate, ICI, shadow fading |
| `platform/mobility.py` | Session types (eMBB/URLLC/mIoT), mobility classes, velocity vectors, per-tier speed limits, beam-tracking derate |
| `platform/handover.py` | A3/A5 event state machine with hysteresis + time-to-trigger, execution outcomes, 6G access steering |
| `platform/twin.py` | Tiers, per-tier radio/MIMO/spectrum config, cells, co-channel interference, trace-driven load, two-timescale `reset`/`step` |
| `platform/agents.py` | Branching PPO and Double DQN — one head per tier, on `engine.ActorCritic` primitives |
| `platform/compare.py` | Scenarios, rule-based baselines, matched comparison, multi-objective optimizer, feasibility verdict, capacity/coverage preview |
| `platform/web/twin.html` | Browser UI |

## Run

```sh
cd platform
python3 server.py                       # http://127.0.0.1:8765/twin.html
python3 -m unittest -v test_twin.py test_engine.py
python3 compare.py                       # matched comparison as JSON
```

## Status

Screening-grade research model — advisory only, not carrier evidence.

Modelled: 3GPP-shaped multi-tier propagation, spatially-correlated shadow
fading, co-channel inter-cell interference (load- and beam-coupling-dependent),
mobility + A3/A5 handover, per-tier RL control, trace-driven time-varying
offered load (`synthetic_profiles/all_profiles.csv`), a guarded decision that
separates *tune the controller* from *change the deployment*.

Not yet modelled: PRACH / random-access overload; explicit per-KPI targets
(eMBB throughput, URLLC latency, mIoT coverage-per-km²) as scenario inputs.
