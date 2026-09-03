# Multi-Tier Digital Twin v2

[**Launch the interactive v2 artifact →**](https://ysrao.github.io/DigitalTwin/multi-tier-twin/twin_console.html)

[Open the v2 workflow and calculation guide (PDF)](https://ysrao.github.io/DigitalTwin/multi-tier-twin/Multi_Tier_Digital_Twin_v2_Workflow.pdf)

> Use the launch link above to run the artifact. Clicking `twin_console.html`
> in GitHub opens its source code because the repository view does not execute HTML.

Authors: Rao Yenamandra (`raosy@digitaltwinsim.com`) and Mubanga Nsofu
(`mubanga.nsofu@vodacom.co.za`).

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
python3 gen_console_data.py --write-release  # v2 dataset + standalone console
```

## v2 standalone evidence package

`twin_console.html` loads `browser_engine_v2.js`, a self-contained editable
aggregate JavaScript screening engine. It recalculates capacity, trains
lightweight linear PPO/Double-DQN approximations, compares five controllers,
performs a five-seed sensitivity check, and runs coordinate search locally.
It is intentionally simpler than, and not numerically identical to, the Python
twin. The generated `validation_dataset_v2.json` is the separate Python
reference record: model version, UTC generation time, configuration hash, five
raw seed-level policy results per scenario, Student-t 95% confidence intervals,
and optimizer outputs.

The five-seed summary is an exploratory sensitivity check. Eight training
episodes per seed do not demonstrate PPO or DQN convergence.

## Status

Screening-grade research model — advisory only, not carrier evidence.

Modelled: 3GPP-shaped multi-tier propagation, spatially-correlated shadow
fading, co-channel inter-cell interference (load- and beam-coupling-dependent),
mobility + A3/A5 handover, per-tier RL control, trace-driven time-varying
offered load (`synthetic_profiles/all_profiles.csv`), a guarded decision that
separates *tune the controller* from *change the deployment*.

Not yet modelled: PRACH / random-access overload; explicit per-KPI targets
(eMBB throughput, URLLC latency, mIoT coverage-per-km²) as scenario inputs.
The mIoT values are therefore offered load only, not random-access capacity or
PRACH-storm performance.
