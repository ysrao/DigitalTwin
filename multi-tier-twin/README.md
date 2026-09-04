# Multi-Tier Digital Twin

## v5 release — regenerated Python-engine results + IEEE paper

[**Launch the v5 interactive console →**](https://ysrao.github.io/DigitalTwin/multi-tier-twin/twin_console_v5.html)

[**Open the v5 method note →**](https://ysrao.github.io/DigitalTwin/multi-tier-twin/twin_method_v5.html)

[**Download the 6-page IEEE paper (PDF) →**](https://ysrao.github.io/DigitalTwin/multi-tier-twin/multi-tier-6g-ran-twin-paper-v5.pdf)

Authors: Rao Yenamandra (`raosy@digitaltwinsim.com`), Mubanga Nsofu
(`mubanga.nsofu@vodacom.co.za`), and Asokan Ram (`asokan.ram@wrc-nc.org`).

v5 runs directly off the Python engine in `platform/` (the same engine as v2 —
`radio.py`, `twin.py`, `handover.py`, `agents.py`, `compare.py`), with
freshly-regenerated, 3-seed comparison results (not the browser-side linear
approximations `browser_engine_v2.js`/`v3.js` use) and an explicit, honest
finding: at a 6-training-episode budget, PPO/Double DQN did **not** cleanly
beat a fixed-static or tuned rule-based baseline, and the per-seed decision
verdict disagreed — reported plainly rather than smoothed over. v5 also adds
a formal §11 RIC/xApp architectural placement (non-RT RIC/rApp vs. near-RT
RIC/xApp correspondence, with the 10 s-vs-10 ms–1 s cadence gap made explicit)
and a References section (`refs_v5.bib`, `paper_v5.qmd` — Quarto/Typst IEEE
source included for reproducibility).

**Scope note:** v5 does **not** include the v3 browser engine's mIoT PRACH
storm simulation, demand-follow controls, or macro/micro-to-Wi-Fi offload
screening (`browser_engine_v3.js`/`prach_engine_v3.js`) — those remain unique
to the v3 artefact below. v5 is the Python-engine-backed, paper-focused
release; v3 is the richer standalone browser engine. They are parallel, not
sequential — until reconciled, treat them as two different scopes of the same
project rather than v5 superseding v3.

---

# Multi-Tier Digital Twin v3

> **Launch link removed here to avoid confusion with v5 above.** v3's browser
> artefact (PRACH storm simulation, demand-follow controls, offload screening)
> is still fully present in this repo at `twin_console_v3.html` — open it
> directly from the file browser if you need those specific v3 features; it
> just isn't the front-page launch link anymore, v5 is.

[Open the v3 Design Document (PDF)](https://ysrao.github.io/DigitalTwin/multi-tier-twin/Multi_Tier_Digital_Twin_v3_Design_Document.pdf)

[Open the Near-RT RIC/xApp v3 Design Document (PDF)](https://ysrao.github.io/DigitalTwin/multi-tier-twin/Near_RT_RIC_xApp_v3_Design_Document.pdf)

Authors: Rao Yenamandra (`raosy@digitaltwinsim.com`), Mubanga Nsofu
(`mubanga.nsofu@vodacom.co.za`), and Asokan Ram (`asokan.ram@wrc-nc.org`).

The v3 browser artefact adds mIoT-only PRACH storm simulation, adaptive
controller comparisons, explicit Demand-follow controls and recommendations,
and separate macro/micro-to-Wi-Fi eMBB offload screening. The v2 links remain
below for reproducibility.

## Previous v2 release

[**Launch the interactive v2 artifact →**](https://ysrao.github.io/DigitalTwin/multi-tier-twin/twin_console.html)

[Open the v2 workflow and calculation guide (PDF)](https://ysrao.github.io/DigitalTwin/multi-tier-twin/Multi_Tier_Digital_Twin_v2_Workflow.pdf)

> Use the launch link above to run the artifact. Clicking `twin_console.html`
> in GitHub opens its source code because the repository view does not execute HTML.

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

The Python v2 reference dataset does not model PRACH. The separate v3 browser
engine adds aggregate PRACH contention, collisions, ACB, retries and access
failures for the cellular mIoT slice. It remains screening-grade and is not a
packet-level 3GPP conformance simulator.
