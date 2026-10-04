# Stage A — Platform check

Date: 2026-10-03. Environment: CPU-only VM (no GPU), Python 3.12, virtualenv
(`.venv/`; system Python is PEP 668 externally managed).

## 1. Reproducible install

- `pip install sionna` → **Sionna 2.2.0** (PyTorch-based; TF-era module paths
  like `sionna.channel` / `sionna.ofdm` no longer exist — v2 layout is
  `sionna.phy`, `sionna.sys`, `sionna.rt`).
- Gotcha: `/tmp` is a 512 MB tmpfs; TF/torch wheels exceed it. Install with
  `TMPDIR=$PWD/.tmp`.

## 2. Multicell smoke test — PASSED

`tests/test_smoke_multicell.py`: 7 sites × 3 sectors (num_rings=1) = **21 cells**,
84 UEs, UMa @ 3.5 GHz, full documented chain
(topology → `set_topology` → `UMa` channel → `get_pathloss` →
`coupling_loss_db` → `geometry_sinr_db`), CPU-only.

| Metric | Value |
| --- | --- |
| Geometry SINR mean / p5 / min / max | 1.54 / −5.44 / −8.17 / 19.50 dB |
| Wall-clock total | 1.30 s (topology 0.04 s, channel 1.26 s) |
| Peak RAM | 982 MB |

SINR range is plausible for a fully loaded 21-cell UMa drop.

## 3. Feature inventory (built-in vs must-build)

Verified against installed 2.2.0 (API probes + package grep).

| Capability (Brief v2) | Status |
| --- | --- |
| 3GPP 38.901 UMa/UMi/RMa/InH/InF scenarios | ✅ built-in (`sionna.phy.channel.tr38901`) |
| Multicell topology generators (TR 38.901, hex grid) | ✅ built-in (`sionna.sys.topology`) |
| PHY abstraction / EESM / effective SINR | ✅ built-in (`sionna.sys`) |
| Inner/outer-loop link adaptation, power control | ✅ built-in (`sionna.sys`) |
| PF scheduler (SU-MIMO) | ✅ built-in (`sionna.sys.scheduling`) |
| OFDM / MIMO / FEC / NR PHY chain | ✅ built-in (`sionna.phy`) |
| Ray tracing | ✅ built-in (`sionna.rt`) |
| Packet queues per UE/slice | ❌ must-build |
| Mobility / handover behavior | ❌ must-build (no hits in package) |
| HetNet macro/small-cell association | ❌ must-build (no hits in package) |
| FDD+TDD carrier aggregation | ❌ must-build (no hits in package) |
| PRACH / mIoT access congestion | ❌ must-build (no hits in package) |
| Slice PRB budgets + borrowing | ❌ must-build (no hits in package) |
| Cluster coordination of budgets | ❌ must-build |

## 4. Open issues for Stage B

1. **Indoor-UT zero-gain edge case**: with default settings, indoor UTs at
   3.5 GHz UMa produce exactly zero channel gain (100% of indoor links;
   both O2I models). All-outdoor drops are unaffected (~4% zero-gain links
   from antenna nulls/deep fades, clipped at 150 dB). Investigate before
   corpus generation — do not silently train on it.
2. **API notes**: `get_pathloss` returns *linear* pathloss with a trailing
   symbol dim — convert `10*log10`, average symbols, then `coupling_loss_db`.
   `set_topology(*topology[0])` when `return_site_positions=True`.
3. CPU feasibility confirmed for small drops; corpus scale-up needs the GPU
   worker per the brief's resource plan.

## 5. Gate decision

**GO for Stage B** with eyes open: Sionna covers channels, topology, PHY
abstraction, link adaptation and scheduling. The system-level dynamics the
brief needs (queues, mobility/handover, HetNet, FDD+TDD CA, PRACH, slicing)
are a genuine build — that is where the engineering effort goes.
