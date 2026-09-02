# RAN Slice Decision Platform MVP

This research MVP implements the minimum two-timescale decision workflow described in Version 4 of the paper:

- calculated link-budget/morphology coverage and IoT-density cell-capacity planning;
- normalized 13-feature per-cell state;
- custom NumPy `13 → 64 → 64 → 7` categorical PPO actor and matching critic;
- hard, tuned-soft, greedy and PPO matched evaluation;
- explicit guarded decision status and negative-gain reporting;
- browser dashboard served by a standard-library Python API.

Run:

```sh
cd platform
python3 server.py
```

Then open <http://127.0.0.1:8765>.

Test:

```sh
cd platform
python3 -m unittest -v test_engine.py
```

The MVP uses reproducible synthetic trace replay. It does not prove carrier performance, five- or seven-nines reliability, or perform direct RIC control. The categorical allocation templates are an initial implementation step before a continuous `3N_cells` action head.

`P6_ppo_advantage_demo` is a deliberately constructed functional demonstration with alternating eMBB-, URLLC-, and IoT-dominant phases. It tests whether the PPO implementation can select phase-appropriate allocations when a rule-based soft baseline preserves protected floors. It automatically uses at least 32,768 training samples for stable reproduction and is explicitly not research or carrier evidence.
