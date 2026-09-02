import math
import unittest

import numpy as np

import radio
from agents import (BranchingDQNAgent, BranchingPPOAgent, DQNAgent, PPOAgent,
                    train_dqn, train_ppo)
from compare import (OBJECTIVES, RULE_BASED, SCENARIOS, build_config,
                     feasibility_verdict, optimize, optimize_thresholds,
                     run_comparison)
from mobility import DEFAULT_SESSION_TYPES, MobilityLimits, SessionPopulation
from twin import (DEFAULT_CELL_COUNTS, MultiTierTwin, TwinConfig, branch_sizes,
                  default_tiers)

FAST = dict(episode_steps=4, control_interval_ticks=15, n_sessions=250,
            traffic_profile="flat")   # steady load — dynamics tested separately


class RadioTests(unittest.TestCase):
    def test_pathloss_grows_with_distance_and_frequency(self):
        near = radio.pathloss("UMa", 100, 25, 1.5, 3.5)
        far = radio.pathloss("UMa", 1000, 25, 1.5, 3.5)
        high = radio.pathloss("UMa", 100, 25, 1.5, 28.0)
        self.assertGreater(far, near)
        self.assertGreater(high, near)

    def test_taller_antenna_reaches_further(self):
        mapl = radio.max_path_loss_db(46, 17, 0, 360e3, 7, 0, 10)
        low = radio.cell_radius_m("UMa", mapl, 15, 1.5, 3.5)
        high = radio.cell_radius_m("UMa", mapl, 40, 1.5, 3.5)
        self.assertGreater(high, low)

    def test_slant_range_matches_geometry(self):
        # Straight overhead the slant range is exactly the orbit altitude.
        self.assertAlmostEqual(radio.slant_range_km(600, 90), 600.0, places=3)
        self.assertGreater(radio.slant_range_km(600, 10), radio.slant_range_km(600, 60))

    def test_doppler_peaks_at_low_elevation_and_rate_peaks_at_zenith(self):
        # TR 38.821 reference for 600 km / 2 GHz: about +/-48 kHz, 544 Hz/s.
        low = abs(radio.doppler_shift_hz(2.0, 600, 10))
        zenith = abs(radio.doppler_shift_hz(2.0, 600, 90))
        self.assertGreater(low, zenith)
        self.assertTrue(40e3 < low < 50e3, low)
        rate_zenith = abs(radio.doppler_rate_hz_s(2.0, 600, 89))
        rate_low = abs(radio.doppler_rate_hz_s(2.0, 600, 10))
        self.assertGreater(rate_zenith, rate_low)
        self.assertTrue(400 < rate_zenith < 700, rate_zenith)

    def test_rain_attenuation_is_negligible_at_s_band_and_severe_at_ka(self):
        s_band = radio.rain_attenuation_db(2.0, 30, 25)
        ka_band = radio.rain_attenuation_db(28.0, 30, 25)
        self.assertLess(s_band, 0.5)
        self.assertGreater(ka_band, 10.0)

    def test_ici_derates_spectral_efficiency(self):
        clean = radio.spectral_efficiency_with_ici(4.0, 0.0)
        noisy = radio.spectral_efficiency_with_ici(4.0, 0.2)
        self.assertAlmostEqual(clean, 4.0, places=6)
        self.assertLess(noisy, clean)

    def test_o2i_high_loss_exceeds_low_loss(self):
        self.assertGreater(radio.o2i_penetration_loss_db(3.5, True),
                           radio.o2i_penetration_loss_db(3.5, False))


class MobilityTests(unittest.TestCase):
    def test_sessions_stay_inside_the_service_area(self):
        pop = SessionPopulation(300, (1.0, 1.0), rng=np.random.default_rng(0))
        for _ in range(200):
            pop.step(1.0)
        self.assertTrue((pop.x_km >= 0).all() and (pop.x_km <= 1.0).all())
        self.assertTrue((pop.y_km >= 0).all() and (pop.y_km <= 1.0).all())

    def test_velocity_vectors_match_speed(self):
        pop = SessionPopulation(200, (1.0, 1.0), rng=np.random.default_rng(1))
        speeds = np.linalg.norm(pop.velocity_kmh, axis=1)
        self.assertTrue(np.allclose(speeds, pop.speed_kmh))

    def test_miot_sessions_respect_their_speed_limit(self):
        pop = SessionPopulation(500, (1.0, 1.0), {"eMBB": 0, "URLLC": 0, "mIoT": 1},
                                rng=np.random.default_rng(2))
        limit = next(t for t in DEFAULT_SESSION_TYPES if t.name == "mIoT").max_speed_kmh
        self.assertLessEqual(pop.speed_kmh.max(), limit)

    def test_interference_lowers_spectral_efficiency_under_load(self):
        from twin import default_tiers
        tiers = [t for t in default_tiers() if t.name == "macro_mid"]
        base = dict(tiers=tiers, cell_counts={"macro_mid": 4}, area_km=(0.6, 0.6),
                    n_sessions=1000, session_mix={"eMBB": 1.0, "URLLC": 0.0, "mIoT": 0.0},
                    episode_steps=4, control_interval_ticks=15,
                    traffic_profile="flat", shadow_fading=False)
        se = {}
        for interf in (False, True):
            tw = MultiTierTwin(TwinConfig(interference=interf, **base))
            tw.reset()
            for _ in range(4):
                _, _, _, info = tw.step([0] * tw.n_tiers + [0])
            se[interf] = info["mean_spectral_efficiency"]
        self.assertLess(se[True], se[False] * 0.9)   # meaningful degradation

    def test_shadow_fading_perturbs_rsrp_and_relaxes_without_it(self):
        on = MultiTierTwin(TwinConfig(shadow_fading=True, **FAST))
        on.reset(1)
        self.assertTrue(np.abs(on.shadow).sum() > 0)
        off = MultiTierTwin(TwinConfig(shadow_fading=False, **FAST))
        off.reset(1)
        self.assertEqual(float(np.abs(off.shadow).sum()), 0.0)

    def test_traffic_profiles_load_and_vary(self):
        from twin import load_traffic_shape
        self.assertIsNone(load_traffic_shape("flat"))
        self.assertIsNone(load_traffic_shape("diurnal"))
        shape = load_traffic_shape("P4_massive_IoT")
        self.assertEqual(shape.shape[0], 3)
        self.assertLessEqual(shape.max(), 1.0 + 1e-9)
        tw = MultiTierTwin(TwinConfig(traffic_profile="diurnal",
                                      episode_steps=8, control_interval_ticks=10,
                                      n_sessions=200))
        tw.reset()
        factors = []
        for _ in range(8):
            tw._refresh_load_factor()
            factors.append(tw.load_factor.copy())
            tw.step([0] * tw.n_tiers + [0])
        spread = np.ptp([f[0] for f in factors])
        self.assertGreater(spread, 0.1)   # eMBB load actually moves over the day

    def test_beam_tracking_derates_at_speed(self):
        limits = MobilityLimits(60.0, 120.0, beam_refinement_ms=0.5)
        self.assertAlmostEqual(limits.derate(0.0, 28.0), 1.0)
        self.assertLess(limits.derate(90.0, 28.0), limits.derate(30.0, 28.0))
        self.assertEqual(limits.derate(150.0, 28.0), 0.0)


class TwinTests(unittest.TestCase):
    def test_step_returns_a_well_formed_transition(self):
        twin = MultiTierTwin(TwinConfig(**FAST))
        obs = twin.reset()
        self.assertEqual(obs.shape, (twin.obs_dim,))
        obs, reward, done, info = twin.step(0)
        self.assertEqual(obs.shape, (twin.obs_dim,))
        self.assertTrue(np.isfinite(obs).all())
        self.assertIsInstance(reward, float)
        for key in ("satisfaction", "handover_delta", "session_continuity",
                    "tier_sessions", "tier_utilization"):
            self.assertIn(key, info)

    def test_episode_terminates_after_configured_steps(self):
        twin = MultiTierTwin(TwinConfig(**FAST))
        twin.reset()
        done = False
        steps = 0
        while not done:
            _, _, done, _ = twin.step(0)
            steps += 1
            self.assertLessEqual(steps, FAST["episode_steps"])
        self.assertEqual(steps, FAST["episode_steps"])

    def test_runs_without_wifi_or_satellite(self):
        counts = {k: v for k, v in DEFAULT_CELL_COUNTS.items()
                  if k not in ("wifi7_indoor", "ntn_leo")}
        tiers = [t for t in default_tiers() if t.name in counts]
        twin = MultiTierTwin(TwinConfig(tiers=tiers, cell_counts=counts, **FAST))
        twin.reset()
        _, _, _, info = twin.step(0)
        self.assertNotIn("wifi7_indoor", info["tier_sessions"])
        self.assertNotIn("ntn_leo", info["tier_sessions"])
        self.assertGreater(info["attached_sessions"], 0)

    def test_runs_with_a_single_tier(self):
        tiers = [t for t in default_tiers() if t.name == "macro_mid"]
        twin = MultiTierTwin(TwinConfig(tiers=tiers, cell_counts={"macro_mid": 7}, **FAST))
        twin.reset()
        _, _, _, info = twin.step(0)
        self.assertEqual(set(info["tier_sessions"]), {"macro_mid"})

    def test_indoor_sessions_never_attach_to_the_satellite(self):
        twin = MultiTierTwin(TwinConfig(indoor_fraction=1.0, **FAST))
        twin.reset()
        twin.step(0)
        ntn_cells = np.flatnonzero(twin.cell_is_ntn)
        if ntn_cells.size:
            self.assertEqual(int(np.isin(twin.ho.serving, ntn_cells).sum()), 0)

    def test_deterministic_for_a_fixed_seed(self):
        rewards = []
        for _ in range(2):
            twin = MultiTierTwin(TwinConfig(**FAST))
            twin.reset(11)
            rewards.append([twin.step(3)[1] for _ in range(3)])
        self.assertEqual(rewards[0], rewards[1])

    def test_tier_overrides_rederive_the_resource_grid(self):
        base = MultiTierTwin(TwinConfig(**FAST))
        base_prbs = base.plan_summary()["tiers"]
        mm0 = next(t for t in base_prbs if t["tier"] == "macro_mid")["prbs_per_cell"]
        over = MultiTierTwin(TwinConfig(
            tier_overrides={"macro_mid": {"bandwidth_mhz": 200, "carriers": 2,
                                          "numerology_khz": 60, "mimo_layers": 8,
                                          "scenario": "UMi"}}, **FAST))
        mm1 = next(t for t in over.plan_summary()["tiers"]
                   if t["tier"] == "macro_mid")
        self.assertGreater(mm1["prbs_per_cell"], mm0)          # 400 MHz re-derived
        self.assertEqual(mm1["morphology"], "UMi")
        self.assertEqual(mm1["pathloss_model"], "TR38.901-UMi")
        self.assertIn("8x", mm1["mimo"])

    def test_mimo_gain_raises_cell_capacity(self):
        lo = MultiTierTwin(TwinConfig(
            tier_overrides={"macro_mid": {"mimo_layers": 1, "mimo_mode": "SU"}}, **FAST))
        hi = MultiTierTwin(TwinConfig(
            tier_overrides={"macro_mid": {"mimo_layers": 8, "mimo_mode": "MU"}}, **FAST))
        ci = lo.tier_index["macro_mid"]
        self.assertLess(lo.cell_mimo_gain[lo.cell_tier == ci][0],
                        hi.cell_mimo_gain[hi.cell_tier == ci][0])

    def test_explicit_cell_coordinates_are_honoured(self):
        from twin import Cell
        cells = [Cell("a", "macro_mid", 0.25, 0.25, 25, 49, 100),
                 Cell("b", "macro_mid", 0.75, 0.75, 25, 49, 100)]
        tiers = [t for t in default_tiers() if t.name == "macro_mid"]
        twin = MultiTierTwin(TwinConfig(tiers=tiers, cells=cells, **FAST))
        self.assertEqual(twin.n_cells, 2)
        self.assertAlmostEqual(float(twin.cell_x[0]), 0.25)

    def test_handover_accounting_is_consistent(self):
        twin = MultiTierTwin(TwinConfig(**FAST))
        twin.reset()
        for _ in range(FAST["episode_steps"]):
            twin.step(0)
        c = twin.ho.counters
        self.assertEqual(c.attempts,
                         c.successes + c.failures_late + c.failures_admission
                         + c.failures_speed)
        self.assertLessEqual(c.successes, c.attempts)


class AgentTests(unittest.TestCase):
    def test_actor_critic_keeps_its_original_shape_by_default(self):
        from engine import ActorCritic, TEMPLATES
        model = ActorCritic()
        self.assertEqual(model.aw[0].shape, (13, 64))
        self.assertEqual(model.aw[2].shape, (64, len(TEMPLATES)))

    def test_agents_have_one_head_per_tier_plus_steering(self):
        twin = MultiTierTwin(TwinConfig(**FAST))
        expected = branch_sizes(twin.n_tiers)
        self.assertEqual(twin.branch_sizes, expected)
        ppo = BranchingPPOAgent(twin.obs_dim, twin.branch_sizes)
        dqn = BranchingDQNAgent(twin.obs_dim, twin.branch_sizes)
        self.assertEqual(ppo.net.tw[0].shape, (twin.obs_dim, 64))
        self.assertEqual([w.shape[1] for w in ppo.net.hw], expected)
        self.assertEqual([w.shape[1] for w in dqn.online.hw], expected)

    def test_each_branch_is_a_distribution_and_action_has_full_length(self):
        twin = MultiTierTwin(TwinConfig(**FAST))
        obs = twin.reset()
        agent = BranchingPPOAgent(twin.obs_dim, twin.branch_sizes)
        for head in agent.net.probs(obs[None, :]):
            self.assertAlmostEqual(float(head[0].sum()), 1.0, places=6)
            self.assertTrue((head[0] >= 0).all())
        action, logp = agent.act(obs)
        self.assertEqual(len(action), twin.n_tiers + 1)
        self.assertLessEqual(action[-1], twin.branch_sizes[-1] - 1)

    def test_training_loops_produce_history_and_beat_nothing_gracefully(self):
        twin = MultiTierTwin(TwinConfig(**FAST))
        p, ppo_info = train_ppo(twin, episodes=2)
        d, dqn_info = train_dqn(twin, episodes=2)
        self.assertEqual(len(ppo_info["reward_history"]), 2)
        self.assertEqual(len(dqn_info["reward_history"]), 2)
        self.assertIn(str(twin.obs_dim), ppo_info["network"])
        # A trained greedy action is a valid per-tier vector.
        obs = twin.reset()
        for agent in (p, d):
            self.assertEqual(len(agent.act(obs, greedy=True)[0]), twin.n_tiers + 1)

    def test_dqn_epsilon_decays(self):
        agent = BranchingDQNAgent(8, [4, 3], epsilon_decay_steps=10)
        start = agent.epsilon
        agent.steps = 20
        self.assertLess(agent.epsilon, start)

    def test_legacy_agent_aliases_exist(self):
        self.assertIs(PPOAgent, BranchingPPOAgent)
        self.assertIs(DQNAgent, BranchingDQNAgent)


class ComparisonTests(unittest.TestCase):
    def test_rule_based_policies_return_valid_actions(self):
        twin = MultiTierTwin(TwinConfig(**FAST))
        obs = twin.reset()
        for name, policy in RULE_BASED.items():
            action = policy(obs)                 # 2-vector: (template, posture)
            self.assertEqual(len(action), 2, name)
            twin.step(action)                    # accepted and broadcast per tier
            self.assertLess(action[0], 9)
            self.assertLess(action[1], twin.branch_sizes[-1])

    def test_every_scenario_builds_and_steps(self):
        for name in SCENARIOS:
            cfg = build_config({"scenario": name, **FAST})
            twin = MultiTierTwin(cfg)
            twin.reset()
            _, _, _, info = twin.step(0)
            self.assertGreater(info["attached_sessions"], 0, name)

    def test_comparison_reports_all_policies_and_a_guarded_decision(self):
        data = run_comparison({"train_episodes": 2, **FAST})
        self.assertEqual(set(data["results"]),
                         {"static", "demand_follow", "rule_based", "ppo", "dqn"})
        self.assertIn(data["decision"]["status"].split(" —")[0],
                      {"PASS", "UNDERPERFORMING", "NO MATERIAL BENEFIT"})
        self.assertIn(data["best_policy"], data["results"])

    def test_every_objective_optimizes_without_regressing_its_score(self):
        for objective in OBJECTIVES:
            data = optimize({"objective": objective, "passes": 1, **FAST})
            self.assertGreaterEqual(data["optimized"]["score"],
                                    data["baseline"]["score"], objective)
            self.assertIn(data["feasibility"]["status"],
                          {"FEASIBLE", "COVERAGE-LIMITED", "CAPACITY-LIMITED",
                           "COVERAGE-AND-CAPACITY-LIMITED"})

    def test_legacy_optimize_thresholds_still_works(self):
        data = optimize_thresholds({"passes": 1, **FAST})
        self.assertEqual(data["objective"], "handover")

    def test_feasibility_flags_an_undersized_deployment(self):
        # One low-band macro over a large rural area: coverage-limited.
        cfg = build_config({"scenario": "coverage_hole", **FAST})
        twin = MultiTierTwin(cfg)
        from compare import evaluate
        result = evaluate(twin, RULE_BASED["rule_based"], cfg.seed + 500)
        verdict = feasibility_verdict(result, twin)
        self.assertNotEqual(verdict["status"], "FEASIBLE")
        self.assertFalse(verdict["thresholds_can_help"])

    def test_preview_reports_capacity_and_coverage_without_training(self):
        from compare import preview
        d = preview({"scenario": "urban_dense", **FAST})
        cap, cov = d["capacity"], d["coverage"]
        self.assertGreater(cap["network_mbps"], 0)
        self.assertEqual(set(cap["by_slice_mbps"]), {"eMBB", "URLLC", "mIoT"})
        # URLLC-eligible capacity excludes the Wi-Fi and NTN tiers.
        self.assertLess(cap["by_slice_mbps"]["URLLC"], cap["by_slice_mbps"]["eMBB"])
        self.assertGreaterEqual(cov["terrestrial_best"], 0.0)
        for t in d["plan"]["tiers"]:
            self.assertGreaterEqual(t["tier_capacity_mbps"], 0)
            self.assertLessEqual(t["area_covered"], 1.0)

    def test_preview_reflects_bandwidth_and_cell_count_overrides(self):
        from compare import preview
        base = preview({"scenario": "urban_dense", **FAST})["capacity"]["network_mbps"]
        bigger = preview({"scenario": "urban_dense",
                          "tier_overrides": {"mmwave": {"bandwidth_mhz": 800, "carriers": 2}},
                          **FAST})["capacity"]["network_mbps"]
        self.assertGreater(bigger, base)
        fewer = preview({"scenario": "urban_dense",
                         "cell_counts": {"macro_mid": 1}, **FAST})
        mm = next(t for t in fewer["plan"]["tiers"] if t["tier"] == "macro_mid")
        self.assertEqual(mm["cells"], 1)

    def test_comparison_carries_feasibility_and_per_tier_action_space(self):
        data = run_comparison({"train_episodes": 2, **FAST})
        self.assertEqual(data["action_space"]["type"], "per_tier_branching")
        self.assertIn("feasibility", data)
        self.assertIn("tier_intent", data["results"]["dqn"])


if __name__ == "__main__":
    unittest.main()
