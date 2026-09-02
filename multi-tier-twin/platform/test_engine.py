import unittest
from engine import Scenario, cell_plan, coverage_plan, load_rows, run_evaluation, state_vector
import numpy as np

class EngineTests(unittest.TestCase):
    def test_cell_plan_respects_iot_density(self):
        p=cell_plan(Scenario(iot_density_km2=2_000_000,area_km2=1,registered_capacity_cell=100_000))
        self.assertEqual(p['required_cells'],20)
    def test_coverage_is_calculated_from_radio_inputs(self):
        urban=coverage_plan(Scenario(morphology='urban'))
        dense=coverage_plan(Scenario(morphology='dense_urban'))
        self.assertGreater(urban['estimated_cell_radius_km'],0)
        self.assertGreaterEqual(dense['coverage_cells'],urban['coverage_cells'])
    def test_state_has_exact_13_features(self):
        row=load_rows('P1_balanced_busy_hour')[0]
        self.assertEqual(state_vector(row,np.array([1,0,0,0,0,0,0]),Scenario()).shape,(13,))
    def test_end_to_end_has_guarded_decision(self):
        d=run_evaluation({'train_steps':512})
        self.assertIn(d['decision']['status'],{'PASS — PPO preferred','NO MATERIAL BENEFIT','UNDERPERFORMING','SLA/FEASIBILITY FAIL','INSUFFICIENT EVIDENCE'})
        self.assertEqual(set(d['results']),{'hard','soft','greedy','ppo'})
    def test_demo_profile_is_explicitly_labeled(self):
        d=run_evaluation({'profile':'P6_ppo_advantage_demo','train_steps':2048,'material_gain_pct':2})
        self.assertEqual(d['evidence_class'],'functional_demonstration_only')
        self.assertIn('demonstration',d['limitations'][0].lower())
        self.assertEqual(d['decision']['status'],'PASS — PPO preferred')

if __name__=='__main__': unittest.main()
