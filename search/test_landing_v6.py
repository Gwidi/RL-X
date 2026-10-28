"""Actual v6 resets and evaluation parity, including a strict speed norm cap."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial.transform import Rotation
from experiments.train_landing_seed606_v6 import extended_profile,make_extended_env
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.domain_randomization.initial_state_functions.horizontal_velocity import sample_horizontal_velocity
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_evaluation import LandingEvaluationWrapper,evaluation_heights

class ExtendedLandingTest(unittest.TestCase):
 def test_legacy_draws_are_bit_identical_and_disk_covers_all_quadrants(self):
  a=np.random.default_rng(17);b=np.random.default_rng(17)
  for _ in range(40):np.testing.assert_array_equal(sample_horizontal_velocity(a,3.),[b.uniform(-3,3),b.uniform(-3,3)])
  rng=np.random.default_rng(24);samples=np.array([sample_horizontal_velocity(rng,5.,'speed_disk') for _ in range(2048)]);r=np.linalg.norm(samples,axis=1)
  self.assertLessEqual(r.max(),5.);self.assertGreater(r.max(),4.99);self.assertAlmostEqual(np.mean(r*r),12.5,delta=.6)
  self.assertEqual(len(set(map(tuple,np.sign(samples)))),4)
  with self.assertRaises(ValueError):sample_horizontal_velocity(rng,5.,'invalid')
 def test_real_training_resets_at_both_curriculum_endpoints(self):
  e=make_extended_env('train');self.addCleanup(e.close);sampler=e.initial_state_function.factorized
  sampler.weights=np.array([0.,0.,0.,1.])
  for difficulty,angle_bound,speed_bound in [(0.,1.,.5),(1.,30.,5.)]:
   sampler.state().update(angle_difficulty=difficulty,velocity_difficulty=difficulty);heights=[];angles=[];speeds=[]
   for _ in range(64):
    e.reset();d=e.internal_state['data'];heights.append(d.qpos[2]);angles.extend(Rotation.from_quat(d.qpos[3:7],scalar_first=True).as_euler('xyz',degrees=True)[:2]);speeds.append(np.linalg.norm(d.qvel[:2]));self.assertEqual(d.qvel[2],0.)
   self.assertTrue(all(1<=h<=3 for h in heights));self.assertIn(3.,heights);self.assertTrue(any(h<2. for h in heights));self.assertLessEqual(max(np.abs(angles)),angle_bound+1e-9);self.assertLessEqual(max(speeds),speed_bound+1e-9)
   if difficulty:self.assertGreater(max(np.abs(angles)),29.);self.assertGreater(max(speeds),4.8)
  self.assertEqual(len(e.policy_observation_indices),80);self.assertEqual(len(e.critic_observation_indices),94)
 def test_full_evaluation_domain_and_paired_zero_policy_resets(self):
  e=make_extended_env();w=LandingEvaluationWrapper(e,0);self.addCleanup(w.close);cfg=e.env_config['landing_evaluation'];self.assertEqual(evaluation_heights(cfg),[3.])
  self.assertEqual(cfg['max_angle_deg'],30.);self.assertEqual(cfg['velocity_sampling'],'speed_disk');seen=[]
  for _ in range(2):
   w.configure_landing_evaluation(dict(cfg,height=3.,perturbed=True));states=[]
   for _ in range(32):
    w.reset();d=e.internal_state['data'];self.assertEqual(d.qpos[2],3.);self.assertLessEqual(np.linalg.norm(d.qvel[:2]),5.+1e-9);self.assertEqual(w.case_info['velocity_sampling'],'speed_disk');states.append(w.case_info['scenario_hash']);seen.append(np.linalg.norm(d.qvel[:2]))
   if 'reference' not in locals():reference=states
   else:self.assertEqual(reference,states)
  self.assertGreater(max(seen),4.8)
 def test_unknown_evaluation_mode_is_rejected(self):
  cfg=extended_profile()['environment']['landing_evaluation'];cfg['velocity_sampling']='bad'
  with self.assertRaises(ValueError):evaluation_heights(cfg)
if __name__=='__main__':unittest.main()
