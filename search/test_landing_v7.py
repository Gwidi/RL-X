"""Noisy velocity is an actor sensor change, not a physical task change."""
import sys,unittest,json,tempfile,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from ml_collections import ConfigDict
from experiments.train_landing_seed606_v6 import extended_profile,make_extended_env
from experiments.train_landing_seed606_v7 import noisy_velocity_profile,make_noisy_env
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_evaluation import LandingEvaluationWrapper

class NoisyVelocityTrainingTest(unittest.TestCase):
 def test_profile_changes_sensor_only(self):
  before=extended_profile();after=noisy_velocity_profile();self.assertEqual(before['algorithm'],after['algorithm']);a=before['environment'];b=after['environment'];sensor=b.pop('policy_velocity');a.pop('policy_velocity');self.assertEqual(a,b)
  self.assertEqual(sensor['noise_std_mps'],.1);self.assertEqual(sensor['delay_steps'],1);self.assertEqual(sensor['bias_std_mps'],0.);self.assertEqual(after['runner']['load_model'],'')
 def test_paired_physics_and_noisy_eval_even_without_physics_randomization(self):
  wrappers=[LandingEvaluationWrapper(make_extended_env(),0),LandingEvaluationWrapper(make_noisy_env(),0)]
  for w in wrappers:self.addCleanup(w.close)
  observations=[]
  for w in wrappers:
   cfg=w.env.env_config['landing_evaluation'];w.configure_landing_evaluation(dict(cfg,height=3.,perturbed=True));observations.append(w.reset()[0])
  self.assertEqual(wrappers[0].case_info['scenario_hash'],wrappers[1].case_info['scenario_hash']);e=wrappers[1].env;indices=e.policy_velocity_obs_idx
  self.assertGreater(np.linalg.norm(observations[0][indices]-observations[1][indices]),0)
  self.assertEqual(e.internal_state['env_curriculum_coeff'],0.)
  for _ in range(12):
   for w in wrappers:w.step(np.zeros(13))
   np.testing.assert_array_equal(wrappers[0].env.internal_state['data'].qpos,e.internal_state['data'].qpos)
  self.assertEqual(len(e.policy_observation_indices),80);self.assertEqual(len(e.critic_observation_indices),94)
 def test_one_step_delay_cache_and_repeatable_sensor_reset(self):
  e=make_noisy_env();self.addCleanup(e.close);e.reset();d=e.internal_state['data'];d.qpos[3:7]=[1,0,0,0];sensor=e.policy_velocity_observation;d.qvel[:3]=[1,0,0];sensor.reset(e);rngstate=json.dumps(e.np_rng.bit_generator.state,sort_keys=True);a=sensor.observe(e)
  np.testing.assert_array_equal(sensor.observe(e),a)
  d.time+=.02;d.qvel[0]=3.;b=sensor.observe(e);np.testing.assert_array_equal(a,b)
  d.time+=.02;c=sensor.observe(e);self.assertLess(abs(c[0]-.75),.15)
  self.assertEqual(json.dumps(e.np_rng.bit_generator.state,sort_keys=True),rngstate)
  d.qvel[0]=1.;sensor.reset(e);np.testing.assert_array_equal(sensor.observe(e),a)
 def test_ideal_v6_checkpoint_is_rejected_by_sensor_contract(self):
  from rl_x.algorithms.ppo.flax.ppo import PPO
  p=noisy_velocity_profile()
  with tempfile.TemporaryDirectory() as folder:
   path=Path(folder)/'ideal.model'
   with zipfile.ZipFile(path,'w') as z:z.writestr('config_environment.json',json.dumps(extended_profile()['environment']))
   cfg=ConfigDict(dict(runner=dict(load_model=str(path)),environment=p['environment']))
   with self.assertRaisesRegex(ValueError,'velocity observation contract'):PPO.load(cfg,None,None,None,None,set())
if __name__=='__main__':unittest.main()
