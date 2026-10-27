"""Velocity semantics, no actor leakage, checkpoint safety and physics parity."""
import json,sys,tempfile,unittest,zipfile
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from ml_collections import ConfigDict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.train_landing_seed606_v5 import velocity_profile,make_velocity_report_env
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.policy_velocity import heading_velocity_xy,tracking_velocity_features,PolicyVelocityObservation,velocity_observation_contract

class PolicyVelocityTest(unittest.TestCase):
    def test_world_axes_yaw_and_vertical_motion(self):
        for yaw,v,expected in [(0,[3,-2,0],[3,-2]),(90,[3,0,0],[0,-3]),(180,[3,0,0],[-3,0])]:
            r=Rotation.from_euler('z',yaw,degrees=True).as_matrix();np.testing.assert_allclose(heading_velocity_xy(v,r),expected,atol=1e-12)
        for roll,pitch in [(10,10),(-10,10),(0,89.99999999),(0,90)]:
            r=Rotation.from_euler('xyz',[roll,pitch,20],degrees=True).as_matrix();np.testing.assert_allclose(heading_velocity_xy([0,0,-5],r),0,atol=1e-12)
    def test_tracking_frame_calibration_and_marker_lever_arm(self):
        r=Rotation.from_euler('z',90,degrees=True).as_matrix();w=np.array([0.,0.,2.]);offset=np.array([.2,0,0]);vbase=np.array([3.,-1.,0]);vmarker=vbase+np.cross(w,r@offset)
        calibration=Rotation.from_euler('x',90,degrees=True).as_matrix();vtracking=calibration.T@vmarker
        np.testing.assert_allclose(tracking_velocity_features(vtracking,calibration,r,w,offset),[-.25,-.75],atol=1e-12)
    def test_masked_and_visible_have_same_actor_and_critic_shapes(self):
        for mode in ['masked','heading_xy']:
            w=make_velocity_report_env(mode);self.addCleanup(w.close);e=w.env
            self.assertEqual(len(e.policy_observation_indices),80);self.assertEqual(len(e.critic_observation_indices),94)
            self.assertEqual(len(np.intersect1d(e.policy_velocity_obs_idx,e.critic_observation_indices)),0)
            self.assertEqual(len(np.intersect1d(e.imu_linear_vel_obs_idx,e.policy_observation_indices)),0)
            w.reset();d=e.internal_state['data'];d.qvel[:3]=[3.,-2.,-5.];d.qpos[3:7]=Rotation.from_euler('xyz',[10,10,0],degrees=True).as_quat(scalar_first=True);mujoco.mj_forward(e.internal_state['mj_model'],d);e.policy_velocity_observation.reset(e);obs=e.get_observation(np.zeros(13))
            np.testing.assert_allclose(obs[e.policy_velocity_obs_idx],[0,0] if mode=='masked' else [.75,-.5],atol=1e-12)
    def test_velocity_features_never_clip_high_speed(self):
        w=make_velocity_report_env();self.addCleanup(w.close);e=w.env;w.reset();d=e.internal_state['data'];d.qvel[:3]=[8,0,0];e.policy_velocity_observation.reset(e)
        self.assertAlmostEqual(e.get_observation(np.zeros(13))[e.policy_velocity_obs_idx[0]],2.)
    def test_identical_actions_produce_identical_physics_across_modes(self):
        aa=make_velocity_report_env('masked');bb=make_velocity_report_env('heading_xy');self.addCleanup(aa.close);self.addCleanup(bb.close)
        cfg=aa.env.env_config['landing_evaluation']
        for w in [aa,bb]:w.configure_landing_evaluation(dict(cfg,height=2.2,perturbed=True,seed=20261029));w.reset()
        self.assertEqual(aa.case_info['scenario_hash'],bb.case_info['scenario_hash'])
        self.assertEqual(aa.case_info['policy_velocity_mode'],'masked')
        self.assertEqual(bb.case_info['policy_velocity_mode'],'heading_xy')
        self.assertEqual(json.loads(bb.case_info['policy_velocity_contract'])['version'],'base_heading_xy_v1')
        for k in range(48):
            action=np.sin(np.arange(13)+k*.03)*.08
            oa,_,ta,xa,_=aa.step(action);ob,_,tb,xb,_=bb.step(action)
            np.testing.assert_allclose(aa.env.internal_state['data'].qpos,bb.env.internal_state['data'].qpos,atol=0,rtol=0)
            np.testing.assert_allclose(oa[aa.env.critic_observation_indices],ob[bb.env.critic_observation_indices],atol=0,rtol=0)
            self.assertEqual((ta,xa),(tb,xb))
            if ta or xa:break
    def test_delay_cache_noise_and_reset_are_independent_of_physics_rng(self):
        w=make_velocity_report_env();self.addCleanup(w.close);e=w.env;w.reset()
        sensor=PolicyVelocityObservation(dict(mode='heading_xy',scale_mps=4.,delay_steps=2,noise_std_mps=0.))
        e.internal_state['data'].qvel[:3]=[1,0,0];before=json.dumps(e.np_rng.bit_generator.state,sort_keys=True);sensor.reset(e);first=sensor.observe(e)
        self.assertEqual(json.dumps(e.np_rng.bit_generator.state,sort_keys=True),before)
        np.testing.assert_array_equal(first,sensor.observe(e))
        for i,speed in enumerate([2,3,4],1):
            e.internal_state['data'].time=.02*i;e.internal_state['data'].qvel[0]=speed;value=sensor.observe(e)
            self.assertAlmostEqual(value[0],.25 if i<3 else .5)
        e.internal_state['data'].qvel[0]=8;sensor.reset(e);self.assertAlmostEqual(sensor.observe(e)[0],2.)
        noise=PolicyVelocityObservation(dict(mode='heading_xy',noise_std_mps=.2));before=json.dumps(e.np_rng.bit_generator.state,sort_keys=True);noise.reset(e);x=noise.observe(e);np.testing.assert_array_equal(x,noise.observe(e));self.assertEqual(json.dumps(e.np_rng.bit_generator.state,sort_keys=True),before)
        self.assertGreater(np.linalg.norm(x-[2,0]),0)
    def test_invalid_configs(self):
        for cfg in [dict(mode='bad'),dict(mode='heading_xy',scale_mps=0),dict(mode='heading_xy',noise_std_mps=-1),dict(mode='heading_xy',delay_steps=1.5),dict(mode='heading_xy',schema='unknown')]:
            with self.assertRaises(ValueError):velocity_observation_contract(cfg)
    def test_checkpoint_rejects_old_or_masked_velocity_contract(self):
        from rl_x.algorithms.ppo.flax.ppo import PPO
        p=velocity_profile()
        for saved in [{},dict(mode='masked',schema='base_heading_xy_v1',scale_mps=4.)]:
            with tempfile.TemporaryDirectory() as folder:
                model=Path(folder)/'test.model'
                with zipfile.ZipFile(model,'w') as z:z.writestr('config_environment.json',json.dumps(dict(imu_history_length=5,landing_physics_sha256=p['environment']['landing_physics_sha256'],policy_velocity=saved)))
                cfg=ConfigDict(dict(runner=dict(load_model=str(model)),environment=p['environment']))
                with self.assertRaisesRegex(ValueError,'velocity observation contract'):PPO.load(cfg,None,None,None,None,set())
    def test_profile_changes_only_observation_and_run_identity(self):
        from experiments.train_landing_seed606_v4 import stable_profile
        old=stable_profile();new=velocity_profile()
        self.assertEqual(old['algorithm'],new['algorithm']);env=dict(new['environment']);env.pop('policy_velocity');self.assertEqual(old['environment'],env)
        self.assertEqual(new['runner']['load_model'],'');self.assertEqual(new['environment']['policy_velocity']['mode'],'heading_xy')
if __name__=='__main__':unittest.main()
