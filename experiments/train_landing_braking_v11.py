"""Experimental V11: low drops with actor-visible smooth braking reference."""
import json,os,sys
from pathlib import Path
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.train_landing_seed606_v9 import focused_profile,focused_environment_config
from experiments.train_landing_seed606_v10 import verify_parent
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_seed606 import cli_flags,verify_controller,ROBOT_DIR
ROOT=Path(__file__).resolve().parents[1]
def braking_profile():
 p=focused_profile();p['environment']['reward']['base_vel_coeff']=0.
 p['environment']['landing_braking']=dict(enabled=True,duration=2.5,tracking_weight=5.,tracking_sigma=.5)
 c=p['environment']['domain_randomization']['initial_state']['energy_dissipation_curriculum']
 c.update(start_height=.75,target_height=1.5,max_lateral_velocity=5.,focused_speed_min=1.,focused_speed_max=5.,focused_boundary_probability=.25)
 p['environment']['landing_evaluation'].update(heights='0.75,1,1.25,1.5',velocity_grid='0,1,2,3,4,5',selection_speed_min=1.,selection_speed_max=5.,frequency_multiplier=32)
 p['runner'].update(exp_name='landing_braking_reference_v11',run_name=os.environ.get('RUN_NAME') or 'braking_reference_v11_'+os.environ.get('SLURM_JOB_ID','local'),load_model=str(Path(os.environ.get('LANDING_BRAKING_CHECKPOINT',str(ROOT/'docs/artifacts/v9_completed_9412383_2026-11-02/best.model'))).resolve()),notes='Experimental branch; v9 parent; drops0.75to1.5m, horizontal norm<=5mps; measured precontact world velocity frozen at touchdown; smoothstep target tozero in2.5s; goal channels headingmps; tracking5*dt*exp(-error2/.5^2); final0.5s stability onlyafter2.5s; no step bonus; original safety/PD/noisy sensor/PPO;5s horizon')
 return p
def braking_environment_config():
 cfg=focused_environment_config()
 def merge(a,b):
  for k,v in b.items():
   if isinstance(v,dict):merge(a[k],v)
   else:a[k]=v
 merge(cfg,braking_profile()['environment']);return cfg
def make_braking_env(mode='test'):
 from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.environment import LocomotionEnv
 from rl_x.environments.custom_mujoco.robot_locomotion.robots.silver_badger.robot_config import robot_config
 cfg=braking_environment_config();cfg['nr_envs']=1;robot=deepcopy(robot_config);robot.update(directory_path=ROBOT_DIR,spine_locked=False)
 env=LocomotionEnv(robot,mode,20261103,False,cfg,1);verify_controller(env.initial_mj_model);return env
if __name__=='__main__':
 p=braking_profile();parent=verify_parent(p['runner']['load_model']);fingerprint=verify_controller()
 if '--check-only' in sys.argv:print(json.dumps(dict(profile=p,parent=parent,fingerprint=fingerprint),indent=2))
 else:
  sys.argv=[sys.argv[0],*cli_flags(p),*sys.argv[1:]]
  from rl_x.runner.runner import Runner
  Runner().run()
