"""V7: unchanged v6 domain with noisy, 20-ms delayed horizontal velocity."""
import json,os,sys
from pathlib import Path
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.train_landing_seed606_v6 import extended_profile
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_seed606 import cli_flags,verify_controller,environment_config,ROBOT_DIR
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.policy_velocity import velocity_observation_contract

def noisy_velocity_profile(mode=None):
    profile=extended_profile(mode)
    profile['environment']['policy_velocity'].update(noise_std_mps=.1,bias_std_mps=0.,delay_steps=1)
    mode=profile['environment']['policy_velocity']['mode']
    profile['runner'].update(exp_name='landing_seed606_noisy_velocity_v7',run_name=os.environ.get('RUN_NAME') or ('seed606_noisy_velocity_v7_'+mode+'_'+os.environ.get('SLURM_JOB_ID','local')),load_model='',notes='V7 fresh weights; v6 domain and PPO unchanged; world XY Gaussian velocity noise sigma0.1mps per component per20ms; no bias; fixed delay1controlstep=20ms; ideal true velocity remains privileged critic input; assumed sensor model not OptiTrack calibration')
    return profile

def noisy_environment_config(mode=None):
    cfg=environment_config()
    def merge(target,updates):
        for k,v in updates.items():
            if isinstance(v,dict):merge(target[k],v)
            else:target[k]=v
    merge(cfg,noisy_velocity_profile(mode)['environment']);return cfg

def make_noisy_env(mode='test',velocity_mode='heading_xy'):
    from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.environment import LocomotionEnv
    from rl_x.environments.custom_mujoco.robot_locomotion.robots.silver_badger.robot_config import robot_config
    cfg=noisy_environment_config(velocity_mode);cfg['nr_envs']=1
    robot=deepcopy(robot_config);robot.update(directory_path=ROBOT_DIR,spine_locked=False)
    env=LocomotionEnv(robot,mode,20261029,False,cfg,1);verify_controller(env.initial_mj_model);return env

if __name__=='__main__':
    profile=noisy_velocity_profile();fingerprint=verify_controller()
    if '--check-only' in sys.argv:
        print(json.dumps(dict(profile=profile,fingerprint=fingerprint,velocity_contract=velocity_observation_contract(profile['environment']['policy_velocity'])),indent=2))
    else:
        sys.argv=[sys.argv[0],*cli_flags(profile),*sys.argv[1:]]
        print('V7: fresh policy; velocity noise sigma0.1m/s per component, delay20ms; v6 task unchanged',flush=True)
        from rl_x.runner.runner import Runner
        Runner().run()
