"""V6: velocity-aware landing, horizontal speed <=5m/s, angles30deg, height3m."""
import json,os,sys
from pathlib import Path
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.train_landing_seed606_v5 import velocity_profile
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_seed606 import cli_flags,verify_controller,environment_config,ROBOT_DIR
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.policy_velocity import velocity_observation_contract

def extended_profile(mode=None):
    profile=velocity_profile(mode)
    curriculum=profile['environment']['domain_randomization']['initial_state']['energy_dissipation_curriculum']
    curriculum.update(target_height=3.,max_angle_deg=30.,max_lateral_velocity=5.,velocity_sampling='speed_disk')
    profile['environment']['landing_evaluation'].update(heights='3.0',max_angle_deg=30.,max_lateral_velocity=5.,velocity_sampling='speed_disk')
    mode=profile['environment']['policy_velocity']['mode']
    profile['runner'].update(exp_name='landing_seed606_extended_v6',run_name=os.environ.get('RUN_NAME') or ('seed606_extended_v6_'+mode+'_'+os.environ.get('SLURM_JOB_ID','local')),load_model='',notes='V6 fresh weights; ideal measured heading velocity /4mps; uniform-area horizontal speed disk <=5mps (not each component +/-5); independent roll/pitch to30deg; independent heights1to3m with25pct at3m; eval3m four conditions; physics/PD/reward/PPO asv5')
    return profile

def extended_environment_config(mode=None):
    cfg=environment_config()
    def merge(target,updates):
        for k,v in updates.items():
            if isinstance(v,dict):merge(target[k],v)
            else:target[k]=v
    merge(cfg,extended_profile(mode)['environment']);return cfg

def make_extended_env(mode='test',velocity_mode='heading_xy'):
    from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.environment import LocomotionEnv
    from rl_x.environments.custom_mujoco.robot_locomotion.robots.silver_badger.robot_config import robot_config
    cfg=extended_environment_config(velocity_mode);cfg['nr_envs']=1
    robot=deepcopy(robot_config);robot.update(directory_path=ROBOT_DIR,spine_locked=False)
    env=LocomotionEnv(robot,mode,20261028,False,cfg,1);verify_controller(env.initial_mj_model);return env

if __name__=='__main__':
    profile=extended_profile();fingerprint=verify_controller()
    if '--check-only' in sys.argv:
        print(json.dumps(dict(profile=profile,fingerprint=fingerprint,velocity_contract=velocity_observation_contract(profile['environment']['policy_velocity'])),indent=2))
    else:
        sys.argv=[sys.argv[0],*cli_flags(profile),*sys.argv[1:]]
        print('V6: fresh policy, speed norm<=5m/s, roll/pitch<=30deg, height<=3m, evaluation at3m',flush=True)
        from rl_x.runner.runner import Runner
        Runner().run()
