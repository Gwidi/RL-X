"""V8: height first, velocity only; fixed PD and v7 noisy velocity observation."""
import json,os,sys
from pathlib import Path
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.train_landing_seed606_v7 import noisy_velocity_profile
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_seed606 import cli_flags,verify_controller,environment_config,ROBOT_DIR

def velocity_only_profile():
    profile=noisy_velocity_profile('heading_xy')
    c=profile['environment']['domain_randomization']['initial_state']['energy_dissipation_curriculum']
    c.update(factorized_enabled=False,height_then_velocity_enabled=True,start_height=1.,target_height=3.,
        min_angle_deg=0.,max_angle_deg=0.,min_lateral_velocity=0.,max_lateral_velocity=5.,velocity_sampling='speed_disk',
        sequential_heights='1,1.5,2,2.5,3',sequential_speeds='0.5,1,2,3,4,5',
        sequential_window=64,sequential_threshold=.8,sequential_frontier_probability=.5,
        nominal_replay_probability=0.,replay_easy_probability=0.)
    profile['environment']['landing_evaluation'].update(heights='1,1.5,2,2.5,3',max_angle_deg=0.,
        max_lateral_velocity=5.,factorized_conditions=False,velocity_grid='0,0.5,1,2,3,4,5',
        success_boundary_threshold=.8,nominal_success_threshold=.8,frequency_multiplier=32)
    profile['runner'].update(exp_name='landing_seed606_velocity_only_v8',
        run_name=os.environ.get('RUN_NAME') or 'seed606_velocity_only_v8_'+os.environ.get('SLURM_JOB_ID','local'),
        load_model='',notes='V8 fresh; zero initial angles; per-worker height then velocity frontier curriculum; strict >80pct over64 frontier episodes; independent fixed-height fixed-speed validation; v7 sensor/PD/physics/reward/PPO unchanged')
    return profile

def velocity_only_environment_config():
    cfg=environment_config()
    def merge(a,b):
        for k,v in b.items():
            if isinstance(v,dict):merge(a[k],v)
            else:a[k]=v
    merge(cfg,velocity_only_profile()['environment']);return cfg

def make_velocity_only_env(mode='test'):
    from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.environment import LocomotionEnv
    from rl_x.environments.custom_mujoco.robot_locomotion.robots.silver_badger.robot_config import robot_config
    cfg=velocity_only_environment_config();cfg['nr_envs']=1
    robot=deepcopy(robot_config);robot.update(directory_path=ROBOT_DIR,spine_locked=False)
    env=LocomotionEnv(robot,mode,20261030,False,cfg,1);verify_controller(env.initial_mj_model);return env

if __name__=='__main__':
    profile=velocity_only_profile();fingerprint=verify_controller()
    if '--check-only' in sys.argv:print(json.dumps(dict(profile=profile,fingerprint=fingerprint),indent=2))
    else:
        sys.argv=[sys.argv[0],*cli_flags(profile),*sys.argv[1:]]
        from rl_x.runner.runner import Runner
        Runner().run()
