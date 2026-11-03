"""V10 comparison arm: retain body-speed penalty, start it3s after touchdown."""
import json,os,sys
from pathlib import Path
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.train_landing_seed606_v10 import free_settling_profile,verify_parent as verify_v9_parent
from experiments.train_landing_seed606_v9 import focused_environment_config
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_seed606 import cli_flags,verify_controller,ROBOT_DIR

def grace3s_profile():
    profile=free_settling_profile()
    profile['environment']['reward'].update(base_vel_coeff=2.,post_touchdown_motion_grace_time=3.)
    profile['runner'].update(exp_name='landing_seed606_grace3s_v10',
        run_name=os.environ.get('RUN_NAME') or 'seed606_grace3s_v10_'+os.environ.get('SLURM_JOB_ID','local'),
        notes='V10 grace3s independent continuation from same v9 parent as no-penalty arm; ONLY environment change versus v9: motion grace0.5to3s; body linear+angular speed coeff remains2; timer starts at first touchdown and does not reset on bounce; success/safety/horizon5s/PPO/sampling unchanged;30M additional transitions')
    return profile

def grace3s_environment_config():
    cfg=focused_environment_config();cfg['reward'].update(base_vel_coeff=2.,post_touchdown_motion_grace_time=3.);return cfg

def make_grace3s_env(mode='test'):
    from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.environment import LocomotionEnv
    from rl_x.environments.custom_mujoco.robot_locomotion.robots.silver_badger.robot_config import robot_config
    cfg=grace3s_environment_config();cfg['nr_envs']=1
    robot=deepcopy(robot_config);robot.update(directory_path=ROBOT_DIR,spine_locked=False)
    env=LocomotionEnv(robot,mode,20261030,False,cfg,1);verify_controller(env.initial_mj_model);return env

def verify_parent(path):
    info=verify_v9_parent(path);info.update(new_reward_coefficient=2.,parent_motion_grace_s=.5,new_motion_grace_s=3.);return info

if __name__=='__main__':
    profile=grace3s_profile();parent=verify_parent(profile['runner']['load_model']);fingerprint=verify_controller()
    if '--check-only' in sys.argv:print(json.dumps(dict(profile=profile,parent=parent,fingerprint=fingerprint),indent=2))
    else:
        sys.argv=[sys.argv[0],*cli_flags(profile),*sys.argv[1:]]
        from rl_x.runner.runner import Runner
        Runner().run()
