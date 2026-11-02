"""V10: continue v9 with zero linear AND angular body-velocity cost."""
import json,os,sys,hashlib,zipfile
from pathlib import Path
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.train_landing_seed606_v9 import focused_profile,focused_environment_config
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_seed606 import cli_flags,verify_controller,ROBOT_DIR
ROOT=Path(__file__).resolve().parents[1]

def free_settling_profile():
    profile=focused_profile()
    profile['environment']['reward']['base_vel_coeff']=0.
    path=Path(os.environ.get('LANDING_V10_CHECKPOINT',str(ROOT/'models/best.model'))).resolve()
    profile['runner'].update(load_model=str(path),exp_name='landing_seed606_free_settling_v10',
        run_name=os.environ.get('RUN_NAME') or 'seed606_free_settling_v10_'+os.environ.get('SLURM_JOB_ID','local'),
        notes='V10 continues v9 policy+critic+Adam and learned std; ONLY task change base_vel_coeff2to0: no linear or angular body-speed penalty anywhere in5s episode; final success still needs>=2s since touchdown and0.5s stability, speed<=0.5mps,angular<=2radps; v9 sampling,PPO,physics,PD,sensor,safety and eval unchanged;30M additional transitions')
    return profile

def free_settling_environment_config():
    cfg=focused_environment_config();cfg['reward']['base_vel_coeff']=0.;return cfg

def make_free_settling_env(mode='test'):
    from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.environment import LocomotionEnv
    from rl_x.environments.custom_mujoco.robot_locomotion.robots.silver_badger.robot_config import robot_config
    cfg=free_settling_environment_config();cfg['nr_envs']=1
    robot=deepcopy(robot_config);robot.update(directory_path=ROBOT_DIR,spine_locked=False)
    env=LocomotionEnv(robot,mode,20261030,False,cfg,1);verify_controller(env.initial_mj_model);return env

def verify_parent(path):
    path=Path(path)
    if not path.is_file():raise FileNotFoundError('Set LANDING_V10_CHECKPOINT to the v9 best.model or latest.model')
    with zipfile.ZipFile(path) as z:
        saved=json.loads(z.read('config_environment.json'));metrics=json.loads(z.read('checkpoint_metrics.json'))
    c=saved['domain_randomization']['initial_state']['energy_dissipation_curriculum']
    if (not c.get('focused_velocity_enabled') or c.get('height_then_velocity_enabled') or
            metrics.get('nr_env_steps') not in (25165824,30015488) or saved['reward'].get('base_vel_coeff')!=2.):
        raise ValueError('V10 requires a completed v9 checkpoint with the original body-velocity penalty; smoke/v8/v10 models are rejected')
    expected=free_settling_profile()['environment']['reward']
    for key,value in expected.items():
        if key!='base_vel_coeff' and saved['reward'].get(key)!=value:raise ValueError(f'Parent reward differs in {key}')
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),parent_v9_steps=metrics['nr_env_steps'],parent_reward_coefficient=2.,new_reward_coefficient=0.)

if __name__=='__main__':
    profile=free_settling_profile();parent=verify_parent(profile['runner']['load_model']);fingerprint=verify_controller()
    if '--check-only' in sys.argv:print(json.dumps(dict(profile=profile,parent=parent,fingerprint=fingerprint),indent=2))
    else:
        sys.argv=[sys.argv[0],*cli_flags(profile),*sys.argv[1:]]
        from rl_x.runner.runner import Runner
        Runner().run()
