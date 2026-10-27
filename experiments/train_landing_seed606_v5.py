"""V5: measured-velocity-compatible actor inputs for emergency landing."""
import json,os,sys
from pathlib import Path
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.train_landing_seed606_v4 import stable_profile
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_seed606 import cli_flags,verify_controller,environment_config,ROBOT_DIR
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.policy_velocity import velocity_observation_contract


def velocity_profile(mode=None):
    mode=mode or os.environ.get('POLICY_VELOCITY_MODE','heading_xy')
    if mode not in ('heading_xy','masked'):
        raise ValueError('V5 requires POLICY_VELOCITY_MODE=heading_xy or masked')
    profile=stable_profile()
    profile['environment']['policy_velocity']=dict(mode=mode,schema='base_heading_xy_v1',source='simulated_base_truth',scale_mps=4.,noise_std_mps=0.,bias_std_mps=0.,delay_steps=0)
    velocity_observation_contract(profile['environment']['policy_velocity'])
    profile['runner'].update(exp_name='landing_seed606_velocity_v5',
        run_name=os.environ.get('RUN_NAME') or ('seed606_velocity_v5_'+mode+'_'+os.environ.get('SLURM_JOB_ID','local')),
        load_model='',notes=f'V5 velocity mode={mode}; base-origin forward/lateral in gravity-aligned heading frame, scaled by4m/s; simulator truth during training, OptiTrack-compatible deployment convention; fresh weights; PD606, implicitfast2.5ms, height1to2.2m, angles10deg, vxy3peraxis, PPO/reward/curriculum as v4')
    return profile


def velocity_environment_config(mode=None):
    cfg=environment_config()
    def merge(target,updates):
        for k,v in updates.items():
            if isinstance(v,dict):merge(target[k],v)
            else:target[k]=v
    merge(cfg,velocity_profile(mode)['environment'])
    return cfg


def make_velocity_report_env(mode='heading_xy'):
    from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.environment import LocomotionEnv
    from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_evaluation import LandingEvaluationWrapper
    from rl_x.environments.custom_mujoco.robot_locomotion.robots.silver_badger.robot_config import robot_config
    cfg=velocity_environment_config(mode);cfg['nr_envs']=1
    robot=deepcopy(robot_config);robot.update(directory_path=ROBOT_DIR,spine_locked=False)
    env=LocomotionEnv(robot,'test',20261027,False,cfg,1);verify_controller(env.initial_mj_model)
    return LandingEvaluationWrapper(env,0)


if __name__=='__main__':
    fingerprint=verify_controller();profile=velocity_profile()
    if '--check-only' in sys.argv:
        print(json.dumps(dict(profile=profile,fingerprint=fingerprint,velocity_contract=velocity_observation_contract(profile['environment']['policy_velocity'])),indent=2))
    else:
        sys.argv=[sys.argv[0],*cli_flags(profile),*sys.argv[1:]]
        print('V5 fixed PD verified; policy velocity:',json.dumps(velocity_observation_contract(profile['environment']['policy_velocity'])),flush=True)
        from rl_x.runner.runner import Runner
        Runner().run()
