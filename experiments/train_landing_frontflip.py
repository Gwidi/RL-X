"""Experimental fresh frontflip task; does not alter V14/V15 profiles."""
import json,os,sys
from pathlib import Path
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.train_landing_braking_v14 import fresh_moving_profile,fresh_moving_environment_config
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_seed606 import cli_flags,verify_controller,ROBOT_DIR

def frontflip_profile():
    p=fresh_moving_profile()
    p['environment']['landing_braking']['enabled']=False
    p['environment']['landing_frontflip']=dict(enabled=True)
    p['environment']['add_goal_arrow']=False # Goal channels now describe phase, not velocity.
    p['environment']['reward'].update(alive_coeff=0.,base_vel_coeff=0.,body_acceleration_coeff=0.,
        collision_coeff=0.,body_support_penalty_coeff=0.,success_confirmation_time=.02)
    p['runner'].update(load_model='',exp_name='landing_frontflip',
        run_name=os.environ.get('RUN_NAME') or 'frontflip_'+os.environ.get('SLURM_JOB_ID','local'),
        notes='Fresh frontflip experimental task; first contact->takeoff->half turn->one full forward aerial turn->second landing and0.5s safety stability;phase bonuses2/3/5,terminal100;energy signed10,positive motor work10,peak foot force20,peak acceleration30,time1/s,slip5;no velocity reference;goal channels now phase/rotation/airtime;all hard safety remains;V14 physical sampling0.75to1.5m,speednorm<=5,angles0;not evidence of energy benefit')
    return p

def frontflip_environment_config():
    cfg=fresh_moving_environment_config()
    def merge(a,b):
        for k,v in b.items():
            if isinstance(v,dict):merge(a.setdefault(k,{}),v)
            else:a[k]=v
    merge(cfg,frontflip_profile()['environment']);return cfg

def make_frontflip_env(mode='test'):
    from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.environment import LocomotionEnv
    from rl_x.environments.custom_mujoco.robot_locomotion.robots.silver_badger.robot_config import robot_config
    cfg=frontflip_environment_config();cfg['nr_envs']=1;robot=deepcopy(robot_config);robot.update(directory_path=ROBOT_DIR,spine_locked=False)
    e=LocomotionEnv(robot,mode,20261106,False,cfg,1);verify_controller(e.initial_mj_model);return e

if __name__=='__main__':
    p=frontflip_profile();fingerprint=verify_controller()
    if '--check-only' in sys.argv:print(json.dumps(dict(profile=p,fingerprint=fingerprint),indent=2))
    else:
        if any(a.startswith('--runner.load_model') for a in sys.argv[1:]):raise ValueError('Frontflip uses new actor goal semantics; train fresh, not a landing checkpoint')
        sys.argv=[sys.argv[0],*cli_flags(p),*sys.argv[1:]]
        from rl_x.runner.runner import Runner
        Runner().run()
