"""V14: fresh PPO, safety success independent of reference tracking."""
import json,os,sys
from pathlib import Path
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.train_landing_braking_v13 import moving_braking_profile, moving_braking_environment_config
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_seed606 import cli_flags, verify_controller, ROBOT_DIR

def fresh_moving_profile():
    p=moving_braking_profile()
    p['environment']['landing_braking'].update(warm_start_from_v11=False,warm_start_from_v12=False,
        success_tracks_reference=False,success_max_horizontal_speed=1.)
    p['runner'].update(load_model='',exp_name='landing_braking_fresh_v14',
        run_name=os.environ.get('RUN_NAME') or 'braking_fresh_v14_'+os.environ.get('SLURM_JOB_ID','local'),
        notes='V14 fresh policy/critic/Adam/std0.15;30M steps;V13 tracking12,sigma0.5,2s to0.5mps and slip5 unchanged;success independently of reference:horizontal norm<=1mps,vertical<=0.3,angular<=2radps,foot support,no forbidden contacts,old physical safety;0.5s stability after2s;heights0.75to1.5,speednorm<=5,zero angles;seed606;PD/physics/sensors unchanged')
    return p

def fresh_moving_environment_config():
    cfg = moving_braking_environment_config()
    def merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict):
                merge(a.setdefault(k, {}), v)
            else:
                a[k] = v
    merge(cfg, fresh_moving_profile()['environment'])
    return cfg


def make_fresh_moving_env(mode='test'):
    from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.environment import LocomotionEnv
    from rl_x.environments.custom_mujoco.robot_locomotion.robots.silver_badger.robot_config import robot_config
    cfg = fresh_moving_environment_config()
    cfg['nr_envs'] = 1
    robot = deepcopy(robot_config)
    robot.update(directory_path=ROBOT_DIR, spine_locked=False)
    env = LocomotionEnv(robot, mode, 20261104, False, cfg, 1)
    verify_controller(env.initial_mj_model)
    return env


if __name__ == '__main__':
    p = fresh_moving_profile()
    parent = dict(fresh_initialization=True, checkpoint=None)
    fingerprint = verify_controller()
    if '--check-only' in sys.argv:
        print(json.dumps(dict(profile=p, parent=parent, fingerprint=fingerprint), indent=2))
    else:
        if any(a.startswith('--runner.load_model') and a not in ('--runner.load_model=',) for a in sys.argv[1:]):
            raise ValueError('V14 is a fresh-training experiment; checkpoint override is disabled')
        sys.argv = [sys.argv[0], *cli_flags(p), *sys.argv[1:]]
        from rl_x.runner.runner import Runner
        Runner().run()
