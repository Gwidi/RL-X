"""V13: continue V12 with stronger tracking and a persistent 0.5m/s goal."""
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.train_landing_braking_v12 import slip_braking_profile as braking_profile, slip_braking_environment_config as braking_environment_config
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_seed606 import cli_flags, verify_controller, ROBOT_DIR

ROOT = Path(__file__).resolve().parents[1]


def moving_braking_profile():
    p = braking_profile()
    p['environment']['landing_braking'].update(duration=2., tracking_weight=12., final_speed=.5, terminal_velocity_tolerance=.25,
        warm_start_from_v11=False, warm_start_from_v12=True, slip=dict(weight=5., force_filter_s=.04,
        load_threshold_n=10., full_load_n=50., impact_grace_s=.1, ramp_s=.2))
    path = os.environ.get('LANDING_V13_CHECKPOINT') or str(ROOT/'docs/artifacts/braking_v12_completed_9447619_2026-11-04/best.model')
    p['runner'].update(load_model=str(Path(path).resolve()),
        exp_name='landing_braking_moving_v13',
        run_name=os.environ.get('RUN_NAME') or 'braking_moving_v13_'+os.environ.get('SLURM_JOB_ID', 'local'),
        notes='V13 continues V12 best policy/critic/Adam/std;30M extra;2s reference ends at0.5mps, direction frozen;tracking12 vs6,sigma0.5 unchanged;slip unchanged;success after2s plus0.5s continuous horizontal tracking error<=0.25mps,vertical<=0.3mps,old angular/contact/safety;nominal direction+x below0.2mps;no step bonus')
    return p


def verify_parent(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'V13 parent not found: {path}. Export LANDING_V13_CHECKPOINT="/full/path/to/v12/models/best.model"')
    with zipfile.ZipFile(path) as z:
        saved = json.loads(z.read('config_environment.json'))
        metrics = json.loads(z.read('checkpoint_metrics.json'))
    b = saved.get('landing_braking', {})
    if (not b.get('enabled') or b.get('duration') != 2. or b.get('tracking_weight') != 6.
            or b.get('final_speed',0) != 0 or b.get('slip', {}).get('weight') != 5.
            or metrics.get('nr_env_steps') not in (29360128,30015488)):
        raise ValueError('V13 requires downloaded V12 best or latest, not an earlier parent')
    expected = braking_environment_config()
    for key in ('reward', 'policy_velocity', 'landing_physics_sha256'):
        if saved.get(key) != expected.get(key):
            raise ValueError(f'V12 parent differs in {key}')
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                parent_steps=metrics['nr_env_steps'])


def moving_braking_environment_config():
    cfg = braking_environment_config()
    def merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict):
                merge(a.setdefault(k, {}), v)
            else:
                a[k] = v
    merge(cfg, moving_braking_profile()['environment'])
    return cfg


def make_moving_braking_env(mode='test'):
    from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.environment import LocomotionEnv
    from rl_x.environments.custom_mujoco.robot_locomotion.robots.silver_badger.robot_config import robot_config
    cfg = moving_braking_environment_config()
    cfg['nr_envs'] = 1
    robot = deepcopy(robot_config)
    robot.update(directory_path=ROBOT_DIR, spine_locked=False)
    env = LocomotionEnv(robot, mode, 20261104, False, cfg, 1)
    verify_controller(env.initial_mj_model)
    return env


if __name__ == '__main__':
    p = moving_braking_profile()
    parent = verify_parent(p['runner']['load_model'])
    fingerprint = verify_controller()
    if '--check-only' in sys.argv:
        print(json.dumps(dict(profile=p, parent=parent, fingerprint=fingerprint), indent=2))
    else:
        sys.argv = [sys.argv[0], *cli_flags(p), *sys.argv[1:]]
        from rl_x.runner.runner import Runner
        Runner().run()
