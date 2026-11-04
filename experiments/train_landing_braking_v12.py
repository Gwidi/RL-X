"""V12: continue completed v11, 2s braking and loaded-support slip cost."""
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.train_landing_braking_v11 import braking_profile, braking_environment_config
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_seed606 import cli_flags, verify_controller, ROBOT_DIR

ROOT = Path(__file__).resolve().parents[1]


def slip_braking_profile():
    p = braking_profile()
    p['environment']['landing_braking'].update(duration=2., tracking_weight=6.,
        warm_start_from_v11=True, slip=dict(weight=5., force_filter_s=.04,
        load_threshold_n=10., full_load_n=50., impact_grace_s=.1, ramp_s=.2))
    path = os.environ.get('LANDING_V12_CHECKPOINT') or str(ROOT/'models/best.model')
    p['runner'].update(load_model=str(Path(path).resolve()),
        exp_name='landing_braking_slip_v12',
        run_name=os.environ.get('RUN_NAME') or 'braking_slip_v12_'+os.environ.get('SLURM_JOB_ID', 'local'),
        notes='V12 continues completed v11 policy/critic/Adam/std; 30M additional steps; braking2s; tracking6 vs5, sigma0.5 unchanged; slip5*dt*sum(load_weight*contact_tangent_speed2)/4, feet+distalcalves; force filter40ms,10to50N; impact grace100ms+ramp200ms; final0.5s stability; heights0.75to1.5m,speednorm<=5,angles0; unchanged PD/physics/sensor/safety/eval grid; success does not certify stepping')
    return p


def verify_parent(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'V12 parent not found: {path}. Export LANDING_V12_CHECKPOINT="/full/path/to/v11/models/best.model"')
    with zipfile.ZipFile(path) as z:
        saved = json.loads(z.read('config_environment.json'))
        metrics = json.loads(z.read('checkpoint_metrics.json'))
    b = saved.get('landing_braking', {})
    if (not b.get('enabled') or b.get('duration') != 2.5 or b.get('tracking_weight') != 5. or b.get('tracking_sigma') != .5
            or b.get('slip', {}).get('weight', 0) or metrics.get('nr_env_steps') != 30015488):
        raise ValueError('V12 requires completed v11 best.model or latest.model (30,015,488 steps)')
    expected = braking_environment_config()
    for key in ('reward', 'policy_velocity', 'landing_physics_sha256'):
        if saved.get(key) != expected.get(key):
            raise ValueError(f'V11 parent differs in {key}')
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                parent_steps=metrics['nr_env_steps'])


def slip_braking_environment_config():
    cfg = braking_environment_config()
    def merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict):
                merge(a.setdefault(k, {}), v)
            else:
                a[k] = v
    merge(cfg, slip_braking_profile()['environment'])
    return cfg


def make_slip_braking_env(mode='test'):
    from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.environment import LocomotionEnv
    from rl_x.environments.custom_mujoco.robot_locomotion.robots.silver_badger.robot_config import robot_config
    cfg = slip_braking_environment_config()
    cfg['nr_envs'] = 1
    robot = deepcopy(robot_config)
    robot.update(directory_path=ROBOT_DIR, spine_locked=False)
    env = LocomotionEnv(robot, mode, 20261104, False, cfg, 1)
    verify_controller(env.initial_mj_model)
    return env


if __name__ == '__main__':
    p = slip_braking_profile()
    parent = verify_parent(p['runner']['load_model'])
    fingerprint = verify_controller()
    if '--check-only' in sys.argv:
        print(json.dumps(dict(profile=p, parent=parent, fingerprint=fingerprint), indent=2))
    else:
        sys.argv = [sys.argv[0], *cli_flags(p), *sys.argv[1:]]
        from rl_x.runner.runner import Runner
        Runner().run()
