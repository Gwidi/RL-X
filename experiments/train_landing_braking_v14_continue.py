"""Continue completed V14 unchanged for another 30,015,488 transitions."""
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.train_landing_braking_v14 import fresh_moving_profile, fresh_moving_environment_config
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_seed606 import cli_flags, verify_controller

ROOT = Path(__file__).resolve().parents[1]


def continuation_profile():
    p = fresh_moving_profile()
    checkpoint = os.environ.get('LANDING_V14_CHECKPOINT') or str(ROOT/'modelsv14/models/latest.model')
    p['runner'].update(load_model=str(Path(checkpoint).resolve()),
        exp_name='landing_braking_continue_v14',
        run_name=os.environ.get('RUN_NAME') or 'braking_continue_v14_'+os.environ.get('SLURM_JOB_ID', 'local'),
        notes='V14 unchanged continuation from final latest at30015488; another30015488 transitions; policy+critic TrainStates including Adam and learned std restored; new run counters and fresh episodes/RNG, not exact simulator resume; same reward/reference/safety success/sampling/PD/physics/sensor/PPO and validation; aggregate exposure60030976')
    return p


def verify_parent(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'V14 checkpoint not found: {path}. Export LANDING_V14_CHECKPOINT="/full/path/to/completed/v14/models/latest.model"')
    with zipfile.ZipFile(path) as z:
        saved = json.loads(z.read('config_environment.json'))
        metrics = json.loads(z.read('checkpoint_metrics.json'))
    if metrics.get('nr_env_steps') != 30015488:
        raise ValueError('Continuation requires the final V14 latest at30,015,488 steps, not a partial or best checkpoint')
    expected = fresh_moving_environment_config()
    for key in ('landing_braking', 'reward', 'policy_velocity', 'domain_randomization',
                'landing_physics_sha256', 'landing_controller_sha256', 'landing_model_sha256',
                'imu_history_length', 'timestep', 'control_type', 'spine_locked'):
        if saved.get(key) != expected.get(key):
            raise ValueError(f'Checkpoint is not the unchanged V14 task: mismatch in {key}')
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                parent_steps=metrics['nr_env_steps'], additional_steps=30015488,
                aggregate_steps=60030976, restores='policy, critic, Adam state and learned std')


if __name__ == '__main__':
    p = continuation_profile()
    parent = verify_parent(p['runner']['load_model'])
    fingerprint = verify_controller()
    if '--check-only' in sys.argv:
        print(json.dumps(dict(profile=p, parent=parent, fingerprint=fingerprint), indent=2))
    else:
        if any(a.startswith('--runner.load_model') for a in sys.argv[1:]):
            raise ValueError('Select the validated parent using LANDING_V14_CHECKPOINT, not --runner.load_model')
        sys.argv = [sys.argv[0], *cli_flags(p), *sys.argv[1:]]
        from rl_x.runner.runner import Runner
        Runner().run()
