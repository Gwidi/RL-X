"""Continue V16 best with low-height hard replay and the same IMU contract."""
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.train_landing_inertial_v16 import inertial_profile, inertial_environment_config
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_seed606 import cli_flags, verify_controller

ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA256 = '16318d229a8a32c641ea559a2ee62e9e68b4f6a5c0e0bebae3d945715737a5db'
ADDITIONAL_STEPS = 30015488


def continuation_profile():
    p = inertial_profile()
    p['algorithm']['total_timesteps'] = ADDITIONAL_STEPS
    e = p['environment']
    e['landing_balanced_replay'] = True
    c = e['domain_randomization']['initial_state']['energy_dissipation_curriculum']
    c.update(height_then_velocity_enabled=False, focused_velocity_enabled=False,
             factorized_enabled=False)
    e['landing_evaluation']['require_nominal_for_best'] = True
    path = os.environ.get('LANDING_V17_CHECKPOINT') or str(ROOT / 'models/best.model')
    p['runner'].update(load_model=str(Path(path).expanduser().resolve()),
        exp_name='landing_inertial_balanced_v17',
        run_name=os.environ.get('RUN_NAME') or 'inertial_balanced_v17_'+os.environ.get('SLURM_JOB_ID', 'local'),
        notes='V17 continues V16 best88080384 with actor+critic+Adam+learned std;30015488 extra transitions;new episodes/RNG,not exact simulation resume;unchanged IMU history/integral/noise/delay/launch/reward/PD/physics/PPO;60pct heights0.75/1/1.25 equal and speed uniform5to8;30pct all4 heights equal and speed uniform0to8;10pct all4 heights equal and zero speed;all azimuths;best by worst-height success plus0.001 global tie-break subject to nominal eligibility;no height observation added')
    return p


def verify_parent(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'V16 best not found: {path}. Export LANDING_V17_CHECKPOINT="/full/path/to/v16/models/best.model" before sbatch.')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != PARENT_SHA256:
        raise ValueError('V17 requires the completed V16 best (job9559569,88,080,384 steps); selected file has a different SHA256')
    with zipfile.ZipFile(path) as z:
        saved = json.loads(z.read('config_environment.json'))
        metrics = json.loads(z.read('checkpoint_metrics.json'))
    expected = inertial_environment_config()
    from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.policy_velocity import velocity_observation_contract
    if velocity_observation_contract(saved['policy_velocity']) != velocity_observation_contract(expected['policy_velocity']):
        raise ValueError('V16 parent velocity observation contract mismatch')
    for key in ('policy_inertial', 'imu_history_length', 'landing_braking',
                'reward', 'domain_randomization', 'timestep', 'control_type', 'spine_locked',
                'landing_physics_sha256', 'landing_controller_sha256', 'landing_model_sha256'):
        if saved.get(key) != expected.get(key):
            raise ValueError(f'V16 parent contract mismatch: {key}')
    if metrics['nr_env_steps'] != 88080384:
        raise ValueError('Unexpected V16 parent transition count')
    return dict(path=str(path.resolve()), sha256=digest, parent_steps=88080384,
                additional_steps=ADDITIONAL_STEPS, aggregate_steps=118095872,
                restores='actor,critic,Adam,learned std; fresh episodes and run counters')


if __name__ == '__main__':
    p = continuation_profile()
    parent = verify_parent(p['runner']['load_model'])
    fingerprint = verify_controller()
    if '--check-only' in sys.argv:
        print(json.dumps(dict(profile=p, parent=parent, fingerprint=fingerprint), indent=2))
    else:
        if any(arg.startswith('--runner.load_model') for arg in sys.argv[1:]):
            raise ValueError('Use LANDING_V17_CHECKPOINT to select the validated V16 best')
        sys.argv = [sys.argv[0], *cli_flags(p), *sys.argv[1:]]
        from rl_x.runner.runner import Runner
        Runner().run()
