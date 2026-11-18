"""V20 warm start from V19 latest, with a trainable Bernoulli handoff to E19."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.train_landing_inertial_speed10_v19 import speed10_profile
from experiments.train_landing_inertial_v16 import inertial_environment_config
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_seed606 import cli_flags, verify_controller

ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA = '5cdf9b332e4a0b68a1f452d73a7810437f847de51e73fcc170e96a2adfd1ecf3'
WALKER_SHA = '1006f7ce5955cb84d56f302aa5175ec69b1b013372736da4ce01dbbae29d063d'


def input_path(variable, default, digest):
    path = Path(os.environ.get(variable, str(ROOT / default))).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Set {variable} to the checkpoint file. Missing: {path}')
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f'{variable}: unexpected checkpoint SHA256: {path}')
    return path


def export_walker(checkpoint, output):
    # Kept in a separate CPU process, so Gym workers never fork a JAX runtime.
    import numpy as np
    from experiments.analysis.landing_walk_handoff import load_walker, read_walker_checkpoint
    from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_handoff import walking_mean

    params, _, metadata = read_walker_checkpoint(checkpoint)
    if metadata['sha256'] != WALKER_SHA:
        raise ValueError('Expected the validated E19 checkpoint')
    weights = {f'{name}/{key}': np.asarray(value)
               for name, layer in params['params'].items() if isinstance(layer, dict)
               for key, value in layer.items()}
    actor, _ = load_walker(checkpoint)
    for obs in np.random.default_rng(19).normal(0., .4, (12, 48)).astype(np.float32):
        np.testing.assert_allclose(walking_mean(weights, obs), actor(obs), rtol=2e-5, atol=1e-5)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix='.npz', delete=False) as temporary:
        temp_path = Path(temporary.name)
    try:
        np.savez(temp_path, source_sha256=WALKER_SHA, **weights)
        temp_path.replace(output)
    finally:
        temp_path.unlink(missing_ok=True)


def handoff_profile(parent, weights):
    profile = speed10_profile()
    profile['environment']['landing_handoff'] = dict(
        enabled=True, walk_weights=str(weights), walk_sha256=WALKER_SHA, walk_speed_limit=.6)
    # An explicit timeout failure is recorded by the landing task at five seconds.
    profile['algorithm'].update(total_timesteps=30_015_488, learning_rate=1e-5, entropy_coef=.002)
    profile['runner'].update(load_model=str(parent), exp_name='landing_walk_handoff_v20',
        run_name=os.environ.get('RUN_NAME') or 'landing_walk_handoff_v20_' + os.environ.get('SLURM_JOB_ID', 'local'),
        notes='V20 warm start V19 latest, actor and critic weights with new zero feature rows, fresh Adam; 13 Gaussian residuals plus Bernoulli switch; frozen E19 and 0.28 s blend; shared world heading command capped at 0.60 m/s; hard landing failures retained; success requires renewed foot support and sustained motion in reference direction')
    return profile


def main():
    if len(sys.argv) == 4 and sys.argv[1] == '--export-e19':
        export_walker(Path(sys.argv[2]), Path(sys.argv[3]))
        return
    parent = input_path('LANDING_V20_CHECKPOINT', 'modelsv19/latest.model', PARENT_SHA)
    walker = input_path('LANDING_V20_WALKER', 'E19SBGppo_hfield_bunker_ruins_foot_velocity.model', WALKER_SHA)
    weights = ROOT / 'runs' / 'handoff_assets' / f'e19_{WALKER_SHA[:12]}.npz'
    weights.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, __file__, '--export-e19', str(walker), str(weights)],
                   check=True, env={**os.environ, 'JAX_PLATFORMS': 'cpu'})
    profile = handoff_profile(parent, weights)
    verify_controller()
    with zipfile.ZipFile(parent) as archive:
        metrics = json.loads(archive.read('checkpoint_metrics.json'))
        saved = json.loads(archive.read('config_environment.json'))
    expected = inertial_environment_config()
    for key in ('landing_braking', 'reward', 'policy_inertial', 'imu_history_length',
                'landing_physics_sha256', 'landing_controller_sha256'):
        if saved.get(key) != expected.get(key):
            raise ValueError(f'Parent task mismatch: {key}')
    if metrics['nr_env_steps'] != 30_015_488:
        raise ValueError('V20 requires the final V19 latest')
    if '--check-only' in sys.argv:
        print(json.dumps(dict(profile=profile, parent_sha256=PARENT_SHA, walker_sha256=WALKER_SHA), indent=2))
        return
    if any(arg.startswith(('--runner.load_model', '--environment.landing_handoff')) for arg in sys.argv[1:]):
        raise ValueError('Use LANDING_V20_CHECKPOINT and LANDING_V20_WALKER')
    sys.argv = [sys.argv[0], *cli_flags(profile), *sys.argv[1:]]
    from rl_x.runner.runner import Runner
    Runner().run()


if __name__ == '__main__':
    main()
