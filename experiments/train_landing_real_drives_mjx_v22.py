"""V22 on MJX-Warp: the same task, reward, limits and PPO, simulated on the GPU.

LANDING_V22_ARM=warm continues V19 latest (LANDING_V22_CHECKPOINT); the default
"fresh" arm starts from new weights. "continue" extends the final V22 warm model
(60,030,976 steps) with a learning rate that decays to zero. Only the rollout shape differs from the CPU
runs: 1024 environments x 128 steps instead of 64 x 2048, the same 131,072 batch.
"""

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NR_ENVS, NR_STEPS = 1024, 128
CONTINUE_STEPS = 39_976_960  # 305 updates


def versions():
    from importlib import metadata
    found = {}
    for name in ('mujoco', 'mujoco-mjx', 'warp-lang', 'jax', 'trimesh'):
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    missing = [name for name, version in found.items() if version is None]
    if missing:
        raise SystemExit(f'Missing packages: {", ".join(missing)}. See experiments/landing_mjx_README.md')
    if found['mujoco'] != found['mujoco-mjx']:
        raise SystemExit(f"mujoco {found['mujoco']} and mujoco-mjx {found['mujoco-mjx']} must have the same version")
    return found


def verify_continue_parent(path):
    """The final V22 warm checkpoint: same task, reward and drive limits, 60,030,976 steps."""
    import zipfile
    from experiments.train_landing_real_drives_v22 import ADDITIONAL_STEPS, real_drive_profile
    with zipfile.ZipFile(path) as archive:
        algorithm, environment, metrics = (json.loads(archive.read(name)) for name in (
            'config_algorithm.json', 'config_environment.json', 'checkpoint_metrics.json'))
    if algorithm['name'] != 'ppo.flax':
        raise ValueError(f"Parent was saved by {algorithm['name']}, expected ppo.flax")
    if metrics['nr_env_steps'] != ADDITIONAL_STEPS:
        raise ValueError(f"The continue arm needs V22 warm latest.model at {ADDITIONAL_STEPS} steps, "
                         f"this checkpoint has {metrics['nr_env_steps']}")
    expected = real_drive_profile('')['environment']
    if environment.get('landing_actuators') != expected['landing_actuators']:
        raise ValueError('Parent was not trained with the 16 Nm HB40 drive profile')
    if not environment.get('landing_real_drive_mix'):
        raise ValueError('Parent was not trained on the V22 speed mixture')
    from experiments.analysis.check_mjx_parity import v22_config
    for key in ('landing_braking', 'reward', 'policy_inertial', 'imu_history_length', 'timestep', 'control_type',
                'spine_locked', 'landing_physics_sha256', 'landing_controller_sha256', 'landing_model_sha256'):
        if environment.get(key) != v22_config().get(key):
            raise ValueError(f'Parent contract mismatch: {key}')
    return dict(steps=metrics['nr_env_steps'], evaluation=metrics.get('eval/landing_success_rate'),
                worst_height=metrics.get('eval/worst_height_success_rate'))


def mjx_profile(arm, parent=''):
    from experiments.train_landing_real_drives_fresh_v22 import TOTAL_STEPS
    from experiments.train_landing_real_drives_v22 import ADDITIONAL_STEPS, real_drive_profile
    steps = dict(warm=ADDITIONAL_STEPS, fresh=TOTAL_STEPS)
    steps['continue'] = CONTINUE_STEPS
    if arm not in steps:
        raise ValueError('LANDING_V22_ARM must be warm, fresh or continue')
    profile = real_drive_profile(parent)
    name = dict(warm='landing_real_drives_mjx_v22', fresh='landing_real_drives_mjx_fresh_v22').get(
        arm, 'landing_real_drives_mjx_continue_v22')
    profile['algorithm'].update(name='ppo.flax_mjx_landing', nr_steps=NR_STEPS, total_timesteps=steps[arm],
                                final_learning_rate=0. if arm == 'continue' else -1.)
    profile['environment'].update(name='custom_mujoco.robot_locomotion.mjx_landing', nr_envs=NR_ENVS)
    origin = dict(warm='continues V19 latest (job 9591668): actor, critic, Adam, learned std',
                  fresh='from scratch: new actor, critic, Adam and std 0.15').get(
        arm, 'continues V22 warm latest (job 9730655, 60,030,976 steps): actor, critic, Adam, learned std; '
             'learning rate 3e-5 decays linearly to 0')
    profile['runner'].update(
        load_model=str(parent), exp_name=name, run_name=os.environ.get('RUN_NAME') or
        f"{name}_{os.environ.get('SLURM_JOB_ID', 'local')}",
        notes=f'V22 on MJX-Warp (float32), {origin}. Same task, 16 Nm limits, reward, IMU contract, evaluation grid and PPO '
              f'as CPU V22; rollouts {NR_ENVS}x{NR_STEPS} instead of 64x2048, batch 131,072.')
    return profile


def check(profile):
    """Build the environment on the GPU and verify limits and a short rollout."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from experiments.analysis.check_mjx_parity import make_mjx_env
    from experiments.train_landing_inertial_v16 import inertial_environment_config
    from experiments.analysis.check_mjx_parity import merge
    if jax.default_backend() != 'gpu':
        raise SystemExit('MJX-Warp training needs a CUDA GPU; JAX uses ' + jax.default_backend())
    config = merge(inertial_environment_config(), profile['environment'])
    config.update(nr_envs=64)
    env = make_mjx_env(config)
    limits = np.asarray(env.model.actuator_forcerange[:, 1] * env.model.actuator_gear[:, 0])
    np.testing.assert_allclose(limits, 16.)
    np.testing.assert_allclose(env.model.jnt_range[env.model.joint('spine_joint').id], [-.785, .785])
    pool = jax.jit(lambda k: env.sample_pool(k, 64))(jax.random.PRNGKey(0))
    assert not bool(pool['invalid'])
    state = jax.jit(lambda p: env.reset(p, 64))(pool)
    step = jax.jit(env.step)
    for _ in range(60):
        state, _ = step(state, jnp.zeros((64, 13)), pool, 0)
    assert not bool(state.overflow) and bool(jnp.isfinite(state.observation).all())
    return dict(device=str(jax.devices()[0]), joint_limit_nm=limits.tolist(),
                mean_speed=float(pool['task']['speed'].mean()), observation=list(state.observation.shape))


def main():
    arm = os.environ.get('LANDING_V22_ARM', 'fresh')
    found = versions()
    from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_seed606 import cli_flags, verify_controller
    parent = ''
    if arm == 'warm':
        from experiments.train_landing_handoff_v20 import PARENT_SHA, input_path
        from experiments.train_landing_real_drives_v22 import verify_parent
        parent = input_path('LANDING_V22_CHECKPOINT', 'modelsv19/latest.model', PARENT_SHA)
        verify_parent(parent)
    elif arm == 'continue':
        parent = Path(os.environ.get('LANDING_V22_CHECKPOINT', '')).expanduser().resolve()
        if not parent.is_file():
            raise FileNotFoundError(f'Set LANDING_V22_CHECKPOINT to the V22 warm latest.model. Missing: {parent}')
        print('Parent:', parent, verify_continue_parent(parent), flush=True)
    profile = mjx_profile(arm, parent)
    fingerprint = verify_controller()
    if '--check-only' in sys.argv:
        print(json.dumps(dict(profile=profile, versions=found, fingerprint=fingerprint, model=check(profile)), indent=2))
        return
    if any(arg.startswith(('--runner.load_model', '--environment.landing_actuators')) for arg in sys.argv[1:]):
        raise ValueError('Use LANDING_V22_ARM, LANDING_V22_CHECKPOINT and the V22 actuator profile')
    sys.argv = [sys.argv[0], *cli_flags(profile), *sys.argv[1:]]
    from rl_x.runner.runner import Runner
    Runner().run()


if __name__ == '__main__':
    main()
