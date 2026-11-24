"""V22 on MJX-Warp: the same task, reward, limits and PPO, simulated on the GPU.

LANDING_V22_ARM=warm continues V19 latest (LANDING_V22_CHECKPOINT); the default
"fresh" arm starts from new weights. Only the rollout shape differs from the CPU
runs: 1024 environments x 128 steps instead of 64 x 2048, the same 131,072 batch.
"""

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NR_ENVS, NR_STEPS = 1024, 128


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


def mjx_profile(arm, parent=''):
    from experiments.train_landing_real_drives_fresh_v22 import TOTAL_STEPS
    from experiments.train_landing_real_drives_v22 import ADDITIONAL_STEPS, real_drive_profile
    if arm not in ('warm', 'fresh'):
        raise ValueError('LANDING_V22_ARM must be warm or fresh')
    profile = real_drive_profile(parent)
    name = 'landing_real_drives_mjx_v22' if arm == 'warm' else 'landing_real_drives_mjx_fresh_v22'
    profile['algorithm'].update(name='ppo.flax_mjx_landing', nr_steps=NR_STEPS,
                                total_timesteps=ADDITIONAL_STEPS if arm == 'warm' else TOTAL_STEPS)
    profile['environment'].update(name='custom_mujoco.robot_locomotion.mjx_landing', nr_envs=NR_ENVS)
    origin = ('continues V19 latest (job 9591668): actor, critic, Adam, learned std' if arm == 'warm'
              else 'from scratch: new actor, critic, Adam and std 0.15')
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
