"""V22 on MJX-Warp: the same task, reward, limits and PPO, simulated on the GPU.

LANDING_V22_ARM=warm continues V19 latest (LANDING_V22_CHECKPOINT); the default
"fresh" arm starts from new weights. "continue" extends the final V22 warm model
(60,030,976 steps) with a learning rate that decays to zero. Only the rollout shape differs from the CPU
runs: 1024 environments x 128 steps instead of 64 x 2048, the same 131,072 batch.
Optional PPO overrides, any arm: LANDING_V22_NR_ENVS (batch = nr_envs x 128), LANDING_V22_MINIBATCH (must divide the
batch), LANDING_V22_ENTROPY_COEF, LANDING_V22_MIN_STD (floor on the learned std), LANDING_V22_LEARNING_RATE.
LANDING_V22_FRESH=1 (soft arm from scratch), LANDING_V22_TOTAL_STEPS,
LANDING_V22_CONSTRAINTS_PER_ENV, LANDING_V22_CONTACTS_PER_ENV (MJX buffers),
LANDING_V22_RANDOM_MASS_KG, RANDOM_COM_M, TORQUE_MIN_NM, TORQUE_MAX_NM, RANDOM_PD, FRICTION_MIN, FRICTION_MAX
(mjx.randomization),
LANDING_V22_POSTURE_REWARD (mjx.posture_reward),
LANDING_V22_SUCCESS_AT_END=1, LANDING_V22_END_ROLL_DEG (mjx.success_at_end, mjx.end_roll_limit_deg), LANDING_V22_GAMMA,
LANDING_V22_FLIGHT_HIP_RATE (mjx.flight_hip_rate_penalty), LANDING_V22_RELEASE_RAMP_STEPS (disturbance ramp, default 20M).
LANDING_V22_MAX_SPEED above 5 extends the training mixture and the evaluation grid.
LANDING_V22_SYMMETRIC_POSE=1 mirrors the right legs to the left in the nominal pose (soft arm or any other);
LANDING_V22_FRONT_HIP then sets the right front hip, rad (default 0.2687, the home value).
LANDING_V22_RELEASE_TILT_DEG / _RELEASE_RATE_DPS: soft arm with an imperfect release, as in the disturbed arm.
LANDING_V22_PARENT_STEPS rejects a parent with a different step count.
"""

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NR_ENVS, NR_STEPS = int(os.environ.get('LANDING_V22_NR_ENVS', 1024)), 128
CONTINUE_STEPS = 39_976_960  # 305 updates
# Imperfect release: roll/pitch up to 30 deg, body angular rate up to 15 deg/s, ramped in over 153 updates.
# Release from the standing pose: moved in from the landing pose over 229 updates, 20% of the episodes stay at home.
STANDING_STEPS, STAND_RELEASE = 90_046_464, dict(stand_fraction=1., home_probability=.2, ramp_steps=30_015_488)
# Soft thigh/calf contact: allowed up to 200 N (10 ms mean); LANDING_V22_REFERENCE selects the braking term (track, upper, off).
SOFT_STEPS = 60_030_976
DISTURBED_STEPS, DISTURBANCE = 60_030_976, dict(max_tilt_deg=30., max_rate_dps=15., ramp_steps=20_054_016)


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


def verify_continue_parent(path, steps=None):
    """A ppo.flax V22 checkpoint (CPU or MJX): same task, reward and 16 Nm drive limits.
    steps: required step count, or None for any."""
    import zipfile
    from experiments.train_landing_real_drives_v22 import ADDITIONAL_STEPS, real_drive_profile
    with zipfile.ZipFile(path) as archive:
        algorithm, environment, metrics = (json.loads(archive.read(name)) for name in (
            'config_algorithm.json', 'config_environment.json', 'checkpoint_metrics.json'))
    if algorithm['name'] != 'ppo.flax':
        raise ValueError(f"Parent was saved by {algorithm['name']}, expected ppo.flax")
    if steps is not None and metrics['nr_env_steps'] != steps:
        raise ValueError(f"The parent must be a checkpoint at {steps} steps, "
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


def front_hip():
    return float(os.environ.get('LANDING_V22_FRONT_HIP') or .2687)


def verify_parent_pose(path, symmetric):
    """The nominal pose is the action origin, so parent and run must agree (experiments/convert_parent_symmetric_pose.py)."""
    import zipfile
    with zipfile.ZipFile(path) as archive:
        mjx = json.loads(archive.read('checkpoint_metrics.json')).get('mjx', {})
    if bool(mjx.get('symmetric_pose', False)) != symmetric:
        raise ValueError('LANDING_V22_SYMMETRIC_POSE=1 needs a parent converted with experiments/convert_parent_symmetric_pose.py'
                         if symmetric else 'The parent uses the symmetric pose: set LANDING_V22_SYMMETRIC_POSE=1')
    if symmetric and abs(mjx.get('symmetric_front_hip', .2687) - front_hip()) > 1e-6:
        raise ValueError(f"The parent was converted for front hip {mjx.get('symmetric_front_hip', .2687)}, "
                         f'this run uses {front_hip()} (LANDING_V22_FRONT_HIP)')


def mjx_profile(arm, parent=''):
    from experiments.train_landing_real_drives_fresh_v22 import TOTAL_STEPS
    from experiments.train_landing_real_drives_v22 import ADDITIONAL_STEPS, real_drive_profile
    steps = dict(warm=ADDITIONAL_STEPS, fresh=TOTAL_STEPS)
    steps.update({'continue': CONTINUE_STEPS, 'disturbed': DISTURBED_STEPS, 'standing': STANDING_STEPS, 'soft': SOFT_STEPS})
    if arm not in steps:
        raise ValueError('LANDING_V22_ARM must be warm, fresh, continue, disturbed, standing or soft')
    profile = real_drive_profile(parent)
    name = dict(warm='landing_real_drives_mjx_v22', fresh='landing_real_drives_mjx_fresh_v22',
                disturbed='landing_real_drives_mjx_disturbed_v22', standing='landing_real_drives_mjx_standing_v22',
                soft='landing_real_drives_mjx_soft_v22').get(
        arm, 'landing_real_drives_mjx_continue_v22')
    profile['algorithm'].update(name='ppo.flax_mjx_landing', nr_steps=NR_STEPS, total_timesteps=steps[arm],
                                final_learning_rate={'continue': 0., 'disturbed': 1e-5, 'standing': 1e-5, 'soft': 1e-5}.get(arm, -1.))
    ppo = {}
    for variable, key, cast in (('MINIBATCH', 'minibatch_size', int), ('ENTROPY_COEF', 'entropy_coef', float),
                                ('MIN_STD', 'min_std', float), ('LEARNING_RATE', 'learning_rate', float), ('GAMMA', 'gamma', float)):
        value = os.environ.get(f'LANDING_V22_{variable}')
        if value:
            ppo[key] = cast(value)
    if NR_ENVS * NR_STEPS % ppo.get('minibatch_size', 64):
        raise ValueError(f'LANDING_V22_MINIBATCH must divide the batch {NR_ENVS * NR_STEPS}')
    profile['algorithm'].update(ppo)
    profile['environment'].update(name='custom_mujoco.robot_locomotion.mjx_landing', nr_envs=NR_ENVS)
    if arm == 'disturbed':
        profile['environment']['mjx'] = dict(release_disturbance=dict(DISTURBANCE))
    if arm == 'standing':
        profile['environment']['mjx'] = dict(release_pose=dict(STAND_RELEASE))
    if arm == 'soft':
        reference = os.environ.get('LANDING_V22_REFERENCE', 'track')
        profile['environment']['mjx'] = dict(limb_contact=dict(soft=True, force_limit=200.), braking_reference=reference)
    origin = dict(warm='continues V19 latest (job 9591668): actor, critic, Adam, learned std',
                  fresh='from scratch: new actor, critic, Adam and std 0.15',
                  soft='continues a V22 checkpoint where thigh and calf touches are allowed up to 200 N (10 ms mean) and end the '
                       f"episode above it; braking term: {os.environ.get('LANDING_V22_REFERENCE', 'track')}; learning rate 3e-5 decays "
                       'linearly to 1e-5',
                  standing='continues a V22 checkpoint released from the symmetric standing pose (moved in from the landing '
                           'pose over 30M steps, 20% of the episodes stay at the landing pose, evaluation always from standing); '
                           'learning rate 3e-5 decays linearly to 1e-5',
                  disturbed='continues a V22 checkpoint with an imperfect release (roll/pitch up to 30 deg, body rate up '
                            'to 15 deg/s, ramped over 20M steps); learning rate 3e-5 decays linearly to 1e-5').get(
        arm, 'continues V22 warm latest (job 9730655, 60,030,976 steps): actor, critic, Adam, learned std; '
             'learning rate 3e-5 decays linearly to 0')
    if os.environ.get('LANDING_V22_FRESH') == '1':
        if arm != 'soft' or parent:
            raise ValueError('LANDING_V22_FRESH=1 is the soft arm without a parent')
        origin = ('from scratch (new actor, critic, Adam, std 0.15) with soft thigh/calf contact up to 200 N; braking term: '
                  f"{os.environ.get('LANDING_V22_REFERENCE', 'track')}; learning rate 3e-5 decays linearly to 1e-5")
    total = int(os.environ.get('LANDING_V22_TOTAL_STEPS') or 0)
    if total:
        if total % (NR_ENVS * NR_STEPS):
            raise ValueError(f'LANDING_V22_TOTAL_STEPS must be a multiple of the batch {NR_ENVS * NR_STEPS}')
        profile['algorithm']['total_timesteps'] = total
        origin += f'; {total:,} steps'
    if ppo:
        origin += '; PPO ' + ', '.join(f'{k}={v:g}' for k, v in ppo.items())
    if os.environ.get('LANDING_V22_SYMMETRIC_POSE') == '1':
        profile['environment'].setdefault('mjx', {}).update(symmetric_pose=True, symmetric_front_hip=front_hip())
        origin += f'; right front hip {front_hip():g} rad, left mirrored'
        origin += '; symmetric nominal pose (left legs mirrored from the right ones: release, action origin, joint observations)'
    tilt = float(os.environ.get('LANDING_V22_RELEASE_TILT_DEG') or 0.)
    rate = float(os.environ.get('LANDING_V22_RELEASE_RATE_DPS') or 0.)
    if tilt or rate:
        if arm != 'soft':
            raise ValueError('LANDING_V22_RELEASE_TILT_DEG/RATE_DPS extend the soft arm')
        ramp = int(os.environ.get('LANDING_V22_RELEASE_RAMP_STEPS') or DISTURBANCE['ramp_steps'])  # 0: full from the start
        profile['environment'].setdefault('mjx', {})['release_disturbance'] = dict(
            max_tilt_deg=tilt, max_rate_dps=rate, ramp_steps=ramp)
        origin += (f'; imperfect release: roll and pitch each up to {tilt:g} deg, body rate up to {rate:g} deg/s, '
                   f'ramped over {ramp:,} steps')
    hip_rate = float(os.environ.get('LANDING_V22_FLIGHT_HIP_RATE') or 0.)
    if hip_rate:
        profile['environment'].setdefault('mjx', {})['flight_hip_rate_penalty'] = hip_rate
        origin += f'; penalty {hip_rate:g} x squared hip action change before touchdown'
    max_speed = float(os.environ.get('LANDING_V22_MAX_SPEED') or 5.)
    if max_speed != 5.:
        profile['environment'].setdefault('mjx', {})['max_speed'] = max_speed
        evaluation = profile['environment']['landing_evaluation']
        grid = [float(v) for v in evaluation['velocity_grid'].split(',')]
        grid += [float(v) for v in range(int(grid[-1]) + 1, int(max_speed) + 1)]
        evaluation['velocity_grid'] = ','.join(f'{v:g}' for v in grid)
        evaluation['selection_speed_max'] = 5.  # best is chosen on the old range plus 3-5 m/s; faster cells are reported only
        origin += f'; release speed extended to {max_speed:g} m/s (20% of episodes in 5-{max_speed:g})'
    for variable, key in (('CONSTRAINTS_PER_ENV', 'constraints_per_env'), ('CONTACTS_PER_ENV', 'contacts_per_env')):
        value = os.environ.get(f'LANDING_V22_{variable}')  # MJX buffer sizes; an overflow stops the run
        if value:
            profile['environment'].setdefault('mjx', {})[key] = int(value)
    randomization = {}
    for variable, key in (('RANDOM_MASS_KG', 'trunk_mass_kg'), ('RANDOM_COM_M', 'com_shift_m'),
                          ('TORQUE_MIN_NM', 'torque_min_nm'), ('TORQUE_MAX_NM', 'torque_max_nm'), ('RANDOM_PD', 'pd_gain'),
                          ('FRICTION_MIN', 'friction_min'), ('FRICTION_MAX', 'friction_max')):
        value = os.environ.get(f'LANDING_V22_{variable}')
        if value:
            randomization[key] = float(value)
    if randomization:
        profile['environment'].setdefault('mjx', {})['randomization'] = randomization
        origin += '; domain randomization per episode ' + ', '.join(f'{k}={v:g}' for k, v in randomization.items())
    posture = float(os.environ.get('LANDING_V22_POSTURE_REWARD') or 0.)
    if posture:
        profile['environment'].setdefault('mjx', {})['posture_reward'] = posture
        origin += f'; reward {posture:g} per step for a level, symmetric stance after touchdown'
    if os.environ.get('LANDING_V22_SUCCESS_AT_END') == '1':
        roll = float(os.environ.get('LANDING_V22_END_ROLL_DEG') or 0.)
        profile['environment'].setdefault('mjx', {}).update(success_at_end=True, end_roll_limit_deg=roll)
        origin += '; success judged at the end of the episode only (no braking time requirements)'
        origin += f', side roll at most {roll:g} deg' if roll else ''
    slip_scale = os.environ.get('LANDING_V22_SLIP_SCALE')  # 0 removes the loaded-slip penalty, any arm
    if slip_scale is not None:
        profile['environment'].setdefault('mjx', {})['slip_penalty_scale'] = float(slip_scale)
        origin += f'; loaded-slip penalty x{float(slip_scale):g}'
    profile['runner'].update(
        load_model=str(parent), exp_name=name, run_name=os.environ.get('RUN_NAME') or
        f"{name}_{os.environ.get('SLURM_JOB_ID', 'local')}",
        notes=f'V22 on MJX-Warp (float32), {origin}. Same task, 16 Nm limits, reward, IMU contract, evaluation grid and PPO '
              f'as CPU V22; rollouts {NR_ENVS}x{NR_STEPS} instead of 64x2048, batch {NR_ENVS * NR_STEPS:,}.')
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
    elif arm == 'soft' and os.environ.get('LANDING_V22_FRESH') == '1':
        if os.environ.get('LANDING_V22_CHECKPOINT'):
            raise ValueError('LANDING_V22_FRESH=1 trains from scratch: unset LANDING_V22_CHECKPOINT')
    elif arm in ('continue', 'disturbed', 'standing', 'soft'):
        parent = Path(os.environ.get('LANDING_V22_CHECKPOINT', '')).expanduser().resolve()
        if not parent.is_file():
            raise FileNotFoundError(f'Set LANDING_V22_CHECKPOINT to the parent latest.model. Missing: {parent}')
        from experiments.train_landing_real_drives_v22 import ADDITIONAL_STEPS
        required = os.environ.get('LANDING_V22_PARENT_STEPS')
        required = int(required) if required else ADDITIONAL_STEPS if arm == 'continue' else None
        print('Parent:', parent, verify_continue_parent(parent, required), flush=True)
        verify_parent_pose(parent, os.environ.get('LANDING_V22_SYMMETRIC_POSE') == '1')
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
