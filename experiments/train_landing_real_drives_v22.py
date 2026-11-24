"""V22: continue V19 latest under the HB40 drive limits (16 Nm, spine +-0.785 rad)."""

import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.train_landing_handoff_v20 import PARENT_SHA, input_path
from experiments.train_landing_inertial_speed10_v19 import speed10_profile
from experiments.train_landing_inertial_v16 import inertial_environment_config, make_inertial_env
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_actuators import (
    motor_torque_limits, read_landing_actuators)
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_seed606 import cli_flags, verify_controller

ACTUATOR_PROFILE = ('rl_x/environments/custom_mujoco/robot_locomotion/robots/silver_badger/data/'
                    'landing_actuators_hb40_16nm.json')
PARENT_STEPS = 30_015_488
ADDITIONAL_STEPS = 60_030_976


def real_drive_profile(parent):
    actuators, digest = read_landing_actuators(ACTUATOR_PROFILE)
    profile = speed10_profile()
    profile['algorithm']['total_timesteps'] = ADDITIONAL_STEPS
    env = profile['environment']
    env.update(landing_uniform_speed10=False, landing_real_drive_mix=True,
               landing_actuators=dict(profile=ACTUATOR_PROFILE, sha256=digest))
    reward = env['reward']
    reward.update(motor_torque_limits(actuators, reward['leg_gear_ratio'], reward['spine_gear_ratio']))
    env['domain_randomization']['initial_state']['energy_dissipation_curriculum'].update(
        start_height=.5, target_height=1.25, max_lateral_velocity=5.)
    env['landing_evaluation'].update(
        heights='0.5,0.75,1,1.25', max_lateral_velocity=5., velocity_grid='0,0.5,1,1.5,2,2.5,3,4,5',
        selection_speed_min=.5, selection_speed_max=3., best_by_worst_cell=False)
    profile['runner'].update(
        load_model=str(parent), exp_name='landing_real_drives_v22',
        run_name=os.environ.get('RUN_NAME') or 'landing_real_drives_v22_' + os.environ.get('SLURM_JOB_ID', 'local'),
        notes=('V22 continues V19 latest (job 9591668): actor, critic, Adam, learned std; 60,030,976 new transitions. '
               'Real drive limits: 16 Nm on every joint instead of 27/28.26 Nm, spine +-0.785 rad instead of +-1.57. '
               'PD gains, home, IMU contract, reward weights, contact and integration unchanged. '
               'Heights 0.5/0.75/1/1.25 m; 10% vertical, 40% 0-2, 30% 2-3.5, 20% 3.5-5 m/s; all azimuths. '
               'Best: worst-height success at 0.5-3 m/s subject to nominal eligibility.'))
    return profile


def verify_parent(parent):
    with zipfile.ZipFile(parent) as archive:
        saved = json.loads(archive.read('config_environment.json'))
        metrics = json.loads(archive.read('checkpoint_metrics.json'))
    expected = inertial_environment_config()
    for key in ('landing_braking', 'reward', 'policy_inertial', 'imu_history_length', 'timestep', 'control_type',
                'spine_locked', 'landing_physics_sha256', 'landing_controller_sha256', 'landing_model_sha256'):
        if saved.get(key) != expected.get(key):
            raise ValueError(f'Parent contract mismatch: {key}')
    if saved.get('landing_actuators', {}).get('profile') or saved.get('landing_handoff', {}).get('enabled'):
        raise ValueError('V22 starts from the 27 Nm V19 landing actor')
    if metrics['nr_env_steps'] != PARENT_STEPS:
        raise ValueError('V22 requires the final V19 latest checkpoint')


def check_model(profile):
    import numpy as np
    config = inertial_environment_config()

    def merge(target, updates):
        for key, value in updates.items():
            if isinstance(value, dict):
                merge(target.setdefault(key, {}), value)
            else:
                target[key] = value
    merge(config, profile['environment'])
    env = make_inertial_env('train', config)
    try:
        heights, speeds = [], []
        for _ in range(200):
            env.reset()
            heights.append(env.internal_state['info']['curriculum/v22_height'])
            speeds.append(env.internal_state['info']['curriculum/v22_speed'])
        model = env.internal_state['mj_model']
        limits = model.actuator_forcerange[:, 1] * model.actuator_gear[:, 0]
        np.testing.assert_allclose(limits, 16.)
        np.testing.assert_allclose(model.jnt_range[model.joint('spine_joint').id], [-.785, .785])
        # The reward must fail a trial only above the same limit as the simulated motor.
        np.testing.assert_allclose(env.reward_function.leg_motor_torque_limit * 9., 16.)
        np.testing.assert_allclose(env.reward_function.spine_motor_torque_limit * 36., 16.)
        assert set(heights) == {.5, .75, 1., 1.25} and min(speeds) == 0. and 3.5 < max(speeds) <= 5.
        return dict(joint_limit_nm=limits.tolist(), mean_speed=float(np.mean(speeds)),
                    vertical_fraction=float(np.mean(np.array(speeds) == 0.)))
    finally:
        env.close()


def main():
    parent = input_path('LANDING_V22_CHECKPOINT', 'modelsv19/latest.model', PARENT_SHA)
    profile = real_drive_profile(parent)
    verify_parent(parent)
    fingerprint = verify_controller()
    if '--check-only' in sys.argv:
        print(json.dumps(dict(profile=profile, parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
                              fingerprint=fingerprint, model=check_model(profile)), indent=2))
        return
    if any(arg.startswith(('--runner.load_model', '--environment.landing_actuators')) for arg in sys.argv[1:]):
        raise ValueError('Use LANDING_V22_CHECKPOINT and the V22 actuator profile')
    sys.argv = [sys.argv[0], *cli_flags(profile), *sys.argv[1:]]
    from rl_x.runner.runner import Runner
    Runner().run()


if __name__ == '__main__':
    main()
