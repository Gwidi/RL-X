"""V16 fresh 90M-step landing PPO: observed launch, IMU history and integral."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.train_landing_braking_v14 import fresh_moving_profile, fresh_moving_environment_config
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_seed606 import cli_flags, verify_controller, ROBOT_DIR
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.policy_inertial import DEFAULTS, inertial_observation_contract


def inertial_profile():
    p = fresh_moving_profile()
    p['algorithm'].update(total_timesteps=90046464)
    e = p['environment']
    e.update(imu_history_length=50, add_goal_arrow=False,
             policy_inertial={**DEFAULTS, 'enabled': True},
             policy_velocity=dict(mode='none'))
    # Inertial schema implies a masked goal and a reward-only truth reference.
    # Do not add defaults to legacy braking dictionaries: archived parents
    # compare those dictionaries to verify that their old tasks are unchanged.
    c = e['domain_randomization']['initial_state']['energy_dissipation_curriculum']
    c.update(focused_velocity_enabled=False, factorized_enabled=False,
             height_then_velocity_enabled=True, start_height=.75, target_height=1.5,
             min_angle_deg=0., max_angle_deg=0., min_lateral_velocity=0., max_lateral_velocity=8.,
             velocity_sampling='speed_disk', sequential_heights='0.75,1,1.25,1.5',
             sequential_speeds='1,2,3,4,5,6,7,8', sequential_window=100,
             sequential_threshold=.8, sequential_frontier_probability=.6,
             nominal_replay_probability=0., replay_easy_probability=0.)
    e['landing_evaluation'].update(heights='0.75,1,1.25,1.5', max_angle_deg=0.,
        max_lateral_velocity=8., velocity_sampling='speed_ring', factorized_conditions=False,
        velocity_grid='0,1,2,3,4,5,5.5,6,6.5,7,7.5,8',
        selection_speed_min=1., selection_speed_max=8., frequency_multiplier=32,
        success_boundary_threshold=.8, nominal_success_threshold=.8, evaluate_final=True)
    p['runner'].update(load_model='', exp_name='landing_inertial_v16',
        run_name=os.environ.get('RUN_NAME') or 'inertial_v16_'+os.environ.get('SLURM_JOB_ID', 'local'),
        notes='V16 fresh actor/critic/Adam/std0.15;90,046,464 flight transitions;observed ideal supported launch from rest0.16to0.32s;forces not qvel jump;400Hz noisy biased IMU integration with gyro attitude and lever-arm correction;1s six-channel IMU history at50Hz plus2 accumulated delta-v channels;20ms delayed;no measured/simulator velocity or velocity reference in actor;privileged truth only reward/critic/diagnostics and fixture servo;V14 braking2s to0.5mps,tracking12,slip5,safety unchanged;per-worker >80pct/100 frontier height0.75to1.5 then horizontal norm1to8mps all directions;eval all heights0to8 from start;zero sampled prelaunch angles,small physical release tilt/spin logged;not a controlled A/B with V15')
    return p


def inertial_environment_config():
    cfg = fresh_moving_environment_config()
    def merge(target, updates):
        for key, value in updates.items():
            if isinstance(value, dict):
                merge(target.setdefault(key, {}), value)
            else:
                target[key] = value
    merge(cfg, inertial_profile()['environment'])
    return cfg


def make_inertial_env(mode='test', config=None, seed=20261110):
    from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.environment import LocomotionEnv
    from rl_x.environments.custom_mujoco.robot_locomotion.robots.silver_badger.robot_config import robot_config
    cfg = inertial_environment_config() if config is None else deepcopy(config)
    cfg['nr_envs'] = 1
    robot = deepcopy(robot_config)
    robot.update(directory_path=ROBOT_DIR, spine_locked=False)
    env = LocomotionEnv(robot, mode, seed, False, cfg, 1)
    verify_controller(env.initial_mj_model)
    return env


if __name__ == '__main__':
    profile = inertial_profile()
    fingerprint = verify_controller()
    if '--check-only' in sys.argv:
        print(json.dumps(dict(profile=profile, fresh_initialization=True,
            inertial_contract=inertial_observation_contract(profile['environment']['policy_inertial']),
            fingerprint=fingerprint), indent=2))
    else:
        if any(arg.startswith('--runner.load_model') for arg in sys.argv[1:]):
            raise ValueError('V16 trains fresh with new IMU/launch semantics; checkpoint overrides are disabled')
        sys.argv = [sys.argv[0], *cli_flags(profile), *sys.argv[1:]]
        from rl_x.runner.runner import Runner
        Runner().run()
