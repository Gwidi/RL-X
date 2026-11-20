"""Closed-loop handoff from the landing actor to a frozen walking actor."""
from functools import lru_cache

import numpy as np

from .policy_velocity import heading_velocity_xy

WALK_HOME = np.array([0., -.1, -.8, 1.5, .1, .8, -1.5, -.1, .8, -1.5, .1, -.8, 1.5])
WALK_JOINT_ORDER = ('spine_joint',) + tuple(
    f'{leg}_{joint}_joint' for leg in ('RL', 'RR', 'FR', 'FL')
    for joint in ('hip', 'thigh', 'calf')
)
WALK_SCALE = .25
# Appended to the raw observation; the first two fields are PPO action masks.
FEATURE_COUNT = 6


@lru_cache(maxsize=2)
def walking_weights(path, source_sha256):
    with np.load(path, allow_pickle=False) as archive:
        if str(archive['source_sha256']) != source_sha256:
            raise ValueError('Frozen walking actor has a different source checkpoint')
        return {key: archive[key] for key in archive.files if key != 'source_sha256'}


def walking_mean(weights, observation):
    x = np.asarray(observation, dtype=np.float32)
    for layer in range(4):
        x = x @ weights[f'Dense_{layer}/kernel'] + weights[f'Dense_{layer}/bias']
        if layer == 0:
            mean = np.mean(x)
            var = max(0., np.mean(x * x) - mean * mean)
            x = (x - mean) / np.sqrt(var + np.float32(1e-6))
            x = x * weights['LayerNorm_0/scale'] + weights['LayerNorm_0/bias']
        if layer < 3:
            x = np.where(x > 0., x, np.expm1(np.minimum(x, 0.)))
    return x


class LandingHandoff:
    def __init__(self, env):
        self.env = env
        cfg = env.env_config.get('landing_handoff', {})
        self.enabled = bool(cfg.get('enabled', False))
        if self.enabled:
            self.speed_limit = float(cfg['walk_speed_limit'])
            self.min_foot_force = float(cfg.get('gate_min_foot_force_n', 10.))
            self.min_base_height = float(cfg.get('gate_min_base_height_m', .16))
            self.max_horizontal_speed = float(cfg.get('gate_max_horizontal_speed_mps', 2.5))
            self.horizontal_speed_inclusive = cfg.get('gate_horizontal_speed_inclusive', True)
            if (not np.all(np.isfinite([self.min_foot_force, self.min_base_height, self.max_horizontal_speed]))
                    or self.min_foot_force < 0. or self.min_base_height < 0.
                    or self.max_horizontal_speed <= 0.):
                raise ValueError('Invalid handoff gate thresholds')
            if not env.landing_braking.enabled or self.speed_limit < env.landing_braking.final_speed:
                raise ValueError('Handoff requires active braking and a cap >= final speed')
            self.weights = walking_weights(cfg['walk_weights'], cfg['walk_sha256'])
        self.blend_duration = .28

    def reset(self):
        if not self.enabled:
            return
        if tuple(self.env.actuator_joint_names) != WALK_JOINT_ORDER:
            raise ValueError('Unexpected actuator order for frozen walker')
        self.env.internal_state['landing_handoff'] = dict(
            time=None, alpha=0., previous=np.zeros(13), low_load=np.zeros(4),
            foot_steps=np.zeros(4, dtype=int), tracking_time=0., delay_cost=0.,
            at_switch_speed=0., at_switch_delay=0., full_time=0.)

    def state(self):
        return self.env.internal_state['landing_handoff']

    def ready(self):
        env = self.env
        state = env.internal_state
        if (self.state()['time'] is not None or not state['has_touched_ground']
                or env.landing_braking.elapsed() + 1e-8 < .08):
            return False
        model, data = state['mj_model'], state['data']
        speed = np.linalg.norm(data.qvel[:2])
        within_speed = (speed <= self.max_horizontal_speed if self.horizontal_speed_inclusive
                        else speed < self.max_horizontal_speed)
        if not (data.qpos[2] > self.min_base_height and within_speed
                and data.site_xmat[env.imu_site_id, 8] > np.cos(np.deg2rad(40.))
                and abs(data.qvel[2]) < .85 and np.linalg.norm(data.qvel[3:6]) < 4.5):
            return False
        if self.min_foot_force > 0.:
            foot_force, *_ = state['landing_safety'].contacts.support_contact_forces(model, data)
            return bool(foot_force > self.min_foot_force)
        return True

    def world_command(self):
        env = self.env
        if not env.internal_state['has_touched_ground']:
            return np.zeros(2)
        world = env.landing_braking.reference_world()
        return world * min(1., self.speed_limit / max(1e-8, np.linalg.norm(world)))

    def local_command(self, world):
        env = self.env
        rotation = env.internal_state['data'].site_xmat[env.imu_site_id].reshape(3, 3)
        xy = heading_velocity_xy(np.r_[world, 0.], rotation)
        return np.r_[xy, 0.]

    def blend_fraction(self):
        time = self.state()['time']
        if time is None:
            return 0.
        elapsed = float(self.env.internal_state['data'].time) - time
        u = np.clip((elapsed + 1e-8) / self.blend_duration, 0., 1.)
        return float(u * u * (3. - 2. * u))

    def observation(self):
        if not self.enabled:
            return np.empty(0)
        alpha = self.blend_fraction()
        return np.r_[float(self.ready()), alpha, float(self.state()['time'] is not None),
                     min(1., self.env.landing_braking.elapsed() / 2.),
                     self.local_command(self.world_command())[:2]]

    def blend(self, action, requested):
        env = self.env
        state = self.state()
        data = env.internal_state['data']
        model = env.internal_state['mj_model']
        gear = model.actuator_gear[:, 0]
        ready = self.ready()
        if ready and requested:
            state.update(time=float(data.time),
                         previous=(data.ctrl / gear - WALK_HOME) / WALK_SCALE,
                         at_switch_speed=float(np.linalg.norm(data.qvel[:2])),
                         at_switch_delay=env.landing_braking.elapsed())
        state['delay_cost'] = .6 * env.dt if ready and state['time'] is None else 0.
        if state['time'] is None:
            return action
        alpha = self.blend_fraction()
        state['alpha'] = alpha
        matrix = data.site_xmat[env.imu_site_id].reshape(3, 3)
        adr = env.imu_angular_velocity_sensor_adr
        observation = np.r_[(data.qpos[env.actuator_joint_mask_qpos] - WALK_HOME) / 3.14,
                            data.qvel[env.actuator_joint_mask_qvel] / 100., state['previous'] / 10.,
                            np.clip(data.sensordata[adr:adr+3] / 50., -1., 1.),
                            self.local_command(self.world_command()), -matrix[2]].clip(-10.,10.)
        raw = walking_mean(self.weights, observation)
        if raw.shape != (13,) or not np.all(np.isfinite(raw)):
            raise ValueError('Frozen walker returned invalid actions')
        state['previous'] = raw
        landing_target = env.control_function.process_action(action) / gear
        target = (1. - alpha) * landing_target + alpha * (WALK_HOME + WALK_SCALE * raw)
        previous = data.ctrl / gear
        target = np.clip(target, previous - 7.*env.dt, previous + 7.*env.dt)
        home = env.internal_state['actuator_joint_nominal_positions'] + env.internal_state['position_offsets']
        return (target - home) / env.internal_state['scaling_factor']

    def reward_and_success(self, braking_reward):
        env = self.env
        state = self.state()
        info = env.internal_state['info']
        alpha = state['alpha']
        data = env.internal_state['data']
        tracking = 0.
        succeeded = False
        if state['time'] is not None:
            command = self.world_command()
            error = float(np.linalg.norm(data.qvel[:2] - command))
            tracking = 12. * env.dt * np.exp(-(error / .40)**2)
            if alpha >= 1.:
                state['full_time'] += env.dt
                model = env.internal_state['mj_model']
                contact_map = env.internal_state['landing_safety'].contacts
                _, _, forces, _, _ = contact_map.support_contact_forces(model, data)
                loaded = forces > 10.
                state['foot_steps'] += loaded & (state['low_load'] >= .06 - 1e-8)
                state['low_load'] = np.where(loaded, 0., state['low_load'] + env.dt)
                stable = (error <= .40 and abs(data.qvel[2]) <= .30
                          and np.linalg.norm(data.qvel[3:6]) <= 2. and np.any(loaded))
                state['tracking_time'] = state['tracking_time'] + env.dt if stable else 0.
                succeeded = bool(state['full_time'] >= .8 - 1e-8
                                 and state['tracking_time'] >= .4 - 1e-8
                                 and np.count_nonzero(state['foot_steps']) >= 2)
        info.update({'handoff/alpha': alpha, 'handoff/accepted': float(state['time'] is not None),
                     'handoff/delay_after_touch_s': state['at_switch_delay'],
                     'handoff/speed_at_switch_mps': state['at_switch_speed'],
                     'handoff/feet_stepped': float(np.count_nonzero(state['foot_steps'])),
                     'reward/handoff_tracking': alpha * tracking,
                     'reward/handoff_wait': -state['delay_cost']})
        # Fade the old tracking term; braking_reward also contains the slip cost.
        return (braking_reward + alpha * (tracking - info['reward/braking_tracking'])
                - state['delay_cost']), succeeded
