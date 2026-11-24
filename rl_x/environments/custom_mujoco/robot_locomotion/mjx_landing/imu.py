"""JAX port of policy_inertial.py, batched over worlds (leading axis)."""

import jax
import jax.numpy as jnp

from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.policy_inertial import inertial_observation_contract


def heading_velocity_xy(velocity, rotation):
    forward = rotation[..., :, 0].at[..., 2].set(0.)
    length = jnp.linalg.norm(forward, axis=-1, keepdims=True)
    left = rotation[..., :, 1].at[..., 2].set(0.)
    left = left / jnp.maximum(jnp.linalg.norm(left, axis=-1, keepdims=True), 1e-12)
    # cross(left, z): used only when body X is vertical
    fallback = jnp.stack([left[..., 1], -left[..., 0], jnp.zeros_like(left[..., 0])], -1)
    forward = jnp.where(length < 1e-6, fallback, forward / jnp.maximum(length, 1e-12))
    left = jnp.stack([-forward[..., 1], forward[..., 0], jnp.zeros_like(forward[..., 0])], -1)
    return jnp.stack([(velocity * forward).sum(-1), (velocity * left).sum(-1)], -1)


def rotation_increment(rotvec):
    x, y, z = rotvec[..., 0], rotvec[..., 1], rotvec[..., 2]
    zero = jnp.zeros_like(x)
    skew = jnp.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(rotvec.shape[:-1] + (3, 3))
    square = (rotvec ** 2).sum(-1)[..., None, None]
    angle = jnp.sqrt(jnp.maximum(square, 1e-20))
    small = square < 1e-2  # Series: 1 - cos cancels in float32. Error below 1e-8.
    a = jnp.where(small, 1. - square / 6. + square ** 2 / 120., jnp.sin(angle) / angle)
    b = jnp.where(small, .5 - square / 24. + square ** 2 / 720., (1. - jnp.cos(angle)) / jnp.maximum(square, 1e-20))
    return jnp.eye(3) + a * skew + b * (skew @ skew)


class InertialSensor:
    def __init__(self, env_config, acceleration_adr, gyro_adr, timestep, gravity):
        self.contract = inertial_observation_contract(env_config['policy_inertial'])
        if not self.contract['enabled']:
            raise ValueError('MJX landing requires policy_inertial')
        if self.contract['delay_steps'] != 1:
            raise ValueError('MJX landing implements a one-step inertial delay')
        if env_config['imu_history_length'] != self.contract['history_steps']:
            raise ValueError('imu_history_length must equal policy_inertial.history_steps')
        self.history_steps = self.contract['history_steps']
        self.acceleration_adr, self.gyro_adr = int(acceleration_adr), int(gyro_adr)
        self.dt, self.gravity = float(timestep), jnp.asarray(gravity, jnp.float32)

    def init(self, keys, site_rotation, site_offset_world):
        n = keys.shape[0]
        split = jax.vmap(lambda k: jax.random.split(k, 3))(keys)
        normal = jax.vmap(lambda k: jax.random.normal(k, (3,)))
        return dict(
            key=split[:, 0], acceleration_bias=normal(split[:, 1]) * self.contract['acceleration_bias_std'],
            gyro_bias=normal(split[:, 2]) * self.contract['gyro_bias_std'],
            rotation=site_rotation, offset=jnp.einsum('nji,nj->ni', site_rotation, site_offset_world),
            previous_gyro=jnp.zeros((n, 3)), has_previous=jnp.zeros(n, bool), delta_v=jnp.zeros((n, 3)),
            # One extra row: rows [:-1] are the snapshot delayed by one control step.
            history=jnp.zeros((n, self.history_steps + 1, 6)),
            heading=jnp.zeros((n, 2)), delayed_heading=jnp.zeros((n, 2)),
            acceleration=jnp.zeros((n, 3)), gyro=jnp.zeros((n, 3)),
            samples=jnp.zeros(n, jnp.int32), clipped=jnp.zeros(n, jnp.int32))

    def physics_step(self, state, sensordata):
        c = self.contract
        split = jax.vmap(lambda k: jax.random.split(k))(state['key'])
        noise = jax.vmap(lambda k: jax.random.normal(k, (6,)))(split[:, 1])
        acc = sensordata[:, self.acceleration_adr:self.acceleration_adr + 3]
        gyro = sensordata[:, self.gyro_adr:self.gyro_adr + 3]
        acc = acc + state['acceleration_bias'] + noise[:, :3] * c['acceleration_noise_std']
        gyro = gyro + state['gyro_bias'] + noise[:, 3:] * c['gyro_noise_std']
        clipped = (jnp.abs(acc) > c['acceleration_clip']).any(-1) | (jnp.abs(gyro) > c['gyro_clip']).any(-1)
        acc = jnp.clip(acc, -c['acceleration_clip'], c['acceleration_clip'])
        gyro = jnp.clip(gyro, -c['gyro_clip'], c['gyro_clip'])
        alpha = jnp.where(state['has_previous'][:, None], (gyro - state['previous_gyro']) / self.dt, 0.)
        r = state['offset']
        base_force = acc - jnp.cross(alpha, r) - jnp.cross(gyro, jnp.cross(gyro, r))
        world = jnp.einsum('nij,nj->ni', state['rotation'], base_force) + self.gravity
        return {**state, 'key': split[:, 0], 'delta_v': state['delta_v'] + world * self.dt,
                'rotation': state['rotation'] @ rotation_increment(gyro * self.dt),
                'previous_gyro': gyro, 'has_previous': jnp.ones_like(state['has_previous']),
                'acceleration': acc, 'gyro': gyro,
                'samples': state['samples'] + 1, 'clipped': state['clipped'] + clipped}

    def publish(self, state):
        row = jnp.concatenate([state['acceleration'] / 98.1, state['gyro'] / 50.], -1)[:, None]
        heading = heading_velocity_xy(state['delta_v'], state['rotation']) / self.contract['delta_v_scale_mps']
        return {**state, 'history': jnp.concatenate([state['history'][:, 1:], row], 1),
                'delayed_heading': state['heading'], 'heading': heading}

    def observe(self, state):
        """(history, delayed acceleration, delayed gyro, delayed delta-v)."""
        history = state['history'][:, :-1]
        last = history[:, -1]
        return history, last[:, :3] * 98.1, last[:, 3:] * 50., state['delayed_heading']
