"""V22 landing task on MJX-Warp.

A port of mujoco/environment.py with simplified_landing_reward.py, restricted to
the options V22 uses. Other options fail in _check_config instead of being
silently ignored. The physics model, observation layout and nominal values are
taken from a CPU LocomotionEnv, so both environments share one definition.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from flax import struct
from mujoco import mjx

from rl_x.environments.custom_mujoco.robot_locomotion.mjx_landing.contacts import (
    ContactTables, active_contacts, contact_metrics, contact_overflow)
from rl_x.environments.custom_mujoco.robot_locomotion.mjx_landing.imu import InertialSensor
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.environment import LocomotionEnv
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_safety import (
    BODY_ACCELERATION_WINDOW, CALF_FORCE_WINDOW, FOOT_FORCE_WINDOW, landing_violations)


FAILURES = ('motor_hard_limit', 'base_crash', 'disallowed_contact', 'calf_force', 'foot_force',
            'body_acceleration', 'base_height', 'unresolved')
# Categories of the V22 mixture: vertical, 0-2, 2-3.5, 3.5-5 m/s.
HEIGHTS = (.5, .75, 1., 1.25)
GROUP_EDGES = (.1, .5, .8)
GROUP_LOW, GROUP_HIGH = (0., 0., 2., 3.5), (0., 2., 3.5, 5.)


@struct.dataclass
class State:
    data: mjx.Data
    imu: dict
    task: dict
    observation: jax.Array
    actual_observation: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    info: dict
    overflow: jax.Array


def select(mask, new, old):
    return jax.tree.map(lambda a, b: jnp.where(mask.reshape(mask.shape + (1,) * (a.ndim - 1)), a, b), new, old)


def tilt_quaternion(tilt):
    """Quaternion (w, x, y, z) of scipy's extrinsic "xyz" rotation by roll, then pitch."""
    half = tilt / 2
    cos, sin = jnp.cos(half), jnp.sin(half)
    roll_w, roll_x, pitch_w, pitch_y = cos[:, 0], sin[:, 0], cos[:, 1], sin[:, 1]
    # q = q_pitch * q_roll
    return jnp.stack([pitch_w * roll_w, pitch_w * roll_x, pitch_y * roll_w, -pitch_y * roll_x], -1)


def highest_precision(function):
    """The runner sets a low default matmul precision for the networks. IMU
    integration would not survive TF32, so tasks are traced at full float32."""
    def wrapped(*args, **kwargs):
        with jax.default_matmul_precision('highest'):
            return function(*args, **kwargs)
    return wrapped


class LandingMjxEnv:
    def __init__(self, robot_config, env_config):
        self.config = env_config
        self._check_config(env_config)
        cpu_config = {**env_config, 'nr_envs': 1}
        cpu = LocomotionEnv(robot_config, 'train', env_config['seed'], False, cpu_config, 1)
        # Reset once: zero-strength randomization still writes a few model fields.
        cpu.reset()
        self._read_cpu_environment(cpu)
        self.nr_envs = env_config['nr_envs']
        disturbance = env_config['mjx']['release_disturbance']
        self.max_tilt, self.max_rate = np.deg2rad(disturbance['max_tilt_deg']), np.deg2rad(disturbance['max_rate_dps'])
        self.disturbed = bool(self.max_tilt or self.max_rate)
        self.warp_model = mjx.put_model(self.model, impl='warp')
        self._data_templates = {}
        cpu.close()

    # ------------------------------------------------------------------ setup

    @staticmethod
    def _check_config(c):
        reward, braking = c['reward'], c['landing_braking']
        initial = c['domain_randomization']['initial_state']['energy_dissipation_curriculum']
        requirements = {
            'landing_real_drive_mix': c['landing_real_drive_mix'],
            'plane terrain': c['terrain']['type'] == 'plane',
            'simplified_landing reward': reward['type'] == 'simplified_landing',
            'pd control': c['control_type'] == 'pd',
            'no render': not c['render'],
            'free spine': not c['spine_locked'],
            'no goal arrow': not c['add_goal_arrow'],
            'no domain randomization': not c['env_curriculum_enabled'] and c['env_curriculum_disabled_coeff'] == 0.,
            'no handoff': not c['landing_handoff']['enabled'],
            'no frontflip': not c['landing_frontflip']['enabled'],
            'no policy height': not c['policy_height']['enabled'],
            'no policy velocity': c['policy_velocity']['mode'] == 'none',
            'no landing mat': not c['landing_mat']['profile'],
            'braking': braking['enabled'] and braking['final_speed'] > 0 and not braking['success_tracks_reference'],
            'privileged masked braking reference': (
                braking.get('reference_source', 'privileged_truth') == 'privileged_truth'
                and braking.get('goal_observation', 'masked') == 'masked'),
            'loaded slip': braking['slip']['weight'] > 0,
            'unused reward terms': all(reward[k] == 0 for k in (
                'base_vel_coeff', 'joint_vel_coeff', 'floor_collision_coeff', 'base_height_coeff',
                'roll_pitch_pos_coeff', 'joint_pos_coeff', 'action_rate_coeff')),
            'no thermal limit': not reward['thermal_limit_enabled'],
            'motor-side torque': reward.get('torque_measurement_mode', 'mujoco_actuator') == 'mujoco_actuator',
            'zero sampled angles': initial['min_angle_deg'] == initial['max_angle_deg'] == 0,
            'V22 heights and speed': (initial['height_reference'] == 'base' and initial['start_height'] == HEIGHTS[0]
                                      and initial['target_height'] == HEIGHTS[-1] and initial['max_lateral_velocity'] == 5.),
            'plain critic exteroception': c['critic_exteroceptive_observation_type'] == 'height_over_ground',
            'no policy exteroception': c['policy_exteroceptive_observation_type'] == 'none',
        }
        missing = [name for name, ok in requirements.items() if not ok]
        if missing:
            raise ValueError('MJX landing does not implement this configuration: ' + ', '.join(missing))

    def _read_cpu_environment(self, cpu):
        state = cpu.internal_state
        self.model = model = state['mj_model']
        self.single_observation_space, self.single_action_space = cpu.observation_space, cpu.action_space
        self.policy_observation_indices = cpu.policy_observation_indices
        self.critic_observation_indices = cpu.critic_observation_indices
        self.dt, self.nr_substeps, self.horizon = cpu.dt, cpu.nr_substeps, cpu.horizon
        self.timestep = float(model.opt.timestep)
        f32 = lambda x: jnp.asarray(x, jnp.float32)
        self.nominal = f32(state['actuator_joint_nominal_positions'])
        self.home_qpos = f32(cpu.initial_qpos)
        self.action_scale = float(np.asarray(state['scaling_factor']))
        if np.any(state['position_offsets']):
            raise ValueError('MJX landing expects zero actuator position offsets')
        joints = cpu.actuator_joint_mask_joints
        if not model.jnt_limited[joints].all():
            raise ValueError('MJX landing expects limited actuator joints')
        self.joint_range = f32(model.jnt_range[joints])
        self.gear = f32(model.actuator_gear[:, 0])
        self.qpos_index, self.qvel_index = jnp.asarray(cpu.actuator_joint_mask_qpos), jnp.asarray(cpu.actuator_joint_mask_qvel)
        max_qvel = np.full(model.nv, 100.)
        max_qvel[cpu.actuator_joint_mask_qvel] = state['actuator_joint_max_velocities']
        self.max_qvel = f32(max_qvel)
        self.imu_site, self.trunk = int(cpu.imu_site_id), int(cpu.trunk_body_id)
        self.lin_vel_adr, self.ang_vel_adr = int(cpu.imu_linear_velocity_sensor_adr), int(cpu.imu_angular_velocity_sensor_adr)
        self.foot_indices = jnp.asarray(cpu.foot_geom_indices)
        self.sphere_ids = jnp.asarray(cpu.reward_collision_sphere_geom_ids)
        radius = model.geom_size[cpu.reward_collision_sphere_geom_ids, 0]
        self.sphere_reach = f32(radius[:, None] + radius[None])
        self.nominal_collisions = int(state['nr_collisions_in_nominal'])
        self.center_height = float(state['center_height'])
        self.leg_actuators, self.spine_actuator = jnp.asarray(cpu.leg_actuator_indices), int(cpu.spine_actuator_index)
        self.body_mass = f32(model.body_mass)
        self.gravity = f32(model.opt.gravity)
        self.reward = cpu.reward_function
        self.braking = cpu.landing_braking
        self.braking_config = self.config['landing_braking']
        self.slip = cpu.landing_braking.slip
        # Degrees of freedom that move each body, for contact-point velocities.
        body_dofs = np.zeros((model.nbody, model.nv))
        for body in range(1, model.nbody):
            body_dofs[body] = body_dofs[model.body_parentid[body]]
            start = model.body_dofadr[body]
            if start >= 0:
                body_dofs[body, start:start + model.body_dofnum[body]] = 1.
        self.body_dofs, self.body_root = f32(body_dofs), jnp.asarray(model.body_rootid)
        self.tables = ContactTables(model, cpu.foot_geom_indices, cpu.landing_body_geom_mask)
        self.sensor = InertialSensor(self.config, cpu.imu_linear_acceleration_sensor_adr, self.ang_vel_adr,
                                     self.timestep, model.opt.gravity)
        window = lambda seconds: max(1, int(np.ceil(seconds / self.timestep)))
        self.acceleration_window = window(BODY_ACCELERATION_WINDOW)
        if self.acceleration_window != self.reward.body_acceleration_window_steps:
            raise ValueError('Safety and reward acceleration windows differ')
        self.force_window = window(FOOT_FORCE_WINDOW)
        if self.force_window != window(CALF_FORCE_WINDOW):
            raise ValueError('Foot and calf force windows differ')
        self.index = SimpleNamespace(**{name: jnp.asarray(getattr(cpu, name)) for name in (
            'joint_positions_obs_idx', 'joint_velocities_obs_idx', 'joint_previous_actions_obs_idx',
            'feet_ground_contact_obs_idx', 'feet_time_on_ground_obs_idx', 'feet_time_in_air_obs_idx',
            'imu_linear_vel_obs_idx', 'imu_angular_vel_obs_idx', 'goal_velocities_obs_idx', 'gravity_vector_obs_idx',
            'critic_exteroception_obs_idx', 'imu_linear_acceleration_obs_idx', 'imu_history_obs_idx',
            'policy_inertial_obs_idx')})
        # Integer thresholds equal the CPU's accumulated-time comparisons.
        steps = lambda seconds: int(round(seconds / self.dt))
        self.ready_steps = steps(self.braking.duration)
        self.success_elapsed_steps = steps(self.braking.duration + self.reward.success_stability_time)
        self.confirmation_steps = steps(self.reward.success_confirmation_time)
        self.stability_steps = steps(self.reward.success_stability_time)
        self.launch_low = steps(self.sensor.contract['launch_min_s'])
        self.launch_high = steps(self.sensor.contract['launch_max_s'])

    def data_template(self, nr_worlds):
        if nr_worlds not in self._data_templates:
            with jax.ensure_compile_time_eval():  # constants, also when first used inside jit
                self._data_templates[nr_worlds] = mjx.make_data(
                    self.model, impl='warp',
                    naconmax=max(1024, nr_worlds * self.config['mjx']['contacts_per_env']),
                    njmax=self.config['mjx']['constraints_per_env'])
        return self._data_templates[nr_worlds]

    # ---------------------------------------------------------------- physics

    def control(self, action):
        target = self.nominal + action * self.action_scale
        return jnp.clip(target, self.joint_range[:, 0], self.joint_range[:, 1]) * self.gear

    def _physics(self, data):
        return jax.vmap(lambda d: mjx.step(self.warp_model, d))(data)

    def _forward(self, data):
        return jax.vmap(lambda d: mjx.forward(self.warp_model, d))(data)

    @highest_precision
    def launch(self, keys, height, velocity, controls=None, tilt=None, rate=None):
        """Batched prepare_release: released (qpos, qvel, qacc_warmstart), IMU
        state, task state and first observation for given height and velocity.

        tilt (n, 2): roll and pitch in rad, applied like the CPU euler "xyz" start.
        rate (n, 3): body angular velocity in rad/s. The fixture applies no torque,
        so the body keeps turning at that rate during the launch."""
        n = keys.shape[0]
        split = jax.vmap(lambda k: jax.random.split(k, 2))(keys)
        if controls is None:
            controls = jax.vmap(lambda k: jax.random.randint(k, (), self.launch_low, self.launch_high + 1))(split[:, 0])
        duration, substeps = controls * self.dt, controls * self.nr_substeps
        template = self.data_template(n)
        qpos = jnp.tile(self.home_qpos, (n, 1)).at[:, 2].set(self.center_height + height)
        qpos = qpos.at[:, self.qpos_index].set(self.nominal)
        qpos = qpos.at[:, :2].add(-velocity * duration[:, None] / 2)
        tilt = jnp.zeros((n, 2)) if tilt is None else tilt
        rate = jnp.zeros((n, 3)) if rate is None else rate
        qpos = qpos.at[:, 3:7].set(tilt_quaternion(tilt))
        qvel = jnp.zeros((n, self.model.nv)).at[:, 3:6].set(rate)
        control = self.control(jnp.zeros(self.gear.shape))
        data = jax.vmap(lambda q, v: template.replace(qpos=q, qvel=v, ctrl=control))(qpos, qvel)
        data = self._forward(data)
        rotation = data.site_xmat[:, self.imu_site].reshape(n, 3, 3)
        imu = self.sensor.init(split[:, 1], rotation,
                               data.site_xpos[:, self.imu_site] - data.xpos[:, self.trunk])
        touched, overflow = jnp.zeros(n, bool), jnp.zeros((), bool)

        def substep(carry, i):
            data, imu, touched, overflow = carry
            running = i < substeps
            u = i / substeps
            phase = 2 * jnp.pi * (i + .5) / substeps
            acceleration = velocity / duration[:, None] * (1 - jnp.cos(phase))[:, None]
            fixture = velocity * (u - jnp.sin(2 * jnp.pi * u) / (2 * jnp.pi))[:, None]
            acceleration = acceleration + (fixture - data.qvel[:, :2]) / self.timestep
            acceleration = jnp.concatenate([acceleration, jnp.zeros((n, 1))], -1) - self.gravity
            force = self.body_mass[None, :, None] * acceleration[:, None]
            stepped = self._physics(data.replace(xfrc_applied=jnp.concatenate([force, jnp.zeros_like(force)], -1)))
            touched |= running & contact_metrics(stepped, self.tables, n)['floor']
            overflow |= contact_overflow(stepped)
            new_imu = self.sensor.physics_step(imu, stepped.sensordata)
            published = self.sensor.publish(new_imu)
            new_imu = select(jnp.full(n, (i + 1) % self.nr_substeps == 0), published, new_imu)
            # Worlds released earlier wait in place: qpos, qvel and warm start are frozen.
            data = stepped.replace(**{k: select(running, getattr(stepped, k), getattr(data, k))
                                      for k in ('qpos', 'qvel', 'qacc_warmstart')})
            return (data, select(running, new_imu, imu), touched, overflow), None

        (data, imu, touched, overflow), _ = jax.lax.scan(
            substep, (data, imu, touched, overflow), jnp.arange(self.launch_high * self.nr_substeps))
        data = self._forward(data.replace(xfrc_applied=jnp.zeros_like(data.xfrc_applied), time=jnp.zeros(n)))
        task = self._initial_task(data, height, velocity, duration)
        task['roll_deg'], task['pitch_deg'] = jnp.rad2deg(tilt[:, 0]), jnp.rad2deg(tilt[:, 1])
        task['rate_dps'] = jnp.rad2deg(jnp.linalg.norm(rate, axis=-1))
        release = dict(qpos=data.qpos, qvel=data.qvel, qacc_warmstart=data.qacc_warmstart)
        observation = self._observation(data, imu, task, jnp.zeros((n,) + self.gear.shape),
                                        jnp.zeros((n, self.tables.nr_feet), bool))
        return dict(release=release, imu=imu, task=task, observation=observation,
                    invalid=touched.any() | overflow)

    @highest_precision
    def sample_disturbance(self, key, size, scale=1.):
        """Roll/pitch (size, 2) and body angular rate (size, 3) of an imperfect release."""
        tilt_key, direction_key, magnitude_key = jax.random.split(key, 3)
        tilt = jax.random.uniform(tilt_key, (size, 2), minval=-1., maxval=1.) * self.max_tilt * scale
        direction = jax.random.normal(direction_key, (size, 3))
        direction = direction / jnp.maximum(jnp.linalg.norm(direction, axis=-1, keepdims=True), 1e-9)
        magnitude = jax.random.uniform(magnitude_key, (size, 1)) * self.max_rate * scale
        return tilt, direction * magnitude

    @highest_precision
    def sample_pool(self, key, size, scale=1.):
        """RealDriveMix conditions followed by the launch fixture."""
        keys = jax.random.split(key, 5)
        height = jnp.asarray(HEIGHTS)[jax.random.randint(keys[0], (size,), 0, len(HEIGHTS))]
        group = jnp.searchsorted(jnp.asarray(GROUP_EDGES), jax.random.uniform(keys[1], (size,)), side='right')
        low, high = jnp.asarray(GROUP_LOW)[group], jnp.asarray(GROUP_HIGH)[group]
        speed = low + (high - low) * jax.random.uniform(keys[2], (size,))
        azimuth = jax.random.uniform(keys[3], (size,), maxval=2 * jnp.pi)
        velocity = speed[:, None] * jnp.stack([jnp.cos(azimuth), jnp.sin(azimuth)], -1)
        tilt, rate = self.sample_disturbance(jax.random.fold_in(key, 7), size, scale)
        pool = self.launch(jax.random.split(keys[4], size), height, velocity, tilt=tilt, rate=rate)
        pool['task']['group'] = group
        return pool

    def _initial_task(self, data, height, velocity, duration):
        n = height.shape[0]
        zeros, false, count = jnp.zeros(n), jnp.zeros(n, bool), jnp.zeros(n, jnp.int32)
        return dict(
            step=count, group=count, height=height, speed=jnp.linalg.norm(velocity, axis=-1),
            launch_duration=duration, release_speed=jnp.linalg.norm(data.qvel[:, :2], axis=-1),
            episode_return=zeros,
            # touchdown and braking
            touched=false, touch_step=count, stable=count, reference=data.qvel[:, :2],
            slip_load=jnp.zeros((n, self.tables.nr_supports)), slip_integral=zeros,
            tracking_error=zeros, tracked_time=zeros,
            feet_on_ground=jnp.zeros((n, self.tables.nr_feet)), feet_in_air=jnp.zeros((n, self.tables.nr_feet)),
            # safety monitor
            contact_started=false, foot_contact=false, forbidden=false, min_height=data.qpos[:, 2],
            foot_history=jnp.zeros((n, self.force_window, self.tables.nr_feet)),
            calf_history=jnp.zeros((n, self.force_window, self.tables.nr_calves)),
            peak_foot=zeros, peak_calf=zeros,
            acceleration_history=jnp.zeros((n, self.acceleration_window)),
            peak_acceleration=zeros, peak_acceleration_raw=zeros, peak_impact=zeros,
            impact_window=count, base_crash=false, hard_limit=false,
            peak_leg=zeros, peak_spine=zeros, leg_tau_squared=zeros, spine_tau_squared=zeros)

    def _state(self, pool, indices):
        n = indices.shape[0]
        chosen = jax.tree.map(lambda x: x[indices], {k: pool[k] for k in ('release', 'imu', 'task', 'observation')})
        template = self.data_template(n)
        control = self.control(jnp.zeros(self.gear.shape))
        data = jax.vmap(lambda r: template.replace(ctrl=control, **r))(chosen['release'])
        data = self._forward(data)
        zeros, false = jnp.zeros(n), jnp.zeros(n, bool)
        info = self._info(chosen['task'], chosen['imu'], data, dict.fromkeys(self.REWARDS, zeros), false,
                          jnp.full(n, -1), zeros)
        info['episode_return'], info['episode_length'] = zeros, zeros
        return State(data=data, imu=chosen['imu'], task=chosen['task'], observation=chosen['observation'],
                     actual_observation=chosen['observation'], reward=zeros, terminated=false, truncated=false,
                     info=info, overflow=jnp.zeros((), bool))

    @highest_precision
    def reset(self, pool, nr_worlds):
        return self._state(pool, jnp.arange(nr_worlds))

    # ------------------------------------------------------------------- step

    def _substep_task(self, task, data, contacts, totals):
        """LandingSafetyMonitor.update and _body_acceleration_reward for one physics step."""
        r = self.reward
        acceleration = jnp.linalg.norm(data.qacc[:, :3], axis=-1)
        started = task['contact_started'] | contacts['floor']
        push = lambda history, value: jnp.concatenate([history[:, 1:], value[:, None]], 1)
        foot_history = push(task['foot_history'], contacts['feet_force'])
        calf_history = push(task['calf_history'], contacts['calf_force'])
        history = jnp.where(started[:, None], push(task['acceleration_history'], acceleration), task['acceleration_history'])
        average = jnp.where(started, history.mean(-1), 0.)
        window = jnp.where(started & contacts['body'], self.acceleration_window, task['impact_window'])
        impact = started & (window > 0)
        excess = jnp.maximum(0., average / r.body_acceleration_limit - 1.)
        totals = dict(
            body_acceleration=totals['body_acceleration'] - r.body_acceleration_coeff * excess ** 2 * self.timestep,
            floor=totals['floor'] | contacts['floor'], body=totals['body'] + contacts['body'])
        task = {**task, 'contact_started': started,
                'foot_contact': task['foot_contact'] | (contacts['feet_force'].sum(-1) > 0),
                'forbidden': task['forbidden'] | contacts['forbidden'],
                'min_height': jnp.minimum(task['min_height'], data.qpos[:, 2]),
                'foot_history': foot_history, 'calf_history': calf_history,
                'peak_foot': jnp.maximum(task['peak_foot'], foot_history.mean(1).max(-1)),
                'peak_calf': jnp.maximum(task['peak_calf'], calf_history.mean(1).max(-1)),
                'acceleration_history': history,
                'peak_acceleration': jnp.maximum(task['peak_acceleration'], average),
                'peak_acceleration_raw': jnp.maximum(task['peak_acceleration_raw'], jnp.where(started, acceleration, 0.)),
                'peak_impact': jnp.maximum(task['peak_impact'], jnp.where(impact, average, 0.)),
                'base_crash': task['base_crash'] | (impact & (average > r.base_impact_acceleration_limit)),
                'impact_window': jnp.where(impact, window - 1, window)}
        return task, totals

    def _reference(self, task, elapsed_steps):
        """LandingBraking.reference_world after touchdown."""
        u = jnp.clip(elapsed_steps * self.dt / self.braking.duration, 0., 1.)
        speed = jnp.linalg.norm(task['reference'], axis=-1)
        direction = jnp.where((speed >= .2)[:, None], task['reference'] / jnp.maximum(speed, 1e-6)[:, None],
                              jnp.asarray([1., 0.]))
        final = self.braking.final_speed
        magnitude = final + (jnp.maximum(speed, final) - final) * (1 - 3 * u ** 2 + 2 * u ** 3)
        return direction * magnitude[:, None]

    def _loaded_slip(self, task, data, elapsed_steps):
        """LandingSlip.step: contact-point slip of the allowed supports."""
        n, size, slip = elapsed_steps.shape[0], self.tables.nr_supports, self.slip
        active, world, geom, force = active_contacts(data, self.tables)
        slot = self.tables.support_slot[geom]
        selected = active & (slot >= 0) & (force > 0)
        body = self.tables.geom_body[geom]
        position = data._impl.contact__pos
        velocity = self._point_velocity(data, world, body, position)
        normal = data._impl.contact__frame[:, 0]
        tangent = velocity - normal * (velocity * normal).sum(-1, keepdims=True)
        index = world * size + jnp.maximum(slot, 0)
        load = jnp.where(selected, force, 0.)
        forces = jax.ops.segment_sum(load, index, n * size).reshape(n, size)
        weighted = jax.ops.segment_sum(load * (tangent ** 2).sum(-1), index, n * size).reshape(n, size)
        alpha = 1. if slip.tau == 0 else -np.expm1(-self.dt / slip.tau)
        filtered = task['slip_load'] + alpha * (forces - task['slip_load'])
        speed2 = jnp.where(forces > 0, weighted / jnp.maximum(forces, 1e-9), 0.)
        weights = jnp.clip((filtered - slip.threshold) / (slip.full_load - slip.threshold), 0, 1) * (forces > 0)
        value = (weights * speed2).sum(-1) / self.tables.nr_foot_geoms
        elapsed = elapsed_steps * self.dt
        ramp = (elapsed >= slip.delay) * 1. if slip.ramp == 0 else jnp.clip((elapsed - slip.delay) / slip.ramp, 0, 1)
        return filtered, value, slip.weight * self.dt * ramp * value

    def _point_velocity(self, data, world, body, position):
        """mj_jac(point) @ qvel with the kinematics of the last forward pass.

        Like the CPU reward, it pairs the pre-integration pose and Jacobian with
        the current (integrated and clipped) joint velocities.
        """
        # cdof is expressed about the root subtree center of mass.
        offset = position - data.subtree_com[world, self.body_root[body]]
        cdof = data.cdof[world]                                  # (contacts, nv, 6)
        active = self.body_dofs[body]                            # (contacts, nv)
        qvel = data.qvel[world] * active
        twist = jnp.einsum('cv,cvk->ck', qvel, cdof)
        return twist[:, 3:] + jnp.cross(twist[:, :3], offset)

    REWARDS = ('alive', 'braking_tracking', 'loaded_slip', 'collision', 'body_acceleration', 'body_support',
               'landing_event', 'total')

    def _reward(self, task, data, totals):
        r, b = self.reward, self.braking
        step = task['step']
        contact = totals['floor']
        first_touch = contact & ~task['touched']
        touched = task['touched'] | contact
        touch_step = jnp.where(first_touch, step, task['touch_step'])
        elapsed_steps = jnp.where(touched, step - touch_step, 0)
        time_steps = jnp.where(touched, elapsed_steps + 1, 0)

        motor = data.actuator_force
        leg, spine = jnp.abs(motor[:, self.leg_actuators]), jnp.abs(motor[:, self.spine_actuator])
        utilization = jnp.maximum(leg.max(-1) / r.leg_motor_torque_limit, spine / r.spine_motor_torque_limit)
        hard_limit = task['hard_limit'] | (utilization > 1. + r.hard_limit_tolerance)

        # Braking
        velocity = data.qvel[:, :2]
        reference = self._reference(task, elapsed_steps)
        error = ((velocity - reference) ** 2).sum(-1)
        tracking = jnp.where(touched, b.weight * self.dt * jnp.exp(-error / b.sigma ** 2), 0.)
        filtered, slip_value, slip_cost = self._loaded_slip(task, data, elapsed_steps)
        slip_cost = jnp.where(touched, slip_cost, 0.)
        slip_value = jnp.where(touched, slip_value, 0.)
        ready = touched & (elapsed_steps >= self.ready_steps)

        angular_speed = jnp.linalg.norm(data.sensordata[:, self.ang_vel_adr:self.ang_vel_adr + 3], axis=-1)
        blocked = ((jnp.linalg.norm(velocity, axis=-1) > self.braking_config['success_max_horizontal_speed'])
                   | (jnp.abs(data.qvel[:, 2]) > .3) | (angular_speed > r.success_max_angular_velocity)
                   | ~contact | (totals['body'] > 0))
        stable = jnp.where(touched & ~blocked & ready, task['stable'] + 1, 0)

        # Self collision of the reward spheres
        center = data.geom_xpos[:, self.sphere_ids]
        distance = jnp.linalg.norm(center[:, :, None] - center[:, None], axis=-1)
        count = ((distance <= self.sphere_reach).sum((1, 2)) - len(self.sphere_ids)) // 2
        collision = -r.self_collision_coeff * jnp.maximum(count - self.nominal_collisions, 0)

        support_time = jnp.where(touched, jnp.clip(time_steps * self.dt - r.body_support_grace_time, 0., self.dt), 0.)
        support = -r.body_support_penalty_coeff * support_time * totals['body'] / self.nr_substeps

        # Outcome, in the CPU priority order
        violations = landing_violations(task['foot_contact'], task['forbidden'], task['min_height'], task['peak_foot'],
                                        task['peak_calf'], task['peak_acceleration'], xp=jnp)
        success = (touched & (time_steps >= self.confirmation_steps) & (elapsed_steps >= self.success_elapsed_steps)
                   & task['foot_contact'] & (stable >= self.stability_steps))
        timeout = step >= self.horizon - 1
        candidates = jnp.stack([hard_limit, task['base_crash'], violations['disallowed_contact'], violations['calf_force'],
                                violations['foot_force'], violations['body_acceleration'], violations['base_height'],
                                timeout & ~success], -1)
        failed = candidates.any(-1)
        reason = jnp.where(failed, jnp.argmax(candidates, -1), -1)
        # A safety violation wins over success in the same step.
        success &= ~candidates[:, :-1].any(-1)
        evaluated = success | failed
        event = jnp.where(success, r.landing_success_coeff, jnp.where(failed, -r.landing_failure_coeff, 0.))

        terms = dict(alive=jnp.full_like(event, r.alive_coeff), braking_tracking=tracking, loaded_slip=-slip_cost,
                     collision=collision * 1., body_acceleration=totals['body_acceleration'], body_support=support,
                     landing_event=event)
        terms['total'] = jnp.nan_to_num(sum(terms.values()), nan=0., posinf=0., neginf=0.)
        task = {**task, 'touched': touched, 'touch_step': touch_step, 'stable': stable, 'hard_limit': hard_limit,
                'slip_load': jnp.where(touched[:, None], filtered, task['slip_load']),
                'slip_integral': task['slip_integral'] + self.dt * slip_value,
                'tracking_error': task['tracking_error'] + jnp.where(touched, error * self.dt, 0.),
                'tracked_time': task['tracked_time'] + jnp.where(touched, self.dt, 0.),
                'peak_leg': jnp.maximum(task['peak_leg'], leg.max(-1)), 'peak_spine': jnp.maximum(task['peak_spine'], spine),
                'leg_tau_squared': task['leg_tau_squared'] + (leg ** 2).sum(-1) * self.dt,
                'spine_tau_squared': task['spine_tau_squared'] + spine ** 2 * self.dt}
        return task, terms, success, evaluated, reason, slip_value

    def _observation(self, data, imu, task, action, feet_contact):
        x, n = self.index, action.shape[0]
        history, acceleration, gyro, heading = self.sensor.observe(imu)
        obs = jnp.zeros((n,) + self.single_observation_space.shape)
        rotation = data.site_xmat[:, self.imu_site].reshape(n, 3, 3)
        put = lambda obs, index, value: obs.at[:, index].set(value)
        obs = put(obs, x.joint_positions_obs_idx, (data.qpos[:, self.qpos_index] - self.nominal) / 3.14)
        obs = put(obs, x.joint_velocities_obs_idx, data.qvel[:, self.qvel_index] / 100.)
        obs = put(obs, x.joint_previous_actions_obs_idx, action / 10.)
        obs = put(obs, x.feet_ground_contact_obs_idx, feet_contact / .5 - 1.)
        obs = put(obs, x.feet_time_on_ground_obs_idx, jnp.clip(task['feet_on_ground'] / 2.5 - 1., -1., 1.))
        obs = put(obs, x.feet_time_in_air_obs_idx, jnp.clip(task['feet_in_air'] / 2.5 - 1., -1., 1.))
        obs = put(obs, x.imu_linear_vel_obs_idx,
                  jnp.clip(data.sensordata[:, self.lin_vel_adr:self.lin_vel_adr + 3] / 10., -1., 1.))
        obs = put(obs, x.imu_angular_vel_obs_idx, jnp.clip(gyro / 50., -1., 1.))
        obs = put(obs, x.gravity_vector_obs_idx, -rotation[:, 2])  # inverse rotation of (0, 0, -1)
        height = data.site_xpos[:, self.imu_site, 2] - self.center_height
        obs = put(obs, x.critic_exteroception_obs_idx, jnp.clip(height[:, None] / 5. - 1., -1., 1.))
        obs = put(obs, x.imu_linear_acceleration_obs_idx, acceleration / 98.1)
        obs = put(obs, x.imu_history_obs_idx, history.reshape(n, -1))
        obs = put(obs, x.policy_inertial_obs_idx, heading)
        return jnp.clip(jnp.nan_to_num(obs, nan=0., posinf=0., neginf=0.), -10., 10.)

    def _info(self, task, imu, data, terms, success, reason, slip_value):
        estimate = imu['delta_v'][:, :2]
        info = {f'reward/{k}': terms[k] for k in self.REWARDS}
        info.update({
            'curriculum/landing_success': success * 1., 'curriculum/v22_group': task['group'] * 1.,
            'curriculum/v22_height': task['height'], 'curriculum/v22_speed': task['speed'],
            'curriculum/initial_height': task['height'], 'curriculum/initial_speed_xy': task['release_speed'],
            'curriculum/initial_roll_deg': task['roll_deg'], 'curriculum/initial_pitch_deg': task['pitch_deg'],
            'curriculum/initial_rate_dps': task['rate_dps'],
            'inertial/launch_duration_s': task['launch_duration'], 'inertial/requested_speed_xy': task['speed'],
            'inertial/release_speed_xy': task['release_speed'],
            'metrics/inertial_velocity_error': jnp.linalg.norm(estimate - data.qvel[:, :2], axis=-1),
            'metrics/inertial_estimated_speed': jnp.linalg.norm(estimate, axis=-1),
            'metrics/inertial_clipped_fraction': imu['clipped'] / jnp.maximum(imu['samples'], 1),
            'metrics/peak_body_acceleration': task['peak_acceleration'],
            'metrics/peak_body_acceleration_raw': task['peak_acceleration_raw'],
            'metrics/peak_base_impact_acceleration': task['peak_impact'],
            'metrics/peak_leg_motor_torque': task['peak_leg'], 'metrics/peak_spine_motor_torque': task['peak_spine'],
            'metrics/leg_tau_squared_integral': task['leg_tau_squared'],
            'metrics/spine_tau_squared_integral': task['spine_tau_squared'],
            'metrics/peak_foot_force_10ms': task['peak_foot'], 'metrics/peak_calf_force_10ms': task['peak_calf'],
            'metrics/safety_min_base_height': task['min_height'],
            'metrics/landing_stable_time': task['stable'] * self.dt,
            'metrics/braking_tracking_mse': task['tracking_error'] / jnp.maximum(task['tracked_time'], self.dt),
            'metrics/loaded_slip_speed2': slip_value, 'metrics/loaded_slip_integral': task['slip_integral']})
        for i, name in enumerate(FAILURES):
            key = 'outcome/unresolved' if name == 'unresolved' else f'outcome/failure_{name}'
            info[key] = (reason == i) * 1.
        return info

    @highest_precision
    def step(self, state, action, pool, next_index):
        """Returns (state, next pool index). Finished worlds restart from the pool.

        The terminal observation and info stay in actual_observation and info.
        """
        n = action.shape[0]
        data = state.data.replace(ctrl=jax.vmap(self.control)(action))
        totals = dict(body_acceleration=jnp.zeros(n), floor=jnp.zeros(n, bool), body=jnp.zeros(n, jnp.int32))

        def substep(carry, _):
            data, imu, task, totals, overflow, _ = carry
            data = self._physics(data)
            imu = self.sensor.physics_step(imu, data.sensordata)
            contacts = contact_metrics(data, self.tables, n)
            task, totals = self._substep_task(task, data, contacts, totals)
            return (data, imu, task, totals, overflow | contact_overflow(data), contacts['feet_contact']), None

        carry = (data, state.imu, state.task, totals, state.overflow, jnp.zeros((n, self.tables.nr_feet), bool))
        (data, imu, task, totals, overflow, feet_contact), _ = jax.lax.scan(substep, carry, None, self.nr_substeps)
        data = data.replace(qvel=jnp.clip(data.qvel, -self.max_qvel, self.max_qvel))
        imu = self.sensor.publish(imu)
        task, terms, success, evaluated, reason, slip_value = self._reward(task, data, totals)
        # The braking goal follows the true velocity until the first contact.
        task['reference'] = jnp.where(task['touched'][:, None], task['reference'], data.qvel[:, :2])
        observation = self._observation(data, imu, task, action, feet_contact)
        terminated = evaluated | (jnp.abs(data.qvel[:, :3]) == 100.).any(-1)
        truncated = task['step'] >= self.horizon - 1
        done = terminated | truncated
        task = {**task, 'step': task['step'] + 1, 'episode_return': task['episode_return'] + terms['total'],
                'feet_on_ground': jnp.where(feet_contact, task['feet_on_ground'] + self.dt, 0.),
                'feet_in_air': jnp.where(feet_contact, 0., task['feet_in_air'] + self.dt)}
        info = self._info(task, imu, data, terms, success, reason, slip_value)
        info['episode_return'], info['episode_length'] = task['episode_return'], task['step'] * 1.

        # Pool entries are handed out in order; a world never repeats one early.
        size = pool['observation'].shape[0]
        indices = (next_index + jnp.cumsum(done) - 1) % size
        fresh = jax.tree.map(lambda x: x[indices], {k: pool[k] for k in ('release', 'imu', 'task', 'observation')})
        data = data.replace(**{k: select(done, v, getattr(data, k)) for k, v in fresh['release'].items()})
        state = State(data=data, imu=select(done, fresh['imu'], imu), task=select(done, fresh['task'], task),
                      observation=select(done, fresh['observation'], observation), actual_observation=observation,
                      reward=terms['total'], terminated=terminated, truncated=truncated, info=info, overflow=overflow)
        return state, next_index + done.sum()

    def close(self):
        pass
