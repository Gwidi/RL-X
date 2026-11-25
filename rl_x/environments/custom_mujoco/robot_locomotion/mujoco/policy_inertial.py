"""Opt-in IMU history and dead reckoning from an observed launch at rest.

The estimator consumes accelerometer/gyro samples, never qvel or qacc.  The
reset fixture applies forces for a finite time before releasing the robot.
All sensor state lives in internal_state so evaluation deepcopy remains safe.
"""
from collections import deque
import hashlib
import json

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .policy_velocity import heading_velocity_xy


DEFAULTS = dict(
    enabled=False, schema="imu_history_delta_v_xy_v1", history_steps=50,
    launch_min_s=.16, launch_max_s=.32,
    acceleration_noise_std=.2, acceleration_bias_std=.03,
    gyro_noise_std=.01, gyro_bias_std=.001,
    acceleration_clip=156.96, gyro_clip=35., delay_steps=1,
    delta_v_scale_mps=4.,
)


def inertial_observation_contract(config):
    if not config.get("enabled", False):
        return {"enabled": False}
    result = {**DEFAULTS, **dict(config)}
    if result["schema"] != DEFAULTS["schema"]:
        raise ValueError("Unsupported inertial observation schema")
    for name, lower in (("history_steps", 1), ("delay_steps", 0)):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < lower:
            raise ValueError(f"policy_inertial.{name} must be an integer >= {lower}")
    for name in ("launch_min_s", "launch_max_s", "acceleration_clip", "gyro_clip", "delta_v_scale_mps"):
        if not np.isfinite(result[name]) or result[name] <= 0:
            raise ValueError(f"policy_inertial.{name} must be positive and finite")
    for name in ("acceleration_noise_std", "acceleration_bias_std", "gyro_noise_std", "gyro_bias_std"):
        if not np.isfinite(result[name]) or result[name] < 0:
            raise ValueError(f"policy_inertial.{name} must be nonnegative and finite")
    if result["launch_max_s"] < result["launch_min_s"]:
        raise ValueError("Inertial launch_max_s must be >= launch_min_s")
    return result


def check_inertial_contract(saved, current):
    a = inertial_observation_contract(saved.get("policy_inertial", {}))
    b = inertial_observation_contract(current.get("policy_inertial", {}))
    if a != b:
        raise ValueError("Checkpoint inertial observation/launch contract differs; use its native profile or train fresh")


def cross3(a, b):
    """np.cross for two 3-vectors without its axis handling overhead.

    Uses the same products and subtractions in the same order as np.cross,
    so results are bit-identical.
    """
    a0, a1, a2 = a.tolist()
    b0, b1, b2 = b.tolist()
    return np.array([a1*b2 - a2*b1, a2*b0 - a0*b2, a0*b1 - a1*b0])


def rotation_increment(rotvec):
    """Rodrigues formula without a scipy object per high-rate IMU sample."""
    x, y, z = rotvec
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-8:
        return np.eye(3) + skew + .5 * skew @ skew
    return np.eye(3) + np.sin(angle) / angle * skew + (1-np.cos(angle)) / angle**2 * (skew @ skew)


class PolicyInertialObservation:
    def __init__(self, env):
        self.env = env
        self.contract = inertial_observation_contract(env.env_config.get("policy_inertial", {}))
        self.enabled = self.contract["enabled"]
        self.dimension = 2 if self.enabled else 0
        if not self.enabled:
            return
        if env.policy_velocity_observation.dimension:
            raise ValueError("Inertial actor cannot also receive policy_velocity")
        if env.imu_history_length != self.contract["history_steps"]:
            raise ValueError("imu_history_length must equal policy_inertial.history_steps")
        braking = env.env_config.get("landing_braking", {})
        if braking.get("enabled") and (braking.get("reference_source", "privileged_truth") != "privileged_truth"
                                       or braking.get("goal_observation", "masked") != "masked"):
            raise ValueError("Inertial braking needs a privileged reward reference and masked actor goal")
        if env.env_config.get("landing_frontflip", {}).get("enabled"):
            raise ValueError("Inertial launch is a landing experiment, not the frontflip task")

    def state(self):
        return self.env.internal_state["policy_inertial_sensor"]

    def _seeded_rng(self):
        blob = json.dumps(self.env.np_rng.bit_generator.state, sort_keys=True,
                          default=lambda x: np.asarray(x).tolist()).encode()
        return np.random.default_rng(int.from_bytes(hashlib.sha256(b"inertial-v1" + blob).digest()[:16], "little"))

    def prepare_release(self, requested_velocity):
        """Ideal distributed launch fixture; policy starts at release, with history.

        Mass-proportional forces supply horizontal acceleration and support
        against gravity to every body. This intentionally avoids an unmodelled
        fixture torque. PD holds home, residual is zero, no policy decisions or
        rewards occur on the fixture. Flight keeps the usual 5-second horizon.
        """
        env = self.env
        model, data = env.internal_state["mj_model"], env.internal_state["data"]
        target = np.asarray(requested_velocity, dtype=float)
        if (target.shape != (model.nv,) or not np.all(np.isfinite(target)) or abs(target[2]) > 1e-10
                or np.any(np.abs(target[6:]) > 1e-10)):
            raise ValueError("Observed inertial launch supports horizontal velocity and body spin only, with zero joint velocity")
        rng = self._seeded_rng()
        low = int(round(self.contract["launch_min_s"] / env.dt))
        high = int(round(self.contract["launch_max_s"] / env.dt))
        if low < 1 or any(abs(self.contract[k] / env.dt-round(self.contract[k] / env.dt)) > 1e-8
                          for k in ("launch_min_s", "launch_max_s")):
            raise ValueError("Inertial launch durations must be positive multiples of the control interval")
        controls = int(rng.integers(low, high+1))
        duration = controls * env.dt
        data.qvel[:] = 0.  # Known rest, never jump from rest directly to target velocity.
        data.qvel[3:6] = target[3:6]  # Release spin: the fixture applies no torque, so it persists.
        data.qpos[:2] -= target[:2] * duration / 2
        data.ctrl[:] = env.control_function.process_action(np.zeros(env.nr_actuator_joints))
        mujoco.mj_forward(model, data)
        rotation = data.site_xmat[env.imu_site_id].reshape(3, 3).copy()
        offset_site = rotation.T @ (data.site_xpos[env.imu_site_id] - data.xpos[env.trunk_body_id])
        env.internal_state["policy_inertial_sensor"] = dict(
            rng=rng, acceleration_bias=rng.normal(0., self.contract["acceleration_bias_std"], 3),
            gyro_bias=rng.normal(0., self.contract["gyro_bias_std"], 3),
            rotation=rotation, offset_site=offset_site, previous_gyro=None,
            delta_v_world=np.zeros(3), history=np.zeros((env.imu_history_length, 6)),
            snapshots=deque(maxlen=self.contract["delay_steps"]+1),
            acceleration=np.zeros(3), gyro=np.zeros(3), samples=0, clipped_samples=0,
        )
        substeps = controls * env.nr_substeps
        for i in range(substeps):
            # The fixture follows a smooth velocity trajectory. Its external
            # servo compensates drag with forces, not a qvel correction. Only
            # the fixture can use this feedback; the actor/estimator cannot.
            acceleration = np.r_[target[:2] / duration, 0.] * (1-np.cos(2*np.pi*(i+.5)/substeps))
            u = i/substeps
            fixture_velocity = target[:2] * (u-np.sin(2*np.pi*u)/(2*np.pi))
            acceleration[:2] += (fixture_velocity-data.qvel[:2]) / model.opt.timestep
            data.xfrc_applied[:, :3] = model.body_mass[:, None] * (acceleration-model.opt.gravity)
            data.xfrc_applied[:, 3:] = 0.
            mujoco.mj_step(model, data)
            self.physics_step()
            if (i+1) % env.nr_substeps == 0:
                self.publish()
        if data.ncon:
            raise ValueError("Inertial launch fixture must not touch terrain")
        data.xfrc_applied[:] = 0.
        data.qfrc_applied[:] = 0.
        data.time = 0.  # Only flight and postcontact stabilization count toward horizon.
        mujoco.mj_forward(model, data)
        info = env.internal_state["info"]
        info.update({"inertial/launch_duration_s": duration,
                     "inertial/requested_speed_xy": float(np.linalg.norm(target[:2])),
                     "inertial/release_speed_xy": float(np.linalg.norm(data.qvel[:2])),
                     "inertial/release_vx": float(data.qvel[0]),
                     "inertial/release_vy": float(data.qvel[1]),
                     "inertial/release_angular_speed": float(np.linalg.norm(data.qvel[3:6]))})
        # The force-driven fixture can generate a small real tilt/spin. Log
        # release state rather than labelling the sampled rest pose as release.
        angles = Rotation.from_quat(data.qpos[3:7], scalar_first=True).as_euler('xyz', degrees=True)
        for axis, value in zip(('roll', 'pitch', 'yaw'), angles):
            info['inertial/release_' + axis + '_deg'] = float(value)
            info['curriculum/initial_' + axis + '_deg'] = float(value)
        info['curriculum/initial_base_height'] = float(data.qpos[2])
        # Keep the existing reset metrics truthful about the physical release.
        for name, value in (("vx", data.qvel[0]), ("vy", data.qvel[1]),
                            ("speed_xy", np.linalg.norm(data.qvel[:2]))):
            info["curriculum/initial_" + name] = float(value)

    def physics_step(self):
        env = self.env
        state = self.state()
        data = env.internal_state["data"]
        dt = float(env.internal_state["mj_model"].opt.timestep)
        rng = state["rng"]
        acc = data.sensordata[env.imu_linear_acceleration_sensor_adr:env.imu_linear_acceleration_sensor_adr+3].copy()
        gyro = data.sensordata[env.imu_angular_velocity_sensor_adr:env.imu_angular_velocity_sensor_adr+3].copy()
        acc += state["acceleration_bias"] + rng.normal(0., self.contract["acceleration_noise_std"], 3)
        gyro += state["gyro_bias"] + rng.normal(0., self.contract["gyro_noise_std"], 3)
        if np.any(np.abs(acc) > self.contract["acceleration_clip"]) or np.any(np.abs(gyro) > self.contract["gyro_clip"]):
            state["clipped_samples"] += 1
        acc = np.clip(acc, -self.contract["acceleration_clip"], self.contract["acceleration_clip"])
        gyro = np.clip(gyro, -self.contract["gyro_clip"], self.contract["gyro_clip"])
        previous = state["previous_gyro"]
        alpha = np.zeros(3) if previous is None else (gyro-previous)/dt
        # IMU is offset from the base origin. Remove rigid-body rotational
        # acceleration using gyro differences, not simulator qacc/angular state.
        r = state["offset_site"]
        base_specific_force = acc - cross3(alpha, r) - cross3(gyro, cross3(gyro, r))
        state["delta_v_world"] += (state["rotation"] @ base_specific_force + env.internal_state["mj_model"].opt.gravity) * dt
        state["rotation"] = state["rotation"] @ rotation_increment(gyro*dt)
        state.update(previous_gyro=gyro, acceleration=acc, gyro=gyro, samples=state["samples"]+1)

    def publish(self):
        """One control-rate sample and one delayed snapshot; never on read."""
        state = self.state()
        history = state["history"]
        history[:-1] = history[1:]
        history[-1] = np.r_[state["acceleration"]/98.1, state["gyro"]/50.]
        state["snapshots"].append(dict(
            history=history.copy(), acceleration=state["acceleration"].copy(),
            gyro=state["gyro"].copy(),
            delta_v=heading_velocity_xy(state["delta_v_world"], state["rotation"]) / self.contract["delta_v_scale_mps"],
        ))

    def observe(self):
        if not self.enabled:
            return np.empty(0)
        return self.state()["snapshots"][0]["delta_v"].copy()

    def snapshot(self):
        return self.state()["snapshots"][0]

    def diagnostics(self):
        state = self.state()
        # Truth is used here only for logging, never to reset/correct the estimate.
        actual = self.env.internal_state["data"].qvel[:2]
        return {"metrics/inertial_velocity_error": float(np.linalg.norm(state["delta_v_world"][:2]-actual)),
                "metrics/inertial_estimated_speed": float(np.linalg.norm(state["delta_v_world"][:2])),
                "metrics/inertial_clipped_fraction": state["clipped_samples"]/max(1, state["samples"])}
