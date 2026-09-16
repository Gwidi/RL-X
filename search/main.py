import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

    # self.nominal_joint_positions = np.array([
    #     -0.1, 0.8, -1.5,
    #     0.1, -0.8, 1.5,
    #     -0.1, -1.0, 1.5,
    #     0.1, 1.0, -1.5,
    #     0.0
    # ])
# Joint targets are expressed in joint-side radians, like JointCommand.t_pos.
NOMINAL_POSE = {
    "rl_j0": -0.1, "rl_j1": -0.8, "rl_j2": 1.5,
    "rr_j0": 0.1, "rr_j1": 0.8, "rr_j2": -1.5,
    "fr_j0": -0.1, "fr_j1": 1.0, "fr_j2": -1.5,
    "fl_j0": 0.1, "fl_j1": -1.0, "fl_j2": 1.5,
    "sp_j0": 0.0,
}

LEG_JOINTS = {
    "hip": ["rl_j0", "rr_j0", "fr_j0", "fl_j0"],
    "thigh": ["rl_j1", "rr_j1", "fr_j1", "fl_j1"],
    "calf": ["rl_j2", "rr_j2", "fr_j2", "fl_j2"],
}

START_HEIGHT = 1.5
# Same dead zone as STATIC_FRICTION in simulation/src/joint_control.cpp.
STATIC_FRICTION = 0.37

# Screen structural load on each leg separately.  The total force across all
# feet is retained as a diagnostic, but does not decide safety.
MAX_LANDING_FOOT_FORCE = 3_000.0               # 10 ms average per foot, N
MAX_LANDING_BODY_ACCELERATION = 10.0 * 9.81   # 50 ms average, m/s^2 (10 g)
FOOT_FORCE_WINDOW = 0.010                      # seconds
BODY_ACCELERATION_WINDOW = 0.050               # seconds
LANDING_QUALITY_PENALTY = 1_000_000.0
NOMINAL_POSITION_REGULARIZATION = 2.0

SIM_DURATION = 2.0
XML_PATH = Path(__file__).resolve().with_name("intention.xml")
OUTPUT_DIR = Path(__file__).resolve().with_name("output")
LEG_GAIN_PARAMETER_NAMES = (
    "kp_hip", "kp_thigh", "kp_calf",
    "kd_hip", "kd_thigh", "kd_calf",
)
LEG_GAIN_BOUNDS = np.array([
    (5.0, 60.0),
    (5.0, 100.0),
    (5.0, 100.0),
    (0.2, 5.0),
    (0.2, 8.0),
    (0.2, 8.0),
], dtype=float)
SPINE_GAIN_BOUNDS = np.array([
    (5.0, 100.0),  # kp_spine
    (0.2, 8.0),    # kd_spine
], dtype=float)
# Nominal leg positions are optimized as offsets from NOMINAL_POSE. Keeping
# the bounds relative makes the flag safe to use if the nominal pose changes.
LEG_NOMINAL_POSITION_NAMES = tuple(
    name for group in ("hip", "thigh", "calf") for name in LEG_JOINTS[group]
)
INDIVIDUAL_GAIN_NAMES = tuple(
    f"{gain}_{name}"
    for gain in ("kp", "kd")
    for name in LEG_NOMINAL_POSITION_NAMES
)
NOMINAL_POSITION_DELTA_BOUNDS = (-0.5, 0.5)
MAX_GP_POINTS = 400
_WORKER_MODEL = None
_WORKER_DATA = None
_WORKER_JMAP = None
_WORKER_CONTACT_MAP = None
_WORKER_STEPS = None
_WORKER_START_HEIGHT = None
_WORKER_OPTIMIZE_NOMINAL_POSITION = False
_WORKER_INDIVIDUAL_GAINS = False


def configure_spine(model, lock_spine):
    """Optionally constrain the spine at its nominal zero position."""
    if not lock_spine:
        return
    spine_joint = model.joint("sp_j0")
    model.jnt_limited[spine_joint.id] = 1
    # A tiny symmetric range gives MuJoCo a bilateral-like joint-limit lock
    # without rebuilding a second XML model.
    model.jnt_range[spine_joint.id] = [-1e-6, 1e-6]
    # A mechanical lock does not consume actuator current.
    spine_actuator = model.actuator("sp_j0")
    model.actuator_ctrlrange[spine_actuator.id] = [0.0, 0.0]


class JointMap:
    """Resolve model indices and transmission data by joint name."""

    def __init__(self, model, joint_names):
        self.names = list(joint_names)
        self.qpos_adr = {}
        self.dof_adr = {}
        self.act_id = {}
        self.gear = {}

        for name in self.names:
            joint = model.joint(name)
            actuator = model.actuator(name)
            act_id = actuator.id
            gear = float(model.actuator_gear[act_id, 0])
            if gear == 0.0:
                raise ValueError(f"Actuator {name!r} has zero transmission gear")

            self.qpos_adr[name] = int(joint.qposadr[0])
            self.dof_adr[name] = int(joint.dofadr[0])
            self.act_id[name] = act_id
            self.gear[name] = gear


class ContactMap:
    """Geometry IDs needed to distinguish feet from lower-leg collisions."""

    def __init__(self, model):
        self.ground_geom_id = model.geom("ground_2").id
        self.support_geom_ids = {self.ground_geom_id}
        landing_box_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "front_landing_box"
        )
        if landing_box_id != -1:
            self.support_geom_ids.add(landing_box_id)
        calf_body_ids = {
            model.body(name).id for name in ("fr_l2", "fl_l2", "rl_l2", "rr_l2")
        }
        self.shin_geom_ids = {
            geom_id
            for geom_id in range(model.ngeom)
            if model.geom_bodyid[geom_id] in calf_body_ids
            and model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CYLINDER
        }
        if not self.shin_geom_ids:
            raise ValueError("No lower-leg collision cylinders found in the MuJoCo model")

        # Each lower leg has a visible non-colliding sphere and a colliding
        # spherical foot. Only the latter has a non-zero contact type.
        self.foot_geom_ids = {
            geom_id
            for geom_id in range(model.ngeom)
            if model.geom_bodyid[geom_id] in calf_body_ids
            and model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE
            and model.geom_contype[geom_id] != 0
        }
        if not self.foot_geom_ids:
            raise ValueError("No colliding foot spheres found in the MuJoCo model")
        self.foot_geom_to_leg = {
            geom_id: model.body(model.geom_bodyid[geom_id]).name[:2]
            for geom_id in self.foot_geom_ids
        }
        self.foot_legs = sorted(set(self.foot_geom_to_leg.values()))

    def support_contact_forces(self, model, data):
        """Measure contacts with either the floor or the front landing box."""
        foot_normal_forces = {
            leg: 0.0 for leg in self.foot_geom_to_leg.values()
        }
        shin_normal_force = 0.0
        contact_force = np.zeros(6)
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            pair = {contact.geom1, contact.geom2}
            if not (pair & self.support_geom_ids):
                continue
            mujoco.mj_contactForce(model, data, contact_id, contact_force)
            normal_force = abs(contact_force[0])
            for foot_geom_id in pair & self.foot_geom_ids:
                foot_normal_forces[
                    self.foot_geom_to_leg[foot_geom_id]
                ] += normal_force
            if pair & self.shin_geom_ids:
                shin_normal_force += normal_force
        non_foot_contact = any(
            bool(
                {data.contact[i].geom1, data.contact[i].geom2}
                & self.support_geom_ids
            )
            and not bool(
                {data.contact[i].geom1, data.contact[i].geom2}
                & self.foot_geom_ids
            )
            for i in range(data.ncon)
        )
        total_foot_force = sum(foot_normal_forces.values())
        max_single_foot_force = max(foot_normal_forces.values(), default=0.0)
        force_by_foot = np.asarray([
            foot_normal_forces[leg] for leg in self.foot_legs
        ])
        return (
            total_foot_force,
            max_single_foot_force,
            force_by_foot,
            shin_normal_force,
            non_foot_contact,
        )


def quaternion_from_euler(roll, pitch, yaw):
    """Return a MuJoCo wxyz quaternion for intrinsic XYZ Euler angles."""
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def reset_drop(
    data, model, jmap, start_height, nominal_pose, initial_state=None,
):
    """Reset an episode, optionally applying validation-only disturbances."""
    mujoco.mj_resetData(model, data)
    initial_state = initial_state or {}
    base_xy = initial_state.get("base_xy", (0.0, 0.0))
    data.qpos[0:3] = [base_xy[0], base_xy[1], start_height]
    data.qpos[3:7] = initial_state.get(
        "quaternion", np.array([1.0, 0.0, 0.0, 0.0])
    )
    data.qvel[0:3] = initial_state.get("linear_velocity", np.zeros(3))
    data.qvel[3:6] = initial_state.get("angular_velocity", np.zeros(3))
    joint_offsets = initial_state.get("joint_offsets", {})

    targets = {}
    for name, nominal in nominal_pose.items():
        target = nominal
        targets[name] = target
        # Spawn without an artificial initial position error/impulse.
        data.qpos[jmap.qpos_adr[name]] = target + joint_offsets.get(name, 0.0)

    mujoco.mj_forward(model, data)
    return targets


def gain_parameter_count(lock_spine, individual_gains=False):
    leg_count = len(INDIVIDUAL_GAIN_NAMES) if individual_gains else len(LEG_GAIN_BOUNDS)
    return leg_count + (0 if lock_spine else len(SPINE_GAIN_BOUNDS))


def build_gains(params, lock_spine, individual_gains=False):
    if individual_gains:
        leg_count = len(LEG_NOMINAL_POSITION_NAMES)
        kp = dict(zip(LEG_NOMINAL_POSITION_NAMES, params[:leg_count]))
        kd = dict(zip(LEG_NOMINAL_POSITION_NAMES, params[leg_count:2 * leg_count]))
    else:
        kp_h, kp_t, kp_c, kd_h, kd_t, kd_c = params[:6]
        group_gains = {
            "hip": (kp_h, kd_h),
            "thigh": (kp_t, kd_t),
            "calf": (kp_c, kd_c),
        }
        kp = {name: group_gains[group][0] for group, names in LEG_JOINTS.items() for name in names}
        kd = {name: group_gains[group][1] for group, names in LEG_JOINTS.items() for name in names}
    if lock_spine:
        # Values are immaterial for a mechanically locked, unpowered joint.
        spine_kp, spine_kd = 0.0, 0.0
    else:
        spine_start = len(INDIVIDUAL_GAIN_NAMES) if individual_gains else len(LEG_GAIN_BOUNDS)
        spine_kp, spine_kd = params[spine_start:spine_start + 2]
    kp["sp_j0"] = spine_kp
    kd["sp_j0"] = spine_kd
    return kp, kd


def nominal_position_names(optimize_nominal_position, lock_spine):
    if not optimize_nominal_position:
        return ()
    names = list(LEG_NOMINAL_POSITION_NAMES)
    if not lock_spine:
        names.append("sp_j0")
    return tuple(names)


def nominal_pose_from_params(params, optimize_nominal_position, lock_spine, individual_gains=False):
    """Build the episode pose from the optional nominal-position parameters."""
    if not optimize_nominal_position:
        return NOMINAL_POSE
    pose = dict(NOMINAL_POSE)
    position_start = gain_parameter_count(lock_spine, individual_gains)
    for name, delta in zip(
        nominal_position_names(optimize_nominal_position, lock_spine),
        params[position_start:],
    ):
        pose[name] += delta
    return pose


def parameter_bounds(optimize_nominal_position, lock_spine, individual_gains=False):
    gain_bounds = np.vstack([np.repeat(LEG_GAIN_BOUNDS[i:i+1], 4, axis=0) for i in range(6)]) if individual_gains else LEG_GAIN_BOUNDS
    if not lock_spine:
        gain_bounds = np.vstack((gain_bounds, SPINE_GAIN_BOUNDS))
    if not optimize_nominal_position:
        return gain_bounds
    position_names = nominal_position_names(optimize_nominal_position, lock_spine)
    position_bounds = np.full(
        (len(position_names), 2), NOMINAL_POSITION_DELTA_BOUNDS,
        dtype=float,
    )
    return np.vstack((gain_bounds, position_bounds))


def apply_position_pd(model, data, jmap, targets, kp, kd):
    """Match JointController::perform(): joint PD torque -> motor-side ctrl."""
    for name in jmap.names:
        q = data.qpos[jmap.qpos_adr[name]]
        qd = data.qvel[jmap.dof_adr[name]]
        joint_torque = kp[name] * (targets[name] - q) - kd[name] * qd

        if abs(joint_torque) < STATIC_FRICTION:
            joint_torque = 0.0

        act_id = jmap.act_id[name]
        motor_command = joint_torque / jmap.gear[name]
        if model.actuator_ctrllimited[act_id]:
            low, high = model.actuator_ctrlrange[act_id]
            motor_command = np.clip(motor_command, low, high)
        data.ctrl[act_id] = motor_command


def controlled_step(model, data, jmap, targets, kp, kd):
    # Same ordering as the ROS simulator: state update, controller, dynamics.
    mujoco.mj_step1(model, data)
    apply_position_pd(model, data, jmap, targets, kp, kd)
    mujoco.mj_step2(model, data)


def episode_steps(model, start_height):
    """Allow enough time to fall from the requested height and settle."""
    gravity = abs(float(model.opt.gravity[2]))
    free_fall_time = np.sqrt(2.0 * start_height / gravity)
    duration = max(SIM_DURATION, free_fall_time + 1.2)
    return int(np.ceil(duration / model.opt.timestep))


def evaluate_drop(
    model, data, jmap, contact_map, params, steps, start_height,
    optimize_nominal_position, lock_spine, initial_state=None, individual_gains=False,
):
    nominal_pose = nominal_pose_from_params(
        params, optimize_nominal_position, lock_spine, individual_gains
    )
    targets = reset_drop(
        data, model, jmap, start_height, nominal_pose, initial_state
    )
    kp, kd = build_gains(params, lock_spine, individual_gains)
    leg_dof_idx = np.array([
        jmap.dof_adr[name]
        for group in ("hip", "thigh", "calf")
        for name in LEG_JOINTS[group]
    ])

    min_body_height = float("inf")
    peak_leg_torque = 0.0
    peak_foot_force = 0.0
    peak_foot_force_raw = 0.0
    peak_total_foot_force = 0.0
    peak_shin_force = 0.0
    peak_body_acceleration = 0.0
    peak_body_acceleration_raw = 0.0
    total_leg_effort = 0.0
    total_spine_effort = 0.0
    peak_leg_current_proxy = 0.0
    peak_spine_current_proxy = 0.0
    foot_contact = False
    shin_contact = False
    non_foot_contact = False
    crashed = False
    acceleration_window_steps = max(
        1, int(np.ceil(BODY_ACCELERATION_WINDOW / model.opt.timestep))
    )
    acceleration_history = []
    foot_force_window_steps = max(
        1, int(np.ceil(FOOT_FORCE_WINDOW / model.opt.timestep))
    )
    foot_force_history = []

    for _ in range(steps):
        controlled_step(model, data, jmap, targets, kp, kd)

        current_height = data.qpos[2]
        min_body_height = min(min_body_height, current_height)
        if current_height < 0.05:
            crashed = True

        # qfrc_actuator is joint-side generalized torque after transmission.
        joint_torque = data.qfrc_actuator[leg_dof_idx]
        leg_ctrl = np.asarray([
            data.ctrl[jmap.act_id[name]]
            for name in jmap.names if name != "sp_j0"
        ])
        spine_ctrl = float(data.ctrl[jmap.act_id["sp_j0"]])
        total_leg_effort += float(np.sum(np.square(leg_ctrl))) * model.opt.timestep
        total_spine_effort += spine_ctrl**2 * model.opt.timestep
        peak_leg_current_proxy = max(peak_leg_current_proxy, float(np.max(np.abs(leg_ctrl))))
        peak_spine_current_proxy = max(peak_spine_current_proxy, abs(spine_ctrl))
        peak_leg_torque = max(peak_leg_torque, float(np.max(np.abs(joint_torque))))
        (
            total_foot_force,
            max_single_foot_force,
            force_by_foot,
            shin_force,
            current_non_foot_contact,
        ) = contact_map.support_contact_forces(model, data)
        non_foot_contact = non_foot_contact or current_non_foot_contact
        peak_foot_force_raw = max(
            peak_foot_force_raw, max_single_foot_force
        )
        peak_total_foot_force = max(
            peak_total_foot_force, total_foot_force
        )
        peak_shin_force = max(peak_shin_force, shin_force)
        if total_foot_force > 0.0:
            foot_contact = True
        if foot_contact:
            foot_force_history.append(force_by_foot)
            if len(foot_force_history) > foot_force_window_steps:
                foot_force_history.pop(0)
            if len(foot_force_history) == foot_force_window_steps:
                window_average_by_foot = np.mean(
                    foot_force_history, axis=0
                )
                peak_foot_force = max(
                    peak_foot_force,
                    float(np.max(window_average_by_foot)),
                )
        # Measure impact acceleration only once landing has started; this
        # excludes the constant gravitational acceleration during free fall.
        if foot_contact or shin_force > 0.0:
            body_acceleration = float(np.linalg.norm(data.qacc[0:3]))
            peak_body_acceleration_raw = max(
                peak_body_acceleration_raw, body_acceleration
            )
            acceleration_history.append(body_acceleration)
            if len(acceleration_history) > acceleration_window_steps:
                acceleration_history.pop(0)
            if len(acceleration_history) == acceleration_window_steps:
                window_average = float(np.mean(acceleration_history))
                peak_body_acceleration = max(
                    peak_body_acceleration, window_average
                )
        if shin_force > 0.0:
            shin_contact = True

    # Minimize total motor-current proxy subject to survival. Including the
    # spine prevents the unlocked model from shifting load there for free.
    cost = total_leg_effort + total_spine_effort
    if optimize_nominal_position:
        # Avoid using a highly asymmetric, boundary pose as a free impact
        # brace.  The position variables are deltas from NOMINAL_POSE.
        position_start = gain_parameter_count(lock_spine, individual_gains)
        position_deltas = np.asarray(params[position_start:], dtype=float)
        cost += NOMINAL_POSITION_REGULARIZATION * float(np.sum(position_deltas**2))
    if not foot_contact:
        cost += LANDING_QUALITY_PENALTY
    if non_foot_contact:
        cost += LANDING_QUALITY_PENALTY
    if crashed:
        penetration = max(0.0, 0.05 - min_body_height)
        cost += LANDING_QUALITY_PENALTY + penetration * LANDING_QUALITY_PENALTY
    if peak_foot_force > MAX_LANDING_FOOT_FORCE:
        excess = peak_foot_force / MAX_LANDING_FOOT_FORCE - 1.0
        cost += LANDING_QUALITY_PENALTY * (1.0 + excess)
    if peak_body_acceleration > MAX_LANDING_BODY_ACCELERATION:
        excess = peak_body_acceleration / MAX_LANDING_BODY_ACCELERATION - 1.0
        cost += LANDING_QUALITY_PENALTY * (1.0 + excess)

    return (
        cost,
        crashed,
        foot_contact,
        non_foot_contact,
        min_body_height,
        peak_leg_torque,
        shin_contact,
        peak_foot_force,
        peak_foot_force_raw,
        peak_total_foot_force,
        peak_shin_force,
        peak_body_acceleration,
        peak_body_acceleration_raw,
        total_leg_effort,
        total_spine_effort,
        peak_leg_current_proxy,
        peak_spine_current_proxy,
    )


def show_best(
    model, data, jmap, params, steps, start_height, optimize_nominal_position,
    lock_spine, playback_speed=1.0, individual_gains=False,
):
    if playback_speed <= 0.0:
        raise ValueError("playback_speed must be positive")
    kp, kd = build_gains(params, lock_spine, individual_gains)
    nominal_pose = nominal_pose_from_params(
        params, optimize_nominal_position, lock_spine, individual_gains
    )
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            targets = reset_drop(data, model, jmap, start_height, nominal_pose)
            for _ in range(steps):
                if not viewer.is_running():
                    return
                step_start = time.time()
                controlled_step(model, data, jmap, targets, kp, kd)
                viewer.sync()
                frame_duration = model.opt.timestep / playback_speed
                remaining = frame_duration - (time.time() - step_start)
                if remaining > 0.0:
                    time.sleep(remaining)
            time.sleep(1.0)


def sample_params(rng, bounds):
    return tuple(rng.uniform(bounds[:, 0], bounds[:, 1]))


def result_from_params(
    model, data, jmap, contact_map, params, steps, start_height,
    optimize_nominal_position, lock_spine, initial_state=None, individual_gains=False,
):
    (
        cost,
        crashed,
        foot_contact,
        non_foot_contact,
        min_height,
        peak_leg_torque,
        shin_contact,
        peak_foot_force,
        peak_foot_force_raw,
        peak_total_foot_force,
        peak_shin_force,
        peak_body_acceleration,
        peak_body_acceleration_raw,
        total_leg_effort,
        total_spine_effort,
        peak_leg_current_proxy,
        peak_spine_current_proxy,
    ) = evaluate_drop(
        model, data, jmap, contact_map, params, steps, start_height,
        optimize_nominal_position, lock_spine, initial_state, individual_gains,
    )
    return {
        "cost": cost,
        "crashed": crashed,
        "foot_contact": foot_contact,
        "non_foot_contact": non_foot_contact,
        "min_height": min_height,
        "peak_leg_torque": peak_leg_torque,
        "shin_contact": shin_contact,
        "peak_foot_force": peak_foot_force,
        "peak_foot_force_raw": peak_foot_force_raw,
        "peak_total_foot_force": peak_total_foot_force,
        "peak_shin_force": peak_shin_force,
        "peak_body_acceleration": peak_body_acceleration,
        "peak_body_acceleration_raw": peak_body_acceleration_raw,
        "total_leg_effort": total_leg_effort,
        "total_spine_effort": total_spine_effort,
        "total_motor_effort": total_leg_effort + total_spine_effort,
        "peak_leg_current_proxy": peak_leg_current_proxy,
        "peak_spine_current_proxy": peak_spine_current_proxy,
        "params": params,
    }


def init_worker(lock_spine, start_height, optimize_nominal_position, individual_gains=False):
    global _WORKER_MODEL, _WORKER_DATA, _WORKER_JMAP
    global _WORKER_CONTACT_MAP, _WORKER_STEPS, _WORKER_START_HEIGHT
    global _WORKER_OPTIMIZE_NOMINAL_POSITION, _WORKER_LOCK_SPINE, _WORKER_INDIVIDUAL_GAINS
    _WORKER_MODEL = mujoco.MjModel.from_xml_path(str(XML_PATH))
    configure_spine(_WORKER_MODEL, lock_spine)
    _WORKER_DATA = mujoco.MjData(_WORKER_MODEL)
    _WORKER_JMAP = JointMap(_WORKER_MODEL, NOMINAL_POSE)
    _WORKER_CONTACT_MAP = ContactMap(_WORKER_MODEL)
    _WORKER_STEPS = episode_steps(_WORKER_MODEL, start_height)
    _WORKER_START_HEIGHT = start_height
    _WORKER_OPTIMIZE_NOMINAL_POSITION = optimize_nominal_position
    _WORKER_LOCK_SPINE = lock_spine
    _WORKER_INDIVIDUAL_GAINS = individual_gains


def evaluate_candidate(params):
    """Evaluate one point using process-local MuJoCo state."""
    return result_from_params(
        _WORKER_MODEL,
        _WORKER_DATA,
        _WORKER_JMAP,
        _WORKER_CONTACT_MAP,
        tuple(params),
        _WORKER_STEPS,
        _WORKER_START_HEIGHT,
        _WORKER_OPTIMIZE_NOMINAL_POSITION,
        _WORKER_LOCK_SPINE,
        individual_gains=_WORKER_INDIVIDUAL_GAINS,
    )


def select_gp_training_data(x_values, y_values, rng):
    """Bound cubic GP fitting cost while retaining good and diverse samples."""
    if len(x_values) <= MAX_GP_POINTS:
        return x_values, y_values
    keep_best = MAX_GP_POINTS // 2
    best_indices = np.argsort(y_values)[:keep_best]
    remaining = np.setdiff1d(np.arange(len(x_values)), best_indices)
    random_indices = rng.choice(
        remaining, size=MAX_GP_POINTS - keep_best, replace=False
    )
    indices = np.concatenate((best_indices, random_indices))
    return x_values[indices], y_values[indices]


def propose_candidates(
    x_values, costs, count, candidate_pool, rng, exploration_fraction, xi,
    parameter_bounds,
):
    import warnings

    from scipy.stats import norm
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    lower = parameter_bounds[:, 0]
    span = parameter_bounds[:, 1] - lower
    x_normalized = (np.asarray(x_values) - lower) / span
    # The log transform separates hard failure penalties from feasible costs.
    y = np.log1p(np.asarray(costs))
    train_x, train_y = select_gp_training_data(x_normalized, y, rng)
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2))
        * Matern(length_scale=np.ones(parameter_bounds.shape[0]), nu=2.5)
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-2))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=0,
        random_state=int(rng.integers(0, 2**31 - 1)),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(train_x, train_y)

    pool = rng.random((candidate_pool, parameter_bounds.shape[0]))
    mean, std = gp.predict(pool, return_std=True)
    std = np.maximum(std, 1e-12)
    improvement = np.min(train_y) - mean - xi
    z = improvement / std
    expected_improvement = improvement * norm.cdf(z) + std * norm.pdf(z)

    exploration_count = min(count, int(np.ceil(count * exploration_fraction)))
    uncertainty_count = exploration_count // 2
    random_count = exploration_count - uncertainty_count
    exploitation_count = count - exploration_count

    selected = []

    def add_ranked(ranking, target_count):
        if target_count <= 0:
            return
        added = 0
        for index in ranking:
            point = pool[index]
            if not selected or min(
                np.linalg.norm(point - other) for other in selected
            ) > 0.05:
                selected.append(point)
                added += 1
            if added == target_count:
                return

    add_ranked(np.argsort(expected_improvement)[::-1], exploitation_count)
    add_ranked(np.argsort(std)[::-1], uncertainty_count)
    for _ in range(random_count):
        selected.append(rng.random(parameter_bounds.shape[0]))
    while len(selected) < count:
        selected.append(rng.random(parameter_bounds.shape[0]))
    return lower + np.asarray(selected) * span


def run_search(
    trials, workers, seed, initial_trials, candidate_pool, batch_size,
    exploration_fraction, xi, lock_spine, start_height,
    optimize_nominal_position, individual_gains=False,
):
    """Bayesian optimization of total actuator effort under survival constraints."""
    workers = min(workers, trials)
    rng = np.random.default_rng(seed)
    initial_trials = min(initial_trials, trials)
    bounds = parameter_bounds(optimize_nominal_position, lock_spine, individual_gains)
    x_values = []
    costs = []
    best = None
    safe_trials = 0
    progress = []
    stagnant_batches = 0
    batch = [sample_params(rng, bounds) for _ in range(initial_trials)]

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(lock_spine, start_height, optimize_nominal_position, individual_gains),
    ) as executor:
        completed = 0
        while completed < trials:
            previous_best_cost = float("inf") if best is None else best["cost"]
            batch = batch[:trials - completed]
            futures = [executor.submit(evaluate_candidate, point) for point in batch]
            # Consume in submission order so process scheduling and --workers
            # cannot change the result.
            for future in futures:
                result = future.result()
                x_values.append(result["params"])
                costs.append(result["cost"])
                result_safe = is_safe(result)
                safe_trials += int(result_safe)
                if best is None or result["cost"] < best["cost"]:
                    best = result
            completed += len(batch)
            if best["cost"] < previous_best_cost * (1.0 - 1e-6):
                stagnant_batches = 0
            else:
                stagnant_batches += 1
            current_exploration = min(
                0.80, exploration_fraction + 0.10 * stagnant_batches
            )
            print(
                f"Ukonczono {completed}/{trials} prob | "
                f"najlepszy koszt: {best['cost']:.3f} | "
                f"bezpieczne: {safe_trials} | eksploracja: {current_exploration:.0%}",
                flush=True,
            )
            progress.append({
                "completed": completed,
                "best_cost": best["cost"],
                "safe_trials": safe_trials,
            })
            if completed < trials:
                batch = propose_candidates(
                    np.asarray(x_values), costs,
                    min(batch_size, trials - completed), candidate_pool,
                    rng, current_exploration, xi,
                    bounds,
                )
    best["safe_trials"] = safe_trials
    best["evaluated_trials"] = trials
    best["progress"] = progress
    return best


def parse_args():
    parser = argparse.ArgumentParser(description="Bayesian optimization of HB40 emergency-drop PD gains")
    parser.add_argument("--trials", type=int, default=300, help="total evaluation budget")
    parser.add_argument("--seed", type=int, default=None, help="random seed for repeatable results")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="number of parallel MuJoCo CPU processes (default: up to 8)",
    )
    parser.add_argument(
        "--initial-trials",
        type=int,
        default=40,
        help="random evaluations before fitting the Gaussian process",
    )
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=4096,
        help="random points considered by expected improvement each iteration",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="candidates proposed per GP fit (default: 128; independent of --workers)",
    )
    parser.add_argument(
        "--exploration",
        type=float,
        default=0.25,
        help="base fraction of uncertainty/random candidates per batch",
    )
    parser.add_argument(
        "--xi",
        type=float,
        default=0.01,
        help="expected-improvement exploration margin in log-cost space",
    )
    parser.add_argument(
        "--optimize-nominal-position",
        action="store_true",
        help=(
            "also optimize nominal leg positions; each joint may move by "
            f"{NOMINAL_POSITION_DELTA_BOUNDS[0]:g} to "
            f"{NOMINAL_POSITION_DELTA_BOUNDS[1]:g} rad"
        ),
    )
    parser.add_argument(
        "--individual-gains", action="store_true",
        help="optimize separate Kp and Kd for each of the 12 leg joints",
    )
    spine_mode = parser.add_mutually_exclusive_group()
    spine_mode.add_argument(
        "--lock-spine",
        action="store_true",
        help="lock sp_j0 at its nominal position",
    )
    spine_mode.add_argument(
        "--compare-spine",
        action="store_true",
        help="compare locked and unlocked spines over a series of drop heights",
    )
    spine_mode.add_argument(
        "--compare-heights",
        choices=("locked", "unlocked"),
        metavar="MODE",
        help=(
            "optimize a series of drop heights for one spine mode "
            "(locked or unlocked)"
        ),
    )
    parser.add_argument(
        "--height",
        type=float,
        default=START_HEIGHT,
        help=f"drop height for a normal single-mode run (default: {START_HEIGHT:g} m)",
    )
    parser.add_argument(
        "--heights",
        type=float,
        nargs="+",
        default=None,
        metavar="M",
        help=(
            "drop heights for --compare-spine or --compare-heights "
            "(default: 1 2 3 4 5)"
        ),
    )
    parser.add_argument("--no-viewer", action="store_true", help="finish without opening the 3D viewer")
    parser.add_argument(
        "--output-dir",
        "--output-folder",
        type=Path,
        default=OUTPUT_DIR,
        metavar="FOLDER",
        help=f"directory for plots and CSV summaries (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--validate-best",
        action="store_true",
        help="Monte Carlo validation of every best height-series candidate",
    )
    parser.add_argument(
        "--validation-trials", type=int, default=300,
        help="Monte Carlo episodes per height and spine mode (default: 300)",
    )
    parser.add_argument(
        "--validation-seed", type=int, default=None,
        help="validation seed (default: search seed plus a fixed offset)",
    )
    parser.add_argument(
        "--validation-height-jitter", type=float, default=0.02, metavar="M",
        help="maximum uniform drop-height disturbance (default: +/-0.02 m)",
    )
    parser.add_argument(
        "--validation-position-jitter", type=float, default=0.005, metavar="M",
        help="maximum uniform horizontal base displacement (default: +/-0.005 m)",
    )
    parser.add_argument(
        "--validation-angle-jitter-deg", type=float, default=2.0, metavar="DEG",
        help="maximum uniform roll/pitch/yaw disturbance (default: +/-2 deg)",
    )
    parser.add_argument(
        "--validation-linear-velocity-jitter", type=float, default=0.10,
        metavar="M/S",
        help=(
            "maximum uniform initial horizontal velocity in X/Y; "
            "Z remains zero (default: +/-0.10 m/s)"
        ),
    )
    parser.add_argument(
        "--validation-angular-velocity-jitter-deg", type=float, default=1.0,
        metavar="DEG/S", help="maximum uniform initial angular velocity (default: +/-1 deg/s)",
    )
    parser.add_argument(
        "--validation-joint-jitter", type=float, default=0.02, metavar="RAD",
        help="maximum uniform initial joint-position error (default: +/-0.02 rad)",
    )
    parser.add_argument(
        "--validation-mass-jitter", type=float, default=0.02, metavar="FRACTION",
        help="maximum relative body mass/inertia disturbance (default: +/-0.02)",
    )
    parser.add_argument(
        "--validation-friction-jitter", type=float, default=0.01, metavar="FRACTION",
        help="maximum relative contact-friction disturbance (default: +/-0.01)",
    )
    return parser.parse_args()


def validate_args(args):
    if args.trials < 1:
        raise ValueError("--trials must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.initial_trials < 1:
        raise ValueError("--initial-trials must be at least 1")
    batch_size = args.batch_size
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.candidate_pool < batch_size:
        raise ValueError("--candidate-pool must be at least the effective batch size")
    if not 0.0 <= args.exploration <= 1.0:
        raise ValueError("--exploration must be between 0 and 1")
    if args.xi < 0.0:
        raise ValueError("--xi must be non-negative")
    if args.height <= 0.0:
        raise ValueError("--height must be positive")
    compare_heights = getattr(args, "compare_heights", None)
    height_series = args.compare_spine or compare_heights is not None
    if args.heights is not None and not height_series:
        raise ValueError(
            "--heights can only be used with --compare-spine or --compare-heights"
        )
    if args.heights is not None and any(height <= 0.0 for height in args.heights):
        raise ValueError("all --heights values must be positive")
    if args.validate_best and not height_series:
        raise ValueError(
            "--validate-best requires --compare-spine or --compare-heights"
        )
    if args.validation_trials < 1:
        raise ValueError("--validation-trials must be at least 1")
    validation_jitters = {
        "--validation-height-jitter": args.validation_height_jitter,
        "--validation-position-jitter": args.validation_position_jitter,
        "--validation-angle-jitter-deg": args.validation_angle_jitter_deg,
        "--validation-linear-velocity-jitter": args.validation_linear_velocity_jitter,
        "--validation-angular-velocity-jitter-deg": args.validation_angular_velocity_jitter_deg,
        "--validation-joint-jitter": args.validation_joint_jitter,
        "--validation-mass-jitter": args.validation_mass_jitter,
        "--validation-friction-jitter": args.validation_friction_jitter,
    }
    for option, value in validation_jitters.items():
        if value < 0.0:
            raise ValueError(f"{option} must be non-negative")
    for option in (
        "--validation-mass-jitter",
        "--validation-friction-jitter",
    ):
        if validation_jitters[option] >= 1.0:
            raise ValueError(f"{option} must be less than 1")
    return batch_size


def print_result(best, model, optimize_nominal_position, lock_spine, individual_gains=False):
    print(
        f"TOP WYNIK - Koszt: {best['cost']:.2f} | "
        f"Wysokosc bazy: {best['min_height']:.3f}m | Safe: {is_safe(best)} | "
        f"Kontakt inny niz stopa: {best['non_foot_contact']}"
    )
    kp, kd = build_gains(best["params"], lock_spine, individual_gains)
    if individual_gains:
        for name in LEG_NOMINAL_POSITION_NAMES:
            print(f"  {name}: Kp={kp[name]:.2f}, Kd={kd[name]:.2f}")
    else:
        print(f"Najlepsze Kp (Hip, Thigh, Calf): {best['params'][0]:.1f}, {best['params'][1]:.1f}, {best['params'][2]:.1f}")
        print(f"Najlepsze Kd (Hip, Thigh, Calf): {best['params'][3]:.1f}, {best['params'][4]:.1f}, {best['params'][5]:.1f}")
    if not lock_spine:
        print(f"Najlepsze Kp/Kd kregoslupa: {kp['sp_j0']:.1f}, {kd['sp_j0']:.1f}")
    if optimize_nominal_position:
        nominal_pose = nominal_pose_from_params(
            best["params"], True, lock_spine, individual_gains
        )
        print("Najlepsza pozycja nominalna:")
        for name in nominal_position_names(True, lock_spine):
            print(f"  {name}: {nominal_pose[name]:.4f} rad")
    print(
        f"Szczytowa srednia sila jednej stopy (okno 10 ms): "
        f"{best['peak_foot_force']:.1f} N "
        f"(limit: {MAX_LANDING_FOOT_FORCE:.0f} N)"
    )
    print(
        f"Surowy chwilowy pik sily jednej stopy: "
        f"{best['peak_foot_force_raw']:.1f} N"
    )
    print(
        f"Szczytowa suma sil wszystkich stop: "
        f"{best['peak_total_foot_force']:.1f} N"
    )
    print(f"Szczytowa sila lydki: {best['peak_shin_force']:.1f} N")
    print(f"Szczytowy moment stawow nog: {best['peak_leg_torque']:.2f} Nm")
    print(f"Calka kwadratu komend wszystkich silnikow: {best['total_motor_effort']:.3f}")
    print(f"Calka kwadratu komend silnikow nog: {best['total_leg_effort']:.3f}")
    print(f"Calka kwadratu komendy kregoslupa: {best['total_spine_effort']:.3f}")
    print(f"Szczytowa komenda silnika nogi: {best['peak_leg_current_proxy']:.3f}")
    print(f"Szczytowa komenda silnika kregoslupa: {best['peak_spine_current_proxy']:.3f}")
    print(
        f"Szczytowe srednie przyspieszenie korpusu (okno 50 ms): "
        f"{best['peak_body_acceleration']:.1f} m/s^2 "
        f"({best['peak_body_acceleration'] / abs(model.opt.gravity[2]):.1f} g)"
    )
    print(
        f"Surowy chwilowy pik przyspieszenia: "
        f"{best['peak_body_acceleration_raw']:.1f} m/s^2 "
        f"({best['peak_body_acceleration_raw'] / abs(model.opt.gravity[2]):.1f} g)"
    )


def optimize_configuration(
    args, batch_size, lock_spine, start_height, seed, optimize_nominal_position,
):
    active_workers = min(args.workers, args.trials)
    label = "zablokowany" if lock_spine else "aktywny"
    print(
        f"\nWysokosc: {start_height:g} m | kregoslup: {label} | "
        f"pozycja nominalna: {'optymalizowana' if optimize_nominal_position else 'stala'} | "
        f"{args.trials} prob, {active_workers} procesow CPU",
        flush=True,
    )
    start_time = time.time()
    best = run_search(
        args.trials,
        active_workers,
        seed,
        args.initial_trials,
        args.candidate_pool,
        batch_size,
        args.exploration,
        args.xi,
        lock_spine,
        start_height,
        optimize_nominal_position,
        args.individual_gains,
    )
    print(f"Zakonczono konfiguracje w {time.time() - start_time:.2f} s.")
    return best


def is_safe(result):
    return (
        result["foot_contact"]
        and not result["crashed"]
        and not result["non_foot_contact"]
        and result["peak_foot_force"] <= MAX_LANDING_FOOT_FORCE
        and result["peak_body_acceleration"] <= MAX_LANDING_BODY_ACCELERATION
    )


def wilson_interval(successes, trials, z=1.959963984540054):
    """Two-sided Wilson score interval for a binomial success probability."""
    probability = successes / trials
    denominator = 1.0 + z**2 / trials
    centre = (probability + z**2 / (2.0 * trials)) / denominator
    radius = z * np.sqrt(
        probability * (1.0 - probability) / trials
        + z**2 / (4.0 * trials**2)
    ) / denominator
    lower = 0.0 if successes == 0 else max(0.0, centre - radius)
    upper = 1.0 if successes == trials else min(1.0, centre + radius)
    return lower, upper


def validation_scenarios(args, model, seed):
    """Generate disturbances once so locked/unlocked receive paired tests."""
    rng = np.random.default_rng(seed)
    angle_limit = np.deg2rad(args.validation_angle_jitter_deg)
    angular_velocity_limit = np.deg2rad(
        args.validation_angular_velocity_jitter_deg
    )
    scenarios = []
    for _ in range(args.validation_trials):
        euler = rng.uniform(-angle_limit, angle_limit, 3)
        scenarios.append({
            "height_offset": rng.uniform(
                -args.validation_height_jitter,
                args.validation_height_jitter,
            ),
            "base_xy": rng.uniform(
                -args.validation_position_jitter,
                args.validation_position_jitter,
                2,
            ),
            "euler": euler,
            "quaternion": quaternion_from_euler(*euler),
            "linear_velocity": np.append(
                rng.uniform(
                    -args.validation_linear_velocity_jitter,
                    args.validation_linear_velocity_jitter,
                    2,
                ),
                0.0,
            ),
            "angular_velocity": rng.uniform(
                -angular_velocity_limit, angular_velocity_limit, 3
            ),
            "joint_offset_values": rng.uniform(
                -args.validation_joint_jitter,
                args.validation_joint_jitter,
                len(NOMINAL_POSE),
            ),
            "body_scales": rng.uniform(
                1.0 - args.validation_mass_jitter,
                1.0 + args.validation_mass_jitter,
                model.nbody,
            ),
            "friction_scales": rng.uniform(
                1.0 - args.validation_friction_jitter,
                1.0 + args.validation_friction_jitter,
                model.ngeom,
            ),
        })
    return scenarios


def validate_candidate(
    best, height, lock_spine, optimize_nominal_position, scenarios, individual_gains=False,
):
    """Evaluate one optimized candidate under a fixed set of disturbances."""
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    configure_spine(model, lock_spine)
    data = mujoco.MjData(model)
    jmap = JointMap(model, NOMINAL_POSE)
    contact_map = ContactMap(model)
    base_mass = model.body_mass.copy()
    base_inertia = model.body_inertia.copy()
    base_friction = model.geom_friction.copy()
    trial_results = []

    for trial_index, scenario in enumerate(scenarios):
        body_scales = scenario["body_scales"]
        model.body_mass[:] = base_mass * body_scales
        model.body_inertia[:] = base_inertia * body_scales[:, None]
        model.geom_friction[:] = (
            base_friction * scenario["friction_scales"][:, None]
        )
        mujoco.mj_setConst(model, data)

        disturbed_height = max(1e-4, height + scenario["height_offset"])
        initial_state = {
            "base_xy": scenario["base_xy"],
            "quaternion": scenario["quaternion"],
            "linear_velocity": scenario["linear_velocity"],
            "angular_velocity": scenario["angular_velocity"],
            "joint_offsets": dict(zip(
                NOMINAL_POSE, scenario["joint_offset_values"]
            )),
        }
        result = result_from_params(
            model, data, jmap, contact_map, best["params"],
            episode_steps(model, disturbed_height), disturbed_height,
            optimize_nominal_position, lock_spine, initial_state, individual_gains,
        )
        result["safe"] = is_safe(result)
        result["trial"] = trial_index
        result["disturbed_height"] = disturbed_height
        result["euler"] = scenario["euler"]
        result["linear_velocity"] = scenario["linear_velocity"]
        result["angular_velocity"] = scenario["angular_velocity"]
        trial_results.append(result)

    successes = sum(result["safe"] for result in trial_results)
    confidence_low, confidence_high = wilson_interval(
        successes, len(trial_results)
    )
    foot_forces = np.asarray([
        result["peak_foot_force"] for result in trial_results
    ])
    raw_foot_forces = np.asarray([
        result["peak_foot_force_raw"] for result in trial_results
    ])
    total_foot_forces = np.asarray([
        result["peak_total_foot_force"] for result in trial_results
    ])
    accelerations = np.asarray([
        result["peak_body_acceleration"] for result in trial_results
    ])
    raw_accelerations = np.asarray([
        result["peak_body_acceleration_raw"] for result in trial_results
    ])
    summary = {
        "trials": len(trial_results),
        "successes": successes,
        "success_rate": successes / len(trial_results),
        "confidence_low": confidence_low,
        "confidence_high": confidence_high,
        "fail_no_foot_contact": sum(
            not result["foot_contact"] for result in trial_results
        ),
        "fail_crashed": sum(result["crashed"] for result in trial_results),
        "fail_non_foot_contact": sum(
            result["non_foot_contact"] for result in trial_results
        ),
        "fail_foot_force": int(np.sum(
            foot_forces > MAX_LANDING_FOOT_FORCE
        )),
        "fail_body_acceleration": int(np.sum(
            accelerations > MAX_LANDING_BODY_ACCELERATION
        )),
        "foot_force_p95": float(np.percentile(foot_forces, 95)),
        "foot_force_p99": float(np.percentile(foot_forces, 99)),
        "foot_force_max": float(np.max(foot_forces)),
        "raw_foot_force_p95": float(np.percentile(raw_foot_forces, 95)),
        "raw_foot_force_p99": float(np.percentile(raw_foot_forces, 99)),
        "raw_foot_force_max": float(np.max(raw_foot_forces)),
        "total_foot_force_p95": float(np.percentile(total_foot_forces, 95)),
        "total_foot_force_p99": float(np.percentile(total_foot_forces, 99)),
        "total_foot_force_max": float(np.max(total_foot_forces)),
        "body_acceleration_p95": float(np.percentile(accelerations, 95)),
        "body_acceleration_p99": float(np.percentile(accelerations, 99)),
        "body_acceleration_max": float(np.max(accelerations)),
        "raw_body_acceleration_p95": float(np.percentile(raw_accelerations, 95)),
        "raw_body_acceleration_p99": float(np.percentile(raw_accelerations, 99)),
        "raw_body_acceleration_max": float(np.max(raw_accelerations)),
        "trials_data": trial_results,
    }
    return summary


def save_validation_results(validation, output_dir, args, validation_seed):
    """Save Monte Carlo summaries, individual trials, and a success plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_columns = [
        "height", "spine", "trials", "successes", "success_rate",
        "confidence_low", "confidence_high", "fail_no_foot_contact",
        "fail_crashed", "fail_non_foot_contact", "fail_foot_force",
        "fail_body_acceleration", "foot_force_p95", "foot_force_p99",
        "foot_force_max", "raw_foot_force_p95", "raw_foot_force_p99",
        "raw_foot_force_max", "total_foot_force_p95", "total_foot_force_p99",
        "total_foot_force_max", "body_acceleration_p95",
        "body_acceleration_p99", "body_acceleration_max",
        "raw_body_acceleration_p95", "raw_body_acceleration_p99",
        "raw_body_acceleration_max",
    ]
    with (output_dir / "walidacja_najlepszych.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_columns)
        writer.writeheader()
        modes = sorted(validation[next(iter(validation))])
        for height in sorted(validation):
            for lock_spine in modes:
                label = "LOCKED" if lock_spine else "UNLOCKED"
                summary = validation[height][lock_spine]
                writer.writerow({
                    "height": height,
                    "spine": label,
                    **{key: summary[key] for key in summary_columns[2:]},
                })

    trial_columns = [
        "height", "spine", "trial", "disturbed_height", "safe",
        "crashed", "foot_contact", "non_foot_contact", "min_height",
        "peak_foot_force", "peak_foot_force_raw", "peak_total_foot_force",
        "peak_body_acceleration", "peak_body_acceleration_raw", "roll_deg",
        "pitch_deg", "yaw_deg", "linear_speed", "angular_speed_deg_s",
    ]
    with (output_dir / "walidacja_proby.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=trial_columns)
        writer.writeheader()
        for height in sorted(validation):
            for lock_spine in modes:
                label = "LOCKED" if lock_spine else "UNLOCKED"
                for result in validation[height][lock_spine]["trials_data"]:
                    roll, pitch, yaw = np.rad2deg(result["euler"])
                    writer.writerow({
                        "height": height,
                        "spine": label,
                        "trial": result["trial"],
                        "disturbed_height": result["disturbed_height"],
                        "safe": result["safe"],
                        "crashed": result["crashed"],
                        "foot_contact": result["foot_contact"],
                        "non_foot_contact": result["non_foot_contact"],
                        "min_height": result["min_height"],
                        "peak_foot_force": result["peak_foot_force"],
                        "peak_foot_force_raw": result["peak_foot_force_raw"],
                        "peak_total_foot_force": result["peak_total_foot_force"],
                        "peak_body_acceleration": result["peak_body_acceleration"],
                        "peak_body_acceleration_raw": result["peak_body_acceleration_raw"],
                        "roll_deg": roll,
                        "pitch_deg": pitch,
                        "yaw_deg": yaw,
                        "linear_speed": np.linalg.norm(result["linear_velocity"]),
                        "angular_speed_deg_s": np.rad2deg(
                            np.linalg.norm(result["angular_velocity"])
                        ),
                    })

    with (output_dir / "walidacja_ustawienia.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["parameter", "value"])
        settings = {
            "validation_seed": validation_seed,
            "validation_trials": args.validation_trials,
            "single_foot_force_limit_n": MAX_LANDING_FOOT_FORCE,
            "foot_force_window_ms": FOOT_FORCE_WINDOW * 1000.0,
            "body_acceleration_limit_g": (
                MAX_LANDING_BODY_ACCELERATION / 9.81
            ),
            "body_acceleration_window_ms": (
                BODY_ACCELERATION_WINDOW * 1000.0
            ),
            "height_jitter_m": args.validation_height_jitter,
            "horizontal_position_jitter_m": args.validation_position_jitter,
            "orientation_jitter_deg": args.validation_angle_jitter_deg,
            "linear_velocity_jitter_m_s": args.validation_linear_velocity_jitter,
            "angular_velocity_jitter_deg_s": args.validation_angular_velocity_jitter_deg,
            "joint_position_jitter_rad": args.validation_joint_jitter,
            "mass_inertia_jitter_fraction": args.validation_mass_jitter,
            "friction_jitter_fraction": args.validation_friction_jitter,
        }
        writer.writerows(settings.items())

    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    colors = {False: "#1976d2", True: "#d32f2f"}
    for lock_spine in modes:
        label = "LOCKED" if lock_spine else "UNLOCKED"
        heights = sorted(validation)
        rates = np.asarray([
            validation[height][lock_spine]["success_rate"] for height in heights
        ])
        lows = np.asarray([
            validation[height][lock_spine]["confidence_low"] for height in heights
        ])
        highs = np.asarray([
            validation[height][lock_spine]["confidence_high"] for height in heights
        ])
        axis.errorbar(
            heights, 100.0 * rates,
            yerr=np.vstack((
                100.0 * np.maximum(0.0, rates - lows),
                100.0 * np.maximum(0.0, highs - rates),
            )),
            marker="o", capsize=3, color=colors[lock_spine], label=label,
        )
    axis.set_title("Walidacja Monte Carlo najlepszych konfiguracji (95% CI)")
    axis.set_xlabel("Nominalna wysokość [m]")
    axis.set_ylabel("Bezpieczne próby [%]")
    axis.set_ylim(-2.0, 102.0)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(output_dir / "walidacja_najlepszych.png", dpi=150)
    plt.close(fig)


def validate_comparison_best(
    results, args, optimize_nominal_position, validation_seed,
):
    """Validate every best height-series candidate using shared disturbances."""
    reference_model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    validation = {}
    print(
        f"\nWALIDACJA MONTE CARLO: {args.validation_trials} prob na "
        f"wysokosc i tryb | seed: {validation_seed}",
        flush=True,
    )
    for height_index, height in enumerate(sorted(results)):
        scenario_seed = np.random.SeedSequence([
            int(validation_seed) % (2**32), height_index
        ])
        scenarios = validation_scenarios(args, reference_model, scenario_seed)
        validation[height] = {}
        for lock_spine in sorted(results[height]):
            label = "LOCKED" if lock_spine else "UNLOCKED"
            summary = validate_candidate(
                results[height][lock_spine], height, lock_spine,
                optimize_nominal_position, scenarios, args.individual_gains,
            )
            validation[height][lock_spine] = summary
            print(
                f"{height:6.2f} m | {label:8s} | "
                f"{summary['successes']}/{summary['trials']} "
                f"({summary['success_rate']:.1%}, 95% CI "
                f"{summary['confidence_low']:.1%}-{summary['confidence_high']:.1%})",
                flush=True,
            )
    save_validation_results(
        validation, args.output_dir, args, validation_seed
    )
    print(f"Raport walidacji zapisano w: {args.output_dir}")
    return validation


def comparison_winner(unlocked, locked):
    unlocked_safe = is_safe(unlocked)
    locked_safe = is_safe(locked)
    if not unlocked_safe and not locked_safe:
        return "NO SAFE"
    if unlocked_safe != locked_safe:
        return "UNLOCKED" if unlocked_safe else "LOCKED"
    if np.isclose(unlocked["cost"], locked["cost"], rtol=1e-6):
        return "TIE"
    return "UNLOCKED" if unlocked["cost"] < locked["cost"] else "LOCKED"


def print_comparison(results):
    available_modes = sorted(results[next(iter(results))])
    if len(available_modes) == 1:
        lock_spine = available_modes[0]
        label = "LOCKED" if lock_spine else "UNLOCKED"
        print(f"\nWYNIKI WZGLEDEM WYSOKOSCI: {label}")
        print("height | safe / total / legs / spine")
        print("-" * 48)
        for height, modes in results.items():
            result = modes[lock_spine]
            print(
                f"{height:6.2f} | {str(is_safe(result)):5s} / "
                f"{result['total_motor_effort']:6.3f} / "
                f"{result['total_leg_effort']:6.3f} / "
                f"{result['total_spine_effort']:6.3f}"
            )
        safe_heights = [
            height for height, modes in results.items()
            if is_safe(modes[lock_spine])
        ]
        maximum = f"{max(safe_heights):g} m" if safe_heights else "none"
        print(f"Highest safe tested height ({label}): {maximum}")
        return

    print("\nPOROWNANIE: minimalny prad przy zachowaniu przezycia")
    print(
        "height | unlocked: safe / total / legs / spine | "
        "locked: safe / total / legs / spine | result"
    )
    print("-" * 112)
    for height, modes in results.items():
        unlocked = modes[False]
        locked = modes[True]
        winner = comparison_winner(unlocked, locked)
        print(
            f"{height:6.2f} | "
            f"{str(is_safe(unlocked)):5s} / {unlocked['total_motor_effort']:6.3f} / "
            f"{unlocked['total_leg_effort']:6.3f} / {unlocked['total_spine_effort']:6.3f} | "
            f"{str(is_safe(locked)):5s} / {locked['total_motor_effort']:6.3f} / "
            f"{locked['total_leg_effort']:6.3f} / {locked['total_spine_effort']:6.3f} | "
            f"{winner}"
        )

    for lock_spine, label in ((False, "UNLOCKED"), (True, "LOCKED")):
        safe_heights = [
            height for height, modes in results.items() if is_safe(modes[lock_spine])
        ]
        maximum = f"{max(safe_heights):g} m" if safe_heights else "none"
        print(f"Highest safe tested height ({label}): {maximum}")


def save_plots(results, output_dir, optimize_nominal_position, individual_gains=False):
    """Save comparison plots and a machine-readable summary without showing a GUI."""
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    heights = sorted(results)
    available_modes = sorted(results[next(iter(results))])
    modes = {
        lock_spine: "LOCKED" if lock_spine else "UNLOCKED"
        for lock_spine in available_modes
    }
    colors = {False: "#1976d2", True: "#d32f2f"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    metrics = [
        ("total_motor_effort", "Całkowity wysiłek silników", "Całka u²"),
        ("total_leg_effort", "Wysiłek nóg", "Całka u²"),
        ("total_spine_effort", "Wysiłek kręgosłupa", "Całka u²"),
    ]
    for axis, (key, title, ylabel) in zip(axes.flat, metrics):
        for lock_spine, label in modes.items():
            values = [results[h][lock_spine][key] for h in heights]
            safe = [is_safe(results[h][lock_spine]) for h in heights]
            axis.plot(heights, values, color=colors[lock_spine], label=label)
            axis.scatter(
                heights, values, c=[colors[lock_spine] if ok else "white" for ok in safe],
                edgecolors=colors[lock_spine], s=55, zorder=3,
            )
        axis.set_title(title)
        axis.set_xlabel("Wysokość [m]")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend()

    axis = axes[1, 1]
    for lock_spine, label in modes.items():
        safe = [is_safe(results[h][lock_spine]) for h in heights]
        axis.plot(
            heights, [int(value) for value in safe], marker="o",
            color=colors[lock_spine], label=label,
        )
    axis.set_title("Przeżycie / bezpieczeństwo")
    axis.set_xlabel("Wysokość [m]")
    axis.set_ylabel("Bezpieczne (1 = tak, 0 = nie)")
    axis.set_yticks([0, 1], ["NIE", "TAK"])
    axis.set_ylim(-0.1, 1.1)
    axis.grid(alpha=0.25)
    axis.legend()
    if len(modes) == 2:
        figure_title = "Porównanie kręgosłupa: odblokowany vs zablokowany"
    else:
        figure_title = f"Wyniki względem wysokości: {next(iter(modes.values()))}"
    fig.suptitle(figure_title)
    comparison_path = output_dir / "porownanie_kregoslupa.png"
    fig.savefig(comparison_path, dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for height in heights:
        for lock_spine, label in modes.items():
            history = results[height][lock_spine].get("progress", [])
            if not history:
                continue
            axis.plot(
                [item["completed"] for item in history],
                [item["best_cost"] for item in history],
                label=f"{height:g} m, {label}", color=colors[lock_spine],
                alpha=0.45 if len(heights) > 1 else 1.0,
                linestyle="--" if lock_spine else "-",
            )
    axis.set_title("Zbieżność optymalizacji")
    axis.set_xlabel("Liczba prób")
    axis.set_ylabel("Najlepszy koszt")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize="small")
    convergence_path = output_dir / "zbieznosc_optymalizacji.png"
    fig.savefig(convergence_path, dpi=150)
    plt.close(fig)

    # Parameter plots are kept separate for both spine modes.  Combining all
    # 12 leg joints and both controller modes in one chart made height trends
    # difficult to read.
    leg_labels = {
        "rl": "tylna lewa",
        "rr": "tylna prawa",
        "fr": "przednia prawa",
        "fl": "przednia lewa",
    }
    joint_colors = {
        "rl": "#1976d2",
        "rr": "#ef6c00",
        "fr": "#388e3c",
        "fl": "#7b1fa2",
    }
    gain_colors = {
        "hip": "#1976d2",
        "thigh": "#ef6c00",
        "calf": "#388e3c",
        "spine": "#7b1fa2",
    }
    gain_labels = {
        "hip": "biodro",
        "thigh": "udo",
        "calf": "łydka",
        "spine": "kręgosłup",
    }

    for lock_spine in modes:
        mode_slug = "locked" if lock_spine else "unlocked"
        mode_title = "zablokowany" if lock_spine else "aktywny"
        mode_results = [results[h][lock_spine] for h in heights]
        nominal_poses = [
            nominal_pose_from_params(
                result["params"], optimize_nominal_position, lock_spine, individual_gains
            )
            for result in mode_results
        ]

        fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        for axis, (group, names) in zip(axes.flat[:3], LEG_JOINTS.items()):
            for name in names:
                leg = name.split("_", 1)[0]
                axis.plot(
                    heights,
                    [pose[name] for pose in nominal_poses],
                    marker="o",
                    color=joint_colors[leg],
                    label=leg_labels[leg],
                )
            axis.set_title(f"Pozycje: {gain_labels[group]}")
            axis.set_xlabel("Wysokość [m]")
            axis.set_ylabel("Pozycja nominalna [rad]")
            axis.grid(alpha=0.25)
            axis.legend(fontsize="small")

        spine_axis = axes[1, 1]
        spine_axis.plot(
            heights,
            [pose["sp_j0"] for pose in nominal_poses],
            marker="o",
            color=gain_colors["spine"],
        )
        spine_axis.set_title(
            "Pozycja: kręgosłup"
            + (" (blokada mechaniczna)" if lock_spine else "")
        )
        spine_axis.set_xlabel("Wysokość [m]")
        spine_axis.set_ylabel("Pozycja nominalna [rad]")
        spine_axis.grid(alpha=0.25)
        fig.suptitle(
            f"Pozycje nominalne stawów — kręgosłup {mode_title}"
        )
        positions_path = output_dir / f"pozycje_stawow_{mode_slug}.png"
        fig.savefig(positions_path, dpi=150)
        plt.close(fig)

        fig, (kp_axis, kd_axis) = plt.subplots(
            1, 2, figsize=(13, 5), constrained_layout=True
        )
        gain_series = list(LEG_NOMINAL_POSITION_NAMES) if individual_gains else list(LEG_JOINTS)
        if not lock_spine:
            gain_series.append("sp_j0")
        for gain_name in gain_series:
            gain_group = gain_name.split("_", 1)[1][1:] if individual_gains and gain_name != "sp_j0" else gain_name
            if gain_group == "sp_j0":
                gain_group = "spine"
            gain_values = [build_gains(result["params"], lock_spine, individual_gains) for result in mode_results]
            kp_axis.plot(
                heights,
                [values[0][gain_name if individual_gains or gain_name == "sp_j0" else LEG_JOINTS[gain_name][0]] for values in gain_values],
                marker="o",
                color=joint_colors.get(gain_name[:2], gain_colors.get(gain_group, "#333333")),
                label=gain_name if individual_gains else gain_labels[gain_group],
            )
            kd_axis.plot(
                heights,
                [values[1][gain_name if individual_gains or gain_name == "sp_j0" else LEG_JOINTS[gain_name][0]] for values in gain_values],
                marker="o",
                color=joint_colors.get(gain_name[:2], gain_colors.get(gain_group, "#333333")),
                label=gain_name if individual_gains else gain_labels[gain_group],
            )
        for axis, symbol in ((kp_axis, "Kp"), (kd_axis, "Kd")):
            axis.set_title(symbol)
            axis.set_xlabel("Wysokość [m]")
            axis.set_ylabel("Wartość regulatora")
            axis.grid(alpha=0.25)
            axis.legend()
        fig.suptitle(
            f"Parametry regulatorów — kręgosłup {mode_title}"
        )
        gains_path = output_dir / f"parametry_regulatorow_{mode_slug}.png"
        fig.savefig(gains_path, dpi=150)
        plt.close(fig)

    with (output_dir / "porownanie_kregoslupa.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "height", "spine", "safe", "cost", "total_motor_effort",
            "total_leg_effort", "total_spine_effort", "peak_foot_force",
            "peak_foot_force_raw", "peak_total_foot_force",
            "peak_body_acceleration",
            "peak_body_acceleration_raw",
        ])
        for height in heights:
            for lock_spine, label in modes.items():
                result = results[height][lock_spine]
                writer.writerow([
                    height, label, is_safe(result), result["cost"],
                    result["total_motor_effort"], result["total_leg_effort"],
                    result["total_spine_effort"], result["peak_foot_force"],
                    result["peak_foot_force_raw"],
                    result["peak_total_foot_force"],
                    result["peak_body_acceleration"],
                    result["peak_body_acceleration_raw"],
                ])

    gain_names = INDIVIDUAL_GAIN_NAMES if individual_gains else LEG_GAIN_PARAMETER_NAMES
    parameter_columns = [*gain_names, "kp_spine", "kd_spine", *NOMINAL_POSE]
    with (output_dir / "parametry_wzgledem_wysokosci.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["height", "spine", "safe", *parameter_columns]
        )
        writer.writeheader()
        for height in heights:
            for lock_spine, label in modes.items():
                result = results[height][lock_spine]
                params = result["params"]
                pose = nominal_pose_from_params(
                    params, optimize_nominal_position, lock_spine, individual_gains
                )
                row = {
                    "height": height,
                    "spine": label,
                    "safe": is_safe(result),
                    **dict(zip(gain_names, params[:len(gain_names)])),
                    "kp_spine": "" if lock_spine else params[len(gain_names)],
                    "kd_spine": "" if lock_spine else params[len(gain_names) + 1],
                    **pose,
                }
                writer.writerow(row)
    print(f"Wykresy i dane zapisano w: {output_dir}")


def save_single_run_plot(best, output_dir, lock_spine, height):
    """Save the convergence chart for a regular single-mode invocation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = best.get("progress", [])
    if not history:
        return
    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.plot(
        [item["completed"] for item in history],
        [item["best_cost"] for item in history],
        color="#1976d2",
    )
    axis.set_title(
        f"Zbieżność optymalizacji — {height:g} m, "
        f"{'zablokowany' if lock_spine else 'odblokowany'} kręgosłup"
    )
    axis.set_xlabel("Liczba prób")
    axis.set_ylabel("Najlepszy koszt")
    axis.grid(alpha=0.25)
    path = output_dir / (
        f"zbieznosc_{height:g}m_{'locked' if lock_spine else 'unlocked'}.png"
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wykres zapisano w: {path}")


def main(args=None):
    if args is None:
        args = parse_args()
    batch_size = validate_args(args)

    compare_heights = getattr(args, "compare_heights", None)
    height_series = args.compare_spine or compare_heights is not None
    if height_series:
        heights = sorted(set(args.heights or [1.0, 2.0, 3.0, 4.0, 5.0]))
        # A concrete shared seed makes heights and, when present, both modes
        # reproducible even when the caller did not request it explicitly.
        seed = args.seed
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])
        print(f"Shared height-series seed: {seed}")
        if args.compare_spine:
            spine_modes = (False, True)
        else:
            spine_modes = (compare_heights == "locked",)
        results = {}
        for height in heights:
            results[height] = {}
            for lock_spine in spine_modes:
                results[height][lock_spine] = optimize_configuration(
                    args, batch_size, lock_spine, height, seed,
                    args.optimize_nominal_position,
                )
        print_comparison(results)
        save_plots(results, args.output_dir, args.optimize_nominal_position, args.individual_gains)
        if args.validate_best:
            validation_seed = args.validation_seed
            if validation_seed is None:
                validation_seed = (int(seed) + 104729) % (2**32)
            validate_comparison_best(
                results, args, args.optimize_nominal_position,
                validation_seed,
            )
        viewer_height = heights[-1]
        highest_modes = results[viewer_height]
        if len(spine_modes) == 1:
            viewer_lock = spine_modes[0]
            winner = None
        else:
            winner = comparison_winner(highest_modes[False], highest_modes[True])
            if winner in ("NO SAFE", "TIE"):
                viewer_lock = min(
                    spine_modes,
                    key=lambda locked: (
                        not is_safe(highest_modes[locked]),
                        highest_modes[locked]["cost"],
                    ),
                )
            else:
                viewer_lock = winner == "LOCKED"
        best = results[viewer_height][viewer_lock]
        selected_lock_spine = viewer_lock
        print(
            f"\nHighest-height "
            f"{'winner' if winner not in (None, 'NO SAFE', 'TIE') else 'best candidate'} details "
            f"({viewer_height:g} m, "
            f"{'LOCKED' if viewer_lock else 'UNLOCKED'}):"
        )
        model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        configure_spine(model, viewer_lock)
        print_result(
            best, model, args.optimize_nominal_position, viewer_lock, args.individual_gains
        )
        start_height = viewer_height
    else:
        start_height = args.height
        best = optimize_configuration(
            args, batch_size, args.lock_spine, start_height, args.seed,
            args.optimize_nominal_position,
        )
        selected_lock_spine = args.lock_spine
        model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        configure_spine(model, args.lock_spine)
        print_result(
            best, model, args.optimize_nominal_position, args.lock_spine, args.individual_gains
        )
        save_single_run_plot(best, args.output_dir, args.lock_spine, start_height)

    if not args.no_viewer:
        print("\nOtwieram symulacje 3D... Zamknij okno, by zakonczyc skrypt.")
        data = mujoco.MjData(model)
        jmap = JointMap(model, NOMINAL_POSE)
        steps = episode_steps(model, start_height)
        show_best(
            model, data, jmap, best["params"], steps, start_height,
            args.optimize_nominal_position, selected_lock_spine,
            individual_gains=args.individual_gains,
        )


if __name__ == "__main__":
    main()
