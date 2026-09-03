import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import time
import warnings

import mujoco
import mujoco.viewer
import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

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

START_HEIGHT = 4.0
SPINE_KP = 40.0
SPINE_KD = 3.0
# Same dead zone as STATIC_FRICTION in simulation/src/joint_control.cpp.
STATIC_FRICTION = 0.37

SIM_DURATION = 2.2
XML_PATH = Path(__file__).resolve().with_name("intention.xml")
PARAMETER_NAMES = (
    "kp_hip", "kp_thigh", "kp_calf",
    "kd_hip", "kd_thigh", "kd_calf",
)
PARAMETER_BOUNDS = np.array([
    (5.0, 60.0),
    (5.0, 100.0),
    (5.0, 100.0),
    (0.2, 5.0),
    (0.2, 8.0),
    (0.2, 8.0),
], dtype=float)
MAX_GP_POINTS = 400
_WORKER_MODEL = None
_WORKER_DATA = None
_WORKER_JMAP = None
_WORKER_CONTACT_MAP = None
_WORKER_STEPS = None
_WORKER_START_HEIGHT = None


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

    def ground_contact_forces(self, model, data):
        foot_normal_force = 0.0
        shin_normal_force = 0.0
        contact_force = np.zeros(6)
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            pair = {contact.geom1, contact.geom2}
            if self.ground_geom_id not in pair:
                continue
            mujoco.mj_contactForce(model, data, contact_id, contact_force)
            normal_force = abs(contact_force[0])
            if pair & self.foot_geom_ids:
                foot_normal_force += normal_force
            if pair & self.shin_geom_ids:
                shin_normal_force += normal_force
        non_foot_contact = any(
            self.ground_geom_id in {data.contact[i].geom1, data.contact[i].geom2}
            and not bool(
                {data.contact[i].geom1, data.contact[i].geom2}
                & self.foot_geom_ids
            )
            for i in range(data.ncon)
        )
        return foot_normal_force, shin_normal_force, non_foot_contact


def reset_drop(data, model, jmap, start_height):
    """Reset to the exact initial state used by every optimization episode."""
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, start_height]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

    targets = {}
    for name, nominal in NOMINAL_POSE.items():
        target = nominal
        targets[name] = target
        # Spawn without an artificial initial position error/impulse.
        data.qpos[jmap.qpos_adr[name]] = target

    mujoco.mj_forward(model, data)
    return targets


def build_gains(params):
    kp_h, kp_t, kp_c, kd_h, kd_t, kd_c = params[:6]
    group_gains = {
        "hip": (kp_h, kd_h),
        "thigh": (kp_t, kd_t),
        "calf": (kp_c, kd_c),
    }
    kp = {"sp_j0": SPINE_KP}
    kd = {"sp_j0": SPINE_KD}
    for group, names in LEG_JOINTS.items():
        group_kp, group_kd = group_gains[group]
        for name in names:
            kp[name] = group_kp
            kd[name] = group_kd
    return kp, kd


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


def evaluate_drop(model, data, jmap, contact_map, params, steps, start_height):
    targets = reset_drop(data, model, jmap, start_height)
    kp, kd = build_gains(params)
    leg_dof_idx = np.array([
        jmap.dof_adr[name]
        for group in ("hip", "thigh", "calf")
        for name in LEG_JOINTS[group]
    ])

    min_body_height = float("inf")
    peak_leg_torque = 0.0
    peak_foot_force = 0.0
    peak_shin_force = 0.0
    peak_body_acceleration = 0.0
    total_leg_effort = 0.0
    total_spine_effort = 0.0
    peak_leg_current_proxy = 0.0
    peak_spine_current_proxy = 0.0
    foot_contact = False
    shin_contact = False
    non_foot_contact = False
    crashed = False

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
        foot_force, shin_force, current_non_foot_contact = contact_map.ground_contact_forces(model, data)
        non_foot_contact = non_foot_contact or current_non_foot_contact
        peak_foot_force = max(peak_foot_force, foot_force)
        peak_shin_force = max(peak_shin_force, shin_force)
        if foot_force > 0.0:
            foot_contact = True
        # Measure impact acceleration only once landing has started; this
        # excludes the constant gravitational acceleration during free fall.
        if foot_contact or shin_force > 0.0:
            body_acceleration = float(np.linalg.norm(data.qacc[0:3]))
            peak_body_acceleration = max(peak_body_acceleration, body_acceleration)
        if shin_force > 0.0:
            shin_contact = True

    # Minimize total motor-current proxy subject to survival. Including the
    # spine prevents the unlocked model from shifting load there for free.
    cost = total_leg_effort + total_spine_effort
    if not foot_contact:
        cost += 1_000_000.0
    if non_foot_contact:
        cost += 1_000_000.0
    if crashed:
        penetration = max(0.0, 0.05 - min_body_height)
        cost += 1_000_000.0 + penetration * 1_000_000.0

    return (
        cost,
        crashed,
        foot_contact,
        non_foot_contact,
        min_body_height,
        peak_leg_torque,
        shin_contact,
        peak_foot_force,
        peak_shin_force,
        peak_body_acceleration,
        total_leg_effort,
        total_spine_effort,
        peak_leg_current_proxy,
        peak_spine_current_proxy,
    )


def show_best(model, data, jmap, params, steps, start_height):
    kp, kd = build_gains(params)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            targets = reset_drop(data, model, jmap, start_height)
            for _ in range(steps):
                if not viewer.is_running():
                    return
                step_start = time.time()
                controlled_step(model, data, jmap, targets, kp, kd)
                viewer.sync()
                remaining = model.opt.timestep - (time.time() - step_start)
                if remaining > 0.0:
                    time.sleep(remaining)
            time.sleep(1.0)


def sample_params(rng):
    return tuple(rng.uniform(PARAMETER_BOUNDS[:, 0], PARAMETER_BOUNDS[:, 1]))


def result_from_params(
    model, data, jmap, contact_map, params, steps, start_height
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
        peak_shin_force,
        peak_body_acceleration,
        total_leg_effort,
        total_spine_effort,
        peak_leg_current_proxy,
        peak_spine_current_proxy,
    ) = evaluate_drop(
        model, data, jmap, contact_map, params, steps, start_height,
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
        "peak_shin_force": peak_shin_force,
        "peak_body_acceleration": peak_body_acceleration,
        "total_leg_effort": total_leg_effort,
        "total_spine_effort": total_spine_effort,
        "total_motor_effort": total_leg_effort + total_spine_effort,
        "peak_leg_current_proxy": peak_leg_current_proxy,
        "peak_spine_current_proxy": peak_spine_current_proxy,
        "params": params,
    }


def init_worker(lock_spine, start_height):
    global _WORKER_MODEL, _WORKER_DATA, _WORKER_JMAP
    global _WORKER_CONTACT_MAP, _WORKER_STEPS, _WORKER_START_HEIGHT
    _WORKER_MODEL = mujoco.MjModel.from_xml_path(str(XML_PATH))
    configure_spine(_WORKER_MODEL, lock_spine)
    _WORKER_DATA = mujoco.MjData(_WORKER_MODEL)
    _WORKER_JMAP = JointMap(_WORKER_MODEL, NOMINAL_POSE)
    _WORKER_CONTACT_MAP = ContactMap(_WORKER_MODEL)
    _WORKER_STEPS = episode_steps(_WORKER_MODEL, start_height)
    _WORKER_START_HEIGHT = start_height


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
    x_values, costs, count, candidate_pool, rng, exploration_fraction, xi
):
    lower = PARAMETER_BOUNDS[:, 0]
    span = PARAMETER_BOUNDS[:, 1] - lower
    x_normalized = (np.asarray(x_values) - lower) / span
    # The log transform separates hard failure penalties from feasible costs.
    y = np.log1p(np.asarray(costs))
    train_x, train_y = select_gp_training_data(x_normalized, y, rng)
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2))
        * Matern(length_scale=np.ones(PARAMETER_BOUNDS.shape[0]), nu=2.5)
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

    pool = rng.random((candidate_pool, PARAMETER_BOUNDS.shape[0]))
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
        selected.append(rng.random(PARAMETER_BOUNDS.shape[0]))
    while len(selected) < count:
        selected.append(rng.random(PARAMETER_BOUNDS.shape[0]))
    return lower + np.asarray(selected) * span


def run_search(
    trials, workers, seed, initial_trials, candidate_pool, batch_size,
    exploration_fraction, xi, lock_spine, start_height,
):
    """Bayesian optimization of total actuator effort under survival constraints."""
    workers = min(workers, trials)
    rng = np.random.default_rng(seed)
    initial_trials = min(initial_trials, trials)
    x_values = []
    costs = []
    best = None
    safe_trials = 0
    stagnant_batches = 0
    batch = [sample_params(rng) for _ in range(initial_trials)]

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(lock_spine, start_height),
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
            if completed < trials:
                batch = propose_candidates(
                    np.asarray(x_values), costs,
                    min(batch_size, trials - completed), candidate_pool,
                    rng, current_exploration, xi,
                )
    best["safe_trials"] = safe_trials
    best["evaluated_trials"] = trials
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
        help="drop heights for --compare-spine (default: 1 2 3 4 5)",
    )
    parser.add_argument("--no-viewer", action="store_true", help="finish without opening the 3D viewer")
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
    if args.heights is not None and not args.compare_spine:
        raise ValueError("--heights can only be used with --compare-spine")
    if args.heights is not None and any(height <= 0.0 for height in args.heights):
        raise ValueError("all --heights values must be positive")
    return batch_size


def print_result(best, model):
    print(
        f"TOP WYNIK - Koszt: {best['cost']:.2f} | "
        f"Wysokosc bazy: {best['min_height']:.3f}m | Safe: {is_safe(best)} | "
        f"Kontakt inny niz stopa: {best['non_foot_contact']}"
    )
    print(f"Najlepsze Kp (Hip, Thigh, Calf): {best['params'][0]:.1f}, {best['params'][1]:.1f}, {best['params'][2]:.1f}")
    print(f"Najlepsze Kd (Hip, Thigh, Calf): {best['params'][3]:.1f}, {best['params'][4]:.1f}, {best['params'][5]:.1f}")
    print(f"Szczytowa sila stop: {best['peak_foot_force']:.1f} N")
    print(f"Szczytowa sila lydki: {best['peak_shin_force']:.1f} N")
    print(f"Szczytowy moment stawow nog: {best['peak_leg_torque']:.2f} Nm")
    print(f"Calka kwadratu komend wszystkich silnikow: {best['total_motor_effort']:.3f}")
    print(f"Calka kwadratu komend silnikow nog: {best['total_leg_effort']:.3f}")
    print(f"Calka kwadratu komendy kregoslupa: {best['total_spine_effort']:.3f}")
    print(f"Szczytowa komenda silnika nogi: {best['peak_leg_current_proxy']:.3f}")
    print(f"Szczytowa komenda silnika kregoslupa: {best['peak_spine_current_proxy']:.3f}")
    print(
        f"Szczytowe przyspieszenie korpusu: {best['peak_body_acceleration']:.1f} m/s^2 "
        f"({best['peak_body_acceleration'] / abs(model.opt.gravity[2]):.1f} g)"
    )


def optimize_configuration(args, batch_size, lock_spine, start_height, seed):
    active_workers = min(args.workers, args.trials)
    label = "zablokowany" if lock_spine else "aktywny"
    print(
        f"\nWysokosc: {start_height:g} m | kregoslup: {label} | "
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
    )
    print(f"Zakonczono konfiguracje w {time.time() - start_time:.2f} s.")
    return best


def is_safe(result):
    return (
        result["foot_contact"]
        and not result["crashed"]
        and not result["non_foot_contact"]
    )


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


def main():
    args = parse_args()
    batch_size = validate_args(args)

    if args.compare_spine:
        heights = sorted(set(args.heights or [1.0, 2.0, 3.0, 4.0, 5.0]))
        # A concrete shared seed makes the two modes directly comparable even
        # when the caller did not request repeatability explicitly.
        seed = args.seed
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])
        print(f"Shared comparison seed: {seed}")
        results = {}
        for height in heights:
            results[height] = {}
            for lock_spine in (False, True):
                results[height][lock_spine] = optimize_configuration(
                    args, batch_size, lock_spine, height, seed
                )
        print_comparison(results)
        viewer_height = heights[-1]
        highest_modes = results[viewer_height]
        winner = comparison_winner(highest_modes[False], highest_modes[True])
        if winner in ("NO SAFE", "TIE"):
            viewer_lock = min(
                (False, True),
                key=lambda locked: (
                    not is_safe(highest_modes[locked]),
                    highest_modes[locked]["cost"],
                ),
            )
        else:
            viewer_lock = winner == "LOCKED"
        best = results[viewer_height][viewer_lock]
        print(
            f"\nHighest-height {'best candidate' if winner in ('NO SAFE', 'TIE') else 'winner'} details "
            f"({viewer_height:g} m, "
            f"{'LOCKED' if viewer_lock else 'UNLOCKED'}):"
        )
        model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        configure_spine(model, viewer_lock)
        print_result(best, model)
        start_height = viewer_height
    else:
        start_height = args.height
        best = optimize_configuration(
            args, batch_size, args.lock_spine, start_height, args.seed
        )
        model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        configure_spine(model, args.lock_spine)
        print_result(best, model)

    if not args.no_viewer:
        print("\nOtwieram symulacje 3D... Zamknij okno, by zakonczyc skrypt.")
        data = mujoco.MjData(model)
        jmap = JointMap(model, NOMINAL_POSE)
        steps = episode_steps(model, start_height)
        show_best(model, data, jmap, best["params"], steps, start_height)


if __name__ == "__main__":
    main()
