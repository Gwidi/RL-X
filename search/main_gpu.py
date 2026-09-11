"""GPU/MJX variant of main.py.

The Bayesian search, costs, reports and viewer are imported from ``main.py``.
Only the batch evaluator is replaced: candidate rollouts are vectorized and
executed by MuJoCo MJX through JAX on one GPU.
"""

import os
import time

# XLA's legacy latency estimator does not contain an H100 SoL table and emits
# the same harmless warning for many kernels.  Keep errors visible while
# making direct ``python main_gpu.py`` runs as quiet as the Slurm launcher.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

import jax

# MJX is designed primarily for float32.  Keeping every contact solve in
# float64 roughly doubles memory traffic and is significantly slower even on
# datacenter GPUs.  Float64 remains available for CPU/MJX parity checks with
# GPU_ENABLE_X64=1.
GPU_ENABLE_X64 = os.environ.get("GPU_ENABLE_X64", "0").lower() in {
    "1", "true", "yes", "on",
}
jax.config.update("jax_enable_x64", GPU_ENABLE_X64)
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

import main as cpu


GPU_NUMPY_DTYPE = np.float64 if GPU_ENABLE_X64 else np.float32
GPU_SOLVER_ITERATIONS = 30
GPU_LINESEARCH_ITERATIONS = 15
_EVALUATOR_CACHE = {}


RESULT_KEYS = (
    "cost",
    "crashed",
    "foot_contact",
    "non_foot_contact",
    "min_height",
    "peak_leg_torque",
    "shin_contact",
    "peak_foot_force",
    "peak_shin_force",
    "peak_body_acceleration",
    "total_leg_effort",
    "total_spine_effort",
    "peak_leg_current_proxy",
    "peak_spine_current_proxy",
)


def require_gpu():
    """Return the first JAX GPU, failing instead of silently using the CPU."""
    try:
        devices = jax.devices("gpu")
    except RuntimeError as exc:
        raise RuntimeError(
            "JAX nie widzi GPU. Na wezle GPU zainstaluj wariant JAX zgodny z "
            "CUDA (np. `pip install \"jax[cuda13]\"`) i sprawdz przydzial GPU."
        ) from exc
    if not devices:
        raise RuntimeError("JAX nie wykryl zadnego GPU; obliczenia nie zostana uruchomione na CPU.")
    return devices[0]


def _prepare_model(lock_spine):
    """Load the same model and make its collision graph compatible with MJX."""
    model = mujoco.MjModel.from_xml_path(str(cpu.XML_PATH))
    cpu.configure_spine(model, lock_spine)

    # The C solver exits as soon as it converges, while MJX benefits greatly
    # from tight static limits.  Profiling this model showed at most 17 Newton
    # iterations; 30/15 retained the CPU results in the validation set.
    model.opt.iterations = min(model.opt.iterations, GPU_SOLVER_ITERATIONS)
    model.opt.ls_iterations = min(
        model.opt.ls_iterations, GPU_LINESEARCH_ITERATIONS
    )

    # MJX builds all collision pairs ahead of time and cannot compile the
    # cylinder-box self-collision pairs present in this XML.  Separate just
    # those two groups while retaining their ground and supported self
    # contacts.  Geoms which had contact disabled remain disabled.
    ground_id = model.geom("ground_2").id
    enabled = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    for geom_id in np.flatnonzero(enabled):
        geom_type = model.geom_type[geom_id]
        if geom_id == ground_id:
            bit, affinity = 1, 2 | 4 | 8
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            bit, affinity = 2, 1 | 2 | 8
        elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            bit, affinity = 4, 1 | 4 | 8
        else:
            bit, affinity = 8, 1 | 2 | 4 | 8
        model.geom_contype[geom_id] = bit
        model.geom_conaffinity[geom_id] = affinity

    # Visual meshes have contact disabled and every body has an explicit
    # inertial.  Replacing them only in the MJX copy avoids unsupported convex
    # mesh preprocessing without changing dynamics, contacts or the viewer.
    mesh_ids = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
    model.geom_type[mesh_ids] = mujoco.mjtGeom.mjGEOM_SPHERE
    model.geom_size[mesh_ids, 0] = 1e-9
    return model


class GpuBatchEvaluator:
    """Compile and evaluate fixed-size batches of complete drop episodes."""

    def __init__(
        self, lock_spine, steps, optimize_nominal_position, gpu_batch_size,
        device,
    ):
        self.batch_size = gpu_batch_size
        self.cpu_model = _prepare_model(lock_spine)
        self.jmap = cpu.JointMap(self.cpu_model, cpu.NOMINAL_POSE)
        self.contact_map = cpu.ContactMap(self.cpu_model)
        self.steps = steps
        self.timestep = float(self.cpu_model.opt.timestep)
        self.lock_spine = lock_spine
        self.optimize_nominal_position = optimize_nominal_position

        self.model = mjx.put_model(self.cpu_model, device=device)
        self.empty_data = mjx.make_data(self.model, device=device)
        evaluate_one = self._build_episode()
        self.evaluate_batch = jax.jit(
            jax.vmap(evaluate_one, in_axes=(0, None)), device=device
        )

    def _build_episode(self):
        model = self.model
        empty_data = self.empty_data
        jmap = self.jmap
        lock_spine = self.lock_spine
        optimize_positions = self.optimize_nominal_position
        timestep = self.timestep

        names = jmap.names
        qpos_idx = jnp.asarray([jmap.qpos_adr[name] for name in names])
        dof_idx = jnp.asarray([jmap.dof_adr[name] for name in names])
        act_idx = jnp.asarray([jmap.act_id[name] for name in names])
        gear = jnp.asarray([jmap.gear[name] for name in names])
        ctrl_low = jnp.asarray(self.cpu_model.actuator_ctrlrange[:, 0])[act_idx]
        ctrl_high = jnp.asarray(self.cpu_model.actuator_ctrlrange[:, 1])[act_idx]
        nominal = jnp.asarray([cpu.NOMINAL_POSE[name] for name in names])
        leg_mask = jnp.asarray([name != "sp_j0" for name in names])
        spine_index = names.index("sp_j0")

        group_idx = jnp.asarray([
            0 if name.endswith("j0") else 1 if name.endswith("j1") else 2
            for name in names
        ])
        position_names = cpu.nominal_position_names(optimize_positions, lock_spine)
        position_target_idx = jnp.asarray([names.index(name) for name in position_names])
        position_start = cpu.gain_parameter_count(lock_spine)

        contact = empty_data._impl.contact
        addresses = np.asarray(contact.efc_address)
        dimensions = np.asarray(contact.dim)
        if np.any(dimensions != 3):
            raise ValueError("Ten wariant GPU wymaga condim=3 dla wszystkich kontaktow.")
        force_idx = jnp.asarray(addresses[:, None] + np.arange(4)[None, :])
        support_ids = jnp.asarray(sorted(self.contact_map.support_geom_ids))
        foot_ids = jnp.asarray(sorted(self.contact_map.foot_geom_ids))
        shin_ids = jnp.asarray(sorted(self.contact_map.shin_geom_ids))

        def contact_metrics(data):
            geoms = data._impl.contact.geom
            # MJX reserves static contact slots.  A slot can already contain
            # geom IDs while the pair is still separated, so distance (the
            # same inclusion test used by MuJoCo's ncon) decides activity.
            active = (
                (geoms[:, 0] >= 0)
                & (data._impl.contact.dist <= data._impl.contact.includemargin)
            )
            support = jnp.any(geoms[:, :, None] == support_ids, axis=(1, 2))
            foot = jnp.any(geoms[:, :, None] == foot_ids, axis=(1, 2))
            shin = jnp.any(geoms[:, :, None] == shin_ids, axis=(1, 2))
            # Equivalent of abs(mj_contactForce(...)[0]) for the pyramidal
            # condim=3 contacts used by intention.xml.
            normal = jnp.abs(jnp.sum(data._impl.efc_force[force_idx], axis=1))
            valid = active & support
            return (
                jnp.sum(jnp.where(valid & foot, normal, 0.0)),
                jnp.sum(jnp.where(valid & shin, normal, 0.0)),
                jnp.any(valid & ~foot),
            )

        def episode(params, start_height):
            targets = nominal
            if optimize_positions:
                targets = targets.at[position_target_idx].add(params[position_start:])

            kp = params[group_idx]
            kd = params[group_idx + 3]
            if lock_spine:
                kp = kp.at[spine_index].set(0.0)
                kd = kd.at[spine_index].set(0.0)
            else:
                kp = kp.at[spine_index].set(params[6])
                kd = kd.at[spine_index].set(params[7])

            qpos = jnp.asarray(self.cpu_model.qpos0)
            qpos = qpos.at[0].set(0.0)
            qpos = qpos.at[1].set(0.0)
            qpos = qpos.at[2].set(start_height)
            qpos = qpos.at[3:7].set(jnp.asarray([1.0, 0.0, 0.0, 0.0]))
            qpos = qpos.at[qpos_idx].set(targets)
            data = empty_data.replace(
                qpos=qpos,
                qvel=jnp.zeros_like(empty_data.qvel),
                ctrl=jnp.zeros_like(empty_data.ctrl),
            )
            data = mjx.forward(model, data)

            # min height, four peaks, two efforts, two current peaks, and four flags
            initial_stats = (
                jnp.inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                jnp.asarray(False), jnp.asarray(False), jnp.asarray(False),
                jnp.asarray(False),
            )

            def step(carry, _):
                data, stats = carry
                q = data.qpos[qpos_idx]
                qd = data.qvel[dof_idx]
                torque = kp * (targets - q) - kd * qd
                torque = jnp.where(jnp.abs(torque) < cpu.STATIC_FRICTION, 0.0, torque)
                command = jnp.clip(torque / gear, ctrl_low, ctrl_high)
                data = data.replace(ctrl=data.ctrl.at[act_idx].set(command))
                data = mjx.step(model, data)

                (
                    min_height, peak_torque, peak_foot_force, peak_shin_force,
                    peak_accel, leg_effort, spine_effort, peak_leg_current,
                    peak_spine_current, foot_contact, shin_contact,
                    non_foot_contact, crashed,
                ) = stats
                foot_force, shin_force, current_non_foot = contact_metrics(data)
                now_foot = foot_force > 0.0
                now_shin = shin_force > 0.0
                landing = foot_contact | now_foot | now_shin
                body_accel = jnp.linalg.norm(data.qacc[0:3])
                joint_torque = data.qfrc_actuator[dof_idx]
                leg_command = jnp.where(leg_mask, command, 0.0)
                return (data, (
                    jnp.minimum(min_height, data.qpos[2]),
                    jnp.maximum(peak_torque, jnp.max(jnp.abs(jnp.where(leg_mask, joint_torque, 0.0)))),
                    jnp.maximum(peak_foot_force, foot_force),
                    jnp.maximum(peak_shin_force, shin_force),
                    jnp.maximum(peak_accel, jnp.where(landing, body_accel, 0.0)),
                    leg_effort + jnp.sum(jnp.square(leg_command)) * timestep,
                    spine_effort + command[spine_index] ** 2 * timestep,
                    jnp.maximum(peak_leg_current, jnp.max(jnp.abs(leg_command))),
                    jnp.maximum(peak_spine_current, jnp.abs(command[spine_index])),
                    foot_contact | now_foot,
                    shin_contact | now_shin,
                    non_foot_contact | current_non_foot,
                    crashed | (data.qpos[2] < 0.05),
                )), None

            (_, stats), _ = jax.lax.scan(
                step, (data, initial_stats), xs=None, length=self.steps
            )
            (
                min_height, peak_torque, peak_foot_force, peak_shin_force,
                peak_accel, leg_effort, spine_effort, peak_leg_current,
                peak_spine_current, foot_contact, shin_contact,
                non_foot_contact, crashed,
            ) = stats

            cost = leg_effort + spine_effort
            if optimize_positions:
                cost += cpu.NOMINAL_POSITION_REGULARIZATION * jnp.sum(
                    jnp.square(params[position_start:])
                )
            cost += jnp.where(foot_contact, 0.0, cpu.LANDING_QUALITY_PENALTY)
            cost += jnp.where(non_foot_contact, cpu.LANDING_QUALITY_PENALTY, 0.0)
            penetration = jnp.maximum(0.0, 0.05 - min_height)
            cost += jnp.where(
                crashed,
                cpu.LANDING_QUALITY_PENALTY * (1.0 + penetration),
                0.0,
            )
            foot_excess = peak_foot_force / cpu.MAX_LANDING_FOOT_FORCE - 1.0
            cost += jnp.where(
                peak_foot_force > cpu.MAX_LANDING_FOOT_FORCE,
                cpu.LANDING_QUALITY_PENALTY * (1.0 + foot_excess),
                0.0,
            )
            accel_excess = peak_accel / cpu.MAX_LANDING_BODY_ACCELERATION - 1.0
            cost += jnp.where(
                peak_accel > cpu.MAX_LANDING_BODY_ACCELERATION,
                cpu.LANDING_QUALITY_PENALTY * (1.0 + accel_excess),
                0.0,
            )
            return jnp.asarray((
                cost, crashed, foot_contact, non_foot_contact, min_height,
                peak_torque, shin_contact, peak_foot_force, peak_shin_force,
                peak_accel, leg_effort, spine_effort, peak_leg_current,
                peak_spine_current,
            ))

        return episode

    def __call__(self, params, start_height):
        """Evaluate a possibly short batch while retaining one compiled shape."""
        params = np.asarray(params, dtype=GPU_NUMPY_DTYPE)
        count = len(params)
        if count > self.batch_size:
            raise ValueError("batch is larger than the compiled GPU batch")
        if count < self.batch_size:
            padding = np.repeat(params[-1:, :], self.batch_size - count, axis=0)
            params = np.concatenate((params, padding), axis=0)
        values = np.asarray(
            self.evaluate_batch(jnp.asarray(params), jnp.asarray(start_height))
        )[:count]
        results = []
        for point, row in zip(params[:count], values):
            result = dict(zip(RESULT_KEYS, row))
            for key in ("crashed", "foot_contact", "non_foot_contact", "shin_contact"):
                result[key] = bool(result[key])
            for key in RESULT_KEYS:
                if key not in ("crashed", "foot_contact", "non_foot_contact", "shin_contact"):
                    result[key] = float(result[key])
            result["total_motor_effort"] = result["total_leg_effort"] + result["total_spine_effort"]
            result["params"] = tuple(float(value) for value in point)
            results.append(result)
        return results


def get_evaluator(
    lock_spine, start_height, optimize_nominal_position, batch_size, device,
):
    """Reuse expensive JIT compilations across equal-length drop episodes."""
    reference_model = _prepare_model(lock_spine)
    steps = cpu.episode_steps(reference_model, start_height)
    key = (
        lock_spine,
        optimize_nominal_position,
        batch_size,
        steps,
        device.platform,
        device.id,
    )
    evaluator = _EVALUATOR_CACHE.get(key)
    if evaluator is None:
        evaluator = GpuBatchEvaluator(
            lock_spine, steps, optimize_nominal_position, batch_size, device
        )
        _EVALUATOR_CACHE[key] = evaluator
    return evaluator


def run_search_gpu(
    trials, workers, seed, initial_trials, candidate_pool, batch_size,
    exploration_fraction, xi, lock_spine, start_height,
    optimize_nominal_position,
):
    """The same Bayesian loop as main.py, with batched MJX evaluation."""
    del workers  # retained in the CLI for compatibility with existing jobs
    rng = np.random.default_rng(seed)
    initial_trials = min(initial_trials, trials)
    bounds = cpu.parameter_bounds(optimize_nominal_position, lock_spine)
    x_values, costs, progress = [], [], []
    best = None
    safe_trials = 0
    stagnant_batches = 0
    batch = [cpu.sample_params(rng, bounds) for _ in range(initial_trials)]
    device = require_gpu()
    initial_evaluator = get_evaluator(
        lock_spine, start_height, optimize_nominal_position, initial_trials,
        device,
    )
    regular_batch_size = min(batch_size, trials)
    regular_evaluator = get_evaluator(
        lock_spine, start_height, optimize_nominal_position,
        regular_batch_size, device,
    )

    completed = 0
    while completed < trials:
        previous_best_cost = float("inf") if best is None else best["cost"]
        batch = batch[:trials - completed]
        evaluator = initial_evaluator if completed == 0 else regular_evaluator
        evaluation_start = time.perf_counter()
        for result in evaluator(batch, start_height):
            x_values.append(result["params"])
            costs.append(result["cost"])
            safe_trials += int(cpu.is_safe(result))
            if best is None or result["cost"] < best["cost"]:
                best = result
        evaluation_seconds = time.perf_counter() - evaluation_start
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
            f"bezpieczne: {safe_trials} | eksploracja: {current_exploration:.0%} | "
            f"GPU: {evaluation_seconds:.2f}s",
            flush=True,
        )
        progress.append({
            "completed": completed,
            "best_cost": best["cost"],
            "safe_trials": safe_trials,
        })
        if completed < trials:
            proposal_start = time.perf_counter()
            batch = cpu.propose_candidates(
                np.asarray(x_values), costs,
                min(batch_size, trials - completed), candidate_pool,
                rng, current_exploration, xi, bounds,
            )
            print(
                f"Dopasowanie GP i wybor kandydatow: "
                f"{time.perf_counter() - proposal_start:.2f}s",
                flush=True,
            )
    best["safe_trials"] = safe_trials
    best["evaluated_trials"] = trials
    best["progress"] = progress
    return best


def optimize_configuration_gpu(
    args, batch_size, lock_spine, start_height, seed,
    optimize_nominal_position,
):
    label = "zablokowany" if lock_spine else "aktywny"
    device = require_gpu()
    print(
        f"\nWysokosc: {start_height:g} m | kregoslup: {label} | "
        f"pozycja nominalna: "
        f"{'optymalizowana' if optimize_nominal_position else 'stala'} | "
        f"{args.trials} prob, GPU: {device}, "
        f"precyzja: {'float64' if GPU_ENABLE_X64 else 'float32'}",
        flush=True,
    )
    start = time.time()
    best = run_search_gpu(
        args.trials, args.workers, seed, args.initial_trials,
        args.candidate_pool, batch_size, args.exploration, args.xi,
        lock_spine, start_height, optimize_nominal_position,
    )
    print(f"Zakonczono konfiguracje w {time.time() - start:.2f} s.")
    return best


def main():
    # Reuse the complete CLI/report/plot/viewer flow.  Python resolves these
    # two replaced globals when cpu.main() invokes the optimization.
    args = cpu.parse_args()
    require_gpu()
    cpu.run_search = run_search_gpu
    cpu.optimize_configuration = optimize_configuration_gpu
    cpu.main(args)


if __name__ == "__main__":
    main()
