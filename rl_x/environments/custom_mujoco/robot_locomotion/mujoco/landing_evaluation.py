"""Paired landing evaluation, independent of the training curriculum."""

from collections import deque
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np


def evaluation_heights(config):
    if config.get("height_reference", "clearance") not in ("clearance", "base"):
        raise ValueError("landing_evaluation.height_reference must be 'clearance' or 'base'")
    from .domain_randomization.initial_state_functions.horizontal_velocity import validate_velocity_sampling
    validate_velocity_sampling(config.get("velocity_sampling", "component_box"))
    values = config["heights"]
    if isinstance(values, str):
        values = values.split(",")
    heights = [float(value) for value in values]
    if not heights or any(not np.isfinite(h) or h <= 0 for h in heights):
        raise ValueError("landing_evaluation.heights must contain positive finite heights")
    if len(set(heights)) != len(heights):
        raise ValueError("landing_evaluation.heights must be unique")
    for key in ("max_angle_deg", "max_lateral_velocity", "randomization_coefficient"):
        value = float(config[key])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"landing_evaluation.{key} must be finite and non-negative")
    if not 0 <= config["randomization_coefficient"] <= 1:
        raise ValueError("landing_evaluation.randomization_coefficient must be in [0, 1]")
    if int(config["frequency_multiplier"]) != config["frequency_multiplier"] or config["frequency_multiplier"] < 1:
        raise ValueError("landing_evaluation.frequency_multiplier must be a positive integer")
    if int(config["seed"]) != config["seed"] or config["seed"] < 0:
        raise ValueError("landing_evaluation.seed must be a non-negative integer")
    return heights


class LandingLoads:
    """Capture foot/proximal calf peaks at every physics step.

    The permitted distal calf third is excluded, as in LandingSafetyMonitor.
    """

    def __init__(self, model, feet):
        self.groups = {"foot": {int(g): i for i, g in enumerate(feet)}, "calf": {}}
        self.calf_names = []
        for leg in ("RL", "RR", "FR", "FL"):
            geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_calf")
            if geom >= 0:
                self.groups["calf"][geom] = len(self.calf_names)
                self.calf_names.append(leg)
        self.floor = model.geom("floor").id
        self.window = max(1, int(np.ceil(0.010 / model.opt.timestep)))
        self.history = {group: deque(maxlen=self.window) for group in self.groups}
        self.raw = {group: np.zeros(len(ids)) for group, ids in self.groups.items()}
        self.average = {group: values.copy() for group, values in self.raw.items()}
        self.calf_contact_duration = 0.0

    def update(self, model, data):
        forces = {group: np.zeros(len(ids)) for group, ids in self.groups.items()}
        force = np.zeros(6)
        for index, contact in enumerate(data.contact):
            pair = {int(contact.geom1), int(contact.geom2)}
            if self.floor not in pair:
                continue
            geom = next((g for g in pair if g != self.floor), -1)
            mujoco.mj_contactForce(model, data, index, force)
            for group, ids in self.groups.items():
                if geom in ids:
                    forces[group][ids[geom]] += abs(float(force[0]))
        for group, values in forces.items():
            self.history[group].append(values)
            self.raw[group] = np.maximum(self.raw[group], values)
            # Zero padding before the first complete 10 ms window.
            mean = np.sum(self.history[group], axis=0) / self.window
            self.average[group] = np.maximum(self.average[group], mean)
        self.calf_contact_duration += float(np.any(forces["calf"] > 1.0)) * model.opt.timestep

    def metrics(self):
        result = {"calf_contact_duration": self.calf_contact_duration}
        for group in self.groups:
            result[f"peak_{group}_force_raw"] = float(np.max(self.raw[group], initial=0.0))
            result[f"peak_{group}_force_10ms"] = float(np.max(self.average[group], initial=0.0))
        for i, leg in enumerate(self.calf_names):
            result[f"peak_calf_force_raw_{leg}"] = float(self.raw["calf"][i])
            result[f"peak_calf_force_10ms_{leg}"] = float(self.average["calf"][i])
        return result


class LandingEvaluationWrapper(gym.Wrapper):
    """Recreate each seeded case regardless of earlier actions or episode lengths."""

    def __init__(self, env, worker_index):
        super().__init__(env)
        self.worker_index = worker_index
        self.case = None
        self.template = deepcopy(env.internal_state)
        self.original_randomization = env.handle_domain_randomization
        env.handle_domain_randomization = self._randomize

    def configure_landing_evaluation(self, case):
        self.case = dict(case)
        self.next_episode = 0

    def _randomize(self, is_episode_start=False):
        if self.case is None:
            return self.original_randomization(is_episode_start=is_episode_start)
        # Freeze physical parameters for the whole case. External disturbances
        # here mean initial angles/velocities, not additional random midair kicks.
        if is_episode_start:
            env = self.env
            env.domain_randomization_unseen_robot_function.sample()
            env.domain_randomization_seen_robot_function.sample()
            env.domain_randomization_mujoco_model_function.sample()
            env.domain_randomization_action_delay_function.sample()
            env.joint_dropout_function.sample()
            env.reward_function.handle_model_change()

    def reset(self, *, seed=None, options=None):
        if self.case is None:
            return self.env.reset(seed=seed)
        env = self.env
        self.episode = self.next_episode
        self.next_episode += 1
        self.case_seed = [int(self.case["seed"]), self.worker_index, self.episode]
        env.np_rng = np.random.default_rng(np.random.SeedSequence(self.case_seed))
        env.internal_state = deepcopy(self.template)
        env.env_curriculum_enabled = False
        perturbed = self.case["perturbed"]
        coefficient = self.case["randomization_coefficient"] if perturbed and self.case["randomize_physics"] else 0.0
        env.internal_state["env_curriculum_coeff"] = coefficient
        env.internal_state["in_eval_mode"] = False
        initial = env.initial_state_function
        initial.height_reference = self.case.get("height_reference", "clearance")
        initial.start_height = initial.target_height = self.case["height"]
        initial.velocity_sampling = self.case.get("velocity_sampling", "component_box")
        initial.min_angle_deg = initial.max_angle_deg = self.case["max_angle_deg"] if perturbed else 0.0
        initial.min_lateral_velocity = initial.max_lateral_velocity = self.case["max_lateral_velocity"] if perturbed else 0.0
        initial.replay_easy_probability = 0.0
        curriculum = initial._get_curriculum_state()
        curriculum.update(fixed_difficulty=True, difficulty=1.0, replay_easy=False)
        observation, info = env.reset()
        self.step_index = 0
        data, model = env.internal_state["data"], env.internal_state["mj_model"]
        fingerprint = hashlib.sha256()
        for value in (data.qpos, data.qvel, model.body_mass, model.body_inertia,
                      model.body_pos, model.body_quat, model.body_ipos, model.body_iquat,
                      model.geom_size, model.geom_pos, model.geom_friction, model.pair_friction,
                      model.jnt_axis, model.jnt_range, model.dof_damping, model.dof_armature,
                      model.dof_frictionloss, model.site_pos,
                      model.actuator_gainprm, model.actuator_biasprm,
                      env.internal_state["actuator_joint_nominal_positions"]):
            fingerprint.update(np.asarray(value).tobytes())
        physics_sha = env.env_config.get("landing_physics_sha256", "")
        if physics_sha:
            fingerprint.update(physics_sha.encode())
            fingerprint.update(np.array([model.opt.timestep, int(model.opt.integrator)], dtype=np.float64).tobytes())
        from .policy_velocity import velocity_observation_contract
        velocity_contract = velocity_observation_contract(env.env_config.get("policy_velocity", {}))
        self.case_info = dict(worker=self.worker_index, episode=self.episode,
                              seed=self.case["seed"], scenario_hash=fingerprint.hexdigest(),
                              velocity_sampling=initial.velocity_sampling,
                              policy_velocity_mode=velocity_contract['mode'],
                              policy_velocity_contract=json.dumps(velocity_contract, sort_keys=True),
                              physics_profile_sha256=physics_sha,
                              physics_timestep=float(model.opt.timestep), physics_integrator=int(model.opt.integrator),
                              base_z=float(data.qpos[2]),
                              height_reference=initial.height_reference,
                              initial_qpos=data.qpos.tolist(), initial_qvel=data.qvel.tolist())
        env._landing_evaluation_loads = LandingLoads(model, env.foot_geom_indices)
        return observation, info

    def step(self, action):
        if self.case is not None:
            # Align sensor/action-delay randomness by simulation time, even if
            # policy and baseline consume different numbers of random draws.
            self.env.np_rng = np.random.default_rng(np.random.SeedSequence(
                self.case_seed + [self.step_index + 1]))
            self.step_index += 1
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self.case is not None and (terminated or truncated):
            info = dict(info)
            info["landing_evaluation_case"] = dict(self.case_info)
            info["landing_evaluation_loads"] = self.env._landing_evaluation_loads.metrics()
        return observation, reward, terminated, truncated, info


def _episode(final_info):
    result = {"episode_return": float(final_info["episode_return"]),
              "episode_length": float(final_info["episode_length"]),
              "success_rate": float(final_info.get("curriculum/landing_success", 0.0)),
              **final_info["landing_evaluation_loads"]}
    for reason in ("base_crash", "motor_hard_limit", "thermal_limit",
                   "disallowed_contact", "calf_force", "foot_force", "body_acceleration", "base_height"):
        result[f"failure_{reason}_rate"] = float(final_info.get(f"outcome/failure_{reason}", 0.0))
    result["unresolved_rate"] = float(final_info.get("outcome/unresolved", 1.0))
    for key in ("peak_body_acceleration", "peak_body_acceleration_raw", "peak_base_impact_acceleration",
                "peak_leg_motor_torque", "peak_spine_motor_torque",
                "leg_tau_squared_integral", "spine_tau_squared_integral"):
        result[key] = float(final_info.get(f"metrics/{key}", 0.0))
    for name in ("safety_peak_body_acceleration", "safety_disallowed_contact", "safety_min_base_height"):
        result[name] = float(final_info.get(f"metrics/{name}", 0.0))
    return result


def evaluate_landing_suite(env, deterministic_action, config, episodes, output_dir, global_step):
    """Run four conditions; quotas prevent selection of only fast failures."""
    if config.get("velocity_grid", ""):
        from .landing_velocity_grid import evaluate_velocity_grid
        return evaluate_velocity_grid(env, deterministic_action, config, episodes, output_dir, global_step)
    heights = evaluation_heights(config)
    if episodes < 1:
        raise ValueError("evaluation_episodes must be positive")
    metrics, rows, summaries = {}, [], []
    conditions = (("nominal", "angles", "velocity", "perturbed")
                  if config.get("factorized_conditions", False) else ("nominal", "perturbed"))
    conditions = config.get("_conditions", conditions)
    for height in heights:
        height_tag = f"h_{height:g}".replace(".", "p")
        for condition in conditions:
            results = {}
            for controller in ("policy", "zero"):
                case = dict(config, height=height, perturbed=condition != "nominal")
                if condition == "angles":
                    case["max_lateral_velocity"] = 0.
                elif condition == "velocity":
                    case["max_angle_deg"] = 0.
                env.call("configure_landing_evaluation", case)
                state, _ = env.reset()
                workers = len(state)
                quotas = np.full(workers, episodes // workers, dtype=int)
                quotas[:episodes % workers] += 1
                completed = np.zeros(workers, dtype=int)
                records = {}
                while np.any(completed < quotas):
                    action = (np.asarray(deterministic_action(state)) if controller == "policy"
                              else np.zeros((workers,) + env.single_action_space.shape, dtype=np.float32))
                    state, _, terminated, truncated, info = env.step(action)
                    for i in np.flatnonzero(terminated | truncated):
                        if completed[i] >= quotas[i]:
                            continue
                        final = info["final_info"][i]
                        identity = final["landing_evaluation_case"]
                        key = (identity["worker"], identity["episode"])
                        record = _episode(final)
                        records[key] = (identity, record)
                        completed[i] += 1
                        rows.append(dict(nr_env_steps=global_step, height=height, condition=condition,
                                         controller=controller, **identity, **record))
                results[controller] = records
                names = next(iter(records.values()))[1]
                summary = {name: float(np.mean([r[name] for _, r in records.values()])) for name in names}
                summary["episode_count"] = len(records)
                for name in ("peak_calf_force_10ms", "peak_foot_force_10ms", "peak_body_acceleration"):
                    values = [r[name] for _, r in records.values()]
                    summary[f"{name}_p95"] = float(np.percentile(values, 95))
                    summary[f"{name}_max"] = float(np.max(values))
                prefix = f"eval/{condition}/{controller}/{height_tag}"
                metrics.update({f"{prefix}/{k}": v for k, v in summary.items()})
                summaries.append(dict(nr_env_steps=global_step, height=height, condition=condition,
                                      controller=controller, **summary))
            if results["policy"].keys() != results["zero"].keys():
                raise RuntimeError("Policy/baseline evaluation case IDs differ")
            gains, wins, losses = [], [], []
            for key, (identity, policy) in results["policy"].items():
                zero_identity, zero = results["zero"][key]
                if identity["scenario_hash"] != zero_identity["scenario_hash"]:
                    raise RuntimeError(f"Unpaired evaluation initial state: {condition}, {height}, {key}")
                gain = policy["success_rate"] - zero["success_rate"]
                gains.append(gain)
                wins.append(gain > 0)
                losses.append(gain < 0)
            prefix = f"eval/{condition}/{height_tag}"
            metrics[f"{prefix}/success_gain"] = float(np.mean(gains))
            metrics[f"{prefix}/policy_only_success_rate"] = float(np.mean(wins))
            metrics[f"{prefix}/zero_only_success_rate"] = float(np.mean(losses))
            for name in ("peak_calf_force_10ms", "peak_foot_force_10ms", "peak_body_acceleration"):
                differences = [policy[name] - results["zero"][key][1][name]
                               for key, (_, policy) in results["policy"].items()]
                metrics[f"{prefix}/{name}_policy_minus_zero"] = float(np.mean(differences))
    for condition in conditions:
        for controller in ("policy", "zero"):
            selected = [s for s in summaries if s["condition"] == condition and s["controller"] == controller]
            for name in ("success_rate", "episode_return"):
                metrics[f"eval/{condition}/{controller}/{name}"] = float(np.mean([s[name] for s in selected]))
        metrics[f"eval/{condition}/success_gain"] = (metrics[f"eval/{condition}/policy/success_rate"]
                                                     - metrics[f"eval/{condition}/zero/success_rate"])
    # Legacy keys now mean policy performance averaged over fixed perturbed heights.
    metrics["eval/episode_return"] = metrics[f"eval/{conditions[-1]}/policy/episode_return"]
    metrics["eval/landing_success_rate"] = metrics[f"eval/{conditions[-1]}/policy/success_rate"]
    metrics["eval/landing_episode_count"] = episodes * len(heights)
    if config.get("require_nominal_for_best", False):
        tags = [f"h_{h:g}".replace(".", "p") for h in heights]
        nominal = [metrics[f"eval/nominal/policy/{tag}/success_rate"] for tag in tags]
        robust = [metrics[f"eval/perturbed/policy/{tag}/success_rate"] for tag in tags]
        metrics["eval/nominal_eligible"] = float(all(x == 1. for x in nominal))
        metrics["eval/worst_height_success_rate"] = float(min(robust))
    directory = Path(output_dir) / "landing_evaluation"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = dict(config, episodes_per_height_per_controller=episodes,
                    height_definition=("base above terrain center" if config.get("height_reference", "clearance") == "base"
                                       else "drop above nominal base height, as in the training curriculum"),
                    perturbed_physics="sampled at reset and frozen within each episode")
    (directory / "config.json").write_text(json.dumps(manifest, indent=2))
    for suffix, records in (("episodes", rows), ("summary", summaries)):
        path = directory / f"{global_step}_{suffix}.csv"
        with path.open("w", newline="") as stream:
            fields = list(records[0])
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    return metrics
