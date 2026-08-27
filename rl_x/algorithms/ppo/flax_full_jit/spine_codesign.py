"""Offline single-critic design search for the configurable Silver Badger spine."""

from __future__ import annotations

import json
import logging
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax


rlx_logger = logging.getLogger("rl_x")


STATE_BANK_FORMAT_VERSION = 1


def _require_codesign_environment(env):
    if getattr(env, "spine_design_size", 0) != 9:
        raise ValueError(
            "Spine co-design requires train_robot=silver_badger_codesign."
        )
    if len(getattr(env, "spine_design_obs_idx", ())) != 10:
        raise ValueError("Expected a 10-value spine design observation patch.")


def collect_nominal_state_bank(
    env,
    policy_apply,
    policy_params,
    get_processed_action,
    key,
    target_size,
    output_path,
):
    """Collect deterministic-policy observations using the nominal morphology."""
    _require_codesign_environment(env)
    if env.spine_design_randomization_enabled:
        raise ValueError(
            "State-bank collection requires spine_design_randomization_enabled=False "
            "so the nominal Silver Badger is the reference."
        )
    target_size = int(target_size)
    if target_size <= 0:
        raise ValueError("spine_state_bank_size must be positive.")
    if not output_path:
        raise ValueError("spine_state_bank_path must be set.")

    nr_envs = int(env.nr_envs)
    nr_steps = max(1, int(np.ceil(target_size / nr_envs)))
    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, nr_envs)
    env_state = env.reset(reset_keys, True)

    @jax.jit
    def step(state):
        action_mean, _ = policy_apply(policy_params, state.next_observation)
        action = get_processed_action(action_mean)
        next_state = env.step(state, action)
        return next_state, next_state.next_observation

    observations = []
    for _ in range(nr_steps):
        env_state, observation = step(env_state)
        observations.append(np.asarray(observation, dtype=np.float32))
    state_bank = np.concatenate(observations, axis=0)
    if state_bank.shape[0] > target_size:
        selection = np.random.default_rng(0).permutation(state_bank.shape[0])[
            :target_size
        ]
        state_bank = state_bank[selection]

    design_obs_indices = np.asarray(env.spine_design_obs_idx, dtype=np.int32)
    nominal_design = np.asarray(env.spine_design_default, dtype=np.float32)
    nominal_design_observation = np.asarray(
        env.spine_design_observation(env.spine_design_default),
        dtype=np.float32,
    )
    design_patches = state_bank[:, design_obs_indices]
    if not np.allclose(
        design_patches,
        nominal_design_observation[None, :],
        rtol=1e-5,
        atol=1e-5,
    ):
        raise RuntimeError(
            "Collected state bank is not nominal-reference: its design patch varies."
        )
    if not np.all(np.isfinite(state_bank)):
        raise RuntimeError("Collected state bank contains non-finite values.")

    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        format_version=np.asarray(STATE_BANK_FORMAT_VERSION, dtype=np.int32),
        state_bank=state_bank,
        spine_design_obs_indices=design_obs_indices,
        nominal_spine_design=nominal_design,
        nominal_spine_design_observation=nominal_design_observation,
        spine_design_parameter_names=np.asarray(
            env.spine_design_parameter_names,
            dtype=np.str_,
        ),
        axis_max_tilt_rad=np.asarray(
            env.spine_axis_max_tilt_rad,
            dtype=np.float32,
        ),
    )
    rlx_logger.info(
        f"Wrote nominal-reference spine state bank to {output_path} "
        f"(shape={state_bank.shape})"
    )
    return key, state_bank


def load_nominal_state_bank(path, env):
    """Load and validate a state bank against the active environment layout."""
    _require_codesign_environment(env)
    if not path:
        raise ValueError("spine_state_bank_path must be set.")
    with np.load(path) as data:
        required = {
            "format_version",
            "state_bank",
            "spine_design_obs_indices",
            "nominal_spine_design",
            "nominal_spine_design_observation",
            "spine_design_parameter_names",
            "axis_max_tilt_rad",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"State bank is missing fields: {missing}")
        format_version = int(np.asarray(data["format_version"]))
        state_bank = np.asarray(data["state_bank"], dtype=np.float32)
        design_obs_indices = np.asarray(
            data["spine_design_obs_indices"], dtype=np.int32
        )
        nominal_design = np.asarray(
            data["nominal_spine_design"], dtype=np.float32
        )
        nominal_design_observation = np.asarray(
            data["nominal_spine_design_observation"], dtype=np.float32
        )
        parameter_names = tuple(str(value) for value in data[
            "spine_design_parameter_names"
        ])
        axis_max_tilt_rad = float(np.asarray(data["axis_max_tilt_rad"]))

    expected_indices = np.asarray(env.spine_design_obs_idx, dtype=np.int32)
    expected_nominal = np.asarray(env.spine_design_default, dtype=np.float32)
    expected_nominal_observation = np.asarray(
        env.spine_design_observation(env.spine_design_default),
        dtype=np.float32,
    )
    if format_version != STATE_BANK_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported state-bank format {format_version}; "
            f"expected {STATE_BANK_FORMAT_VERSION}."
        )
    if state_bank.ndim != 2 or state_bank.shape[1] != env.single_observation_space.shape[0]:
        raise ValueError(
            "State-bank observation shape does not match the active environment."
        )
    if not np.array_equal(design_obs_indices, expected_indices):
        raise ValueError("State-bank design observation indices do not match.")
    if parameter_names != tuple(env.spine_design_parameter_names):
        raise ValueError("State-bank design parameter order does not match.")
    if not np.allclose(nominal_design, expected_nominal):
        raise ValueError("State-bank nominal design does not match.")
    if not np.allclose(
        nominal_design_observation,
        expected_nominal_observation,
    ):
        raise ValueError("State-bank nominal design observation does not match.")
    if not np.isclose(axis_max_tilt_rad, env.spine_axis_max_tilt_rad):
        raise ValueError("State-bank hinge-axis cap does not match.")
    if not np.all(np.isfinite(state_bank)):
        raise ValueError("State bank contains non-finite values.")
    if not np.allclose(
        state_bank[:, design_obs_indices],
        nominal_design_observation[None, :],
        rtol=1e-5,
        atol=1e-5,
    ):
        raise ValueError("State bank is not a nominal-morphology reference bank.")
    return state_bank, design_obs_indices


def _design_delta_geometry(env):
    default = jnp.asarray(env.spine_design_default, dtype=jnp.float32)
    design_min = jnp.asarray(
        env.spine_design_randomization_min, dtype=jnp.float32
    )
    design_max = jnp.asarray(
        env.spine_design_randomization_max, dtype=jnp.float32
    )
    span = design_max - design_min
    if np.any(np.asarray(span) <= 0.0):
        raise ValueError("Every spine search dimension must have a nonzero span.")
    delta_min = (design_min - default) / span
    delta_max = (design_max - default) / span
    return default, span, delta_min, delta_max


def _design_from_delta(env, delta, default, span):
    return env.project_spine_design(default + delta * span)


def _delta_from_design(design, default, span):
    return (design - default) / span


def search_spine_design(
    env,
    critic_apply,
    critic_params,
    state_bank_path,
    output_path,
    nr_steps=200,
    minibatch_size=1024,
    learning_rate=0.05,
    max_step=0.05,
    l2_weight=0.01,
    seed=0,
):
    """Ascend one PPO critic from the nominal design and nominal state bank."""
    state_bank, design_obs_indices = load_nominal_state_bank(
        state_bank_path,
        env,
    )
    if not output_path:
        raise ValueError("spine_search_output_path must be set.")
    if int(nr_steps) < 0:
        raise ValueError("spine_search_steps must be nonnegative.")
    if float(learning_rate) <= 0.0 or float(max_step) <= 0.0:
        raise ValueError("Search learning rate and max step must be positive.")
    if float(l2_weight) < 0.0:
        raise ValueError("spine_search_l2_weight must be nonnegative.")

    state_bank_jax = jnp.asarray(state_bank, dtype=jnp.float32)
    design_obs_indices_jax = jnp.asarray(design_obs_indices, dtype=jnp.int32)
    default, span, delta_min, delta_max = _design_delta_geometry(env)
    bank_size = state_bank.shape[0]
    minibatch_size = max(1, min(int(minibatch_size), bank_size))
    optimizer = optax.adam(float(learning_rate))
    delta = jnp.zeros(env.spine_design_size, dtype=jnp.float32)
    optimizer_state = optimizer.init(delta)

    def project_delta(candidate):
        candidate = jnp.clip(candidate, delta_min, delta_max)
        design = _design_from_delta(env, candidate, default, span)
        return jnp.clip(
            _delta_from_design(design, default, span),
            delta_min,
            delta_max,
        )

    def values_and_objective(candidate, states):
        design = _design_from_delta(env, candidate, default, span)
        design_observation = env.spine_design_observation(design)
        substituted_states = states.at[:, design_obs_indices_jax].set(
            design_observation[None, :]
        )
        critic_values = critic_apply(critic_params, substituted_states)
        raw_critic = jnp.mean(critic_values)
        objective = raw_critic - float(l2_weight) * jnp.mean(candidate ** 2)
        return raw_critic, objective

    @jax.jit
    def score(candidate, states):
        return values_and_objective(candidate, states)

    @jax.jit
    def step(candidate, current_optimizer_state, states):
        def loss(search_delta):
            _, objective = values_and_objective(search_delta, states)
            return -objective

        _, gradient = jax.value_and_grad(loss)(candidate)
        updates, next_optimizer_state = optimizer.update(
            gradient,
            current_optimizer_state,
            candidate,
        )
        updates = jnp.clip(updates, -float(max_step), float(max_step))
        next_candidate = project_delta(
            optax.apply_updates(candidate, updates)
        )
        return next_candidate, next_optimizer_state

    nominal_raw, nominal_objective = score(delta, state_bank_jax)
    nominal_raw = float(nominal_raw)
    nominal_objective = float(nominal_objective)
    if not np.isfinite(nominal_objective):
        raise RuntimeError("Nominal critic objective is non-finite.")
    best_delta = np.asarray(delta, dtype=np.float32)
    best_raw = nominal_raw
    best_objective = nominal_objective
    trace = [{
        "step": 0,
        "critic": nominal_raw,
        "objective": nominal_objective,
        "best_objective": nominal_objective,
    }]

    rng = np.random.default_rng(int(seed))
    for step_index in range(int(nr_steps)):
        indices = rng.choice(
            bank_size,
            size=minibatch_size,
            replace=False,
        )
        delta, optimizer_state = step(
            delta,
            optimizer_state,
            state_bank_jax[indices],
        )
        raw, objective = score(delta, state_bank_jax)
        raw = float(raw)
        objective = float(objective)
        if not np.isfinite(objective):
            raise RuntimeError(
                f"Spine search produced a non-finite objective at step {step_index + 1}."
            )
        if objective > best_objective:
            best_delta = np.asarray(delta, dtype=np.float32)
            best_raw = raw
            best_objective = objective
        trace.append({
            "step": step_index + 1,
            "critic": raw,
            "objective": objective,
            "best_objective": best_objective,
        })

    best_design = np.asarray(
        _design_from_delta(
            env,
            jnp.asarray(best_delta),
            default,
            span,
        ),
        dtype=np.float32,
    )
    best_axis = np.asarray(
        env.spine_axis_from_design(best_design),
        dtype=np.float32,
    )
    best_design_observation = np.asarray(
        env.spine_design_observation(best_design),
        dtype=np.float32,
    )
    if not np.isclose(np.linalg.norm(best_axis), 1.0, atol=1e-5):
        raise RuntimeError("Optimized hinge axis is not unit length.")
    if np.linalg.norm(best_design[-2:]) > env.spine_axis_max_tilt_rad + 1e-5:
        raise RuntimeError("Optimized hinge axis exceeds its tilt cap.")

    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    trace_step = np.asarray([item["step"] for item in trace], dtype=np.int32)
    trace_critic = np.asarray([item["critic"] for item in trace], dtype=np.float32)
    trace_objective = np.asarray(
        [item["objective"] for item in trace], dtype=np.float32
    )
    trace_best_objective = np.asarray(
        [item["best_objective"] for item in trace], dtype=np.float32
    )
    np.savez_compressed(
        output_path,
        best_normalized_delta=best_delta,
        best_spine_design=best_design,
        best_rotation_axis=best_axis,
        best_spine_design_observation=best_design_observation,
        nominal_critic=np.asarray(nominal_raw, dtype=np.float32),
        nominal_objective=np.asarray(nominal_objective, dtype=np.float32),
        best_critic=np.asarray(best_raw, dtype=np.float32),
        best_objective=np.asarray(best_objective, dtype=np.float32),
        trace_step=trace_step,
        trace_critic=trace_critic,
        trace_objective=trace_objective,
        trace_best_objective=trace_best_objective,
        spine_design_parameter_names=np.asarray(
            env.spine_design_parameter_names,
            dtype=np.str_,
        ),
        state_bank_path=np.asarray(os.path.abspath(state_bank_path)),
    )
    json_path = os.path.splitext(output_path)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump({
            "state_bank_path": os.path.abspath(state_bank_path),
            "nominal_critic": nominal_raw,
            "nominal_objective": nominal_objective,
            "best_critic": best_raw,
            "best_objective": best_objective,
            "best_normalized_delta": best_delta.tolist(),
            "best_spine_design": dict(zip(
                env.spine_design_parameter_names,
                best_design.tolist(),
            )),
            "best_rotation_axis": best_axis.tolist(),
            "settings": {
                "steps": int(nr_steps),
                "minibatch_size": minibatch_size,
                "learning_rate": float(learning_rate),
                "max_step": float(max_step),
                "l2_weight": float(l2_weight),
                "seed": int(seed),
            },
            "trace": trace,
        }, file, indent=2)
    rlx_logger.info(
        f"Wrote spine design search to {output_path} "
        f"(objective {nominal_objective:.6f} -> {best_objective:.6f})"
    )
    return {
        "best_normalized_delta": best_delta,
        "best_spine_design": best_design,
        "best_rotation_axis": best_axis,
        "nominal_critic": nominal_raw,
        "best_critic": best_raw,
        "nominal_objective": nominal_objective,
        "best_objective": best_objective,
        "trace": trace,
    }
