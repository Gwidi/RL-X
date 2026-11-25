"""PPO of ppo.flax with rollouts compiled on the GPU for the MJX landing task.

Networks, losses, optimizer, checkpoints and log names come from ppo.flax. The
rollout uses many short trajectories instead of a few long ones.
"""

import logging
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from rl_x.algorithms.ppo.flax.ppo import PPO as FlaxPPO
from rl_x.algorithms.ppo.flax_mjx_landing.general_properties import GeneralProperties
from rl_x.environments.custom_mujoco.robot_locomotion.mjx_landing.contacts import solver_limit_worlds
from rl_x.environments.custom_mujoco.robot_locomotion.mjx_landing.default_config import CPU_ENVIRONMENT
from rl_x.environments.custom_mujoco.robot_locomotion.mjx_landing.environment import FAILURES
from rl_x.environments.custom_mujoco.robot_locomotion.mjx_landing.evaluation import LandingGrid

rlx_logger = logging.getLogger("rl_x")

GROUPS = {"landing_nominal_replay": (0,), "landing_replay": (1,), "landing_current": (2, 3)}


def with_learning_rate(state, learning_rate):
    """The learning rate lives in the Adam state, so a checkpoint overrides the config."""
    clip, inject = state.opt_state
    current = inject.hyperparams["learning_rate"]
    inject = inject._replace(hyperparams={**inject.hyperparams,
                                          "learning_rate": jnp.asarray(learning_rate, jnp.asarray(current).dtype)})
    return state.replace(opt_state=(clip, inject))


class PPO(FlaxPPO):
    def __init__(self, config, train_env, eval_env, run_path, writer):
        super().__init__(config, train_env, eval_env, run_path, writer)
        if not self.paired_landing_evaluation or self.learned_handoff:
            raise ValueError("ppo.flax_mjx_landing requires the landing grid evaluation and no handoff")
        if self.batch_size % self.minibatch_size:
            raise ValueError("The minibatch size must divide nr_envs * nr_steps")
        if self.anneal_learning_rate:
            raise ValueError("Use final_learning_rate: a schedule inside Adam changes the checkpoint layout")
        self.final_learning_rate = config.algorithm.final_learning_rate
        self.disturbance_ramp_steps = config.environment.mjx.release_disturbance.ramp_steps
        self.pool_size = config.environment.mjx.reset_pool_size
        if self.pool_size < self.nr_envs:
            raise ValueError("mjx.reset_pool_size must be at least nr_envs")

    def scheduled_learning_rate(self, global_step):
        if self.final_learning_rate < 0:
            return self.learning_rate
        remaining = 1 - global_step / self.total_timesteps
        return self.final_learning_rate + (self.learning_rate - self.final_learning_rate) * remaining

    def disturbance_scale(self, global_step):
        """Release disturbance grows linearly from 0 to its maximum over ramp_steps."""
        return min(1., global_step / self.disturbance_ramp_steps) if self.disturbance_ramp_steps else 1.

    def train(self):
        env = self.train_env

        @jax.jit
        def sample_pool(key, scale):
            return env.sample_pool(key, self.pool_size, scale)

        @jax.jit
        def start(pool):
            return env.reset(pool, self.nr_envs)

        @jax.jit
        def rollout(policy_state, critic_state, env_state, pool, first_index, key):
            def step(carry, _):
                env_state, index, key = carry
                key, subkey = jax.random.split(key)
                observation = env_state.observation
                mean, logstd = self.policy.apply(policy_state.params, observation)
                action = mean + jnp.exp(logstd) * jax.random.normal(subkey, shape=mean.shape)
                log_prob, _ = self.action_statistics((mean, logstd), observation, action)
                value = self.critic.apply(critic_state.params, observation).squeeze(-1)
                env_state, index = env.step(env_state, self.get_processed_action(action), pool, index)
                transition = (observation, env_state.actual_observation, action, env_state.reward, value,
                              env_state.terminated.astype(jnp.float32), log_prob)
                done = env_state.terminated | env_state.truncated
                return (env_state, index, key), (transition, (done, env_state.info))

            (env_state, index, key), (batch, (done, info)) = jax.lax.scan(
                step, (env_state, first_index, key), None, self.nr_steps)
            weight = done.astype(jnp.float32)
            # Episode values are read at the last step of each finished episode.
            group = info["curriculum/v22_group"].astype(jnp.int32)
            count = jnp.maximum(weight.sum(), 1.)
            episode = {k: (v * weight).sum() / count for k, v in info.items()}
            groups = {name: sum((group == g).astype(jnp.float32) for g in members) * weight for name, members in GROUPS.items()}
            success = info["curriculum/landing_success"]
            group_metrics = {name: (jnp.sum(success * w) / jnp.maximum(w.sum(), 1.), w.sum()) for name, w in groups.items()}
            metrics = dict(episodes=weight.sum(), episode=episode, groups=group_metrics,
                           step={k: v.mean() for k, v in info.items()}, resets=index - first_index,
                           overflow=env_state.overflow, solver_limit=solver_limit_worlds(env_state.data))
            return env_state, batch, metrics, key

        @jax.jit
        def calculate_gae_advantages(critic_state, next_states, rewards, terminations, values):
            next_values = self.critic.apply(critic_state.params, next_states).squeeze(-1)
            delta = rewards + self.gamma * next_values * (1. - terminations) - values

            def backward(advantage, inputs):
                delta_t, termination_t = inputs
                advantage = delta_t + self.gamma * self.gae_lambda * (1. - termination_t) * advantage
                return advantage, advantage

            _, advantages = jax.lax.scan(backward, jnp.zeros_like(delta[0]), (delta, terminations), reverse=True)
            return advantages, advantages + values

        @jax.jit
        def update(policy_state, critic_state, states, actions, advantages, returns, values, log_probs, key):
            def loss_fn(policy_params, critic_params, state_b, action_b, log_prob_b, return_b, advantage_b):
                output = self.policy.apply(policy_params, state_b)
                new_log_prob, entropy = self.action_statistics(output, state_b, action_b)
                logratio = new_log_prob - log_prob_b
                ratio = jnp.exp(logratio)
                pg_loss = jnp.maximum(-advantage_b * ratio,
                                      -advantage_b * jnp.clip(ratio, 1 - self.clip_range, 1 + self.clip_range)).mean()
                new_value = self.critic.apply(critic_params, state_b).squeeze(-1)
                critic_loss = (.5 * (new_value - return_b) ** 2).mean()
                entropy_loss = entropy.mean()
                metrics = {
                    "loss/policy_gradient_loss": pg_loss, "loss/critic_loss": critic_loss, "loss/entropy_loss": entropy_loss,
                    "policy_ratio/approx_kl": ((ratio - 1) - logratio).mean(),
                    "policy_ratio/clip_fraction": (jnp.abs(ratio - 1) > self.clip_range).astype(jnp.float32).mean()}
                return pg_loss - self.entropy_coef * entropy_loss + self.critic_coef * critic_loss, metrics

            flat = lambda x: x.reshape((self.batch_size,) + x.shape[2:])
            batch = tuple(flat(x) for x in (states, actions, log_probs, returns, advantages))
            grad = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)
            key, subkey = jax.random.split(key)
            indices = jnp.tile(jnp.arange(self.batch_size), (self.nr_epochs, 1))
            indices = jax.random.permutation(subkey, indices, axis=1, independent=True)
            indices = indices.reshape((self.nr_epochs * self.nr_minibatches, self.minibatch_size))

            def minibatch_update(carry, minibatch):
                policy_state, critic_state = carry
                state_b, action_b, log_prob_b, return_b, advantage_b = (x[minibatch] for x in batch)
                advantage_b = (advantage_b - advantage_b.mean()) / (advantage_b.std() + 1e-8)
                (_, metrics), (policy_grad, critic_grad) = grad(
                    policy_state.params, critic_state.params, state_b, action_b, log_prob_b, return_b, advantage_b)
                metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_grad)
                metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_grad)
                return (policy_state.apply_gradients(grads=policy_grad),
                        critic_state.apply_gradients(grads=critic_grad)), metrics

            (policy_state, critic_state), metrics = jax.lax.scan(minibatch_update, (policy_state, critic_state), indices)
            metrics = {k: v.mean() for k, v in metrics.items()}
            metrics["lr/learning_rate"] = policy_state.opt_state[1].hyperparams["learning_rate"]
            metrics["v_value/explained_variance"] = 1 - jnp.var(returns - values) / (jnp.var(returns) + 1e-8)
            metrics["policy/std_dev"] = jnp.mean(jnp.exp(policy_state.params["params"]["policy_logstd"]))
            return policy_state, critic_state, metrics, key

        grid = LandingGrid(self.eval_env, self.landing_evaluation_config.to_dict(), self.evaluation_episodes,
                           self.deterministic_action)
        self.key, pool_key = jax.random.split(self.key)
        env_state = start(sample_pool(jax.random.fold_in(pool_key, 0), self.disturbance_scale(0)))
        # All worlds start at release. Without a burn-in the first rollouts contain
        # only early failures, because no landing can succeed within nr_steps.
        for i in range(-(-env.horizon // self.nr_steps)):
            self.key, burn_in_key = jax.random.split(self.key)
            burn_in_pool = sample_pool(jax.random.fold_in(pool_key, 2 ** 30 + i), self.disturbance_scale(0))
            env_state, *_ = rollout(self.policy_state, self.critic_state, env_state, burn_in_pool, 0, burn_in_key)
        global_step = nr_updates = nr_episodes = 0
        previous_end = logging_time = None

        while global_step < self.total_timesteps:
            start_time = time.time()
            pool = sample_pool(jax.random.fold_in(pool_key, 1 + global_step // self.batch_size),
                               self.disturbance_scale(global_step))
            env_state, batch, rollout_metrics, self.key = rollout(
                self.policy_state, self.critic_state, env_state, pool, 0, self.key)
            rollout_metrics = jax.device_get(rollout_metrics)
            if rollout_metrics["overflow"] or bool(pool["invalid"]):
                raise RuntimeError("MJX contact buffer overflow or invalid launch: increase environment.mjx limits")
            global_step += self.batch_size
            nr_episodes += int(rollout_metrics["episodes"])
            acting_end = time.time()

            states, next_states, actions, rewards, values, terminations, log_probs = batch
            advantages, returns = jax.block_until_ready(calculate_gae_advantages(
                self.critic_state, next_states, rewards, terminations, values))
            advantage_end = time.time()

            learning_rate = self.scheduled_learning_rate(global_step - self.batch_size)
            policy_state, critic_state, optimization_metrics, self.key = update(
                with_learning_rate(self.policy_state, learning_rate), with_learning_rate(self.critic_state, learning_rate),
                states, actions, advantages, returns, values, log_probs, self.key)
            # Checkpoints keep the base rate; a decayed one would silently stop later training.
            self.policy_state = with_learning_rate(policy_state, self.learning_rate)
            self.critic_state = with_learning_rate(critic_state, self.learning_rate)
            optimization_metrics = {k: float(v) for k, v in optimization_metrics.items()}
            nr_updates += self.nr_epochs * self.nr_minibatches
            if not all(np.isfinite(v) for v in optimization_metrics.values()):
                raise RuntimeError(f"Non-finite PPO metrics at step {global_step}: {optimization_metrics}")
            optimizing_end = time.time()

            evaluation_metrics = {}
            final = self.landing_evaluation_config.get("evaluate_final", False) and global_step >= self.total_timesteps
            if self.evaluation_frequency != -1 and (global_step % self.evaluation_frequency == 0 or final):
                evaluation_metrics = grid.evaluate(self.policy_state.params, os.path.dirname(self.save_path), global_step)
            evaluating_end = time.time()

            if self.save_model:
                self._save_best_and_latest(evaluation_metrics, global_step)
            saving_end = time.time()

            self.logged_step = global_step - self.batch_size  # the scale used for this rollout's pool
            metrics = self._rollout_log(rollout_metrics)
            metrics.update(evaluation_metrics)
            metrics.update({"steps/nr_env_steps": global_step, "steps/nr_updates": nr_updates, "steps/nr_episodes": nr_episodes,
                            "time/acting_time": acting_end - start_time,
                            "time/calc_adv_and_return_time": advantage_end - acting_end,
                            "time/optimizing_time": optimizing_end - advantage_end,
                            "time/evaluating_time": evaluating_end - optimizing_end,
                            "time/saving_time": saving_end - evaluating_end})
            if logging_time is not None:
                metrics["time/logging_time_prev"] = logging_time
            if previous_end is not None:
                metrics["time/sps"] = int(self.batch_size / (saving_end - previous_end))
            previous_end = saving_end
            metrics.update(optimization_metrics)
            self.start_logging(global_step)
            for name, value in metrics.items():
                self.log(name, value, global_step)
            self.end_logging()
            logging_time = time.time() - saving_end

        if self.save_model:
            self.save("latest.model")

    def _save_best_and_latest(self, evaluation_metrics, global_step):
        if evaluation_metrics:
            self.last_evaluation_metrics = {"evaluation_nr_env_steps": global_step, **evaluation_metrics}
        self.checkpoint_metrics = {"nr_env_steps": global_step, **self.last_evaluation_metrics}
        if not evaluation_metrics:
            return
        config = self.landing_evaluation_config
        score = evaluation_metrics["eval/landing_success_rate"]
        if config.get("require_nominal_for_best", False):
            key = "eval/worst_cell_success_rate" if config.get("best_by_worst_cell", False) else "eval/worst_height_success_rate"
            score = (evaluation_metrics[key] + .001 * score if evaluation_metrics["eval/nominal_eligible"] == 1. else None)
        if score is not None and score > self.best_mean_return:
            self.best_mean_return = score
            self.save()
        self.save("latest.model")

    def _rollout_log(self, m):
        count, result = float(m["episodes"]), {}
        if count:
            result["rollout/landing_episode_count"] = int(count)
            result["rollout/landing_success_rate"] = m["episode"]["curriculum/landing_success"]
            for name in FAILURES:
                key = "outcome/unresolved" if name == "unresolved" else f"outcome/failure_{name}"
                log = "landing_unresolved_rate" if name == "unresolved" else f"failure_{name}_rate"
                result[f"rollout/{log}"] = m["episode"][key]
            result["rollout/landing_initial_height"] = m["episode"]["curriculum/initial_height"]
            for name, (rate, size) in m["groups"].items():
                if size:
                    result[f"rollout/{name}_success_rate"], result[f"rollout/{name}_count"] = rate, int(size)
            for name in ("episode_return", "episode_length"):
                result[f"rollout/{name}"] = m["episode"][name]
        result["rollout/disturbance_scale"] = self.disturbance_scale(self.logged_step)
        result["rollout/solver_limit_worlds"] = float(m["solver_limit"])
        result["rollout/reset_pool_use"] = float(m["resets"]) / self.pool_size
        for name, value in m["step"].items():
            if name not in ("episode_return", "episode_length"):
                # Episode integrals are only meaningful at the end of an episode.
                result[f"env_info/{name}"] = m["episode"][name] if name.endswith("tau_squared_integral") else value
        return {k: float(v) for k, v in result.items()}

    def save(self, file_name=None):
        """Save a checkpoint the CPU environment and ppo.flax can load."""
        environment, algorithm = self.config.environment, self.config.algorithm
        names = environment.name, algorithm.name, environment.nr_envs
        mjx = environment.mjx.to_dict()
        with environment.unlocked(), algorithm.unlocked():
            environment.name, algorithm.name, environment.nr_envs = CPU_ENVIRONMENT, "ppo.flax", 64
            del environment.mjx
            del algorithm.final_learning_rate
        try:
            self.checkpoint_metrics = {**self.checkpoint_metrics, "physics_backend": "mjx-warp",
                                       "mjx": mjx, "mjx_nr_envs": names[2]}
            super().save(file_name)
        finally:
            with environment.unlocked(), algorithm.unlocked():
                environment.name, algorithm.name, environment.nr_envs = names
                environment.mjx = mjx
                algorithm.final_learning_rate = self.final_learning_rate

    def load(config, train_env, eval_env, run_path, writer, explicitly_set_algorithm_params):
        return FlaxPPO.load(config, train_env, eval_env, run_path, writer, explicitly_set_algorithm_params, model_class=PPO)

    def test(self, episodes):
        grid = LandingGrid(self.eval_env, self.landing_evaluation_config.to_dict(), self.evaluation_episodes,
                           self.deterministic_action)
        metrics = grid.evaluate(self.policy_state.params, os.path.dirname(self.save_path), 0)
        for name, value in metrics.items():
            if name.endswith("policy/success_rate") or "boundary" in name or name.startswith("eval/landing"):
                rlx_logger.info(f"{name}: {value}")

    def general_properties():
        return GeneralProperties
