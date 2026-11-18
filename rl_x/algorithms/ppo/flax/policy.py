from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_policy(config, env):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_observation_indices = getattr(env, "policy_observation_indices", jnp.arange(env.single_observation_space.shape[0]))

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        handoff = getattr(config, 'environment', {}).get('landing_handoff', {}).get('enabled', False)
        motor_shape = (env.single_action_space.shape[0] - 1,) if handoff else env.single_action_space.shape
        motor_slice = slice(None, -1) if handoff else slice(None)
        return (Policy(motor_shape, config.algorithm.std_dev, config.algorithm.nr_hidden_units, policy_observation_indices, handoff),
                get_processed_action_function(
                    config.algorithm.action_clipping_and_rescaling,
                    jnp.array(env.single_action_space.low[motor_slice]),
                    jnp.array(env.single_action_space.high[motor_slice])
                ))


class Policy(nn.Module):
    as_shape: Sequence[int]
    std_dev: float
    nr_hidden_units: int
    policy_observation_indices: Sequence[int]
    handoff: bool = False

    @nn.compact
    def __call__(self, x):
        x = x[..., self.policy_observation_indices]
        policy_mean = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        policy_mean = nn.LayerNorm()(policy_mean)
        policy_mean = nn.elu(policy_mean)
        policy_mean = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(policy_mean)
        policy_mean = nn.elu(policy_mean)
        policy_mean = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(policy_mean)
        policy_mean = nn.elu(policy_mean)
        features = policy_mean
        policy_mean = nn.Dense(np.prod(self.as_shape).item(), kernel_init=orthogonal(0.01), bias_init=constant(0.0))(features)
        policy_logstd = self.param("policy_logstd", constant(jnp.log(self.std_dev)), (1, np.prod(self.as_shape).item()))
        if self.handoff:
            logit = nn.Dense(1, kernel_init=constant(0.), bias_init=constant(-2.5), name='handoff_head')(features)
            return policy_mean, policy_logstd, logit
        return policy_mean, policy_logstd


def get_processed_action_function(action_clipping_and_rescaling, env_as_low, env_as_high):
    if action_clipping_and_rescaling:
        def get_clipped_and_scaled_action(action, env_as_low=env_as_low, env_as_high=env_as_high):
            clipped_action = jnp.clip(action, -1, 1)
            return env_as_low + (0.5 * (clipped_action + 1.0) * (env_as_high - env_as_low))
        return jax.jit(get_clipped_and_scaled_action)
    else:
        return jax.jit(lambda x: x)
