from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType
from rl_x.algorithms.ppo.height_map import (
    HeightMapObservationEncoder,
    split_height_map_observation_indices,
)


def get_policy(config, env):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_observation_indices = getattr(env, "policy_observation_indices", jnp.arange(env.single_observation_space.shape[0]))
    proprioception_indices, height_map_indices = split_height_map_observation_indices(
        config,
        policy_observation_indices,
        "policy",
    )
    height_map_cnn_output_dim = getattr(
        config.algorithm,
        "height_map_cnn_output_dim",
        8,
    )

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return (Policy(
                    env.single_action_space.shape,
                    config.algorithm.std_dev,
                    proprioception_indices,
                    height_map_indices,
                    height_map_cnn_output_dim,
                ),
                get_processed_action_function(
                    config.algorithm.action_clipping_and_rescaling,
                    jnp.array(env.single_action_space.low), jnp.array(env.single_action_space.high)
                ))


class Policy(nn.Module):
    as_shape: Sequence[int]
    std_dev: float
    proprioception_indices: Sequence[int]
    height_map_indices: Sequence[int]
    height_map_cnn_output_dim: int

    @nn.compact
    def __call__(self, x):
        x = HeightMapObservationEncoder(
            proprioception_indices=self.proprioception_indices,
            height_map_indices=self.height_map_indices,
            height_map_output_dim=self.height_map_cnn_output_dim,
        )(x)
        policy_mean = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        policy_mean = nn.LayerNorm()(policy_mean)
        policy_mean = nn.elu(policy_mean)
        policy_mean = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(policy_mean)
        policy_mean = nn.elu(policy_mean)
        policy_mean = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(policy_mean)
        policy_mean = nn.elu(policy_mean)
        policy_mean = nn.Dense(np.prod(self.as_shape).item(), kernel_init=orthogonal(0.01), bias_init=constant(0.0))(policy_mean)
        policy_logstd = self.param("policy_logstd", constant(jnp.log(self.std_dev)), (1, np.prod(self.as_shape).item()))
        return policy_mean, policy_logstd


def get_processed_action_function(action_clipping_and_rescaling, env_as_low, env_as_high):
    if action_clipping_and_rescaling:
        def get_clipped_and_scaled_action(action, env_as_low=env_as_low, env_as_high=env_as_high):
            clipped_action = jnp.clip(action, -1, 1)
            return env_as_low + (0.5 * (clipped_action + 1.0) * (env_as_high - env_as_low))
        return jax.jit(get_clipped_and_scaled_action)
    else:
        return jax.jit(lambda x: x)
