from typing import Sequence
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

from rl_x.environments.observation_space_type import ObservationSpaceType
from rl_x.algorithms.ppo.height_map import (
    HeightMapObservationEncoder,
    split_height_map_observation_indices,
)


def get_critic(config, env):
    observation_space_type = env.general_properties.observation_space_type
    critic_observation_indices = getattr(env, "critic_observation_indices", jnp.arange(env.single_observation_space.shape[0]))
    proprioception_indices, height_map_indices = split_height_map_observation_indices(
        config,
        critic_observation_indices,
        "critic",
    )
    height_map_cnn_output_dim = getattr(
        config.algorithm,
        "height_map_cnn_output_dim",
        8,
    )

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return Critic(
            config.algorithm.nr_hidden_units,
            proprioception_indices,
            height_map_indices,
            height_map_cnn_output_dim,
        )


class Critic(nn.Module):
    nr_hidden_units: int
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
        critic = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = nn.LayerNorm()(critic)
        critic = nn.elu(critic)
        critic = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = nn.elu(critic)
        critic = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = nn.elu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(critic)
        return critic
