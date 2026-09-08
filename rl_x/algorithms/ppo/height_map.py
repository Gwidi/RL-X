from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp
import numpy as np


HEIGHT_MAP_SHAPE = (17, 11)
HEIGHT_MAP_OBSERVATION_TYPE = "height_samples"


def split_height_map_observation_indices(
    config,
    observation_indices: Sequence[int],
    observer: str,
):
    """Separates proprioception and the flattened height map for an observer."""
    observation_indices = np.asarray(observation_indices, dtype=np.int32)
    exteroception_type = getattr(
        config.environment,
        f"{observer}_exteroceptive_observation_type",
        None,
    )
    cnn_enabled = getattr(config.algorithm, "height_map_cnn_enabled", True)

    if not cnn_enabled or exteroception_type != HEIGHT_MAP_OBSERVATION_TYPE:
        return observation_indices, np.empty((0,), dtype=np.int32)

    height_map_size = int(np.prod(HEIGHT_MAP_SHAPE))
    if observation_indices.size < height_map_size:
        raise ValueError(
            f"{observer} uses height_samples, but its observation contains "
            f"only {observation_indices.size} values; expected at least "
            f"{height_map_size} for a {HEIGHT_MAP_SHAPE} height map."
        )

    return (
        observation_indices[:-height_map_size],
        observation_indices[-height_map_size:],
    )


class HeightMapEncoder(nn.Module):
    output_dim: int

    @nn.compact
    def __call__(self, height_map):
        height_map = height_map.reshape(
            height_map.shape[:-1] + HEIGHT_MAP_SHAPE + (1,)
        )

        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
        )(height_map)
        x = nn.elu(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
        )(x)
        x = nn.elu(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
        )(x)
        x = nn.elu(x)

        x = x.reshape(x.shape[:-3] + (-1,))
        x = nn.Dense(self.output_dim)(x)
        x = nn.LayerNorm()(x)
        return nn.elu(x)


class HeightMapObservationEncoder(nn.Module):
    proprioception_indices: Sequence[int]
    height_map_indices: Sequence[int]
    height_map_output_dim: int

    @nn.compact
    def __call__(self, observation):
        proprioception = observation[..., self.proprioception_indices]
        if len(self.height_map_indices) == 0:
            return proprioception

        height_map = observation[..., self.height_map_indices]
        height_map_embedding = HeightMapEncoder(
            output_dim=self.height_map_output_dim,
        )(height_map)
        return jnp.concatenate(
            (proprioception, height_map_embedding),
            axis=-1,
        )
