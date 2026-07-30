import numpy as np
import jax.numpy as jnp

from rl_x.environments.custom_mujoco.robot_locomotion.mjx.terrain_functions.hfield_bunker_ruins import (
    HFieldBunkerRuinsTerrainGeneration,
)


class HFieldTruncatedPyramidTerrainGeneration(
    HFieldBunkerRuinsTerrainGeneration
):
    """Square frustum with a flat center platform and linear sides."""

    config_prefix = "truncated_pyramid"
    terrain_label = "truncated pyramid"
    inverted = False

    def __init__(self, env):
        super().__init__(env)

        terrain_config = self.env.env_config["terrain"]
        self.center_half_width_m = terrain_config.get(
            f"{self.config_prefix}_center_half_width_m",
            0.8,
        )
        self.outer_half_width_m = terrain_config.get(
            f"{self.config_prefix}_outer_half_width_m",
            3.2,
        )
        self.height_m = terrain_config.get(
            f"{self.config_prefix}_height_m",
            1.0,
        )
        self.use_curriculum = terrain_config.get(
            f"{self.config_prefix}_use_curriculum",
            True,
        )

        if (
            not np.isfinite(self.center_half_width_m)
            or self.center_half_width_m < 0.0
        ):
            raise ValueError(
                f"terrain.{self.config_prefix}_center_half_width_m must "
                "be finite and non-negative."
            )
        if (
            not np.isfinite(self.outer_half_width_m)
            or self.outer_half_width_m <= self.center_half_width_m
        ):
            raise ValueError(
                f"terrain.{self.config_prefix}_outer_half_width_m must "
                "be finite and greater than the center half-width."
            )
        if self.outer_half_width_m > self.hfield_half_length_in_meters:
            raise ValueError(
                f"terrain.{self.config_prefix}_outer_half_width_m must "
                "not exceed the heightfield half-width."
            )
        if (
            not np.isfinite(self.height_m)
            or not 0.0 <= self.height_m <= self.max_possible_height
        ):
            raise ValueError(
                f"terrain.{self.config_prefix}_height_m must be finite "
                "and between zero and the heightfield height range."
            )

    @property
    def height_state_key(self):
        return f"terrain/{self.config_prefix}_height_m"

    @property
    def max_height_state_key(self):
        return f"terrain/{self.config_prefix}_max_height_m"

    def init(self, internal_state):
        super().init(internal_state)
        internal_state[self.height_state_key] = 0.0
        internal_state[self.max_height_state_key] = 0.0

    def sample(self, mjx_model, internal_state, key):
        curriculum_coeff = jnp.where(
            self.use_curriculum,
            internal_state["env_curriculum_coeff"],
            1.0,
        )
        curriculum_coeff = jnp.where(
            internal_state["in_eval_mode"],
            1.0,
            curriculum_coeff,
        )

        height_field = self.pyramid_terrain(curriculum_coeff)
        new_height_field_data = self.isaac_hf_to_mujoco_hf(height_field)
        new_mjx_model = mjx_model.replace(
            hfield_data=new_height_field_data,
        )

        center_idx = (
            self.hfield_half_length * self.hfield_length
            + self.hfield_half_length
        )
        internal_state["center_height"] = (
            new_height_field_data[center_idx] * self.mujoco_height_scaling
        )
        internal_state["current_height_field_data"] = (
            new_height_field_data.reshape(
                self.hfield_length,
                self.hfield_length,
            )
        )
        internal_state[self.height_state_key] = (
            curriculum_coeff * self.height_m
        )
        internal_state[self.max_height_state_key] = jnp.max(
            height_field
        )

        return new_mjx_model

    def pyramid_terrain(self, curriculum_coeff):
        """Build a square frustum or its inverted counterpart."""
        grid_spacing_m = (
            2.0 * self.hfield_half_length_in_meters
            / (self.hfield_length - 1)
        )
        coordinates_m = (
            jnp.arange(self.hfield_length)
            - (self.hfield_length - 1) / 2.0
        ) * grid_spacing_m
        x_grid_m = coordinates_m[None, :]
        y_grid_m = coordinates_m[:, None]

        square_radius_m = jnp.maximum(
            jnp.abs(x_grid_m),
            jnp.abs(y_grid_m),
        )
        center_fraction = jnp.clip(
            (self.outer_half_width_m - square_radius_m)
            / (
                self.outer_half_width_m
                - self.center_half_width_m
            ),
            0.0,
            1.0,
        )
        height_fraction = (
            1.0 - center_fraction
            if self.inverted
            else center_fraction
        )

        return curriculum_coeff * self.height_m * height_fraction
