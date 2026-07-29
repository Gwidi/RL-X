import numpy as np
import jax.numpy as jnp

from rl_x.environments.custom_mujoco.robot_locomotion.mjx.terrain_functions.hfield_bunker_ruins import (
    HFieldBunkerRuinsTerrainGeneration,
)


class HFieldBowlTerrainGeneration(HFieldBunkerRuinsTerrainGeneration):
    """Circular bowl with a flat spawn area and parabolic sides."""

    def __init__(self, env):
        super().__init__(env)

        terrain_config = self.env.env_config["terrain"]
        self.radius_m = terrain_config.get(
            "bowl_radius_m",
            self.hfield_half_length_in_meters,
        )
        self.flat_radius_m = terrain_config.get(
            "bowl_flat_radius_m",
            0.8,
        )
        self.depth_m = terrain_config.get(
            "bowl_depth_m",
            1.0,
        )
        self.use_curriculum = terrain_config.get(
            "bowl_use_curriculum",
            True,
        )

        if (
            not np.isfinite(self.flat_radius_m)
            or self.flat_radius_m < 0.0
        ):
            raise ValueError(
                "terrain.bowl_flat_radius_m must be finite and non-negative."
            )
        if (
            not np.isfinite(self.radius_m)
            or self.radius_m <= self.flat_radius_m
        ):
            raise ValueError(
                "terrain.bowl_radius_m must be finite and greater than "
                "terrain.bowl_flat_radius_m."
            )
        if self.radius_m > self.hfield_half_length_in_meters:
            raise ValueError(
                "terrain.bowl_radius_m must not exceed the heightfield "
                "half-width."
            )
        if (
            not np.isfinite(self.depth_m)
            or not 0.0 <= self.depth_m <= self.max_possible_height
        ):
            raise ValueError(
                "terrain.bowl_depth_m must be finite and between zero "
                "and the heightfield height range."
            )

    def init(self, internal_state):
        super().init(internal_state)
        internal_state["terrain/bowl_depth_m"] = 0.0
        internal_state["terrain/bowl_max_height_m"] = 0.0

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

        height_field = self.bowl_terrain(curriculum_coeff)
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
        internal_state["terrain/bowl_depth_m"] = (
            curriculum_coeff * self.depth_m
        )
        internal_state["terrain/bowl_max_height_m"] = jnp.max(
            height_field
        )

        return new_mjx_model

    def bowl_terrain(self, curriculum_coeff):
        """Build a radial quadratic profile rising from the flat center."""
        coordinates_m = jnp.linspace(
            -self.hfield_half_length_in_meters,
            self.hfield_half_length_in_meters,
            self.hfield_length,
        )
        x_grid_m = coordinates_m[None, :]
        y_grid_m = coordinates_m[:, None]

        radial_distance_m = jnp.sqrt(
            x_grid_m**2 + y_grid_m**2
        )
        normalized_slope_position = jnp.clip(
            (radial_distance_m - self.flat_radius_m)
            / (self.radius_m - self.flat_radius_m),
            0.0,
            1.0,
        )

        return (
            curriculum_coeff
            * self.depth_m
            * normalized_slope_position**2
        )
