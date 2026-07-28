import jax
import jax.numpy as jnp

from rl_x.environments.custom_mujoco.robot_locomotion.mjx.terrain_functions.hfield_bunker_ruins import (
    HFieldBunkerRuinsTerrainGeneration,
)


class HFieldInvertedPyramidStairsTerrainGeneration(
    HFieldBunkerRuinsTerrainGeneration
):
    """Square stair basin with a flat spawn platform at its lowest point."""

    def __init__(self, env):
        super().__init__(env)

        terrain_config = self.env.env_config["terrain"]
        self.center_half_width_m = terrain_config.get(
            "inverted_pyramid_center_half_width_m",
            0.8,
        )
        self.tread_depth_m = terrain_config.get(
            "inverted_pyramid_tread_depth_m",
            0.4,
        )
        self.tread_depth_scale_min = terrain_config.get(
            "inverted_pyramid_tread_depth_scale_min",
            0.65,
        )
        self.tread_depth_scale_max = terrain_config.get(
            "inverted_pyramid_tread_depth_scale_max",
            1.35,
        )
        self.randomize_tread_depth = terrain_config.get(
            "inverted_pyramid_randomize_tread_depth",
            True,
        )
        self.step_height_m = terrain_config.get(
            "inverted_pyramid_step_height_m",
            0.15,
        )
        self.use_curriculum = terrain_config.get(
            "inverted_pyramid_use_curriculum",
            True,
        )

        if not 0.0 <= self.center_half_width_m < self.hfield_half_length_in_meters:
            raise ValueError(
                "terrain.inverted_pyramid_center_half_width_m must be non-negative "
                "and smaller than the heightfield half-width."
            )
        if self.tread_depth_m <= 0.0:
            raise ValueError(
                "terrain.inverted_pyramid_tread_depth_m must be positive."
            )
        if self.tread_depth_scale_min <= 0.0:
            raise ValueError(
                "terrain.inverted_pyramid_tread_depth_scale_min must be positive."
            )
        if self.tread_depth_scale_max < self.tread_depth_scale_min:
            raise ValueError(
                "terrain.inverted_pyramid_tread_depth_scale_max must be >= "
                "inverted_pyramid_tread_depth_scale_min."
            )
        if self.step_height_m < 0.0:
            raise ValueError(
                "terrain.inverted_pyramid_step_height_m must be non-negative."
            )

    def init(self, internal_state):
        super().init(internal_state)
        internal_state["terrain/inverted_pyramid_step_height_m"] = 0.0
        internal_state["terrain/inverted_pyramid_max_height_m"] = 0.0
        internal_state[
            "terrain/inverted_pyramid_tread_depth_m"
        ] = self.tread_depth_m
        internal_state[
            "terrain/inverted_pyramid_nr_steps"
        ] = self.number_of_steps(self.tread_depth_m)

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

        sampled_tread_depth_m = self.sample_tread_depth(
            curriculum_coeff,
            key,
        )
        nr_steps = self.number_of_steps(sampled_tread_depth_m)
        height_field = self.inverted_pyramid_stairs_terrain(
            curriculum_coeff,
            sampled_tread_depth_m,
            nr_steps,
        )
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

        applied_step_height_m = curriculum_coeff * self.step_height_m
        internal_state[
            "terrain/inverted_pyramid_step_height_m"
        ] = applied_step_height_m
        internal_state[
            "terrain/inverted_pyramid_max_height_m"
        ] = jnp.max(height_field)
        internal_state[
            "terrain/inverted_pyramid_tread_depth_m"
        ] = sampled_tread_depth_m
        internal_state[
            "terrain/inverted_pyramid_nr_steps"
        ] = nr_steps

        return new_mjx_model

    def sample_tread_depth(self, curriculum_coeff, key):
        if not self.randomize_tread_depth:
            return jnp.asarray(self.tread_depth_m)

        randomization_coeff = jnp.clip(curriculum_coeff, 0.0, 1.0)
        min_scale = 1.0 + randomization_coeff * (
            self.tread_depth_scale_min - 1.0
        )
        max_scale = 1.0 + randomization_coeff * (
            self.tread_depth_scale_max - 1.0
        )
        scale = jax.random.uniform(
            key,
            shape=(),
            minval=min_scale,
            maxval=max_scale,
        )
        return self.tread_depth_m * scale

    def number_of_steps(self, tread_depth_m):
        available_width_m = (
            self.hfield_half_length_in_meters - self.center_half_width_m
        )
        return jnp.ceil(
            available_width_m / tread_depth_m
        ).astype(jnp.int32)

    def inverted_pyramid_stairs_terrain(
        self,
        curriculum_coeff,
        tread_depth_m,
        nr_steps,
    ):
        """Builds concentric square treads that rise away from the center."""
        coordinates_m = (
            jnp.arange(self.hfield_length) - self.hfield_half_length
        ) / self.one_meter_length
        x_grid_m = coordinates_m[None, :]
        y_grid_m = coordinates_m[:, None]

        # Chebyshev distance produces square contours, i.e. pyramid-like sides.
        square_radius_m = jnp.maximum(
            jnp.abs(x_grid_m),
            jnp.abs(y_grid_m),
        )
        distance_from_platform_m = jnp.maximum(
            square_radius_m - self.center_half_width_m,
            0.0,
        )
        # The tolerance keeps exact tread boundaries on the same level in
        # NumPy float64 and JAX float32.
        step_index = jnp.ceil(
            distance_from_platform_m / tread_depth_m - 1e-6
        )
        step_index = jnp.clip(step_index, 0, nr_steps)

        return (
            step_index
            * self.step_height_m
            * curriculum_coeff
        )
