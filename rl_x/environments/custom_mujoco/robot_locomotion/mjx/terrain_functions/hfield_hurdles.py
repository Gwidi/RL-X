import numpy as np
import jax.numpy as jnp

from rl_x.environments.custom_mujoco.robot_locomotion.mjx.terrain_functions.hfield_bunker_ruins import (
    HFieldBunkerRuinsTerrainGeneration,
)


class HFieldHurdlesTerrainGeneration(
    HFieldBunkerRuinsTerrainGeneration
):
    """Thin concentric square walls surrounding the robot spawn area."""

    def __init__(self, env):
        super().__init__(env)

        terrain_config = self.env.env_config["terrain"]
        self.first_wall_distance_m = terrain_config.get(
            "hurdles_first_wall_distance_m",
            1.0,
        )
        self.wall_spacing_m = terrain_config.get(
            "hurdles_wall_spacing_m",
            0.8,
        )
        self.wall_count = terrain_config.get(
            "hurdles_wall_count",
            4,
        )
        self.wall_thickness_m = terrain_config.get(
            "hurdles_wall_thickness_m",
            0.025,
        )
        self.wall_height_m = terrain_config.get(
            "hurdles_wall_height_m",
            0.20,
        )
        self.use_curriculum = terrain_config.get(
            "hurdles_use_curriculum",
            True,
        )

        self.grid_spacing_m = (
            2.0 * self.hfield_half_length_in_meters
            / (self.hfield_length - 1)
        )

        self._validate_config()
        self._configure_rasterized_walls()
        self._validate_wall_fit()

    def _validate_config(self):
        if (
            not np.isfinite(self.first_wall_distance_m)
            or self.first_wall_distance_m <= 0.0
        ):
            raise ValueError(
                "terrain.hurdles_first_wall_distance_m must be finite "
                "and positive."
            )
        if (
            not np.isfinite(self.wall_spacing_m)
            or self.wall_spacing_m <= 0.0
        ):
            raise ValueError(
                "terrain.hurdles_wall_spacing_m must be finite and "
                "positive."
            )
        if (
            isinstance(self.wall_count, bool)
            or not isinstance(self.wall_count, (int, np.integer))
            or self.wall_count <= 0
        ):
            raise ValueError(
                "terrain.hurdles_wall_count must be a positive integer."
            )
        if (
            not np.isfinite(self.wall_thickness_m)
            or self.wall_thickness_m <= 0.0
        ):
            raise ValueError(
                "terrain.hurdles_wall_thickness_m must be finite and "
                "positive."
            )
        if (
            not np.isfinite(self.wall_height_m)
            or not 0.0 <= self.wall_height_m <= self.max_possible_height
        ):
            raise ValueError(
                "terrain.hurdles_wall_height_m must be finite and "
                "between zero and the heightfield height range."
            )

    def _configure_rasterized_walls(self):
        """Snap flat wall tops to heightfield vertices."""
        wall_top_interval_count = max(
            1,
            int(np.ceil(
                self.wall_thickness_m / self.grid_spacing_m
            )),
        )
        self.effective_wall_thickness_m = (
            wall_top_interval_count * self.grid_spacing_m
        )

        radius_grid_origin_m = (
            0.0
            if self.hfield_length % 2 == 1
            else self.grid_spacing_m / 2.0
        )
        wall_center_grid_origin_m = (
            radius_grid_origin_m
            + (wall_top_interval_count % 2)
            * self.grid_spacing_m
            / 2.0
        )
        requested_wall_centers_m = (
            self.first_wall_distance_m
            + np.arange(self.wall_count) * self.wall_spacing_m
        )
        wall_center_grid_indices = np.rint(
            (
                requested_wall_centers_m
                - wall_center_grid_origin_m
            )
            / self.grid_spacing_m
        )
        self.wall_centers_m = (
            wall_center_grid_origin_m
            + wall_center_grid_indices * self.grid_spacing_m
        )

    def _validate_wall_fit(self):
        inner_wall_edge_m = (
            self.wall_centers_m[0]
            - self.effective_wall_thickness_m / 2.0
        )
        if inner_wall_edge_m <= 0.0:
            raise ValueError(
                "The first hurdle wall must leave a flat spawn area."
            )

        outer_wall_edge_m = (
            self.wall_centers_m[-1]
            + self.effective_wall_thickness_m / 2.0
        )
        if (
            outer_wall_edge_m > self.hfield_half_length_in_meters
        ):
            raise ValueError(
                "The outer hurdle wall must fit inside the heightfield."
            )

    def init(self, internal_state):
        super().init(internal_state)
        internal_state["terrain/hurdles_height_m"] = 0.0
        internal_state[
            "terrain/hurdles_effective_thickness_m"
        ] = self.effective_wall_thickness_m

    def sample(self, mjx_model, internal_state, key):
        del key

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

        height_field = self.hurdles_terrain(curriculum_coeff)
        new_height_field_data = self.isaac_hf_to_mujoco_hf(
            height_field
        )
        new_mjx_model = mjx_model.replace(
            hfield_data=new_height_field_data,
        )

        center_idx = (
            self.hfield_half_length * self.hfield_length
            + self.hfield_half_length
        )
        internal_state["center_height"] = (
            new_height_field_data[center_idx]
            * self.mujoco_height_scaling
        )
        internal_state["current_height_field_data"] = (
            new_height_field_data.reshape(
                self.hfield_length,
                self.hfield_length,
            )
        )
        internal_state["terrain/hurdles_height_m"] = (
            curriculum_coeff * self.wall_height_m
        )

        return new_mjx_model

    def hurdles_terrain(self, curriculum_coeff):
        """Build repeated square hurdle walls around the heightfield center."""
        coordinates_m = (
            jnp.arange(self.hfield_length)
            - (self.hfield_length - 1) / 2.0
        ) * self.grid_spacing_m
        x_grid_m = coordinates_m[None, :]
        y_grid_m = coordinates_m[:, None]
        square_radius_m = jnp.maximum(
            jnp.abs(x_grid_m),
            jnp.abs(y_grid_m),
        )

        wall_centers_m = jnp.asarray(self.wall_centers_m)
        distance_to_wall_m = jnp.min(
            jnp.abs(
                square_radius_m[None, :, :]
                - wall_centers_m[:, None, None]
            ),
            axis=0,
        )
        wall_mask = distance_to_wall_m <= (
            self.effective_wall_thickness_m / 2.0
            + self.grid_spacing_m * 1e-4
        )

        return (
            curriculum_coeff
            * self.wall_height_m
            * wall_mask.astype(jnp.float32)
        )
