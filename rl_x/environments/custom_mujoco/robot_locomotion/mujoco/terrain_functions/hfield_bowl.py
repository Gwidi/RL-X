import numpy as np

from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.terrain_functions.hfield_bunker_ruins import (
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

    def init(self):
        super().init()
        self.env.internal_state["terrain/bowl_depth_m"] = 0.0
        self.env.internal_state["terrain/bowl_max_height_m"] = 0.0

    def sample(self):
        curriculum_coeff = (
            self.env.internal_state["env_curriculum_coeff"]
            if self.use_curriculum
            else 1.0
        )
        curriculum_coeff = np.where(
            self.env.internal_state["in_eval_mode"],
            1.0,
            curriculum_coeff,
        )

        height_field = self.bowl_terrain(curriculum_coeff)
        new_height_field_data = self.isaac_hf_to_mujoco_hf(height_field)

        self.env.internal_state["mj_model"].hfield_data = (
            new_height_field_data
        )

        center_idx = (
            self.hfield_half_length * self.hfield_length
            + self.hfield_half_length
        )
        self.env.internal_state["center_height"] = (
            new_height_field_data[center_idx] * self.mujoco_height_scaling
        )
        self.env.internal_state["current_height_field_data"] = (
            new_height_field_data.reshape(
                self.hfield_length,
                self.hfield_length,
            )
        )
        self.env.internal_state["terrain/bowl_depth_m"] = (
            curriculum_coeff * self.depth_m
        )
        self.env.internal_state["terrain/bowl_max_height_m"] = np.max(
            height_field
        )

    def bowl_terrain(self, curriculum_coeff):
        """Build a radial quadratic profile rising from the flat center."""
        coordinates_m = np.linspace(
            -self.hfield_half_length_in_meters,
            self.hfield_half_length_in_meters,
            self.hfield_length,
        )
        x_grid_m = coordinates_m[None, :]
        y_grid_m = coordinates_m[:, None]

        radial_distance_m = np.sqrt(
            x_grid_m**2 + y_grid_m**2
        )
        normalized_slope_position = np.clip(
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

    def get_debug_overlay(self):
        applied_depth_m = self.env.internal_state[
            "terrain/bowl_depth_m"
        ]
        max_height_m = self.env.internal_state[
            "terrain/bowl_max_height_m"
        ]
        return [
            ("Terrain", "bowl"),
            ("Radius", f"{self.radius_m:.3f} m"),
            ("Flat radius", f"{self.flat_radius_m:.3f} m"),
            ("Depth", f"{applied_depth_m:.3f} m"),
            ("Max height", f"{max_height_m:.3f} m"),
        ]
