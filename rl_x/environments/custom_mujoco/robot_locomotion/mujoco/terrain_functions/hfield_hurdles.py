import numpy as np

from rl_x.environments.custom_mujoco.robot_locomotion.hurdles_boxes import (
    DEFAULT_HALF_WIDTH_M,
    number_of_walls,
)


class HFieldHurdlesTerrainGeneration:
    """Thin concentric square walls built from MuJoCo box geoms."""

    def __init__(self, env):
        self.env = env
        terrain_config = self.env.env_config["terrain"]
        self.first_wall_distance_m = terrain_config.get(
            "hurdles_first_wall_distance_m", 1.0
        )
        self.wall_spacing_m = terrain_config.get(
            "hurdles_wall_spacing_m", 0.8
        )
        self.wall_count = number_of_walls(terrain_config)
        self.wall_thickness_m = terrain_config.get(
            "hurdles_wall_thickness_m", 0.025
        )
        self.wall_height_m = terrain_config.get(
            "hurdles_wall_height_m", 0.20
        )
        self.use_curriculum = terrain_config.get(
            "hurdles_use_curriculum", True
        )
        self.half_width_m = terrain_config.get(
            "hurdles_half_width_m", DEFAULT_HALF_WIDTH_M
        )
        self.wall_centers_m = (
            self.first_wall_distance_m
            + np.arange(self.wall_count) * self.wall_spacing_m
        )
        self.effective_wall_thickness_m = self.wall_thickness_m
        self.terrain_geom_ids = np.asarray(
            getattr(env, "terrain_geom_ids", ()), dtype=np.int32
        )
        self._validate_config()

    def _validate_config(self):
        positive_values = (
            ("hurdles_first_wall_distance_m", self.first_wall_distance_m),
            ("hurdles_wall_spacing_m", self.wall_spacing_m),
            ("hurdles_wall_thickness_m", self.wall_thickness_m),
            ("hurdles_half_width_m", self.half_width_m),
        )
        for name, value in positive_values:
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"terrain.{name} must be finite and positive.")
        if not np.isfinite(self.wall_height_m) or self.wall_height_m < 0.0:
            raise ValueError(
                "terrain.hurdles_wall_height_m must be finite and "
                "non-negative."
            )
        inner_wall_edge_m = (
            self.wall_centers_m[0] - self.wall_thickness_m / 2.0
        )
        if inner_wall_edge_m <= 0.0:
            raise ValueError(
                "The first hurdle wall must leave a flat spawn area."
            )
        outer_wall_edge_m = (
            self.wall_centers_m[-1] + self.wall_thickness_m / 2.0
        )
        if outer_wall_edge_m > self.half_width_m:
            raise ValueError(
                "The outer hurdle wall must fit inside the terrain."
            )

    def init(self):
        self.env.internal_state["center_height"] = 0.0
        self.env.internal_state["robot_imu_height_over_ground"] = (
            self.env.initial_imu_height
        )
        self.env.internal_state["terrain/hurdles_height_m"] = 0.0
        self.env.internal_state[
            "terrain/hurdles_effective_thickness_m"
        ] = self.wall_thickness_m

    def sample(self):
        if self.terrain_geom_ids.size != self.wall_count * 4:
            raise RuntimeError(
                "The model does not contain the expected hurdle box geoms."
            )
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
        positions, sizes, rbounds = self.box_geometry(curriculum_coeff)
        model = self.env.internal_state["mj_model"]
        model.geom_pos[self.terrain_geom_ids] = positions
        model.geom_size[self.terrain_geom_ids] = sizes
        model.geom_rbound[self.terrain_geom_ids] = rbounds
        self.env.internal_state["terrain/hurdles_height_m"] = (
            float(curriculum_coeff) * self.wall_height_m
        )

    def box_geometry(self, curriculum_coeff):
        """Returns four non-overlapping boxes for each square wall."""
        half_thickness = self.wall_thickness_m / 2.0
        inner = self.wall_centers_m - half_thickness
        outer = self.wall_centers_m + half_thickness
        half_height = np.full(
            self.wall_count,
            self.wall_height_m * float(curriculum_coeff) / 2.0,
        )
        active = half_height > 0.0
        zeros = np.zeros_like(inner)
        positions = np.stack(
            (
                np.stack((zeros, self.wall_centers_m, half_height), axis=1),
                np.stack((zeros, -self.wall_centers_m, half_height), axis=1),
                np.stack((self.wall_centers_m, zeros, half_height), axis=1),
                np.stack((-self.wall_centers_m, zeros, half_height), axis=1),
            ),
            axis=1,
        )
        sizes = np.stack(
            (
                np.stack(
                    (outer, np.full_like(outer, half_thickness), half_height),
                    axis=1,
                ),
                np.stack(
                    (outer, np.full_like(outer, half_thickness), half_height),
                    axis=1,
                ),
                np.stack(
                    (np.full_like(inner, half_thickness), inner, half_height),
                    axis=1,
                ),
                np.stack(
                    (np.full_like(inner, half_thickness), inner, half_height),
                    axis=1,
                ),
            ),
            axis=1,
        )
        hidden = np.asarray([0.0, 0.0, -(self.wall_height_m + 1.0)])
        positions = np.where(active[:, None, None], positions, hidden)
        sizes = np.where(active[:, None, None], sizes, 0.001)
        positions = positions.reshape(-1, 3)
        sizes = sizes.reshape(-1, 3)
        return positions, sizes, np.linalg.norm(sizes, axis=1)

    def ground_height_at(self, x_in_m, y_in_m):
        square_radius_m = np.maximum(np.abs(x_in_m), np.abs(y_in_m))
        distance_to_wall_m = np.min(
            np.abs(
                np.asarray(square_radius_m)[None, ...]
                - self.wall_centers_m.reshape(
                    (-1,) + (1,) * np.ndim(square_radius_m)
                )
            ),
            axis=0,
        )
        return np.where(
            distance_to_wall_m <= self.wall_thickness_m / 2.0,
            self.env.internal_state["terrain/hurdles_height_m"],
            0.0,
        )

    def check_feet_floor_contact(self):
        contacts = self.env.internal_state["data"].contact.geom
        if contacts.shape[0] == 0:
            return np.zeros(self.env.nr_feet, dtype=bool)
        return np.asarray(
            [
                np.any(
                    (
                        (contacts[:, 0] == foot_id)
                        & np.isin(contacts[:, 1], self.env.ground_geom_ids)
                    )
                    | (
                        (contacts[:, 1] == foot_id)
                        & np.isin(contacts[:, 0], self.env.ground_geom_ids)
                    )
                )
                for foot_id in self.env.foot_geom_indices
            ]
        )

    def check_flat_feet_floor_missing_contacts(self):
        if self.env.foot_type == "sphere":
            return np.zeros(self.env.nr_feet)
        data = self.env.internal_state["data"]
        feet_xpos = data.geom_xpos[self.env.foot_geom_indices]
        feet_xmat = data.geom_xmat[self.env.foot_geom_indices].reshape(-1, 3, 3)
        feet_sizes = self.env.internal_state["mj_model"].geom_size[
            self.env.foot_geom_indices
        ]
        corners = np.asarray(
            [[1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]]
        )[None, :, :] * feet_sizes[:, None, :]
        global_corners = (
            np.einsum("fij,fgj->fgi", feet_xmat, corners)
            + feet_xpos[:, None, :]
        )
        floor_height = self.ground_height_at(
            global_corners[:, :, 0], global_corners[:, :, 1]
        )
        return np.sum(global_corners[:, :, 2] > floor_height, axis=1)

    def pre_step(self):
        imu_pos = self.env.internal_state["data"].site_xpos[self.env.imu_site_id]
        self.env.internal_state["robot_imu_height_over_ground"] = (
            imu_pos[2] - self.ground_height_at(imu_pos[0], imu_pos[1])
        )

    def post_step(self):
        return

    def get_debug_overlay(self):
        return [
            ("Terrain", "hurdles (boxes)"),
            ("Wall count", str(self.wall_count)),
            ("First wall", f"{self.first_wall_distance_m:.3f} m"),
            ("Wall spacing", f"{self.wall_spacing_m:.3f} m"),
            ("Wall thickness", f"{self.wall_thickness_m:.3f} m"),
            (
                "Wall height",
                f'{self.env.internal_state["terrain/hurdles_height_m"]:.3f} m',
            ),
        ]
