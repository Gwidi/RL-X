import numpy as np

from rl_x.environments.custom_mujoco.robot_locomotion.inverted_pyramid_boxes import (
    DEFAULT_HALF_WIDTH_M,
    maximum_number_of_steps,
)


class HFieldInvertedPyramidStairsTerrainGeneration:
    """Square stair basin built from box geoms around a flat center."""

    def __init__(self, env):
        self.env = env
        terrain_config = self.env.env_config["terrain"]

        self.half_width_m = terrain_config.get(
            "inverted_pyramid_half_width_m",
            DEFAULT_HALF_WIDTH_M,
        )
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
            0.85,
        )
        self.tread_depth_scale_max = terrain_config.get(
            "inverted_pyramid_tread_depth_scale_max",
            1.15,
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
        self.max_nr_steps = maximum_number_of_steps(terrain_config)
        self.terrain_geom_ids = np.asarray(
            getattr(env, "terrain_geom_ids", ()),
            dtype=np.int32,
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

    def rng(self):
        get_rng = getattr(self.env, "get_terrain_rng", None)
        return get_rng() if get_rng is not None else self.env.np_rng

    def init(self):
        self.env.internal_state["center_height"] = 0.0
        self.env.internal_state["robot_imu_height_over_ground"] = (
            self.env.initial_imu_height
        )
        self.env.internal_state[
            "terrain/inverted_pyramid_step_height_m"
        ] = 0.0
        self.env.internal_state[
            "terrain/inverted_pyramid_max_height_m"
        ] = 0.0
        self.env.internal_state[
            "terrain/inverted_pyramid_tread_depth_m"
        ] = self.tread_depth_m
        initial_tread_depths_m = np.full(
            self.max_nr_steps,
            self.tread_depth_m,
        )
        self.env.internal_state[
            "terrain/inverted_pyramid_tread_depths_m"
        ] = initial_tread_depths_m
        self.env.internal_state[
            "terrain/inverted_pyramid_step_inner_edges_m"
        ] = self.step_inner_edges(initial_tread_depths_m)
        self.env.internal_state[
            "terrain/inverted_pyramid_nr_steps"
        ] = self.number_of_steps(initial_tread_depths_m)

    def sample(self):
        if self.terrain_geom_ids.size != self.max_nr_steps * 4:
            raise RuntimeError(
                "The model does not contain the expected inverted-pyramid "
                "box geoms."
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
        sampled_tread_depths_m = self.sample_tread_depths(curriculum_coeff)
        positions, sizes, rbounds, nr_steps = self.box_geometry(
            curriculum_coeff,
            sampled_tread_depths_m,
        )
        inner_edges_m = self.step_inner_edges(sampled_tread_depths_m)
        outer_edges_m = np.minimum(
            inner_edges_m + sampled_tread_depths_m,
            self.half_width_m,
        )
        active = np.arange(self.max_nr_steps) < nr_steps
        applied_tread_depths_m = np.where(
            active,
            outer_edges_m - inner_edges_m,
            0.0,
        )

        model = self.env.internal_state["mj_model"]
        model.geom_pos[self.terrain_geom_ids] = positions
        model.geom_size[self.terrain_geom_ids] = sizes
        model.geom_rbound[self.terrain_geom_ids] = rbounds

        applied_step_height_m = float(curriculum_coeff) * self.step_height_m
        self.env.internal_state[
            "terrain/inverted_pyramid_step_height_m"
        ] = applied_step_height_m
        self.env.internal_state[
            "terrain/inverted_pyramid_max_height_m"
        ] = nr_steps * applied_step_height_m
        self.env.internal_state[
            "terrain/inverted_pyramid_tread_depth_m"
        ] = np.sum(applied_tread_depths_m) / max(nr_steps, 1)
        self.env.internal_state[
            "terrain/inverted_pyramid_tread_depths_m"
        ] = applied_tread_depths_m
        self.env.internal_state[
            "terrain/inverted_pyramid_step_inner_edges_m"
        ] = inner_edges_m
        self.env.internal_state[
            "terrain/inverted_pyramid_nr_steps"
        ] = nr_steps

    def sample_tread_depths(self, curriculum_coeff):
        if not self.randomize_tread_depth:
            return np.full(self.max_nr_steps, self.tread_depth_m)

        randomization_coeff = np.clip(curriculum_coeff, 0.0, 1.0)
        min_scale = 1.0 + randomization_coeff * (
            self.tread_depth_scale_min - 1.0
        )
        max_scale = 1.0 + randomization_coeff * (
            self.tread_depth_scale_max - 1.0
        )
        return self.tread_depth_m * self.rng().uniform(
            low=min_scale,
            high=max_scale,
            size=self.max_nr_steps,
        )

    def step_inner_edges(self, tread_depths_m):
        tread_depths_m = np.asarray(tread_depths_m)
        if tread_depths_m.ndim == 0:
            tread_depths_m = np.full(
                self.max_nr_steps,
                tread_depths_m,
            )
        return np.minimum(
            self.center_half_width_m
            + np.concatenate(
                (np.zeros(1), np.cumsum(tread_depths_m[:-1]))
            ),
            self.half_width_m,
        )

    def number_of_steps(self, tread_depths_m):
        return int(
            np.count_nonzero(
                self.step_inner_edges(tread_depths_m)
                < self.half_width_m - 1e-6
            )
        )

    def box_geometry(self, curriculum_coeff, tread_depths_m):
        """Returns four non-overlapping rectangles for every square tread."""
        tread_depths_m = np.asarray(tread_depths_m)
        if tread_depths_m.ndim == 0:
            tread_depths_m = np.full(
                self.max_nr_steps,
                tread_depths_m,
            )
        nr_steps = self.number_of_steps(tread_depths_m)
        step_indices = np.arange(self.max_nr_steps)
        inner = self.step_inner_edges(tread_depths_m)
        outer = np.minimum(
            inner + tread_depths_m,
            self.half_width_m,
        )
        half_depth = np.maximum((outer - inner) / 2.0, 0.001)
        middle = (inner + outer) / 2.0
        half_height = (
            (step_indices + 1)
            * self.step_height_m
            * float(curriculum_coeff)
            / 2.0
        )
        active = (step_indices < nr_steps) & (half_height > 0.0)

        positions = np.stack(
            (
                np.stack((np.zeros_like(middle), middle, half_height), axis=1),
                np.stack((np.zeros_like(middle), -middle, half_height), axis=1),
                np.stack((middle, np.zeros_like(middle), half_height), axis=1),
                np.stack((-middle, np.zeros_like(middle), half_height), axis=1),
            ),
            axis=1,
        )
        sizes = np.stack(
            (
                np.stack((outer, half_depth, half_height), axis=1),
                np.stack((outer, half_depth, half_height), axis=1),
                np.stack((half_depth, inner, half_height), axis=1),
                np.stack((half_depth, inner, half_height), axis=1),
            ),
            axis=1,
        )

        hidden_z = -(self.max_nr_steps * self.step_height_m + 1.0)
        positions = np.where(
            active[:, None, None],
            positions,
            np.asarray([0.0, 0.0, hidden_z]),
        )
        sizes = np.where(active[:, None, None], sizes, 0.001)
        positions = positions.reshape(-1, 3)
        sizes = sizes.reshape(-1, 3)
        rbounds = np.linalg.norm(sizes, axis=1)
        return positions, sizes, rbounds, nr_steps

    def ground_height_at(self, x_in_m, y_in_m):
        radius = np.maximum(np.abs(x_in_m), np.abs(y_in_m))
        inner_edges_m = self.env.internal_state[
            "terrain/inverted_pyramid_step_inner_edges_m"
        ]
        nr_steps = self.env.internal_state[
            "terrain/inverted_pyramid_nr_steps"
        ]
        step_index = np.clip(
            np.sum(
                radius[..., None] > inner_edges_m + 1e-6,
                axis=-1,
            ),
            0,
            nr_steps,
        )
        inside = radius <= self.half_width_m
        return np.where(
            inside,
            step_index
            * self.env.internal_state[
                "terrain/inverted_pyramid_step_height_m"
            ],
            0.0,
        )

    def check_feet_floor_contact(self):
        contacts = self.env.internal_state["data"].contact.geom
        if contacts.shape[0] == 0:
            return np.zeros(self.env.nr_feet, dtype=bool)
        ground_ids = self.env.ground_geom_ids
        return np.asarray(
            [
                np.any(
                    ((contacts[:, 0] == foot_id) & np.isin(contacts[:, 1], ground_ids))
                    | ((contacts[:, 1] == foot_id) & np.isin(contacts[:, 0], ground_ids))
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
            global_corners[:, :, 0],
            global_corners[:, :, 1],
        )
        return np.sum(global_corners[:, :, 2] > floor_height, axis=1)

    def pre_step(self):
        imu_pos = self.env.internal_state["data"].site_xpos[
            self.env.imu_site_id
        ]
        self.env.internal_state["robot_imu_height_over_ground"] = (
            imu_pos[2] - self.ground_height_at(imu_pos[0], imu_pos[1])
        )

    def post_step(self):
        min_edge = self.half_width_m - 0.5
        qpos = self.env.internal_state["data"].qpos
        reached_edge = np.any(
            (min_edge < np.abs(qpos[:2]))
            & (np.abs(qpos[:2]) < self.half_width_m)
        )
        if reached_edge:
            qpos, qvel = self.env.initial_state_function.setup()
            self.env.internal_state["data"].qpos = qpos
            self.env.internal_state["data"].qvel = qvel

    def get_debug_overlay(self):
        state = self.env.internal_state
        return [
            ("Terrain", "inverted pyramid stairs (boxes)"),
            (
                "Mean tread depth",
                f'{state["terrain/inverted_pyramid_tread_depth_m"]:.3f} m',
            ),
            (
                "Number of steps",
                str(state["terrain/inverted_pyramid_nr_steps"]),
            ),
            (
                "Step height",
                f'{state["terrain/inverted_pyramid_step_height_m"]:.3f} m',
            ),
            ("Center half-width", f"{self.center_half_width_m:.3f} m"),
            (
                "Max height",
                f'{state["terrain/inverted_pyramid_max_height_m"]:.3f} m',
            ),
        ]
