import jax
import jax.numpy as jnp

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
        self.max_nr_steps = maximum_number_of_steps(terrain_config)
        self.terrain_geom_ids = jnp.asarray(
            getattr(env, "terrain_geom_ids", ()),
            dtype=jnp.int32,
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
        internal_state["center_height"] = 0.0
        internal_state["robot_imu_height_over_ground"] = (
            self.env.initial_imu_height
        )
        internal_state["terrain/inverted_pyramid_step_height_m"] = 0.0
        internal_state["terrain/inverted_pyramid_max_height_m"] = 0.0
        internal_state[
            "terrain/inverted_pyramid_tread_depth_m"
        ] = jnp.asarray(self.tread_depth_m)
        internal_state[
            "terrain/inverted_pyramid_nr_steps"
        ] = self.number_of_steps(self.tread_depth_m)

    def sample(self, mjx_model, internal_state, key):
        if self.terrain_geom_ids.size != self.max_nr_steps * 4:
            raise RuntimeError(
                "The model does not contain the expected inverted-pyramid "
                "box geoms."
            )
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
        positions, sizes, rbounds, nr_steps = self.box_geometry(
            curriculum_coeff,
            sampled_tread_depth_m,
        )
        new_mjx_model = mjx_model.replace(
            geom_pos=mjx_model.geom_pos.at[self.terrain_geom_ids].set(
                positions
            ),
            geom_size=mjx_model.geom_size.at[self.terrain_geom_ids].set(
                sizes
            ),
            geom_rbound=mjx_model.geom_rbound.at[
                self.terrain_geom_ids
            ].set(rbounds),
        )

        applied_step_height_m = curriculum_coeff * self.step_height_m
        internal_state[
            "terrain/inverted_pyramid_step_height_m"
        ] = applied_step_height_m
        internal_state[
            "terrain/inverted_pyramid_max_height_m"
        ] = nr_steps * applied_step_height_m
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
        return self.tread_depth_m * jax.random.uniform(
            key,
            shape=(),
            minval=min_scale,
            maxval=max_scale,
        )

    def number_of_steps(self, tread_depth_m):
        return jnp.ceil(
            (self.half_width_m - self.center_half_width_m)
            / tread_depth_m
        ).astype(jnp.int32)

    def box_geometry(self, curriculum_coeff, tread_depth_m):
        nr_steps = self.number_of_steps(tread_depth_m)
        step_indices = jnp.arange(self.max_nr_steps)
        inner = jnp.minimum(
            self.center_half_width_m + step_indices * tread_depth_m,
            self.half_width_m,
        )
        outer = jnp.minimum(inner + tread_depth_m, self.half_width_m)
        half_depth = jnp.maximum((outer - inner) / 2.0, 0.001)
        middle = (inner + outer) / 2.0
        half_height = (
            (step_indices + 1)
            * self.step_height_m
            * curriculum_coeff
            / 2.0
        )
        active = (step_indices < nr_steps) & (half_height > 0.0)
        zeros = jnp.zeros_like(middle)

        positions = jnp.stack(
            (
                jnp.stack((zeros, middle, half_height), axis=1),
                jnp.stack((zeros, -middle, half_height), axis=1),
                jnp.stack((middle, zeros, half_height), axis=1),
                jnp.stack((-middle, zeros, half_height), axis=1),
            ),
            axis=1,
        )
        sizes = jnp.stack(
            (
                jnp.stack((outer, half_depth, half_height), axis=1),
                jnp.stack((outer, half_depth, half_height), axis=1),
                jnp.stack((half_depth, inner, half_height), axis=1),
                jnp.stack((half_depth, inner, half_height), axis=1),
            ),
            axis=1,
        )
        hidden = jnp.asarray(
            [
                0.0,
                0.0,
                -(self.max_nr_steps * self.step_height_m + 1.0),
            ]
        )
        positions = jnp.where(active[:, None, None], positions, hidden)
        sizes = jnp.where(active[:, None, None], sizes, 0.001)
        positions = positions.reshape(-1, 3)
        sizes = sizes.reshape(-1, 3)
        rbounds = jnp.linalg.norm(sizes, axis=1)
        return positions, sizes, rbounds, nr_steps

    def ground_height_at(self, internal_state, x_in_m, y_in_m):
        radius = jnp.maximum(jnp.abs(x_in_m), jnp.abs(y_in_m))
        distance = jnp.maximum(
            radius - self.center_half_width_m,
            0.0,
        )
        step_index = jnp.clip(
            jnp.ceil(
                distance
                / internal_state[
                    "terrain/inverted_pyramid_tread_depth_m"
                ]
                - 1e-6
            ),
            0,
            internal_state["terrain/inverted_pyramid_nr_steps"],
        )
        return jnp.where(
            radius <= self.half_width_m,
            step_index
            * internal_state[
                "terrain/inverted_pyramid_step_height_m"
            ],
            0.0,
        )

    def check_feet_floor_contact(self, data):
        contact_geoms = data._impl.contact.geom
        contact_dist = data._impl.contact.dist
        foot_ids = self.env.foot_geom_indices[:, None]
        ground_ids = self.env.ground_geom_ids
        geom1_is_foot = contact_geoms[None, :, 0] == foot_ids
        geom2_is_foot = contact_geoms[None, :, 1] == foot_ids
        geom1_is_ground = jnp.isin(contact_geoms[:, 0], ground_ids)[None, :]
        geom2_is_ground = jnp.isin(contact_geoms[:, 1], ground_ids)[None, :]
        matches = (geom1_is_foot & geom2_is_ground) | (
            geom2_is_foot & geom1_is_ground
        )
        return jnp.any(matches & (contact_dist[None, :] < 0.0), axis=1)

    def check_flat_feet_floor_missing_contacts(
        self,
        data,
        mjx_model,
        internal_state,
    ):
        if self.env.foot_type == "sphere":
            return jnp.zeros(self.env.nr_feet)
        feet_xpos = data.geom_xpos[self.env.foot_geom_indices]
        feet_xmat = data.geom_xmat[
            self.env.foot_geom_indices
        ].reshape(-1, 3, 3)
        feet_sizes = mjx_model.geom_size[self.env.foot_geom_indices]
        corners = jnp.asarray(
            [[1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]]
        )[None, :, :] * feet_sizes[:, None, :]
        global_corners = (
            jnp.einsum("fij,fgj->fgi", feet_xmat, corners)
            + feet_xpos[:, None, :]
        )
        floor_height = self.ground_height_at(
            internal_state,
            global_corners[:, :, 0],
            global_corners[:, :, 1],
        )
        return jnp.sum(global_corners[:, :, 2] > floor_height, axis=1)

    def pre_step(self, data, internal_state):
        imu_pos = data.site_xpos[self.env.imu_site_id]
        internal_state["robot_imu_height_over_ground"] = (
            imu_pos[2]
            - self.ground_height_at(
                internal_state,
                imu_pos[0],
                imu_pos[1],
            )
        )

    def post_step(self, data, mjx_model, internal_state, key):
        min_edge = self.half_width_m - 0.5
        reached_edge = jnp.any(
            (min_edge < jnp.abs(data.qpos[:2]))
            & (jnp.abs(data.qpos[:2]) < self.half_width_m)
        )
        qpos, qvel = self.env.initial_state_function.setup(
            mjx_model,
            internal_state,
            key,
        )
        initial_state_data = data.replace(qpos=qpos, qvel=qvel)
        return jax.lax.cond(
            reached_edge,
            lambda _: initial_state_data,
            lambda _: data,
            None,
        )
