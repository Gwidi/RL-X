import numpy as np
import jax.numpy as jnp

from rl_x.environments.custom_mujoco.robot_locomotion.hurdles_boxes import (
    DEFAULT_HALF_WIDTH_M,
    number_of_walls,
)


class HFieldHurdlesTerrainGeneration:
    """Thin concentric square walls built from MJX box geoms."""

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
        self.terrain_geom_ids = jnp.asarray(
            getattr(env, "terrain_geom_ids", ()), dtype=jnp.int32
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
        if self.wall_centers_m[0] - self.wall_thickness_m / 2.0 <= 0.0:
            raise ValueError(
                "The first hurdle wall must leave a flat spawn area."
            )
        if (
            self.wall_centers_m[-1] + self.wall_thickness_m / 2.0
            > self.half_width_m
        ):
            raise ValueError(
                "The outer hurdle wall must fit inside the terrain."
            )

    def init(self, internal_state):
        internal_state["center_height"] = 0.0
        internal_state["robot_imu_height_over_ground"] = (
            self.env.initial_imu_height
        )
        internal_state["terrain/hurdles_height_m"] = 0.0
        internal_state[
            "terrain/hurdles_effective_thickness_m"
        ] = self.wall_thickness_m

    def sample(self, mjx_model, internal_state, key):
        del key
        if self.terrain_geom_ids.size != self.wall_count * 4:
            raise RuntimeError(
                "The model does not contain the expected hurdle box geoms."
            )
        curriculum_coeff = jnp.where(
            self.use_curriculum,
            internal_state["env_curriculum_coeff"],
            1.0,
        )
        curriculum_coeff = jnp.where(
            internal_state["in_eval_mode"], 1.0, curriculum_coeff
        )
        positions, sizes, rbounds = self.box_geometry(curriculum_coeff)
        internal_state["terrain/hurdles_height_m"] = (
            curriculum_coeff * self.wall_height_m
        )
        return mjx_model.replace(
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

    def box_geometry(self, curriculum_coeff):
        half_thickness = self.wall_thickness_m / 2.0
        centers = jnp.asarray(self.wall_centers_m)
        inner = centers - half_thickness
        outer = centers + half_thickness
        half_height = jnp.full(
            (self.wall_count,),
            self.wall_height_m * curriculum_coeff / 2.0,
        )
        active = half_height > 0.0
        zeros = jnp.zeros_like(inner)
        thicknesses = jnp.full_like(inner, half_thickness)
        positions = jnp.stack(
            (
                jnp.stack((zeros, centers, half_height), axis=1),
                jnp.stack((zeros, -centers, half_height), axis=1),
                jnp.stack((centers, zeros, half_height), axis=1),
                jnp.stack((-centers, zeros, half_height), axis=1),
            ),
            axis=1,
        )
        sizes = jnp.stack(
            (
                jnp.stack((outer, thicknesses, half_height), axis=1),
                jnp.stack((outer, thicknesses, half_height), axis=1),
                jnp.stack((thicknesses, inner, half_height), axis=1),
                jnp.stack((thicknesses, inner, half_height), axis=1),
            ),
            axis=1,
        )
        hidden = jnp.asarray([0.0, 0.0, -(self.wall_height_m + 1.0)])
        positions = jnp.where(active[:, None, None], positions, hidden)
        sizes = jnp.where(active[:, None, None], sizes, 0.001)
        positions = positions.reshape(-1, 3)
        sizes = sizes.reshape(-1, 3)
        return positions, sizes, jnp.linalg.norm(sizes, axis=1)

    def ground_height_at(self, internal_state, x_in_m, y_in_m):
        square_radius_m = jnp.maximum(jnp.abs(x_in_m), jnp.abs(y_in_m))
        centers = jnp.asarray(self.wall_centers_m).reshape(
            (-1,) + (1,) * jnp.ndim(square_radius_m)
        )
        distance_to_wall_m = jnp.min(
            jnp.abs(jnp.asarray(square_radius_m)[None, ...] - centers), axis=0
        )
        return jnp.where(
            distance_to_wall_m <= self.wall_thickness_m / 2.0,
            internal_state["terrain/hurdles_height_m"],
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
        self, data, mjx_model, internal_state
    ):
        if self.env.foot_type == "sphere":
            return jnp.zeros(self.env.nr_feet)
        feet_xpos = data.geom_xpos[self.env.foot_geom_indices]
        feet_xmat = data.geom_xmat[self.env.foot_geom_indices].reshape(-1, 3, 3)
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
            - self.ground_height_at(internal_state, imu_pos[0], imu_pos[1])
        )

    def post_step(self, data, mjx_model, internal_state, key):
        return data
