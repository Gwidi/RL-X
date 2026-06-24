import jax
import jax.numpy as jnp

from rl_x.environments.custom_mujoco.robot_locomotion.mjx.terrain_functions.hfield_bunker_ruins import (
    HFieldBunkerRuinsTerrainGeneration,
)


class HFieldBunkerStairsTerrainGeneration(HFieldBunkerRuinsTerrainGeneration):
    def bunker_ruins_terrain(self, max_obstacle_height, noise_height, key):
        ruins_key, stairs_key = jax.random.split(key)

        terrain = super().bunker_ruins_terrain(
            max_obstacle_height=max_obstacle_height,
            noise_height=noise_height,
            key=ruins_key,
        )

        terrain = self.add_stair_bands(terrain, max_obstacle_height, stairs_key)
        terrain = self.clear_safe_start_zone(terrain)

        return terrain

    def add_stair_bands(self, terrain, max_obstacle_height, key):
        num_stair_bands = max(2, int(8 * (self.hfield_half_length_in_meters / 10.0) ** 2))
        keys = jax.random.split(key, 8)

        y_idx, x_idx = jnp.meshgrid(
            jnp.arange(self.hfield_length),
            jnp.arange(self.hfield_length),
            indexing="ij",
        )

        cx = jax.random.uniform(keys[0], shape=(num_stair_bands, 1, 1), minval=0, maxval=self.hfield_length)
        cy = jax.random.uniform(keys[1], shape=(num_stair_bands, 1, 1), minval=0, maxval=self.hfield_length)

        yaw = jax.random.uniform(keys[2], shape=(num_stair_bands, 1, 1), minval=0.0, maxval=2 * jnp.pi)
        cos_yaw = jnp.cos(yaw)
        sin_yaw = jnp.sin(yaw)

        tread_depth_px = jax.random.uniform(
            keys[3],
            shape=(num_stair_bands, 1, 1),
            minval=0.35 * self.one_meter_length,
            maxval=0.65 * self.one_meter_length,
        )
        band_width_px = jax.random.uniform(
            keys[4],
            shape=(num_stair_bands, 1, 1),
            minval=0.8 * self.one_meter_length,
            maxval=1.6 * self.one_meter_length,
        )
        step_height = jax.random.uniform(
            keys[5],
            shape=(num_stair_bands, 1, 1),
            minval=0.35 * max_obstacle_height,
            maxval=0.75 * max_obstacle_height,
        )
        num_steps = jax.random.randint(keys[6], shape=(num_stair_bands, 1, 1), minval=3, maxval=7)

        dx = x_idx[None, :, :] - cx
        dy = y_idx[None, :, :] - cy

        longitudinal_px = dx * cos_yaw + dy * sin_yaw
        lateral_px = -dx * sin_yaw + dy * cos_yaw

        step_index = jnp.floor(longitudinal_px / tread_depth_px) + 1.0
        step_index = jnp.clip(step_index, 0.0, num_steps.astype(jnp.float32))
        stair_heights = step_index * step_height

        in_stair_band = (
            (longitudinal_px >= 0.0)
            & (longitudinal_px < num_steps * tread_depth_px)
            & (jnp.abs(lateral_px) < band_width_px / 2.0)
        )
        stair_heights = jnp.where(in_stair_band, stair_heights, -1000.0)
        stair_layer = jnp.max(stair_heights, axis=0)

        return jnp.maximum(terrain, stair_layer)

    def clear_safe_start_zone(self, terrain):
        y_idx, x_idx = jnp.meshgrid(
            jnp.arange(self.hfield_length),
            jnp.arange(self.hfield_length),
            indexing="ij",
        )
        safe_radius_px = int(0.8 * self.one_meter_length)
        dist_from_center = jnp.sqrt((x_idx - self.hfield_half_length) ** 2 + (y_idx - self.hfield_half_length) ** 2)

        return jnp.where(dist_from_center < safe_radius_px, 0.0, terrain)
