import numpy as np

from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.terrain_functions.hfield_bunker_ruins import (
    HFieldBunkerRuinsTerrainGeneration,
)


class HFieldBunkerStairsTerrainGeneration(HFieldBunkerRuinsTerrainGeneration):
    def bunker_ruins_terrain(self, max_obstacle_height, max_slope_height, noise_height):
        terrain = super().bunker_ruins_terrain(
            max_obstacle_height=max_obstacle_height,
            max_slope_height=max_slope_height,
            noise_height=noise_height,
        )

        terrain = self.add_stair_bands(terrain, max_obstacle_height)
        terrain = self.clear_safe_start_zone(terrain)

        return terrain

    def add_stair_bands(self, terrain, max_obstacle_height):
        num_stair_bands = max(2, int(8 * (self.hfield_half_length_in_meters / 10.0) ** 2))

        y_idx, x_idx = np.meshgrid(
            np.arange(self.hfield_length),
            np.arange(self.hfield_length),
            indexing="ij",
        )

        cx = self.env.np_rng.uniform(low=0, high=self.hfield_length, size=(num_stair_bands, 1, 1))
        cy = self.env.np_rng.uniform(low=0, high=self.hfield_length, size=(num_stair_bands, 1, 1))

        yaw = self.env.np_rng.uniform(low=0.0, high=2 * np.pi, size=(num_stair_bands, 1, 1))
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        tread_depth_px = self.env.np_rng.uniform(
            low=0.35 * self.one_meter_length,
            high=0.65 * self.one_meter_length,
            size=(num_stair_bands, 1, 1),
        )
        band_width_px = self.env.np_rng.uniform(
            low=0.8 * self.one_meter_length,
            high=1.6 * self.one_meter_length,
            size=(num_stair_bands, 1, 1),
        )
        step_height = self.env.np_rng.uniform(
            low=0.35 * max_obstacle_height,
            high=0.75 * max_obstacle_height,
            size=(num_stair_bands, 1, 1),
        )
        num_steps = self.env.np_rng.integers(low=3, high=7, size=(num_stair_bands, 1, 1))

        dx = x_idx[None, :, :] - cx
        dy = y_idx[None, :, :] - cy

        longitudinal_px = dx * cos_yaw + dy * sin_yaw
        lateral_px = -dx * sin_yaw + dy * cos_yaw

        step_index = np.floor(longitudinal_px / tread_depth_px) + 1.0
        step_index = np.clip(step_index, 0.0, num_steps.astype(np.float32))
        stair_heights = step_index * step_height

        in_stair_band = (
            (longitudinal_px >= 0.0)
            & (longitudinal_px < num_steps * tread_depth_px)
            & (np.abs(lateral_px) < band_width_px / 2.0)
        )
        stair_heights = np.where(in_stair_band, stair_heights, -1000.0)
        stair_layer = np.max(stair_heights, axis=0)

        return np.maximum(terrain, stair_layer)

    def clear_safe_start_zone(self, terrain):
        y_idx, x_idx = np.meshgrid(
            np.arange(self.hfield_length),
            np.arange(self.hfield_length),
            indexing="ij",
        )
        safe_radius_px = int(0.8 * self.one_meter_length)
        dist_from_center = np.sqrt((x_idx - self.hfield_half_length) ** 2 + (y_idx - self.hfield_half_length) ** 2)

        return np.where(dist_from_center < safe_radius_px, 0.0, terrain)
