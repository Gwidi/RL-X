import numpy as np


class HFieldCurbsTerrainGeneration:
    def __init__(self, env):
        self.env = env

        self.wave_fn_min = self.env.env_config["terrain"]["wave_fn_min"]
        self.wave_fn_max = self.env.env_config["terrain"]["wave_fn_max"]
        self.wave_height_max_per_m_factor = self.env.env_config["terrain"]["wave_height_max_per_m_factor"]
        self.random_height_max_per_m_factor = self.env.env_config["terrain"]["random_height_max_per_m_factor"]
        self.block_probability = self.env.env_config["terrain"]["block_probability"]
        self.block_length_in_meters = self.env.env_config["terrain"]["block_length_in_meters"]
        self.block_height_max_per_m_factor = self.env.env_config["terrain"]["block_height_max_per_m_factor"]

        hfield_size = self.env.initial_mj_model.hfield_size[0]
        if hfield_size[0] != hfield_size[1]:
            raise ValueError("The heightfield is not square.")

        self.hfield_length = self.env.initial_mj_model.hfield_ncol[0]
        self.hfield_half_length_in_meters = hfield_size[0]
        self.max_possible_height = hfield_size[2]

        self.one_meter_length = int(self.hfield_length / (self.hfield_half_length_in_meters * 2))
        self.hfield_half_length = self.hfield_length // 2
        self.mujoco_height_scaling = self.max_possible_height


    def init(self):
        self.env.internal_state["center_height"] = 0.0
        self.env.internal_state["robot_imu_height_over_ground"] = self.env.initial_imu_height - self.env.internal_state["center_height"]
        self.env.internal_state["current_height_field_data"] = self.env.initial_mj_model.hfield_data.reshape((self.hfield_length, self.hfield_length))


    def check_feet_floor_contact(self):
        contact_geom_pairs = self.env.internal_state["data"].contact.geom
        possible_contact_pairs = np.stack([np.full_like(self.env.foot_geom_indices, self.env.floor_geom_id), self.env.foot_geom_indices], axis=1)
        in_contact = np.any(np.all(contact_geom_pairs == possible_contact_pairs[:, None, :], axis=2), axis=1)

        return in_contact


    def check_flat_feet_floor_missing_contacts(self):
        if self.env.foot_type == "sphere":
            return np.zeros(self.env.nr_feet)
        elif self.env.foot_type == "box":
            feet_xpos = self.env.internal_state["data"].geom_xpos[self.env.foot_geom_indices]
            feet_xmat = self.env.internal_state["data"].geom_xmat[self.env.foot_geom_indices].reshape(-1, 3, 3)
            feet_sizes = self.env.internal_state["mj_model"].geom_size[self.env.foot_geom_indices]
            lower_base_corners = np.array([
                [1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]
            ])
            corners = lower_base_corners[None, :, :] * feet_sizes[:, None, :]
            global_corners = np.einsum("fij,fgj->fgi", feet_xmat, corners) + feet_xpos[:, None, :]
            floor_height_at_corners = self.ground_height_at(global_corners[:, :, 0], global_corners[:, :, 1])
            in_contact = np.sum(global_corners[:, :, 2] > floor_height_at_corners, axis=1)
        return in_contact


    def ground_height_at(self, x_in_m, y_in_m):
        x = np.clip(np.round(x_in_m * self.one_meter_length + self.hfield_half_length).astype(np.int32), 0, self.hfield_length-1)
        y = np.clip(np.round(y_in_m * self.one_meter_length + self.hfield_half_length).astype(np.int32), 0, self.hfield_length-1)
        return self.env.internal_state["current_height_field_data"][y, x] * self.mujoco_height_scaling


    def pre_step(self):
        self.env.internal_state["robot_imu_height_over_ground"] = self.env.internal_state["data"].site_xpos[self.env.imu_site_id, 2] - self.ground_height_at(self.env.internal_state["data"].site_xpos[self.env.imu_site_id, 0], self.env.internal_state["data"].site_xpos[self.env.imu_site_id, 1])
    

    def post_step(self):
        min_edge = self.hfield_half_length_in_meters - 0.5
        max_edge = self.hfield_half_length_in_meters
        reached_edge = np.array(((min_edge < np.abs(self.env.internal_state["data"].qpos[0])) & (np.abs(self.env.internal_state["data"].qpos[0]) < max_edge)) | ((min_edge < np.abs(self.env.internal_state["data"].qpos[1])) & (np.abs(self.env.internal_state["data"].qpos[1]) < max_edge)))
        if reached_edge:
            qpos, qvel = self.env.initial_state_function.setup()
            self.env.internal_state["data"].qpos = qpos
            self.env.internal_state["data"].qvel = qvel
    

    def sample(self):
        # Curb and noise heights increase with the Curriculum coefficient
        curriculum_coeff = self.env.internal_state["env_curriculum_coeff"]
        
        curb_height = curriculum_coeff * self.env.np_rng.uniform(
            low=0, 
            high=self.env.internal_state["robot_dimensions_mean"] * self.block_height_max_per_m_factor
        )
        
        noise_height = curriculum_coeff * self.env.np_rng.uniform(
            low=0, 
            high=self.env.internal_state["robot_dimensions_mean"] * self.random_height_max_per_m_factor
        )

        # Call the new function generating random curbs and trenches to step over
        isaac_height_field = self.dynamic_curbs_terrain(
            curb_height=curb_height,
            noise_height=noise_height
        )
        new_height_field_data = self.isaac_hf_to_mujoco_hf(isaac_height_field)

        self.env.internal_state["mj_model"].hfield_data = new_height_field_data

        self.env.internal_state["center_height"] = new_height_field_data[self.hfield_half_length * self.hfield_length + self.hfield_half_length] * self.mujoco_height_scaling
        self.env.internal_state["current_height_field_data"] = new_height_field_data.reshape(self.hfield_length, self.hfield_length)
    

    def isaac_hf_to_mujoco_hf(self, isaac_hf):
        hf = isaac_hf + np.abs(np.min(isaac_hf))
        hf /= self.mujoco_height_scaling
        return hf.reshape(-1)


    def dynamic_curbs_terrain(self, curb_height, noise_height):
        """
        Generates curbs and trenches perpendicular to the X-axis.
        The width of the obstacles and the gaps between them are randomized FOR EACH obstacle.
        """
        height_field_raw = np.zeros((self.hfield_length, self.hfield_length))

        # --- Adjustable parameters (you can move these to env_config) ---
        min_width_m = 0.05   # Min curb width (5 cm)
        max_width_m = 0.40   # Max curb width (40 cm)
        min_space_m = 0.60   # Min gap between curbs
        max_space_m = 1.40   # Max gap between curbs
        # ----------------------------------------------------------------

        current_x_px = 0
        
        # Iterate over the entire X-axis (robot's forward movement direction)
        while current_x_px < self.hfield_length:
            # 1. Randomize the width of the current obstacle
            obs_width_m = self.env.np_rng.uniform(low=min_width_m, high=max_width_m)
            obs_width_px = max(1, int(obs_width_m * self.one_meter_length))
            
            # 2. Randomize the gap to the next obstacle
            space_m = self.env.np_rng.uniform(low=min_space_m, high=max_space_m)
            space_px = max(1, int(space_m * self.one_meter_length))
            
            end_x_px = min(current_x_px + obs_width_px, self.hfield_length)
            
            # 3. Randomize whether it's a ridge (up) or a trench (down)
            # (Great for training blind policy on sudden ground drops/elevation changes)
            sign = self.env.np_rng.choice([1.0, -1.0])
            
            # Apply height to all Y values (creating a horizontal barrier across the map)
            # Note: In MuJoCo and this class, the first axis is Y, the second is X
            height_field_raw[:, current_x_px:end_x_px] = sign * curb_height
            
            # Shift the index to the start of the next obstacle
            current_x_px = end_x_px + space_px
        
        # 4. Add noise (rubble/roughness) on top of the terrain
        if noise_height > 0:
            # Size of the "rock" in pixels (if 1 px = 1 cm, then 10 px = 10 cm)
            # Increase this value if you want larger rocky blocks
            chunk_size = int(0.10 * self.one_meter_length) 
            chunk_size = max(1, chunk_size)
            
            # Create noise with lower resolution
            small_size_y = self.hfield_length // chunk_size
            small_size_x = self.hfield_length // chunk_size
            
            small_noise = self.env.np_rng.uniform(
                low=-noise_height, 
                high=noise_height, 
                size=(small_size_y, small_size_x)
            )
            
            # Scale the noise up (duplicate pixels), creating discrete blocks (rocks)
            blocky_noise = np.repeat(np.repeat(small_noise, chunk_size, axis=0), chunk_size, axis=1)
            
            # Adjust the size to match the full grid (in case of remainder from division)
            pad_y = self.hfield_length - blocky_noise.shape[0]
            pad_x = self.hfield_length - blocky_noise.shape[1]
            
            if pad_y > 0 or pad_x > 0:
                blocky_noise = np.pad(blocky_noise, ((0, pad_y), (0, pad_x)), mode='edge')
                
            height_field_raw += blocky_noise

        # 5. CRUCIAL FOR MUJOCO: Flatten the starting zone in the center of the map!
        # The robot must have a flat area to spawn, otherwise it might initialize inside a curb wall.
        safe_radius_m = 0.8 # Leave a 1.6m x 1.6m flat square in the center
        safe_radius_px = int(safe_radius_m * self.one_meter_length)
        
        center_x = self.hfield_half_length
        center_y = self.hfield_half_length
        
        # Zero out the terrain around the spawn point
        height_field_raw[
            max(0, center_y - safe_radius_px) : min(self.hfield_length, center_y + safe_radius_px),
            max(0, center_x - safe_radius_px) : min(self.hfield_length, center_x + safe_radius_px)
        ] = 0.0

        return height_field_raw