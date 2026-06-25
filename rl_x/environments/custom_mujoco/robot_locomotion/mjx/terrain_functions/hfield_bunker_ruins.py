import jax
import jax.numpy as jnp


class HFieldBunkerRuinsTerrainGeneration:
    def __init__(self, env):
        self.env = env

        terrain_config = self.env.env_config["terrain"]
        self.random_height_max_per_m_factor = terrain_config["random_height_max_per_m_factor"]
        self.block_height_max_per_m_factor = terrain_config["block_height_max_per_m_factor"]
        self.block_slope_height_max_per_m_factor = terrain_config.get("block_slope_height_max_per_m_factor", self.block_height_max_per_m_factor)

        hfield_size = self.env.initial_mjx_model.hfield_size[0]
        if hfield_size[0] != hfield_size[1]:
            raise ValueError("The heightfield is not square.")

        self.hfield_length = self.env.initial_mjx_model.hfield_ncol[0]
        self.hfield_half_length_in_meters = hfield_size[0]
        self.max_possible_height = hfield_size[2]

        self.one_meter_length = int(self.hfield_length / (self.hfield_half_length_in_meters * 2))
        self.hfield_half_length = self.hfield_length // 2
        self.mujoco_height_scaling = self.max_possible_height


    def init(self, internal_state):
        internal_state["center_height"] = 0.0
        internal_state["robot_imu_height_over_ground"] = self.env.initial_imu_height - internal_state["center_height"]
        internal_state["current_height_field_data"] = self.env.initial_mjx_model.hfield_data.reshape((self.hfield_length, self.hfield_length))


    def check_feet_floor_contact(self, data):
        contact_pairs = jnp.stack([jnp.full_like(self.env.foot_geom_indices, self.env.floor_geom_id), self.env.foot_geom_indices], axis=1)
        contact_pairs_rev = jnp.stack([self.env.foot_geom_indices, jnp.full_like(self.env.foot_geom_indices, self.env.floor_geom_id)], axis=1)
        mask1 = (data._impl.contact.geom[None, :, :] == contact_pairs[:, None, :]).all(axis=2)
        mask2 = (data._impl.contact.geom[None, :, :] == contact_pairs_rev[:, None, :]).all(axis=2)
        mask = mask1 | mask2
        masekd_dist = jnp.where(mask, data._impl.contact.dist[None, :], 1e4)
        indices = masekd_dist.argmin(axis=1)
        dists = data._impl.contact.dist[indices] * mask[jnp.arange(mask.shape[0]), indices]
        in_contact = dists < 0.0

        return in_contact


    def check_flat_feet_floor_missing_contacts(self, data, mjx_model, internal_state):
        if self.env.foot_type == "sphere":
            return jnp.zeros(self.env.nr_feet)
        elif self.env.foot_type == "box":
            feet_xpos = data.geom_xpos[self.env.foot_geom_indices]
            feet_xmat = data.geom_xmat[self.env.foot_geom_indices].reshape(-1, 3, 3)
            feet_sizes = mjx_model.geom_size[self.env.foot_geom_indices]
            lower_base_corners = jnp.array([
                [1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]
            ])
            corners = lower_base_corners[None, :, :] * feet_sizes[:, None, :]
            global_corners = jnp.einsum("fij,fgj->fgi", feet_xmat, corners) + feet_xpos[:, None, :]
            floor_height_at_corners = self.ground_height_at(internal_state, global_corners[:, :, 0], global_corners[:, :, 1])
            in_contact = jnp.sum(global_corners[:, :, 2] > floor_height_at_corners, axis=1)
        return in_contact


    def ground_height_at(self, internal_state, x_in_m, y_in_m):
        x = jnp.clip(jnp.round(x_in_m * self.one_meter_length + self.hfield_half_length).astype(jnp.int32), 0, self.hfield_length-1)
        y = jnp.clip(jnp.round(y_in_m * self.one_meter_length + self.hfield_half_length).astype(jnp.int32), 0, self.hfield_length-1)
        return internal_state["current_height_field_data"][y, x] * self.mujoco_height_scaling


    def pre_step(self, data, internal_state):
        internal_state["robot_imu_height_over_ground"] = data.site_xpos[self.env.imu_site_id, 2] - self.ground_height_at(internal_state, data.site_xpos[self.env.imu_site_id, 0], data.site_xpos[self.env.imu_site_id, 1])


    def post_step(self, data, mjx_model, internal_state, key):
        min_edge = self.hfield_half_length_in_meters - 0.5
        max_edge = self.hfield_half_length_in_meters
        reached_edge = jnp.array(((min_edge < jnp.abs(data.qpos[0])) & (jnp.abs(data.qpos[0]) < max_edge)) | ((min_edge < jnp.abs(data.qpos[1])) & (jnp.abs(data.qpos[1]) < max_edge)))
        qpos, qvel = self.env.initial_state_function.setup(mjx_model, internal_state, key)
        initial_state_data = data.replace(qpos=qpos, qvel=qvel)
        new_data = jax.lax.cond(reached_edge, lambda _: initial_state_data, lambda _: data, None)
        return new_data
    

    def sample(self, mjx_model, internal_state, key):
        curriculum_coeff = internal_state["env_curriculum_coeff"]

        key, noise_key, terrain_key = jax.random.split(key, 3)

        max_obstacle_height = curriculum_coeff * internal_state["robot_dimensions_mean"] * self.block_height_max_per_m_factor
        max_slope_height = curriculum_coeff * internal_state["robot_dimensions_mean"] * self.block_slope_height_max_per_m_factor

        noise_height = curriculum_coeff * jax.random.uniform(
            noise_key,
            shape=(),
            minval=0.0,
            maxval=internal_state["robot_dimensions_mean"] * self.random_height_max_per_m_factor,
        )

        isaac_height_field = self.bunker_ruins_terrain(
            max_obstacle_height=max_obstacle_height,
            max_slope_height=max_slope_height,
            noise_height=noise_height,
            key=terrain_key
        )

        new_height_field_data = self.isaac_hf_to_mujoco_hf(isaac_height_field)

        new_mjx_model = mjx_model.replace(hfield_data=new_height_field_data)

        internal_state["center_height"] = new_height_field_data[self.hfield_half_length * self.hfield_length + self.hfield_half_length] * self.mujoco_height_scaling
        internal_state["current_height_field_data"] = new_height_field_data.reshape(self.hfield_length, self.hfield_length)

        return new_mjx_model
    

    def isaac_hf_to_mujoco_hf(self, isaac_hf):
        hf = isaac_hf + jnp.abs(jnp.min(isaac_hf))
        hf /= self.mujoco_height_scaling
        return hf.reshape(-1)

    def bunker_ruins_terrain(self, max_obstacle_height, max_slope_height, noise_height, key):
        """
        V2: Improved, chaotic version with block rotation and sharp concrete edges.
        """
        # --- ENVIRONMENT PARAMETERS ---
        num_blocks = int(250 * (self.hfield_half_length_in_meters / 10.0)**2) # Increased density
        min_block_size_px = max(1, int(0.6 * self.one_meter_length))
        max_block_size_px = max(1, int(2.0 * self.one_meter_length))
        hill_height = max_slope_height * 0.4
        
        keys = jax.random.split(key, 8)

        # ---------------------------------------------------------------------
        # 1. BASE LAYER: Hills and undulations
        # ---------------------------------------------------------------------
        y_idx, x_idx = jnp.meshgrid(jnp.arange(self.hfield_length), jnp.arange(self.hfield_length), indexing='ij')
        
        x_norm = x_idx * (6 * jnp.pi / self.hfield_length)
        y_norm = y_idx * (6 * jnp.pi / self.hfield_length)
        
        base_hills = (jnp.sin(x_norm) * jnp.cos(y_norm) + jnp.sin(x_norm * 0.7 + 1.0)) * hill_height

        # ---------------------------------------------------------------------
        # 2. RUBBLE LAYER: Concrete slabs rotated at various angles (YAW)
        # ---------------------------------------------------------------------
        # Randomize positions (Block centers)
        cx = jax.random.uniform(keys[0], shape=(num_blocks, 1, 1), minval=0, maxval=self.hfield_length)
        cy = jax.random.uniform(keys[1], shape=(num_blocks, 1, 1), minval=0, maxval=self.hfield_length)
        
        # Randomize slab dimensions
        w_px = jax.random.uniform(keys[2], shape=(num_blocks, 1, 1), minval=min_block_size_px, maxval=max_block_size_px)
        l_px = jax.random.uniform(keys[3], shape=(num_blocks, 1, 1), minval=min_block_size_px, maxval=max_block_size_px)
        
        # NEW: Randomize rotation around Z-axis (Yaw)
        yaw = jax.random.uniform(keys[4], shape=(num_blocks, 1, 1), minval=0.0, maxval=2 * jnp.pi)
        cos_yaw = jnp.cos(yaw)
        sin_yaw = jnp.sin(yaw)

        # Randomize base height and slab inclination (Pitch/Roll)
        min_obstacle_height = 0.2 * max_obstacle_height
        block_base_z = jax.random.uniform(keys[5], shape=(num_blocks, 1, 1), minval=min_obstacle_height, maxval=max_obstacle_height)
        slope_x = jax.random.uniform(keys[6], shape=(num_blocks, 1, 1), minval=-0.8, maxval=0.8) * max_slope_height / self.one_meter_length
        slope_y = jax.random.uniform(keys[7], shape=(num_blocks, 1, 1), minval=-0.8, maxval=0.8) * max_slope_height / self.one_meter_length

        # Distance vectors of each map pixel from the center of each block
        dx = x_idx[None, :, :] - cx
        dy = y_idx[None, :, :] - cy

        # NEW: Coordinate transformation - point rotation relative to block center
        # This creates skewed blocks
        dx_rot = dx * cos_yaw - dy * sin_yaw
        dy_rot = dx * sin_yaw + dy * cos_yaw

        # Calculate slab height including inclination (using rotated coordinates)
        block_heights = block_base_z + (dx_rot * slope_x) + (dy_rot * slope_y)
        
        # Mask: is the pixel inside the rotated rectangle?
        in_block_mask = (jnp.abs(dx_rot) < (w_px / 2.0)) & (jnp.abs(dy_rot) < (l_px / 2.0))
        
        # Apply mask
        block_heights = jnp.where(in_block_mask, block_heights, -1000.0)

        # Stack blocks on top of each other (Max over N axis)
        # Ground underneath is base_hills
        all_layers = jnp.concatenate([base_hills[None, :, :], base_hills[None, :, :] + block_heights], axis=0)
        terrain = jnp.max(all_layers, axis=0)

        # ---------------------------------------------------------------------
        # 3. MICRO-UNEVENNESS LAYER: Hard rubble and stones
        # ---------------------------------------------------------------------
        def add_fractal_rubble(terr, noise_k):
            # Instead of smooth noise, we create discrete, rough stones (Fractal-like noise)
            k1, k2 = jax.random.split(noise_k)
            
            # Large stones
            c_size_1 = max(1, int(0.2 * self.one_meter_length))
            noise_1 = jax.random.uniform(k1, shape=(self.hfield_length // c_size_1, self.hfield_length // c_size_1), minval=-noise_height, maxval=noise_height)
            noise_1 = jnp.repeat(jnp.repeat(noise_1, c_size_1, axis=0), c_size_1, axis=1)
            
            # Medium stones (gravel)
            c_size_2 = max(1, int(0.05 * self.one_meter_length))
            noise_2 = jax.random.uniform(k2, shape=(self.hfield_length // c_size_2, self.hfield_length // c_size_2), minval=-noise_height*0.5, maxval=noise_height*0.5)
            noise_2 = jnp.repeat(jnp.repeat(noise_2, c_size_2, axis=0), c_size_2, axis=1)

            # Size matching (Padding, if division with remainder cut off pixels)
            pad_y = self.hfield_length - noise_1.shape[0]
            pad_x = self.hfield_length - noise_1.shape[1]
            noise_1 = jnp.pad(noise_1, ((0, pad_y), (0, pad_x)), mode='edge')
            
            pad_y = self.hfield_length - noise_2.shape[0]
            pad_x = self.hfield_length - noise_2.shape[1]
            noise_2 = jnp.pad(noise_2, ((0, pad_y), (0, pad_x)), mode='edge')

            # Noise applied as absolute values so rubble protrudes outside concrete blocks
            return terr + jnp.abs(noise_1) + jnp.abs(noise_2)

        terrain_key, noise_key = jax.random.split(keys[0])
        
        terrain = jax.lax.cond(
            noise_height > 0,
            lambda t: add_fractal_rubble(t, noise_key),
            lambda t: t,
            terrain
        )

        # ---------------------------------------------------------------------
        # 4. SAFE STARTING ZONE (Overwrite only, no funnel)
        # ---------------------------------------------------------------------
        safe_radius_px = int(0.8 * self.one_meter_length)
        center_x = self.hfield_half_length
        center_y = self.hfield_half_length
        
        dist_from_center = jnp.sqrt((x_idx - center_x)**2 + (y_idx - center_y)**2)
        
        # Instead of multiplying everything, we set flat terrain exactly at height 0 
        # only where the distance is less than the radius.
        terrain = jnp.where(dist_from_center < safe_radius_px, 0.0, terrain)

        # No Gaussian blur at the very end! We want to preserve sharp concrete edges.
        
        return terrain
