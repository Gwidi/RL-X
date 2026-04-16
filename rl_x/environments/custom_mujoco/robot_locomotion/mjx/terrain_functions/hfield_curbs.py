import jax
import jax.numpy as jnp


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
        # Curb and noise heights increase with the Curriculum coefficient
        curriculum_coeff = internal_state["env_curriculum_coeff"]

        key, curb_key, noise_key, terrain_key = jax.random.split(key, 4)
        
        curb_height = curriculum_coeff * jax.random.uniform(
            curb_key,
            shape=(),
            minval=0.0,
            maxval=internal_state["robot_dimensions_mean"] * self.block_height_max_per_m_factor,
        )

        noise_height = curriculum_coeff * jax.random.uniform(
            noise_key,
            shape=(),
            minval=0.0,
            maxval=internal_state["robot_dimensions_mean"] * self.random_height_max_per_m_factor,
        )

        isaac_height_field = self.dynamic_curbs_terrain(
            curb_height=curb_height,
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


    def diverse_terrain(self,
            wave_fn_min, wave_fn_max, wave_height,
            random_height,
            block_probability, block_height, block_length_in_meters,
            key
        ):
        keys = jax.random.split(key, 5)

        I, J = jnp.meshgrid(jnp.arange(self.hfield_length),
                            jnp.arange(self.hfield_length),
                            indexing='ij')
        wave1 = jnp.sin(2 * jnp.pi * jax.random.uniform(keys[0], minval=wave_fn_min, maxval=wave_fn_max) * I / self.hfield_length)
        wave2 = jnp.sin(2 * jnp.pi * jax.random.uniform(keys[1], minval=wave_fn_min, maxval=wave_fn_max) * J / self.hfield_length)
        height_field_raw = wave_height * (wave1 + wave2)

        height_field_raw += jax.random.uniform(keys[2], shape=(self.hfield_length, self.hfield_length), minval=-random_height, maxval=random_height)

        block_length = int(block_length_in_meters * self.one_meter_length)
        block_repeat = int(self.hfield_length * block_length)
        layer1 = jnp.repeat(jax.random.bernoulli(keys[3], block_probability, shape=((self.hfield_length * self.hfield_length) // block_repeat,)), block_repeat).reshape(self.hfield_length, self.hfield_length)
        layer2 = jnp.repeat(jax.random.bernoulli(keys[4], block_probability, shape=((self.hfield_length * self.hfield_length) // block_repeat,)), block_repeat).reshape(self.hfield_length, self.hfield_length).T
        height_field_raw += layer1.astype(float) * block_height + layer2.astype(float) * block_height

        return height_field_raw
    
    def dynamic_curbs_terrain(self, curb_height, noise_height, key):
        """
        MJX/JAX-friendly version:
        - no Python int() on tracers
        - no dynamic slice assignment
        - fixed scan instead of data-dependent Python while
        """
        terrain_key, noise_key = jax.random.split(key)

        height_field_raw = jnp.zeros((self.hfield_length, self.hfield_length))
        col_idx = jnp.arange(self.hfield_length)

        min_width_px = max(1, int(0.05 * self.one_meter_length))
        max_width_px = max(min_width_px, int(0.40 * self.one_meter_length))
        min_space_px = max(1, int(0.60 * self.one_meter_length))
        max_space_px = max(min_space_px, int(1.40 * self.one_meter_length))

        def step(carry, step_key):
            terrain, current_x = carry

            def update(carry):
                terrain, current_x = carry
                width_key, space_key, sign_key = jax.random.split(step_key, 3)

                obs_width_px = jax.random.randint(
                    width_key,
                    shape=(),
                    minval=min_width_px,
                    maxval=max_width_px + 1,
                )
                space_px = jax.random.randint(
                    space_key,
                    shape=(),
                    minval=min_space_px,
                    maxval=max_space_px + 1,
                )

                end_x = jnp.minimum(current_x + obs_width_px, self.hfield_length)
                sign = jnp.where(jax.random.bernoulli(sign_key, p=0.5), 1.0, -1.0)

                segment_mask = (col_idx >= current_x) & (col_idx < end_x)
                terrain = jnp.where(segment_mask[None, :], sign * curb_height, terrain)

                return (terrain, end_x + space_px), None

            def no_update(carry):
                return carry, None

            return jax.lax.cond(current_x < self.hfield_length, update, no_update, carry)

        step_keys = jax.random.split(terrain_key, self.hfield_length)
        (height_field_raw, _), _ = jax.lax.scan(
            step,
            (height_field_raw, jnp.array(0, dtype=jnp.int32)),
            step_keys,
        )

        def add_noise(terrain):
            chunk_size = max(1, int(0.10 * self.one_meter_length))
            small_size_y = self.hfield_length // chunk_size
            small_size_x = self.hfield_length // chunk_size

            small_noise = jax.random.uniform(
                noise_key,
                shape=(small_size_y, small_size_x),
                minval=-noise_height,
                maxval=noise_height,
            )

            blocky_noise = jnp.repeat(
                jnp.repeat(small_noise, chunk_size, axis=0),
                chunk_size,
                axis=1,
            )

            pad_y = self.hfield_length - blocky_noise.shape[0]
            pad_x = self.hfield_length - blocky_noise.shape[1]

            blocky_noise = jnp.pad(blocky_noise, ((0, pad_y), (0, pad_x)), mode="edge")
            return terrain + blocky_noise

        height_field_raw = jax.lax.cond(
            noise_height > 0,
            add_noise,
            lambda terrain: terrain,
            height_field_raw,
        )

        safe_radius_m = 0.8
        safe_radius_px = int(safe_radius_m * self.one_meter_length)

        center_x = self.hfield_half_length
        center_y = self.hfield_half_length

        height_field_raw = height_field_raw.at[
            max(0, center_y - safe_radius_px) : min(self.hfield_length, center_y + safe_radius_px),
            max(0, center_x - safe_radius_px) : min(self.hfield_length, center_x + safe_radius_px),
        ].set(0.0)

        return height_field_raw
