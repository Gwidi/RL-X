import numpy as np

class SimplifiedLandingReward:
    def __init__(self, env):
        self.env = env
        dt = env.dt

        self.alive_coeff = env.env_config["reward"].get("alive_coeff", 1.0) * dt
        self.base_height_coeff = env.env_config["reward"].get("base_height_coeff", 5.0) * dt
        self.roll_pitch_pos_coeff = env.env_config["reward"].get("roll_pitch_pos_coeff", 3.0) * dt
        self.base_vel_coeff = env.env_config["reward"].get("base_vel_coeff", 2.0) * dt 
        self.joint_torque_coeff = env.env_config["reward"].get("joint_torque_coeff", 0.05) * dt
        self.joint_vel_coeff = env.env_config["reward"].get("joint_vel_coeff", 0.1) * dt 
        self.action_rate_coeff = env.env_config["reward"].get("action_rate_coeff", 0.05) * dt
        
        # ZMIANA 1: Rozdzielenie kolizji
        self.self_collision_coeff = env.env_config["reward"].get("collision_coeff", 20.0) * dt
        self.floor_collision_coeff = 2.0 * dt  # Łagodne ostrzeżenie za łydki na ziemi
        
        self.joint_pos_coeff = env.env_config["reward"].get("joint_pos_coeff", 5.0) * dt

        self.nominal_landing_height = env.env_config["reward"]["nominal_landing_height"]
        self.soft_joint_position_limit = env.env_config["reward"].get("soft_joint_position_limit", 0.9)

    def init(self):
        self.env.internal_state["joint_position_limits"] = self.calculate_joint_position_limits()
        self.setup()

    def handle_model_change(self):
        self.env.internal_state["joint_position_limits"] = self.calculate_joint_position_limits()

    def calculate_joint_position_limits(self):
        joint_limits = self.env.internal_state["mj_model"].jnt_range[1:]
        joint_limits_midpoint = (joint_limits[:, 0] + joint_limits[:, 1]) / 2
        joint_limits_range = joint_limits[:, 1] - joint_limits[:, 0]
        lower_joint_limits = joint_limits_midpoint - joint_limits_range / 2 * self.soft_joint_position_limit
        upper_joint_limits = joint_limits_midpoint + joint_limits_range / 2 * self.soft_joint_position_limit
        return np.stack([lower_joint_limits, upper_joint_limits], axis=1)

    def setup(self):
        self.env.internal_state["feet_time_on_ground"] = np.zeros(self.env.nr_feet)
        self.env.internal_state["feet_time_in_air"] = np.zeros(self.env.nr_feet)
        self.env.internal_state["previous_imu_linear_velocity"] = np.zeros(self.env.imu_linear_velocity_sensor_dim)
        self.env.internal_state["sum_tracking_performance_percentage"] = 0.0
        self.env.internal_state["previous_actuator_joint_velocities"] = np.zeros(self.env.nr_actuator_joints)
        self.env.internal_state["landing_evaluated"] = False
        self.env.internal_state["landing_success"] = False
        
        self.env.internal_state["has_touched_ground"] = False
        self.env.internal_state["time_since_touchdown"] = 0.0

    def step(self):
        feet_floor_contacts = self.env.terrain_function.check_feet_floor_contact()
        
        self.env.internal_state["feet_time_on_ground"] = np.where(feet_floor_contacts, self.env.internal_state["feet_time_on_ground"] + self.env.dt, 0.0)
        self.env.internal_state["feet_time_in_air"] = np.where(feet_floor_contacts, 0.0, self.env.internal_state["feet_time_in_air"] + self.env.dt)
        self.env.internal_state["previous_actuator_joint_velocities"] = self.env.internal_state["data"].qvel[self.env.actuator_joint_mask_qvel]
        self.env.internal_state["previous_imu_linear_velocity"] = self.env.internal_state["data"].sensordata[self.env.imu_linear_velocity_sensor_adr:self.env.imu_linear_velocity_sensor_adr + self.env.imu_linear_velocity_sensor_dim]

        if np.any(feet_floor_contacts):
            self.env.internal_state["has_touched_ground"] = True
            
        if self.env.internal_state.get("has_touched_ground", False):
            self.env.internal_state["time_since_touchdown"] += self.env.dt

    def _evaluate_landing(
        self,
        height,
        euler,
        lin_vel,
        ang_vel,
    ):
        """
        Ocena jakości lądowania ~0.3 s po touchdown.

        Zwraca True tylko wtedy, gdy robot:
        - nie leży na brzuchu,
        - jest względnie pionowy,
        - ma małą prędkość liniową,
        - ma małą prędkość kątową.
        """

        max_roll_pitch = np.deg2rad(15.0)

        max_linear_velocity = 1.5
        max_angular_velocity = 3.0

        height_ok = height > 0.18

        orientation_ok = (
            abs(euler[0]) < max_roll_pitch
            and abs(euler[1]) < max_roll_pitch
        )

        linear_velocity_ok = (
            np.linalg.norm(lin_vel) < max_linear_velocity
        )

        angular_velocity_ok = (
            np.linalg.norm(ang_vel) < max_angular_velocity
        )

        return bool(
            height_ok
            and orientation_ok
            and linear_velocity_ok
            and angular_velocity_ok
        )

    def reward_and_info(self, action):
        qpos = self.env.internal_state["data"].qpos[self.env.actuator_joint_mask_qpos]
        qvel = self.env.internal_state["data"].qvel[self.env.actuator_joint_mask_qvel]
        tau = self.env.internal_state["data"].qfrc_actuator[self.env.actuator_joint_mask_qvel]
        lin_vel = self.env.internal_state["data"].sensordata[self.env.imu_linear_velocity_sensor_adr:self.env.imu_linear_velocity_sensor_adr + self.env.imu_linear_velocity_sensor_dim]
        ang_vel = self.env.internal_state["data"].sensordata[self.env.imu_angular_velocity_sensor_adr:self.env.imu_angular_velocity_sensor_adr + self.env.imu_angular_velocity_sensor_dim]
        euler = self.env.internal_state["imu_orientation_euler"]

        has_touched = self.env.internal_state.get("has_touched_ground", False)
        time_since_touch = self.env.internal_state.get("time_since_touchdown", 0.0)
        height = self.env.internal_state["robot_imu_height_over_ground"]
        target_height = self.nominal_landing_height

        landing_evaluated = self.env.internal_state.get(
            "landing_evaluated",
            False,
        )

        if (
            has_touched
            and time_since_touch >= 0.50
            and not landing_evaluated
        ):

            landing_success = self._evaluate_landing(
                height=height,
                euler=euler,
                lin_vel=lin_vel,
                ang_vel=ang_vel,
            )

            self.env.internal_state[
                "landing_evaluated"
            ] = True

            self.env.internal_state[
                "landing_success"
            ] = landing_success

            # ----------------------------------------------------------
            # UPDATE CURRICULUM
            # ----------------------------------------------------------
            #
            # Curriculum state jest przechowywany w internal_state,
            # więc nie musimy mieć bezpośredniego reference do klasy
            # initial-state curriculum.
            #

            curriculum = self.env.internal_state.get(
                "landing_curriculum"
            )

            if curriculum is not None:

                success = float(landing_success)

                alpha = curriculum.get(
                    "ema_alpha",
                    0.05,
                )

                curriculum["success_ema"] = (
                    (1.0 - alpha)
                    * curriculum["success_ema"]
                    + alpha * success
                )

                curriculum["last_success"] = success

                curriculum["nr_evaluated_landings"] += 1

                ema = curriculum["success_ema"]

                if ema > 0.85:

                    curriculum["difficulty"] = min(
                        1.0,
                        curriculum["difficulty"]
                        + 0.01,
                    )

                    curriculum["last_update"] = "increase"

                elif ema < 0.55:

                    curriculum["difficulty"] = max(
                        0.0,
                        curriculum["difficulty"]
                        - 0.005,
                    )

                    curriculum["last_update"] = "decrease"

                else:

                    curriculum["last_update"] = "hold"

        if height < 0.15:
            base_crash_reward = -20.0 * self.env.dt
        else:
            base_crash_reward = 0.0

        angular_position_reward = self.roll_pitch_pos_coeff * -np.sum(np.square(euler[:2]))
        
        if not has_touched:

            # ----------------------------------------------------------
            # FREE FALL
            # ----------------------------------------------------------
            #
            # Nie karzemy robota za prędkość XY/Z ani angular velocity.
            # Jest to naturalna część spadania.
            #

            base_vel_xy_reward = 0.0
            base_vel_z_reward = 0.0

            # Stabilność orientacji nadal jest ważna.
            angular_position_reward = (
                self.roll_pitch_pos_coeff
                * -np.sum(np.square(euler[:2]))
            )

            # Pozwalamy nogom bardziej się poruszać.
            nominal_joint_pos = (
                self.env.internal_state[
                    "actuator_joint_nominal_positions"
                ]
            )

            joint_pos_reward = (
                0.1
                * self.joint_pos_coeff
                * -np.mean(
                    np.square(
                        qpos - nominal_joint_pos
                    )
                )
            )

            # Mała kara za ekstremalne prędkości stawów.
            joint_vel_reward = (
                0.25
                * self.joint_vel_coeff
                * -np.mean(np.square(qvel))
            )

            # Nie próbujemy utrzymywać wysokości podczas lotu.
            base_height_reward = 0.0

        else:

            # ----------------------------------------------------------
            # POST TOUCHDOWN
            # ----------------------------------------------------------

            base_vel_xy_reward = (
                self.base_vel_coeff
                * -(
                    np.sum(np.square(lin_vel[:2]))
                    + np.sum(np.square(ang_vel))
                )
            )

            angular_position_reward = (
                self.roll_pitch_pos_coeff
                * -np.sum(np.square(euler[:2]))
            )

            joint_pos_reward = 0.0

            # ----------------------------------------------------------
            # FIRST 0.3 s = ENERGY ABSORPTION
            # ----------------------------------------------------------

            if time_since_touch < 0.3:

                # Podczas amortyzacji nie karzemy za ruch w dół.
                if lin_vel[2] > 0.0:
                    base_vel_z_reward = (
                        self.base_vel_coeff
                        * -np.square(lin_vel[2])
                        * 2.0
                    )
                else:
                    base_vel_z_reward = 0.0

                # Pozwalamy nogom szybko pracować podczas amortyzacji.
                joint_vel_reward = 0.0

                squat_penalty_weight = np.clip(
                    time_since_touch / 0.3,
                    0.0,
                    1.0,
                )

                base_height_reward = (
                    squat_penalty_weight
                    * self.base_height_coeff
                    * -np.square(
                        height - target_height
                    )
                )

            else:

                # ------------------------------------------------------
                # STABILIZATION
                # ------------------------------------------------------

                base_vel_z_reward = (
                    self.base_vel_coeff
                    * -np.square(lin_vel[2])
                )

                joint_vel_reward = (
                    self.joint_vel_coeff
                    * -np.mean(np.square(qvel))
                )

                base_height_reward = (
                    self.base_height_coeff
                    * -np.square(
                        height - target_height
                    )
                )

        base_vel_reward = (
            base_vel_xy_reward
            + base_vel_z_reward
        )

        if getattr(self.env, "spine_locked", False):
            leg_tau = tau
            spine_tau = 0.0
        else:
            spine_tau = tau[0]
            leg_tau = tau[1:]

        torque_reward = self.joint_torque_coeff * -np.mean(np.square(leg_tau))

        info = self.env.internal_state["info"]
        info["metrics/leg_torque_penalty"] = torque_reward
        
        if getattr(self.env, "spine_locked", False):
            info["metrics/spine_torque_penalty"] = 0.0
        else:
            info["metrics/spine_torque_penalty"] = self.joint_torque_coeff * -np.square(spine_tau)
            
        action_rate_reward = self.action_rate_coeff * -np.mean(np.square(action - self.env.internal_state["last_action"]))

        # =====================================================================
        # ZMIANA 3: ROZDZIELONE KOLIZJE
        # =====================================================================
        all_geom_xpos = self.env.internal_state["data"].geom_xpos[self.env.reward_collision_sphere_geom_ids]
        all_geom_sizes = self.env.internal_state["mj_model"].geom_size[self.env.reward_collision_sphere_geom_ids, 0]
        
        # A) Zderzenia ciała ze sobą -> Ogromna kara (np. 20.0 z basha)
        distance_between_geoms = np.linalg.norm(all_geom_xpos[:, None] - all_geom_xpos[None], axis=-1)
        contact_between_geoms = distance_between_geoms <= (all_geom_sizes[:, None] + all_geom_sizes[None])
        nr_self_collisions = (np.sum(contact_between_geoms) - len(self.env.reward_collision_sphere_geom_ids)) // 2
        nr_self_collisions = np.maximum(nr_self_collisions - self.env.internal_state["nr_collisions_in_nominal"], 0)
        self_collision_reward = self.self_collision_coeff * -nr_self_collisions

        # B) Łydki/brzuch dotykające gleby -> Mała kara (2.0) zachęcająca do nie zderzania się, ale dopuszczająca głęboki przysiad
        geom_ground_heights = self.env.terrain_function.ground_height_at(all_geom_xpos[:, 0], all_geom_xpos[:, 1])
        geom_clearance = all_geom_xpos[:, 2] - geom_ground_heights
        nr_floor_collisions = np.sum(geom_clearance < all_geom_sizes)
        floor_collision_reward = self.floor_collision_coeff * -nr_floor_collisions
        
        collision_reward = self_collision_reward + floor_collision_reward
        # =====================================================================

        alive_reward = self.alive_coeff * 1.0

        reward = (
            alive_reward
            + base_vel_reward
            + angular_position_reward
            + base_height_reward
            + base_crash_reward
            + joint_pos_reward
            + joint_vel_reward
            + torque_reward
            + action_rate_reward
            + collision_reward
        )
        
        reward = np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

        info = self.env.internal_state["info"]
        info["reward/alive"] = alive_reward
        info["reward/base_vel"] = base_vel_reward
        info["reward/angular_position"] = angular_position_reward
        info["reward/base_height"] = base_height_reward
        info["reward/base_crash"] = base_crash_reward
        info["reward/joint_pos"] = joint_pos_reward
        info["reward/joint_vel"] = joint_vel_reward
        info["reward/joint_torque"] = torque_reward
        info["reward/action_rate"] = action_rate_reward
        info["reward/collision"] = collision_reward
        info["reward/total"] = reward
        info["curriculum/landing_success"] = float(
            self.env.internal_state.get(
                "landing_success",
                False,
            )
        )

        curriculum = self.env.internal_state.get(
            "landing_curriculum"
        )

        if curriculum is not None:

            info["curriculum/difficulty"] = (
                curriculum["difficulty"]
            )

            info["curriculum/success_ema"] = (
                curriculum["success_ema"]
            )

            info["curriculum/nr_evaluated_landings"] = (
                curriculum["nr_evaluated_landings"]
            )

        desired_imu_linear_velocity_xy = self.env.internal_state["goal_velocities"][:2]
        xy_difference = desired_imu_linear_velocity_xy - lin_vel[:2]
        max_xy_velocity_diff_abs = np.mean(2 * self.env.internal_state["max_command_velocities"][:2])
        info["env_info/xy_vel_diff_abs"] = np.nan_to_num(
            np.mean(np.minimum(np.abs(xy_difference), 2 * self.env.internal_state["max_command_velocities"][:2])),
            nan=max_xy_velocity_diff_abs, posinf=max_xy_velocity_diff_abs, neginf=max_xy_velocity_diff_abs,
        )

        feet_positions = self.env.internal_state["data"].geom_xpos[self.env.foot_geom_indices]
        feet_ground_heights = self.env.terrain_function.ground_height_at(feet_positions[:, 0], feet_positions[:, 1])
        feet_clearance = feet_positions[:, 2] - feet_ground_heights
        feet_floor_contacts = self.env.terrain_function.check_feet_floor_contact()
        info["env_info/mean_foot_height_in_air"] = np.mean(feet_clearance[~feet_floor_contacts]) if np.any(~feet_floor_contacts) else 0.0

        return reward