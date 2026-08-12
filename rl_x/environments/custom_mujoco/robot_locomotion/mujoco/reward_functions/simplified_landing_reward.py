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
        self.collision_coeff = env.env_config["reward"].get("collision_coeff", 1.0) * dt

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
        
        # Wznowione śledzenie momentu uderzenia
        self.env.internal_state["has_touched_ground"] = False
        self.env.internal_state["time_since_touchdown"] = 0.0

    def step(self):
        feet_floor_contacts = self.env.terrain_function.check_feet_floor_contact()
        
        self.env.internal_state["feet_time_on_ground"] = np.where(feet_floor_contacts, self.env.internal_state["feet_time_on_ground"] + self.env.dt, 0.0)
        self.env.internal_state["feet_time_in_air"] = np.where(feet_floor_contacts, 0.0, self.env.internal_state["feet_time_in_air"] + self.env.dt)
        self.env.internal_state["previous_actuator_joint_velocities"] = self.env.internal_state["data"].qvel[self.env.actuator_joint_mask_qvel]
        self.env.internal_state["previous_imu_linear_velocity"] = self.env.internal_state["data"].sensordata[self.env.imu_linear_velocity_sensor_adr:self.env.imu_linear_velocity_sensor_adr + self.env.imu_linear_velocity_sensor_dim]

        # Logika czasu po uderzeniu
        if np.any(feet_floor_contacts):
            self.env.internal_state["has_touched_ground"] = True
            
        if self.env.internal_state.get("has_touched_ground", False):
            self.env.internal_state["time_since_touchdown"] += self.env.dt

    def reward_and_info(self, action):
        qvel = self.env.internal_state["data"].qvel[self.env.actuator_joint_mask_qvel]
        tau = self.env.internal_state["data"].qfrc_actuator[self.env.actuator_joint_mask_qvel]
        lin_vel = self.env.internal_state["data"].sensordata[self.env.imu_linear_velocity_sensor_adr:self.env.imu_linear_velocity_sensor_adr + self.env.imu_linear_velocity_sensor_dim]
        ang_vel = self.env.internal_state["data"].sensordata[self.env.imu_angular_velocity_sensor_adr:self.env.imu_angular_velocity_sensor_adr + self.env.imu_angular_velocity_sensor_dim]
        euler = self.env.internal_state["imu_orientation_euler"]

        angular_position_reward = self.roll_pitch_pos_coeff * -np.sum(np.square(euler[:2]))

        # =====================================================================
        # INTELIGENTNE KARY ZA PRĘDKOŚĆ (Tolerancja podczas amortyzacji)
        # =====================================================================
        has_touched = self.env.internal_state.get("has_touched_ground", False)
        time_since_touch = self.env.internal_state.get("time_since_touchdown", 0.0)

        # 1. Prędkość korpusu: Zawsze karzemy za ślizganie się w poziomie (X, Y) i rotacje
        base_vel_xy_reward = self.base_vel_coeff * -(np.sum(np.square(lin_vel[:2])) + np.sum(np.square(ang_vel)))
        
        # Karę za ruch góra/dół (Z) i ruch stawów włączamy na pełnej mocy 
        # dopiero po fazie amortyzacji (np. po 0.3 sekundach).
        if has_touched and time_since_touch < 0.3:
            # W fazie "deep squat" pozwalamy na opadanie w osi Z i zginanie nóg!
            base_vel_z_reward = 0.0
            joint_vel_reward = 0.0
        else:
            # Po amortyzacji (lub w locie) - robot ma zastygnąć w bezruchu
            base_vel_z_reward = self.base_vel_coeff * -np.square(lin_vel[2])
            joint_vel_reward = self.joint_vel_coeff * -np.mean(np.square(qvel))

        base_vel_reward = base_vel_xy_reward + base_vel_z_reward

        # =====================================================================
        # INTELIGENTNA KARA ZA WYSOKOŚĆ (3 fazy lądowania)
        # =====================================================================
        height = self.env.internal_state["robot_imu_height_over_ground"]
        target_height = self.nominal_landing_height
        has_touched = self.env.internal_state.get("has_touched_ground", False)
        time_since_touch = self.env.internal_state.get("time_since_touchdown", 0.0)

        base_height_reward = 0.0

        if not has_touched:
            # FAZA 1: LOT W DÓŁ
            # Karzemy TYLKO jeśli zwinie nogi pod siebie będąc wysoko w powietrzu
            if height < target_height:
                base_height_reward = self.base_height_coeff * -np.square(height - target_height)
        else:
            # FAZA 2 i 3: PO UDERZENIU
            if height > target_height:
                # Bezwzględna kara za odbijanie się od ziemi jak piłka (powrót w górę)
                base_height_reward = self.base_height_coeff * -np.square(height - target_height)
            else:
                # Robot kuca w celu amortyzacji (height <= target_height)
                # Dajemy mu swobodę przez pierwsze 0.4s po kontakcie z ziemią.
                # Po 0.4s kara zaczyna rosnąć od zera do maksa przez kolejne 0.6 sekundy.
                squat_penalty_weight = np.clip((time_since_touch - 0.4) / 0.6, 0.0, 1.0)
                base_height_reward = squat_penalty_weight * self.base_height_coeff * -np.square(height - target_height)

        # Reszta kar
        torque_reward = self.joint_torque_coeff * -np.mean(np.square(tau))
        action_rate_reward = self.action_rate_coeff * -np.mean(np.square(action - self.env.internal_state["last_action"]))

        all_contact_relevant_geom_xpos = self.env.internal_state["data"].geom_xpos[self.env.reward_collision_sphere_geom_ids]
        all_contact_relevant_geom_sizes = self.env.internal_state["mj_model"].geom_size[self.env.reward_collision_sphere_geom_ids, 0]
        distance_between_geoms = np.linalg.norm(all_contact_relevant_geom_xpos[:, None] - all_contact_relevant_geom_xpos[None], axis=-1)
        contact_between_geoms = distance_between_geoms <= (all_contact_relevant_geom_sizes[:, None] + all_contact_relevant_geom_sizes[None])
        nr_collisions = (np.sum(contact_between_geoms) - len(self.env.reward_collision_sphere_geom_ids)) // 2
        nr_collisions = np.maximum(nr_collisions - self.env.internal_state["nr_collisions_in_nominal"], 0)
        
        collision_reward = self.collision_coeff * -nr_collisions
        alive_reward = self.alive_coeff * 1.0

        reward = (
            alive_reward
            + base_vel_reward
            + angular_position_reward
            + base_height_reward
            + joint_vel_reward
            + torque_reward
            + action_rate_reward
            + collision_reward
        )
        
        reward = np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

        # Logowanie
        info = self.env.internal_state["info"]
        info["reward/alive"] = alive_reward
        info["reward/base_vel"] = base_vel_reward
        info["reward/angular_position"] = angular_position_reward
        info["reward/base_height"] = base_height_reward
        info["reward/joint_vel"] = joint_vel_reward
        info["reward/joint_torque"] = torque_reward
        info["reward/action_rate"] = action_rate_reward
        info["reward/collision"] = collision_reward
        info["reward/total"] = reward

        # Diagnostyka
        feet_floor_contacts = self.env.terrain_function.check_feet_floor_contact()
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
        info["env_info/mean_foot_height_in_air"] = np.mean(feet_clearance[~feet_floor_contacts]) if np.any(~feet_floor_contacts) else 0.0

        return reward