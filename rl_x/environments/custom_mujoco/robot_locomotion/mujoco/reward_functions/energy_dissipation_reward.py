import numpy as np


class EnergyDissipationReward:
    def __init__(self, env):
        self.env = env

        # 1. Pobranie współczynników z env_config i przeskalowanie przez dt (dokładnie jak w Twoim kodzie)
        self.alive_coeff = env.env_config["reward"]["alive_coeff"] * env.dt
        
        # Nagrody/kary orientacji i stabilności bazy
        self.roll_pitch_pos_coeff = env.env_config["reward"]["roll_pitch_pos_coeff"] * env.dt
        self.roll_pitch_vel_coeff = env.env_config["reward"]["roll_pitch_vel_coeff"] * env.dt
        self.base_height_coeff = env.env_config["reward"]["base_height_coeff"] * env.dt
        self.nominal_landing_height = env.env_config["reward"]["nominal_landing_height"]

        # Kary/nagrody energetyczne (prądy i dysypacja)
        self.power_draw_penalty_coeff = env.env_config["reward"]["power_draw_penalty_coeff"] * env.dt
        self.regen_braking_reward_coeff = env.env_config["reward"]["regen_braking_reward_coeff"] * env.dt
        self.current_spike_penalty_coeff = env.env_config["reward"]["current_spike_penalty_coeff"] * env.dt
        self.max_safe_current = env.env_config["reward"]["max_safe_current"]
        self.motor_kt = env.env_config["reward"]["motor_kt"]

        # Kary dynamiczne uderzenia i gładkości
        self.post_impact_bounce_coeff = env.env_config["reward"]["post_impact_bounce_coeff"] * env.dt
        self.joint_torque_coeff = env.env_config["reward"]["joint_torque_coeff"] * env.dt
        self.action_rate_coeff = env.env_config["reward"]["action_rate_coeff"] * env.dt
        self.action_smoothness_coeff = env.env_config["reward"]["action_smoothness_coeff"] * env.dt
        self.collision_coeff = env.env_config["reward"]["collision_coeff"] * env.dt

        self.feet_symmetry_pairs = env.feet_symmetry_pairs

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
        # 1. Twoje zmienne do śledzenia fazy uderzenia
        self.env.internal_state["has_touched_ground"] = False
        self.env.internal_state["time_since_touchdown"] = 0.0
        self.env.internal_state["previous_actuator_joint_velocities"] = np.zeros(self.env.nr_actuator_joints)
        
        # 2. KLUCZOWE: Przywróć zmienne wymagane przez generator obserwacji (get_observation)
        self.env.internal_state["feet_time_on_ground"] = np.zeros(self.env.nr_feet)
        self.env.internal_state["feet_time_in_air"] = np.zeros(self.env.nr_feet)
        self.env.internal_state["previous_imu_linear_velocity"] = np.zeros(self.env.imu_linear_velocity_sensor_dim)
        self.env.internal_state["sum_tracking_performance_percentage"] = 0.0

    def step(self):
        feet_floor_contacts = self.env.terrain_function.check_feet_floor_contact()
        is_contact = np.any(feet_floor_contacts)

        if is_contact:
            self.env.internal_state["has_touched_ground"] = True
            self.env.internal_state["time_since_touchdown"] += self.env.dt

        # Aktualizacja zmiennych dla obserwacji środowiska:
        self.env.internal_state["feet_time_on_ground"] = np.where(feet_floor_contacts, self.env.internal_state["feet_time_on_ground"] + self.env.dt, 0.0)
        self.env.internal_state["feet_time_in_air"] = np.where(feet_floor_contacts, 0.0, self.env.internal_state["feet_time_in_air"] + self.env.dt)
        self.env.internal_state["previous_actuator_joint_velocities"] = self.env.internal_state["data"].qvel[self.env.actuator_joint_mask_qvel]
        self.env.internal_state["previous_imu_linear_velocity"] = self.env.internal_state["data"].sensordata[self.env.imu_linear_velocity_sensor_adr:self.env.imu_linear_velocity_sensor_adr + self.env.imu_linear_velocity_sensor_dim]

    def reward_and_info(self, action):
        curriculum_coeff = self.env.internal_state.get("env_curriculum_coeff", 1.0)
        
        # Wyciągnij kluczowe wektory stanu ze struktury danych
        qvel = self.env.internal_state["data"].qvel[self.env.actuator_joint_mask_qvel]
        tau = self.env.internal_state["data"].qfrc_actuator[self.env.actuator_joint_mask_qvel]
        current_imu_linear_velocity = self.env.internal_state["data"].sensordata[self.env.imu_linear_velocity_sensor_adr:self.env.imu_linear_velocity_sensor_adr + self.env.imu_linear_velocity_sensor_dim]
        current_imu_angular_velocity = self.env.internal_state["data"].sensordata[self.env.imu_angular_velocity_sensor_adr:self.env.imu_angular_velocity_sensor_adr + self.env.imu_angular_velocity_sensor_dim]
        
        has_touched = self.env.internal_state["has_touched_ground"]
        time_since_touchdown = self.env.internal_state["time_since_touchdown"]

        # =====================================================================
        # 1. KOMPONENTY ENERGETYCZNE I PRĄDOWE (Kluczowe dla spadania z 1m)
        # =====================================================================
        
        # A) Kara za pobór energii z baterii (tylko praca dodatnia: tau * w > 0)
        power_draw = np.mean(np.maximum(tau * qvel, 0.0))
        power_draw_penalty_reward = curriculum_coeff * self.power_draw_penalty_coeff * -power_draw

        # B) Nagroda za dysypację / hamowanie odzyskowe (praca ujemna: tau * w < 0)
        # Nagradzana TYLKO w pierwszych 300 ms po uderzeniu stóp w ziemię
        regen_power = np.mean(np.maximum(-tau * qvel, 0.0))
        is_in_impact_window = (has_touched and time_since_touchdown <= 0.30)
        regen_braking_reward = curriculum_coeff * self.regen_braking_reward_coeff * (regen_power if is_in_impact_window else 0.0)

        # C) Kara za przekroczenie bezpiecznego prądu (ochrona zębatki i falownika)
        # I = tau / kt -> karzemy prądy przekraczające limit (np. 40A / 60A)
        estimated_currents = np.abs(tau) / self.motor_kt
        current_spikes_norm = np.mean(np.square(np.maximum(estimated_currents - self.max_safe_current, 0.0)))
        current_spike_penalty_reward = curriculum_coeff * self.current_spike_penalty_coeff * -current_spikes_norm

        # D) Ogólna kara za momenty w stawach
        torque_norm = np.mean(np.square(tau))
        torque_reward = curriculum_coeff * self.joint_torque_coeff * -torque_norm

        # =====================================================================
        # 2. KOMPONENTY STABILNOŚCI I KONTROLI LOTU / LĄDOWANIA
        # =====================================================================

        # A) Kąty orientacji (Roll/Pitch muszą być poziomo)
        roll_pitch_position_norm = np.sum(np.square(self.env.internal_state["imu_orientation_euler"][:2]))
        angular_position_reward = curriculum_coeff * self.roll_pitch_pos_coeff * -roll_pitch_position_norm

        # B) Prędkości kątowe bazy (brak koziołkowania w locie i po uderzeniu)
        angular_velocity_norm = np.sum(np.square(current_imu_angular_velocity[:2]))
        angular_velocity_reward = curriculum_coeff * self.roll_pitch_vel_coeff * -angular_velocity_norm

        # C) Kara za sprężyste odbicie po uderzeniu (Bounce penalty)
        # Jeśli robot dotknął ziemi, prędkość w osi Z nie może być dodatnia (z > 0 oznacza wyskok w górę)
        z_vel = current_imu_linear_velocity[2]
        bounce_norm = np.square(np.maximum(z_vel, 0.0))
        post_impact_bounce_reward = curriculum_coeff * self.post_impact_bounce_coeff * -(bounce_norm if has_touched else 0.0)

        # D) Śledzenie docelowej wysokości bazy nad ziemią (aktywowane DOPIERO po wygaszeniu uderzenia)
        if time_since_touchdown > 0.25:
            height_difference_squared = (self.env.internal_state["robot_imu_height_over_ground"] - self.nominal_landing_height) ** 2
            base_height_reward = curriculum_coeff * self.base_height_coeff * -height_difference_squared
        else:
            base_height_reward = 0.0

        # =====================================================================
        # 3. KOMPONENTY REGULARYZACJI AKCJI I KOLIZJI
        # =====================================================================

        # Action rate
        action_rate_norm = np.mean(np.square(action - self.env.internal_state["last_action"]))
        action_rate_reward = curriculum_coeff * self.action_rate_coeff * -action_rate_norm
        
        # Action smoothness
        action_smoothness_norm = np.mean(np.square(action - 2 * self.env.internal_state["last_action"] + self.env.internal_state["second_last_action"]))
        action_smoothness_reward = curriculum_coeff * self.action_smoothness_coeff * -action_smoothness_norm

        # Collision reward (korpus/nogi nie mogą uderzyć w ziemię)
        all_contact_relevant_geom_xpos = self.env.internal_state["data"].geom_xpos[self.env.reward_collision_sphere_geom_ids]
        all_contact_relevant_geom_sizes = self.env.internal_state["mj_model"].geom_size[self.env.reward_collision_sphere_geom_ids, 0]
        distance_between_geoms = np.linalg.norm(all_contact_relevant_geom_xpos[:, None] - all_contact_relevant_geom_xpos[None], axis=-1)
        contact_between_geoms = distance_between_geoms <= (all_contact_relevant_geom_sizes[:, None] + all_contact_relevant_geom_sizes[None])
        nr_collisions = (np.sum(contact_between_geoms) - len(self.env.reward_collision_sphere_geom_ids)) // 2
        nr_collisions = np.maximum(nr_collisions - self.env.internal_state["nr_collisions_in_nominal"], 0)
        collision_reward = curriculum_coeff * self.collision_coeff * -nr_collisions

        # Alive reward
        alive_reward = curriculum_coeff * self.alive_coeff * 1.0

        # =====================================================================
        # AGREGACJA NAGRODY
        # =====================================================================
        reward = (
            alive_reward
            + power_draw_penalty_reward
            + regen_braking_reward
            + current_spike_penalty_reward
            + torque_reward
            + angular_position_reward
            + angular_velocity_reward
            + post_impact_bounce_reward
            + base_height_reward
            + action_rate_reward
            + action_smoothness_reward
            + collision_reward
        )
        
        reward = np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

        # =====================================================================
        # LOGOWANIE DO INFO (identyczna struktura kluczy jak w Twoim przykładzie)
        # =====================================================================
        self.env.internal_state["info"]["reward/alive"] = alive_reward
        self.env.internal_state["info"]["reward/power_draw_penalty"] = power_draw_penalty_reward
        self.env.internal_state["info"]["reward/regen_braking"] = regen_braking_reward
        self.env.internal_state["info"]["reward/current_spike_penalty"] = current_spike_penalty_reward
        self.env.internal_state["info"]["reward/joint_torque"] = torque_reward
        self.env.internal_state["info"]["reward/angular_position"] = angular_position_reward
        self.env.internal_state["info"]["reward/angular_velocity"] = angular_velocity_reward
        self.env.internal_state["info"]["reward/post_impact_bounce"] = post_impact_bounce_reward
        self.env.internal_state["info"]["reward/base_height"] = base_height_reward
        self.env.internal_state["info"]["reward/action_rate"] = action_rate_reward
        self.env.internal_state["info"]["reward/action_smoothness"] = action_smoothness_reward
        self.env.internal_state["info"]["reward/collision"] = collision_reward
        self.env.internal_state["info"]["reward/total"] = reward

        # =====================================================================
        # DIAGNOSTYKA WYMAGANA PRZEZ ENVIRONMENT.PY (uniknięcie KeyError)
        # =====================================================================
        # Obliczenie błędu śledzenia prędkości XY (nawet jeśli komenda to 0.0)
        feet_floor_contacts = self.env.terrain_function.check_feet_floor_contact()
        desired_imu_linear_velocity_xy = self.env.internal_state["goal_velocities"][:2]
        xy_difference = desired_imu_linear_velocity_xy - current_imu_linear_velocity[:2]
        max_xy_velocity_diff_abs = np.mean(2 * self.env.internal_state["max_command_velocities"][:2])
        self.env.internal_state["info"]["env_info/xy_vel_diff_abs"] = np.nan_to_num(
            np.mean(np.minimum(np.abs(xy_difference), 2 * self.env.internal_state["max_command_velocities"][:2])),
            nan=max_xy_velocity_diff_abs,
            posinf=max_xy_velocity_diff_abs,
            neginf=max_xy_velocity_diff_abs,
        )

        # Obliczenie średniej wysokości stóp w powietrzu
        feet_positions = self.env.internal_state["data"].geom_xpos[self.env.foot_geom_indices]
        feet_ground_heights = self.env.terrain_function.ground_height_at(
            feet_positions[:, 0],
            feet_positions[:, 1],
        )
        feet_clearance = feet_positions[:, 2] - feet_ground_heights
        mean_foot_height_in_air = np.mean(feet_clearance[~feet_floor_contacts]) if np.any(~feet_floor_contacts) else 0.0
        self.env.internal_state["info"]["env_info/mean_foot_height_in_air"] = mean_foot_height_in_air

        return reward
    