from ml_collections import config_dict


def get_config(environment_name):
    config = {
        "name": environment_name,
        "nr_envs": 1,
        "seed": 1,
        "render": False,
        "copy_train_env_for_eval": True,
        "train_robot": "unitree_go2",
        "spine_locked": False,
        "control_type": "pd",
        "command": {
            "type": "random",
            "sampling_type": "step_probability",
            "max_velocity_per_m_factor": 2.0,
            "clip_max_velocity": 1.0,
            "x_velocity_multiplier": 1.0,
            "zero_clip_threshold_percentage": 0.1,
            "all_zero_chance": 0.04,
            "single_zero_chance": 0.005,
        },
        "env_curriculum_enabled": True,
        "env_curriculum_disabled_coeff": 0.99,
        "env_curriculum_nr_levels": 100,
        "env_curriculum_level_success_episode_return": 8.0,
        "domain_randomization": {
            "sampling_type": "step_probability_and_reset",
            "action_delay": {
                "type": "default",
                "max_nr_delay_steps": 1,
                "mixed_chance": 0.05,
            },
            "initial_state": {
                "type": "random",
                "roll_angle_pi_factor": 0.05,
                "pitch_angle_pi_factor": 0.05,
                "yaw_angle_pi_factor": 1.0,
                "actuator_joint_position_offset_to_nominal": 0.01,
                "actuator_joint_nominal_position_factor": 0.5,
                "joint_velocity_max_factor": 0.5,
                "trunk_velocity_clip_mass_factor": 0.1,
                "trunk_velocity_clip_limit": 0.5,
                "initial_height_offset": 10.0,
            },
            "joint_dropout": {
                "type": "default",
                "dropout_open_chance": 0.001,
                "dropout_lock_chance": 0.001,
            },
            "mujoco_model": {
                "type": "default",
                "friction_tangential_factor": 1.0,
                "friction_torsional_factor": 1.0,
                "friction_rolling_factor": 1.0,
                "stiffness_factor": 0.5,
                "damping_factor": 0.6,
                "foot_solimp_factor": 0.8,
                "add_impratio": 1.0,
                "xy_gravity": 0.5,
                "z_gravity_factor": 0.1,
                "density_factor": 0.1,
                "viscosity_factor": 0.1,
            },
            "observation_noise": {
                "type": "default",
                "joint_position": 0.01,
                "joint_velocity": 1.5,
                "imu_angular_velocity": 0.2,
                "gravity_vector": 0.05,
                "exteroception": 0.03,
            },
            "perturbation": {
                "sampling_type": "step_probability",
                "type": "default",
                "trunk_velocity_clip_mass_factor": 0.1,
                "trunk_velocity_clip_limit": 1.0,
                "trunk_velocity_add_chance": 0.5,
                "max_joint_velocity": 0.5,
                "max_joint_position": 0.01,
            },
            "seen_robot": {
                "type": "default",
                "robot_size_scaling_factor": 0.0,
                "coupled_mass_inertia_factor": 0.1,
                "decoupled_mass_inertia_factor": 0.05,
                "add_com_displacement": 0.005,
                "add_inertia_orientation_rad": 0.01,
                "add_body_position": 0.0,
                "add_body_orientation_rad": 0.01,
                "add_imu_position": 0.05,
                "foot_size_factor": 0.05,
                "joint_axis_angle_rad": 0.01,
                "torque_limit_factor": 0.15,
                "add_actuator_joint_nominal_position": 0.01,
                "joint_velocity_max_factor": 0.15,
                "add_joint_range": 0.05,
                "joint_damping_factor": 0.3,
                "add_joint_damping": 0.003,
                "joint_armature_factor": 0.5,
                "add_joint_armature": 0.001,
                "joint_stiffness_factor": 0.1,
                "add_joint_stiffness": 0.1,
                "joint_friction_loss_factor": 1.0,
                "add_joint_friction_loss": 0.00001,
                "p_gain_factor": 0.1,
                "d_gain_factor": 0.1,
                "scaling_factor_factor": 0.1,
            },
            "unseen_robot": {
                "type": "default",
                "mass_inertia_factor": 0.25,
                "com_factor": 0.1,
                "body_position_factor": 0.01,
                "joint_damping_factor": 0.2,
                "joint_armature_factor": 0.2,
                "joint_stiffness_factor": 0.2,
                "joint_friction_loss_factor": 0.3,
                "p_gain_factor": 0.2,
                "d_gain_factor": 0.2,
                "position_offset": 0.03,
            },
        },
        "policy_exteroceptive_observation_type": "none",
        "critic_exteroceptive_observation_type": "height_over_ground",
        "reward": {
            "type": "default",
            "tracking_xy_velocity_command_coeff": 2.0,
            "tracking_xy_temperature": 0.25,
            "tracking_yaw_velocity_command_coeff": 1.0,
            "tracking_yaw_temperature": 0.25,
            "alive_clipped_coeff": 0.05,
            "alive_unclipped_coeff": 0.05,
            "z_velocity_coeff": 2.0,
            "imu_acceleration_coeff": 1e-4,
            "roll_pitch_vel_coeff": 0.05,
            "roll_pitch_pos_coeff": 10.0,
            "actuator_joint_nominal_diff_coeff": 5.0,
            "joint_position_limit_coeff": 40.0,
            "soft_joint_position_limit": 0.9,
            "actuator_joint_velocity_limit_coeff": 5.0,
            "soft_actuator_joint_velocity_limit": 0.9,
            "joint_velocity_coeff": 4e-4,
            "joint_acceleration_coeff": 5e-6,
            "joint_torque_coeff": 4e-4,
            "power_draw_penalty_coeff": 4e-4,
            "action_rate_coeff": 10.0,
            "action_smoothness_coeff": 0.1,
            "collision_coeff": 2.0,
            "base_height_coeff": 30.0,
            "foot_clearance_coeff": 0.01,
            "foot_clearance_max_height_m": 0.10,
            "foot_air_time_coeff": 3.0,
            "foot_air_time_per_robot_size_m": 0.4,
            "symmetry_air_coeff": 1.0,
            "foot_slip_coeff": 0.1,
            "foot_z_velocity_coeff": 0.2,
            "foot_velocity_coeff": 0.2,
            "foot_flat_contact_coeff": 0.01,
        },
        "reward": {
            "type": "drop_landing_energy_dissipation",

            # -----------------------------------------------------------------
            # 1. PARAMETRY SPRZĘTOWE (MAB Robotics HB40 / falowniki MD80 / seria MA-p)
            # -----------------------------------------------------------------
            # KT silnika (Nm/A) – służy do estymacji prądu z momentu (I = tau / kt)
            "motor_kt": 0.1,  
            # Bezpieczny limit prądu przed spaleniem lub uszkodzeniem zębatek planetarnych
            "max_safe_current": 40.0,  
            # Docelowa wysokość nad ziemią PO amortyzacji i zatrzymaniu uderzenia
            "nominal_landing_height": 0.28,  

            # -----------------------------------------------------------------
            # 2. KOMPONENTY ENERGETYCZNE I PRĄDOWE (Kluczowe dla spadku z 1m)
            # -----------------------------------------------------------------
            # WYSOKA NAGRODA: Zmusza silniki do pracy jako generator w fazie uderzenia
            # i zrzucania energii do Brake Resistor w PDS
            "regen_braking_reward_coeff": 5.0,  
            
            # BARDZO SUROWA KARA: Ścina piki prądowe przekraczające limit 40A,
            # chroniąc mechanikę przekładni planetarnej przed udarem
            "current_spike_penalty_coeff": -0.5,  
            
            # Standardowa kara za dodatnią pracę z baterii (w locie i po lądowaniu)
            "power_draw_penalty_coeff": -0.001,  
            
            # Lekka regularyzacja momentu obrotowego (przeciwdziała sztywności)
            "joint_torque_coeff": -0.0001,  

            # -----------------------------------------------------------------
            # 3. KOMPONENTY STABILNOŚCI I AMORTYZACJI
            # -----------------------------------------------------------------
            # BARDZO WYSOKA KARA: Wymusza krytyczne tłumienie (brak wybijania w górę po kontakcie)
            "post_impact_bounce_coeff": -15.0,  
            
            # Silne trzymanie poziomu tułowia (Roll i Pitch bliskie 0)
            "roll_pitch_pos_coeff": -10.0,  
            
            # Tłumienie rotacji tułowia w locie i podczas uderzenia
            "roll_pitch_vel_coeff": -0.5,  
            
            # Aktywowane dopiero po 250ms od uderzenia – uczy powrotu do postawy stojącej
            "base_height_coeff": -10.0,  
            
            # Nagroda za przetrwanie epizodu (nieprzewrócenie się)
            "alive_coeff": 1.0,  

            # -----------------------------------------------------------------
            # 4. REGULARYZACJA AKCJI I BEZPIECZEŃSTWO
            # -----------------------------------------------------------------
            # Kary za gwałtowne drgania wyjścia sieci (ochrona przed oscylacjami)
            "action_rate_coeff": -1.0,  
            "action_smoothness_coeff": -0.1,  
            
            # Kara za uderzenie korpusu / kolan o podłoże
            "collision_coeff": -5.0,  
        },
        "termination": {
            "type": "below_height",
            "height_percentage_threshold": 0.1,
        },
        "terrain": {
            "type": "hfield_bunker_ruins",
            "wave_fn_min": 0,
            "wave_fn_max": 2,
            "wave_height_max_per_m_factor": 0.05,
            "random_height_max_per_m_factor": 0.04,
            "block_probability": 0.8,
            "block_length_in_meters": 0.5,
            "block_height_max_per_m_factor": 0.3,
            "block_slope_height_max_per_m_factor": 0.2,
            # Used by hfield_bunker_ruins_unbounded. Difficulty 1.0 matches
            # hfield_bunker_ruins at env_curriculum_coeff == 1.0.
            "curriculum_initial_difficulty": 2.0,
            "curriculum_eval_difficulty": 1.0,
            "curriculum_step_scale": 1.0,
            "curriculum_nr_levels": 100,
            "curriculum_level_success_episode_return": 8.0
        },
        "add_goal_arrow": False,
        "timestep": 0.005,
        "episode_length_in_seconds": 20,
    }

    return config_dict.ConfigDict(config)
