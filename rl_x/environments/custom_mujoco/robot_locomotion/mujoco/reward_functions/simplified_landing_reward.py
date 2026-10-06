import numpy as np


class SimplifiedLandingReward:
    """
    Emergency landing reward v2.

    Główne cele:
    1. Robot ma przeżyć zrzut.
    2. Nie może przekroczyć fizycznego limitu momentu silników.
    3. Nie może nadmiernie obciążać termicznie silników.
    4. Duży, krótki moment podczas amortyzacji jest dozwolony.
    5. Sukces jest zatwierdzany dopiero po okresie stabilizacji po touchdown.
    6. Spine locked nie ma aktywnego silnika spine i nie jest monitorowany
       jako actuator torque.

    WAŻNE:
    ----------
    torque_measurement_mode == "mujoco_actuator":
        data.actuator_force traktujemy jako moment po stronie silnika.
        To jest rekomendowany tryb, jeśli MJCF ma fizyczne gear, np. gear=9.

    torque_measurement_mode == "joint_equivalent":
        Tryb kompatybilności ze starym modelem gear=1, w którym
        actuator_force reprezentuje ekwiwalentny moment jointu.
        Wtedy:
            tau_motor = tau_joint / physical_gear_ratio

        Jeśli używasz tego trybu, sam model powinien mieć odpowiednio
        zwiększony forcerange joint-side, np.:
            3 Nm * 9 = 27 Nm
        dla nóg.
    """

    def __init__(self, env):
        self.env = env
        self.dt = env.dt

        cfg = env.env_config.get("reward", {})

        # ==============================================================
        # PHYSICAL MOTOR PARAMETERS
        # ==============================================================

        # Twarde fizyczne limity. NIE RANDOMIZOWAĆ.
        self.leg_motor_torque_limit = float(
            cfg.get("leg_motor_torque_limit", 3.0)
        )

        self.spine_motor_torque_limit = float(
            cfg.get("spine_motor_torque_limit", 0.785)
        )

        self.leg_gear_ratio = float(
            cfg.get("leg_gear_ratio", 9.0)
        )

        self.spine_gear_ratio = float(
            cfg.get("spine_gear_ratio", 36.0)
        )

        # "mujoco_actuator" = actuator_force jest motor-side.
        # "joint_equivalent" = actuator_force interpretujemy jako joint-side.
        self.torque_measurement_mode = cfg.get(
            "torque_measurement_mode",
            "mujoco_actuator",
        )

        if self.torque_measurement_mode not in (
            "mujoco_actuator",
            "joint_equivalent",
        ):
            raise ValueError(
                "torque_measurement_mode must be "
                "'mujoco_actuator' or 'joint_equivalent'"
            )

        # Minimalna tolerancja numeryczna.
        self.hard_limit_tolerance = float(
            cfg.get("hard_limit_tolerance", 1e-3)
        )

        # ==============================================================
        # THERMAL MODEL
        # ==============================================================

        # thermal_dose ma jednostkę przybliżonych
        # "sekund równoważnego obciążenia przy 100% torque".
        #
        # D += (tau / tau_max)^2 * dt
        #
        # 0.5 oznacza np.:
        # ~0.5 s przy 100%
        # ~2.0 s przy 50%
        #
        # Jest to PROXY termiczne. Docelowo skalibrować z hardware.
        self.thermal_dose_limit = float(
            cfg.get("thermal_dose_limit", 0.5)
        )

        # 0 = brak chłodzenia podczas krótkiego epizodu.
        # >0 daje prosty model wykładniczego chłodzenia.
        self.thermal_cooling_rate = float(
            cfg.get("thermal_cooling_rate", 0.0)
        )

        # Delikatny continuous shaping.
        self.thermal_penalty_coeff = float(
            cfg.get("thermal_penalty_coeff", 2.0)
        )

        # Kara zaczyna działać dopiero np. powyżej 90% momentu.
        self.saturation_start_ratio = float(
            cfg.get("saturation_start_ratio", 0.90)
        )

        self.saturation_penalty_coeff = float(
            cfg.get("saturation_penalty_coeff", 1.0)
        )

        # ==============================================================
        # LANDING / SAFETY
        # ==============================================================

        self.base_crash_height = float(
            cfg.get("base_crash_height", 0.05)
        )

        # Sukces zatwierdzamy dopiero po tym czasie od touchdown.
        # 2 s to wartość startowa.
        self.success_confirmation_time = float(
            cfg.get("success_confirmation_time", 2.0)
        )

        # Terminal event rewards.
        # Nie mnożymy przez dt.
        self.landing_success_coeff = float(
            cfg.get("landing_success_coeff", 100.0)
        )

        self.landing_failure_coeff = float(
            cfg.get("landing_failure_coeff", 100.0)
        )

        # ==============================================================
        # CONTINUOUS SHAPING
        # ==============================================================

        # Mały alive reward. Nie powinien dominować sukcesu.
        self.alive_coeff = (
            float(cfg.get("alive_coeff", 0.5))
            * self.dt
        )

        self.base_height_coeff = (
            float(cfg.get("base_height_coeff", 1.5))
            * self.dt
        )

        self.roll_pitch_pos_coeff = (
            float(cfg.get("roll_pitch_pos_coeff", 0.0))
            * self.dt
        )

        self.base_vel_coeff = (
            float(cfg.get("base_vel_coeff", 0.5))
            * self.dt
        )

        self.joint_vel_coeff = (
            float(cfg.get("joint_vel_coeff", 0.05))
            * self.dt
        )

        self.action_rate_coeff = (
            float(cfg.get("action_rate_coeff", 0.02))
            * self.dt
        )

        self.self_collision_coeff = (
            float(cfg.get("collision_coeff", 5.0))
            * self.dt
        )

        self.floor_collision_coeff = (
            float(cfg.get("floor_collision_coeff", 0.05))
            * self.dt
        )

        self.joint_pos_coeff = (
            float(cfg.get("joint_pos_coeff", 0.0))
            * self.dt
        )

        self.nominal_landing_height = cfg[
            "nominal_landing_height"
        ]

        self.soft_joint_position_limit = float(
            cfg.get("soft_joint_position_limit", 0.9)
        )

    # ==================================================================
    # INITIALIZATION
    # ==================================================================

    def init(self):
        self.env.internal_state[
            "joint_position_limits"
        ] = self.calculate_joint_position_limits()

        self.setup()

    def handle_model_change(self):
        self.env.internal_state[
            "joint_position_limits"
        ] = self.calculate_joint_position_limits()

    def calculate_joint_position_limits(self):
        joint_limits = (
            self.env.internal_state["mj_model"].jnt_range[1:]
        )

        midpoint = (
            joint_limits[:, 0] + joint_limits[:, 1]
        ) / 2.0

        joint_range = (
            joint_limits[:, 1] - joint_limits[:, 0]
        )

        lower = (
            midpoint
            - joint_range
            / 2.0
            * self.soft_joint_position_limit
        )

        upper = (
            midpoint
            + joint_range
            / 2.0
            * self.soft_joint_position_limit
        )

        return np.stack([lower, upper], axis=1)

    def setup(self):
        state = self.env.internal_state

        state["feet_time_on_ground"] = np.zeros(
            self.env.nr_feet
        )

        state["feet_time_in_air"] = np.zeros(
            self.env.nr_feet
        )

        state["previous_imu_linear_velocity"] = np.zeros(
            self.env.imu_linear_velocity_sensor_dim
        )

        state["previous_actuator_joint_velocities"] = np.zeros(
            self.env.nr_actuator_joints
        )

        # --------------------------------------------------------------
        # Landing state
        # --------------------------------------------------------------

        state["has_touched_ground"] = False
        state["time_since_touchdown"] = 0.0

        state["landing_evaluated"] = False
        state["landing_success"] = False
        state["landing_failure_reason"] = "none"

        # --------------------------------------------------------------
        # Safety state
        # --------------------------------------------------------------

        state["base_crash_detected"] = False
        state["motor_hard_limit_detected"] = False
        state["thermal_failure_detected"] = False

        # kompatybilność z wcześniejszym kodem / loggingiem
        state["actuator_overload_detected"] = False

        # --------------------------------------------------------------
        # Peak torque
        # --------------------------------------------------------------

        state["peak_leg_motor_torque"] = 0.0
        state["peak_spine_motor_torque"] = 0.0

        state["peak_leg_joint_torque_est"] = 0.0
        state["peak_spine_joint_torque_est"] = 0.0

        # stare nazwy dla kompatybilności
        state["peak_leg_torque"] = 0.0
        state["peak_spine_torque"] = 0.0

        # --------------------------------------------------------------
        # Thermal dose PER MOTOR
        # --------------------------------------------------------------

        nr_leg_actuators = len(
            self.env.leg_actuator_indices
        )

        state["leg_thermal_dose"] = np.zeros(
            nr_leg_actuators,
            dtype=float,
        )

        if self.env.spine_actuator_index == -1:
            state["spine_thermal_dose"] = np.zeros(
                0,
                dtype=float,
            )
        else:
            state["spine_thermal_dose"] = np.zeros(
                1,
                dtype=float,
            )

        # compatibility with previous logging
        state["leg_tau_squared_integral"] = 0.0
        state["spine_tau_squared_integral"] = 0.0

        # --------------------------------------------------------------
        # Landing statistics
        # --------------------------------------------------------------

        state.setdefault(
            "nr_successful_landings",
            0,
        )

    # ==================================================================
    # POST-STEP STATE UPDATE
    # ==================================================================

    def step(self):
        state = self.env.internal_state

        feet_floor_contacts = (
            self.env.terrain_function
            .check_feet_floor_contact()
        )

        state["feet_time_on_ground"] = np.where(
            feet_floor_contacts,
            state["feet_time_on_ground"] + self.env.dt,
            0.0,
        )

        state["feet_time_in_air"] = np.where(
            feet_floor_contacts,
            0.0,
            state["feet_time_in_air"] + self.env.dt,
        )

        state[
            "previous_actuator_joint_velocities"
        ] = state["data"].qvel[
            self.env.actuator_joint_mask_qvel
        ]

        state[
            "previous_imu_linear_velocity"
        ] = state["data"].sensordata[
            self.env.imu_linear_velocity_sensor_adr:
            self.env.imu_linear_velocity_sensor_adr
            + self.env.imu_linear_velocity_sensor_dim
        ]

        if np.any(feet_floor_contacts):
            state["has_touched_ground"] = True

        if state["has_touched_ground"]:
            state["time_since_touchdown"] += self.env.dt

    # ==================================================================
    # TORQUE CONVERSION
    # ==================================================================

    def _get_motor_and_joint_torque(self):
        """
        Zwraca:
            motor_tau
            joint_tau_est

        Kolejność obu tablic == kolejność actuatorów MuJoCo.
        """

        model = self.env.internal_state["mj_model"]
        data = self.env.internal_state["data"]

        actuator_force = np.asarray(
            data.actuator_force,
            dtype=float,
        )

        # pierwszy element gear vector dla prostych hinge actuatorów
        mujoco_gear = np.asarray(
            model.actuator_gear[:, 0],
            dtype=float,
        )

        if self.torque_measurement_mode == "mujoco_actuator":
            # Rekomendowany model:
            #
            # actuator_force = motor-side torque
            #
            motor_tau = actuator_force.copy()

            joint_tau_est = (
                actuator_force * mujoco_gear
            )

        else:
            # Legacy / joint-equivalent model.
            #
            # Zakładamy, że actuator_force reprezentuje joint torque,
            # a fizyczny gear nie występuje bezpośrednio w MJCF.

            motor_tau = actuator_force.copy()

            leg_idx = np.asarray(
                self.env.leg_actuator_indices,
                dtype=int,
            )

            motor_tau[leg_idx] /= self.leg_gear_ratio

            if self.env.spine_actuator_index != -1:
                motor_tau[
                    self.env.spine_actuator_index
                ] /= self.spine_gear_ratio

            joint_tau_est = actuator_force.copy()

        return motor_tau, joint_tau_est

    # ==================================================================
    # THERMAL MODEL
    # ==================================================================

    def _update_thermal_dose(
        self,
        leg_motor_tau,
        spine_motor_tau,
    ):
        state = self.env.internal_state

        # --------------------------------------------------------------
        # cooling
        # --------------------------------------------------------------

        if self.thermal_cooling_rate > 0.0:
            decay = np.exp(
                -self.thermal_cooling_rate
                * self.dt
            )

            state["leg_thermal_dose"] *= decay
            state["spine_thermal_dose"] *= decay

        # --------------------------------------------------------------
        # heating ~ torque^2
        # --------------------------------------------------------------

        if leg_motor_tau.size:
            leg_util = (
                np.abs(leg_motor_tau)
                / self.leg_motor_torque_limit
            )

            state["leg_thermal_dose"] += (
                np.square(leg_util)
                * self.dt
            )

        if spine_motor_tau.size:
            spine_util = (
                np.abs(spine_motor_tau)
                / self.spine_motor_torque_limit
            )

            state["spine_thermal_dose"] += (
                np.square(spine_util)
                * self.dt
            )

    # ==================================================================
    # CURRICULUM / FINAL RESULT
    # ==================================================================

    def _update_curriculum(self, success):
        curriculum = self.env.internal_state.get(
            "landing_curriculum"
        )

        if curriculum is None:
            return

        success_float = float(bool(success))

        alpha = curriculum["ema_alpha"]

        curriculum["success_ema"] = (
            (1.0 - alpha)
            * curriculum["success_ema"]
            + alpha
            * success_float
        )

        curriculum["last_success"] = success_float
        curriculum["nr_evaluated_landings"] += 1

        ema = curriculum["success_ema"]

        if ema > curriculum["success_threshold_up"]:
            curriculum["difficulty"] = min(
                1.0,
                curriculum["difficulty"]
                + curriculum["difficulty_step_up"],
            )

            curriculum["last_update"] = "increase"

        elif ema < curriculum["success_threshold_down"]:
            curriculum["difficulty"] = max(
                0.0,
                curriculum["difficulty"]
                - curriculum["difficulty_step_down"],
            )

            curriculum["last_update"] = "decrease"

        else:
            curriculum["last_update"] = "hold"

    def _finalize_landing(
        self,
        success,
        failure_reason="none",
    ):
        """
        Finalizuje wynik dokładnie raz.
        """

        state = self.env.internal_state

        if state["landing_evaluated"]:
            return 0.0

        success = bool(success)

        state["landing_evaluated"] = True
        state["landing_success"] = success

        state["landing_failure_reason"] = (
            "none"
            if success
            else failure_reason
        )

        if success:
            state["nr_successful_landings"] += 1

        self._update_curriculum(success)

        return (
            self.landing_success_coeff
            if success
            else -self.landing_failure_coeff
        )

    # ==================================================================
    # MAIN REWARD
    # ==================================================================

    def reward_and_info(self, action):
        state = self.env.internal_state
        data = state["data"]

        qpos = data.qpos[
            self.env.actuator_joint_mask_qpos
        ]

        qvel = data.qvel[
            self.env.actuator_joint_mask_qvel
        ]

        leg_qvel = qvel[
            self.env.leg_actuator_indices
        ]

        lin_vel = data.sensordata[
            self.env.imu_linear_velocity_sensor_adr:
            self.env.imu_linear_velocity_sensor_adr
            + self.env.imu_linear_velocity_sensor_dim
        ]

        ang_vel = data.sensordata[
            self.env.imu_angular_velocity_sensor_adr:
            self.env.imu_angular_velocity_sensor_adr
            + self.env.imu_angular_velocity_sensor_dim
        ]

        euler = state["imu_orientation_euler"]

        has_touched = state.get(
            "has_touched_ground",
            False,
        )

        time_since_touch = state.get(
            "time_since_touchdown",
            0.0,
        )

        height = state[
            "robot_imu_height_over_ground"
        ]

        target_height = self.nominal_landing_height

        # ==============================================================
        # MOTOR / JOINT TORQUE
        # ==============================================================

        motor_tau, joint_tau_est = (
            self._get_motor_and_joint_torque()
        )

        leg_idx = np.asarray(
            self.env.leg_actuator_indices,
            dtype=int,
        )

        leg_motor_tau = motor_tau[leg_idx]
        leg_joint_tau = joint_tau_est[leg_idx]

        if self.env.spine_actuator_index == -1:
            spine_motor_tau = np.zeros(
                0,
                dtype=float,
            )

            spine_joint_tau = np.zeros(
                0,
                dtype=float,
            )

        else:
            spine_idx = self.env.spine_actuator_index

            spine_motor_tau = motor_tau[
                spine_idx:spine_idx + 1
            ]

            spine_joint_tau = joint_tau_est[
                spine_idx:spine_idx + 1
            ]

        # ==============================================================
        # PEAK TORQUE
        # ==============================================================

        if leg_motor_tau.size:
            state["peak_leg_motor_torque"] = max(
                state["peak_leg_motor_torque"],
                float(
                    np.max(
                        np.abs(leg_motor_tau)
                    )
                ),
            )

            state["peak_leg_joint_torque_est"] = max(
                state["peak_leg_joint_torque_est"],
                float(
                    np.max(
                        np.abs(leg_joint_tau)
                    )
                ),
            )

        if spine_motor_tau.size:
            state["peak_spine_motor_torque"] = max(
                state["peak_spine_motor_torque"],
                float(
                    np.max(
                        np.abs(spine_motor_tau)
                    )
                ),
            )

            state["peak_spine_joint_torque_est"] = max(
                state["peak_spine_joint_torque_est"],
                float(
                    np.max(
                        np.abs(spine_joint_tau)
                    )
                ),
            )

        # compatibility
        state["peak_leg_torque"] = (
            state["peak_leg_motor_torque"]
        )

        state["peak_spine_torque"] = (
            state["peak_spine_motor_torque"]
        )

        # ==============================================================
        # RAW tau^2 DIAGNOSTICS
        # ==============================================================

        if leg_motor_tau.size:
            state[
                "leg_tau_squared_integral"
            ] += (
                float(
                    np.sum(
                        np.square(leg_motor_tau)
                    )
                )
                * self.dt
            )

        if spine_motor_tau.size:
            state[
                "spine_tau_squared_integral"
            ] += (
                float(
                    np.sum(
                        np.square(spine_motor_tau)
                    )
                )
                * self.dt
            )

        # ==============================================================
        # MOTOR UTILIZATION
        # ==============================================================

        if leg_motor_tau.size:
            leg_util = (
                np.abs(leg_motor_tau)
                / self.leg_motor_torque_limit
            )
        else:
            leg_util = np.zeros(0)

        if spine_motor_tau.size:
            spine_util = (
                np.abs(spine_motor_tau)
                / self.spine_motor_torque_limit
            )
        else:
            spine_util = np.zeros(0)

        max_leg_util = (
            float(np.max(leg_util))
            if leg_util.size
            else 0.0
        )

        max_spine_util = (
            float(np.max(spine_util))
            if spine_util.size
            else 0.0
        )

        # ==============================================================
        # HARD MOTOR LIMIT
        # ==============================================================

        hard_limit = (
            np.any(
                leg_util
                > 1.0 + self.hard_limit_tolerance
            )
            or
            np.any(
                spine_util
                > 1.0 + self.hard_limit_tolerance
            )
        )

        if hard_limit:
            state["motor_hard_limit_detected"] = True

        # compatibility with old metric
        state["actuator_overload_detected"] = (
            state["motor_hard_limit_detected"]
            or state["thermal_failure_detected"]
        )

        # ==============================================================
        # THERMAL DOSE
        # ==============================================================

        self._update_thermal_dose(
            leg_motor_tau,
            spine_motor_tau,
        )

        leg_dose = state["leg_thermal_dose"]
        spine_dose = state["spine_thermal_dose"]

        max_leg_dose = (
            float(np.max(leg_dose))
            if leg_dose.size
            else 0.0
        )

        max_spine_dose = (
            float(np.max(spine_dose))
            if spine_dose.size
            else 0.0
        )

        thermal_failure = (
            max_leg_dose
            >= self.thermal_dose_limit
            or
            max_spine_dose
            >= self.thermal_dose_limit
        )

        if thermal_failure:
            state["thermal_failure_detected"] = True

        state["actuator_overload_detected"] = (
            state["motor_hard_limit_detected"]
            or state["thermal_failure_detected"]
        )

        # ==============================================================
        # BASE CRASH
        # ==============================================================

        if (
            has_touched
            and height < self.base_crash_height
        ):
            state["base_crash_detected"] = True

        # ==============================================================
        # CONTINUOUS MOTOR COST
        # ==============================================================

        all_util_parts = []

        if leg_util.size:
            all_util_parts.append(leg_util)

        if spine_util.size:
            all_util_parts.append(spine_util)

        if all_util_parts:
            all_util = np.concatenate(
                all_util_parts
            )

            thermal_reward = (
                -self.thermal_penalty_coeff
                * np.mean(
                    np.square(all_util)
                )
                * self.dt
            )

            sat_normalized = np.maximum(
                0.0,
                (
                    all_util
                    - self.saturation_start_ratio
                )
                / max(
                    1e-6,
                    1.0
                    - self.saturation_start_ratio,
                )
            )

            saturation_reward = (
                -self.saturation_penalty_coeff
                * np.mean(
                    np.square(
                        sat_normalized
                    )
                )
                * self.dt
            )

        else:
            thermal_reward = 0.0
            saturation_reward = 0.0

        # ==============================================================
        # FREE FALL / POST TOUCHDOWN SHAPING
        # ==============================================================

        angular_position_reward = (
            -self.roll_pitch_pos_coeff
            * np.sum(
                np.square(euler[:2])
            )
        )

        if not has_touched:
            # ----------------------------------------------------------
            # FREE FALL
            # ----------------------------------------------------------

            base_vel_xy_reward = 0.0
            base_vel_z_reward = 0.0
            base_height_reward = 0.0

            nominal_joint_pos = state[
                "actuator_joint_nominal_positions"
            ]

            joint_pos_reward = (
                -0.1
                * self.joint_pos_coeff
                * np.mean(
                    np.square(
                        qpos - nominal_joint_pos
                    )
                )
            )

            joint_vel_reward = (
                -0.25
                * self.joint_vel_coeff
                * np.mean(
                    np.square(
                        leg_qvel
                    )
                )
            )

        else:
            # ----------------------------------------------------------
            # POST TOUCHDOWN
            # ----------------------------------------------------------

            base_vel_xy_reward = (
                -self.base_vel_coeff
                * (
                    np.sum(
                        np.square(
                            lin_vel[:2]
                        )
                    )
                    + np.sum(
                        np.square(
                            ang_vel
                        )
                    )
                )
            )

            joint_pos_reward = 0.0

            # ----------------------------------------------------------
            # ENERGY ABSORPTION
            # ----------------------------------------------------------

            if time_since_touch < 0.5:
                # Nie karzemy ruchu w dół podczas kompresji.
                if lin_vel[2] > 0.0:
                    base_vel_z_reward = (
                        -2.0
                        * self.base_vel_coeff
                        * np.square(
                            lin_vel[2]
                        )
                    )
                else:
                    base_vel_z_reward = 0.0

                # nogi mogą bardzo szybko pracować
                joint_vel_reward = 0.0

                base_height_reward = 0.0

            else:
                # ------------------------------------------------------
                # STABILIZATION
                # ------------------------------------------------------

                base_vel_z_reward = (
                    -self.base_vel_coeff
                    * np.square(
                        lin_vel[2]
                    )
                )

                joint_vel_reward = (
                    -self.joint_vel_coeff
                    * np.mean(
                        np.square(
                            leg_qvel
                        )
                    )
                )

                base_height_reward = (
                    -self.base_height_coeff
                    * np.square(
                        height
                        - target_height
                    )
                )

        base_vel_reward = (
            base_vel_xy_reward
            + base_vel_z_reward
        )

        # ==============================================================
        # COLLISIONS
        # ==============================================================

        all_geom_xpos = data.geom_xpos[
            self.env.reward_collision_sphere_geom_ids
        ]

        all_geom_sizes = (
            state["mj_model"]
            .geom_size[
                self.env.reward_collision_sphere_geom_ids,
                0,
            ]
        )

        distance_between_geoms = np.linalg.norm(
            all_geom_xpos[:, None]
            - all_geom_xpos[None],
            axis=-1,
        )

        contact_between_geoms = (
            distance_between_geoms
            <= (
                all_geom_sizes[:, None]
                + all_geom_sizes[None]
            )
        )

        nr_self_collisions = (
            np.sum(contact_between_geoms)
            - len(
                self.env.reward_collision_sphere_geom_ids
            )
        ) // 2

        nr_self_collisions = np.maximum(
            nr_self_collisions
            - state["nr_collisions_in_nominal"],
            0,
        )

        self_collision_reward = (
            -self.self_collision_coeff
            * nr_self_collisions
        )

        geom_ground_heights = (
            self.env.terrain_function
            .ground_height_at(
                all_geom_xpos[:, 0],
                all_geom_xpos[:, 1],
            )
        )

        geom_clearance = (
            all_geom_xpos[:, 2]
            - geom_ground_heights
        )

        nr_floor_collisions = np.sum(
            geom_clearance
            < all_geom_sizes
        )

        floor_collision_reward = (
            -self.floor_collision_coeff
            * nr_floor_collisions
        )

        collision_reward = (
            self_collision_reward
            + floor_collision_reward
        )

        # ==============================================================
        # ACTION RATE
        # ==============================================================

        action_rate_reward = (
            -self.action_rate_coeff
            * np.mean(
                np.square(
                    action
                    - state["last_action"]
                )
            )
        )

        # ==============================================================
        # ALIVE
        # ==============================================================

        alive_reward = self.alive_coeff

        # ==============================================================
        # FINAL LANDING RESULT
        # ==============================================================

        landing_event_reward = 0.0

        if not state["landing_evaluated"]:

            # Hard physical/model violation.
            if state["motor_hard_limit_detected"]:
                landing_event_reward = (
                    self._finalize_landing(
                        success=False,
                        failure_reason="motor_hard_limit",
                    )
                )

            elif state["thermal_failure_detected"]:
                landing_event_reward = (
                    self._finalize_landing(
                        success=False,
                        failure_reason="thermal_limit",
                    )
                )

            elif state["base_crash_detected"]:
                landing_event_reward = (
                    self._finalize_landing(
                        success=False,
                        failure_reason="base_crash",
                    )
                )

            # Success dopiero po przeżyciu całego safety window.
            elif (
                has_touched
                and
                time_since_touch
                >= self.success_confirmation_time
            ):
                landing_event_reward = (
                    self._finalize_landing(
                        success=True,
                    )
                )

        # ==============================================================
        # TOTAL REWARD
        # ==============================================================

        reward = (
            alive_reward
            + base_vel_reward
            + angular_position_reward
            + base_height_reward
            + joint_pos_reward
            + joint_vel_reward
            + thermal_reward
            + saturation_reward
            + landing_event_reward
            + action_rate_reward
            + collision_reward
        )

        reward = np.nan_to_num(
            reward,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # ==============================================================
        # LOGGING
        # ==============================================================

        info = state["info"]

        # rewards
        info["reward/alive"] = alive_reward
        info["reward/base_vel"] = base_vel_reward
        info["reward/angular_position"] = (
            angular_position_reward
        )
        info["reward/base_height"] = (
            base_height_reward
        )
        info["reward/joint_pos"] = (
            joint_pos_reward
        )
        info["reward/joint_vel"] = (
            joint_vel_reward
        )
        info["reward/thermal"] = (
            thermal_reward
        )
        info["reward/saturation"] = (
            saturation_reward
        )
        info["reward/landing_event"] = (
            landing_event_reward
        )
        info["reward/action_rate"] = (
            action_rate_reward
        )
        info["reward/collision"] = (
            collision_reward
        )
        info["reward/total"] = reward

        # motor torque
        info["metrics/peak_leg_motor_torque"] = (
            state["peak_leg_motor_torque"]
        )

        info["metrics/peak_spine_motor_torque"] = (
            state["peak_spine_motor_torque"]
        )

        info["metrics/peak_leg_joint_torque_est"] = (
            state["peak_leg_joint_torque_est"]
        )

        info["metrics/peak_spine_joint_torque_est"] = (
            state["peak_spine_joint_torque_est"]
        )

        info["metrics/max_leg_motor_utilization"] = (
            max_leg_util
        )

        info["metrics/max_spine_motor_utilization"] = (
            max_spine_util
        )

        # thermal
        info["metrics/leg_thermal_dose_max"] = (
            max_leg_dose
        )

        info["metrics/spine_thermal_dose_max"] = (
            max_spine_dose
        )

        info["metrics/leg_tau_squared_integral"] = (
            state["leg_tau_squared_integral"]
        )

        info["metrics/spine_tau_squared_integral"] = (
            state["spine_tau_squared_integral"]
        )

        # failures
        info["metrics/base_crash_detected"] = float(
            state["base_crash_detected"]
        )

        info[
            "metrics/motor_hard_limit_detected"
        ] = float(
            state["motor_hard_limit_detected"]
        )

        info[
            "metrics/thermal_failure_detected"
        ] = float(
            state["thermal_failure_detected"]
        )

        info[
            "metrics/actuator_overload_detected"
        ] = float(
            state["actuator_overload_detected"]
        )

        # landing
        info["curriculum/landing_evaluated"] = float(
            state["landing_evaluated"]
        )

        info["curriculum/landing_success"] = float(
            state["landing_success"]
        )
        # Mutually exclusive episode outcomes, sampled at episode end by PPO.
        failure_reason = state["landing_failure_reason"]
        info["outcome/failure_motor_hard_limit"] = float(failure_reason == "motor_hard_limit")
        info["outcome/failure_thermal_limit"] = float(failure_reason == "thermal_limit")
        info["outcome/failure_base_crash"] = float(failure_reason == "base_crash")
        info["outcome/unresolved"] = float(not state["landing_evaluated"])

        # Event metrics — dużo łatwiejsze do poprawnej agregacji.
        info["events/landing_success"] = float(
            landing_event_reward > 0.0
        )

        info["events/landing_failure"] = float(
            landing_event_reward < 0.0
        )

        curriculum = state.get(
            "landing_curriculum"
        )

        if curriculum is not None:
            info["curriculum/difficulty"] = (
                curriculum["difficulty"]
            )

            info["curriculum/success_ema"] = (
                curriculum["success_ema"]
            )

            info[
                "curriculum/nr_evaluated_landings"
            ] = curriculum[
                "nr_evaluated_landings"
            ]

            nr_eval = curriculum[
                "nr_evaluated_landings"
            ]

            if nr_eval > 0:
                info[
                    "curriculum/local_success_rate"
                ] = (
                    state[
                        "nr_successful_landings"
                    ]
                    / nr_eval
                )
            else:
                info[
                    "curriculum/local_success_rate"
                ] = 0.0

        return reward
