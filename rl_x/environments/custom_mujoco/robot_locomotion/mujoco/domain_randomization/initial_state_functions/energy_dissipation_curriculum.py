import numpy as np
from scipy.spatial.transform import Rotation
import mujoco

class EnergyDissipationCurriculumInitialState:

    def __init__(self, env):
        self.env = env

        domain_rand_config = env.env_config.get("domain_randomization", {})
        initial_state_config = domain_rand_config.get("initial_state", {})
        curriculum_config = initial_state_config.get(
            "energy_dissipation_curriculum", {}
        )

        self.start_height = curriculum_config.get("start_height", 0.5)
        self.target_height = curriculum_config.get("target_height", 6.0)
        self.height_reference = curriculum_config.get("height_reference", "clearance")
        if self.height_reference not in ("clearance", "base"):
            raise ValueError("height_reference must be 'clearance' or 'base'")
        self.eval_difficulty = float(curriculum_config.get("eval_difficulty", 1.0))
        if not np.isfinite(self.eval_difficulty) or not 0.0 <= self.eval_difficulty <= 1.0:
            raise ValueError("eval_difficulty must be between 0 and 1")

        self.min_angle_deg = curriculum_config.get("min_angle_deg", 5.0)
        self.max_angle_deg = curriculum_config.get("max_angle_deg", 25.0)

        self.min_lateral_velocity = curriculum_config.get(
            "min_lateral_velocity", 0.2
        )
        self.max_lateral_velocity = curriculum_config.get(
            "max_lateral_velocity", 2.5
        )

        # Curriculum parameters
        self.ema_alpha = curriculum_config.get("ema_alpha", 0.05)

        self.success_threshold_up = curriculum_config.get(
            "success_threshold_up", 0.85
        )
        self.success_threshold_down = curriculum_config.get(
            "success_threshold_down", 0.55
        )

        self.difficulty_step_up = curriculum_config.get(
            "difficulty_step_up", 0.01
        )
        self.difficulty_step_down = curriculum_config.get(
            "difficulty_step_down", 0.005
        )

        # Keep practicing the shortest drop after the curriculum advances.
        self.replay_easy_probability = float(
            curriculum_config.get("replay_easy_probability", 0.15)
        )
        if (
            not np.isfinite(self.replay_easy_probability)
            or not 0.0 <= self.replay_easy_probability <= 1.0
        ):
            raise ValueError("replay_easy_probability must be between 0 and 1")

        self.is_energy_curriculum = (
            initial_state_config.get("type")
            == "energy_dissipation_curriculum"
        )
        self.nominal_replay_probability = float(curriculum_config.get("nominal_replay_probability", 0.0))
        self.nominal_replay_top_probability = float(curriculum_config.get("nominal_replay_top_probability", 0.5))
        for value in (self.nominal_replay_probability, self.nominal_replay_top_probability):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("Nominal replay probabilities must be between 0 and 1")
        self.factorized = None
        if curriculum_config.get("factorized_enabled", False):
            from .factorized_landing import FactorizedLanding
            self.factorized = FactorizedLanding(self, curriculum_config)

    # ------------------------------------------------------------------
    # CURRICULUM STATE
    # ------------------------------------------------------------------

    def _get_curriculum_state(self):
        state = self.env.internal_state

        if "landing_curriculum" not in state:
            state["landing_curriculum"] = {
                "difficulty": 0.0,

                "success_ema": 0.0,
                "nr_evaluated_landings": 0,
                "last_success": 0.0,
                "last_update": "none",
                "fixed_difficulty": self.env.runner_mode == "test",
                "replay_easy": False,
                "applied_difficulty": 0.0,

                # Curriculum parameters
                "ema_alpha": self.ema_alpha,
                "success_threshold_up":
                    self.success_threshold_up,
                "success_threshold_down":
                    self.success_threshold_down,
                "difficulty_step_up":
                    self.difficulty_step_up,
                "difficulty_step_down":
                    self.difficulty_step_down,
            }
            if state["landing_curriculum"]["fixed_difficulty"]:
                state["landing_curriculum"]["difficulty"] = self.eval_difficulty

        return state["landing_curriculum"]


    def update_landing_result(self, success):

        curriculum = self._get_curriculum_state()

        if self.factorized is not None and self.env.runner_mode == "train" and not curriculum["fixed_difficulty"]:
            self.factorized.update(success)
            return

        if curriculum["replay_easy"] or curriculum.get("replay_nominal", False):
            return

        success = float(bool(success))

        # EMA
        curriculum["success_ema"] = (
            (1.0 - self.ema_alpha) * curriculum["success_ema"]
            + self.ema_alpha * success
        )

        curriculum["last_success"] = success
        curriculum["nr_evaluated_landings"] += 1

        if curriculum["fixed_difficulty"]:
            return

        ema = curriculum["success_ema"]
        difficulty = curriculum["difficulty"]

        # Increase difficulty
        if ema > self.success_threshold_up:
            difficulty += self.difficulty_step_up

            curriculum["last_update"] = "increase"

        # Decrease difficulty
        elif ema < self.success_threshold_down:
            difficulty -= self.difficulty_step_down

            curriculum["last_update"] = "decrease"

        else:
            curriculum["last_update"] = "hold"

        curriculum["difficulty"] = np.clip(
            difficulty,
            0.0,
            1.0,
        )

    # ------------------------------------------------------------------
    # DIFFICULTY -> INITIAL CONDITIONS
    # ------------------------------------------------------------------

    def _get_curriculum_schedule(self, difficulty=None):

        curriculum = self._get_curriculum_state()

        if difficulty is None:
            difficulty = curriculum["difficulty"]

        # Height
        actual_height = (
             self.start_height
            + difficulty * (
                self.target_height - self.start_height
            )
        )

        # Roll / pitch
        angle_deg = (
            self.min_angle_deg
            + difficulty**2 * (
                self.max_angle_deg - self.min_angle_deg
            )
        )

        angle_rad = np.deg2rad(angle_deg)

        # Lateral velocity
        lateral_velocity = (
            self.min_lateral_velocity
            + difficulty * (
                self.max_lateral_velocity
                - self.min_lateral_velocity
            )
        )

        return (
            actual_height,
            angle_rad,
            lateral_velocity,
            difficulty,
        )

    # ------------------------------------------------------------------
    # INITIAL STATE
    # ------------------------------------------------------------------

    def setup(self):

        if not self.is_energy_curriculum:
            return None

        curriculum = self._get_curriculum_state()
        curriculum["replay_nominal"] = bool(
            self.env.runner_mode == "train"
            and self.factorized is None
            and self.nominal_replay_probability > 0.0
            and self.env.np_rng.random() < self.nominal_replay_probability
        )
        curriculum["replay_easy"] = bool(
            self.env.runner_mode == "train"
            and self.factorized is None
            and not curriculum["replay_nominal"]
            and curriculum["difficulty"] > 0.0
            and self.env.np_rng.random() < self.replay_easy_probability
        )
        curriculum["applied_difficulty"] = (
            0.0 if curriculum["replay_easy"] else curriculum["difficulty"]
        )

        (
            height,
            max_angle,
            max_lateral_velocity,
            difficulty,
        ) = self._get_curriculum_schedule(curriculum["applied_difficulty"])

        if self.factorized is not None and self.env.runner_mode == "train" and not curriculum["fixed_difficulty"]:
            height, max_angle, max_lateral_velocity, difficulty = self.factorized.sample(curriculum)
        elif curriculum["replay_nominal"]:
            height = (self.target_height if self.env.np_rng.random() < self.nominal_replay_top_probability
                      else self.env.np_rng.uniform(self.start_height, self.target_height))
            max_angle = max_lateral_velocity = 0.0

        # --------------------------------------------------------------
        # ORIENTATION
        # --------------------------------------------------------------

        roll_angle = self.env.np_rng.uniform(
            -max_angle,
            max_angle,
        )

        pitch_angle = self.env.np_rng.uniform(
            -max_angle,
            max_angle,
        )

        # Na początku zostawiamy yaw = 0.
        yaw_angle = 0.0

        quaternion = Rotation.from_euler(
            "xyz",
            [
                roll_angle,
                pitch_angle,
                yaw_angle,
            ],
        ).as_quat(scalar_first=True)

        # --------------------------------------------------------------
        # JOINTS
        # --------------------------------------------------------------

        actuator_joint_positions = (
            self.env.internal_state[
                "actuator_joint_nominal_positions"
            ].copy()
        )

        actuator_joint_velocities = np.zeros(
            self.env.actuator_joint_max_velocities.size
        )

        # --------------------------------------------------------------
        # LATERAL VELOCITY
        # --------------------------------------------------------------

        linear_vel_x = self.env.np_rng.uniform(
            -max_lateral_velocity,
            max_lateral_velocity,
        )

        linear_vel_y = self.env.np_rng.uniform(
            -max_lateral_velocity,
            max_lateral_velocity,
        )

        # Swobodny spadek -> początkowe v_z = 0.
        linear_vel_z = 0.0

        # Brak początkowego spinu.
        angular_velocities = np.zeros(3)

        # --------------------------------------------------------------
        # POSITION
        # --------------------------------------------------------------

        nominal_height = self.env.internal_state[
            "robot_nominal_qpos_height_over_ground"
        ]

        center_height = self.env.internal_state[
            "center_height"
        ]

        initial_z = center_height + height
        if self.height_reference == "clearance":
            initial_z += nominal_height

        linear_positions = np.array([
            0.0,
            0.0,
            initial_z,
        ])

        # --------------------------------------------------------------
        # QPOS / QVEL
        # --------------------------------------------------------------

        qpos = self.env.initial_qpos.copy()

        qpos[:3] = linear_positions
        qpos[3:7] = quaternion

        qpos[
            self.env.actuator_joint_mask_qpos
        ] = actuator_joint_positions

        qvel = np.zeros(
            self.env.initial_mj_model.nv
        )

        qvel[:3] = [
            linear_vel_x,
            linear_vel_y,
            linear_vel_z,
        ]

        qvel[3:6] = angular_velocities

        qvel[
            self.env.actuator_joint_mask_qvel
        ] = actuator_joint_velocities

        # --------------------------------------------------------------
        # MUJOCO FORWARD
        # --------------------------------------------------------------

        data = mujoco.MjData(
            self.env.internal_state["mj_model"]
        )

        data.qpos = qpos
        data.qvel = qvel

        data.ctrl = np.zeros(
            self.env.nr_actuator_joints
        )

        mujoco.mj_forward(
            self.env.internal_state["mj_model"],
            data,
        )

        # --------------------------------------------------------------
        # PREVENT FEET FROM STARTING UNDERGROUND
        # --------------------------------------------------------------

        feet_x_pos = data.geom_xpos[
            self.env.foot_geom_indices,
            0,
        ]

        feet_y_pos = data.geom_xpos[
            self.env.foot_geom_indices,
            1,
        ]

        correction = np.max(
            self.env.terrain_function.ground_height_at(
                feet_x_pos,
                feet_y_pos,
            )
            - data.geom_xpos[
                self.env.foot_geom_indices,
                2,
            ]
        )

        if correction > 0:
            qpos[2] += correction

        # --------------------------------------------------------------
        # LOGGING
        # --------------------------------------------------------------

        info = self.env.internal_state["info"]

        info["curriculum/difficulty"] = curriculum["difficulty"]
        info["curriculum/applied_difficulty"] = difficulty
        info["curriculum/replay_easy"] = float(curriculum["replay_easy"])
        info["curriculum/replay_nominal"] = float(curriculum["replay_nominal"])
        info["curriculum/initial_height"] = height
        info["curriculum/initial_base_height"] = float(qpos[2])
        info["curriculum/initial_roll_deg"] = float(np.rad2deg(roll_angle))
        info["curriculum/initial_pitch_deg"] = float(np.rad2deg(pitch_angle))
        info["curriculum/initial_vx"] = float(linear_vel_x)
        info["curriculum/initial_vy"] = float(linear_vel_y)
        info["curriculum/initial_speed_xy"] = float(np.hypot(linear_vel_x, linear_vel_y))
        info["curriculum/max_angle_deg"] = np.rad2deg(max_angle)
        info["curriculum/max_lateral_velocity"] = (
            max_lateral_velocity
        )

        info["curriculum/success_ema"] = (
            curriculum["success_ema"]
        )

        info["curriculum/last_success"] = (
            curriculum["last_success"]
        )

        info["curriculum/nr_evaluated_landings"] = (
            curriculum["nr_evaluated_landings"]
        )

        return qpos, qvel
