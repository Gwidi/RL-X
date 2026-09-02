import numpy as np


class ForwardCommands:
    """Random forward and yaw commands without lateral velocity."""

    def __init__(self, env):
        self.env = env
        command_config = env.env_config["command"]
        self.max_velocity_per_m_factor = command_config[
            "max_velocity_per_m_factor"
        ]
        self.clip_max_velocity = command_config["clip_max_velocity"]
        self.zero_clip_threshold_percentage = command_config[
            "zero_clip_threshold_percentage"
        ]
        self.all_zero_chance = command_config["all_zero_chance"]
        self.single_zero_chance = command_config["single_zero_chance"]

        self.default_actuator_joint_keep_nominal = np.zeros(
            env.nr_actuator_joints, dtype=bool
        )
        self.default_actuator_joint_keep_nominal[
            env.robot_config["actuator_joints_to_stay_near_nominal"]
        ] = True

    def init(self):
        self.env.internal_state[
            "actuator_joint_keep_nominal"
        ] = self.default_actuator_joint_keep_nominal

    def get_next_command(self):
        max_velocities = self.env.internal_state["max_command_velocities"]
        goal_velocities = self.env.np_rng.uniform(
            size=(3,), low=-max_velocities, high=max_velocities
        )
        goal_velocities[0] = abs(goal_velocities[0])
        goal_velocities[1] = 0.0
        goal_velocities = np.where(
            np.abs(goal_velocities)
            < self.zero_clip_threshold_percentage * max_velocities,
            0.0,
            goal_velocities,
        )
        goal_velocities = np.where(
            self.env.np_rng.binomial(n=1, p=self.all_zero_chance),
            np.zeros(3),
            goal_velocities,
        )
        goal_velocities = np.where(
            self.env.np_rng.uniform(size=(3,)) < self.single_zero_chance,
            0.0,
            goal_velocities,
        )

        self.env.internal_state["goal_velocities"] = goal_velocities

        actuator_joint_keep_nominal = np.where(
            np.all(goal_velocities == 0.0),
            np.ones(self.env.nr_actuator_joints, dtype=bool),
            self.default_actuator_joint_keep_nominal,
        )
        self.env.internal_state[
            "actuator_joint_keep_nominal"
        ] = actuator_joint_keep_nominal
