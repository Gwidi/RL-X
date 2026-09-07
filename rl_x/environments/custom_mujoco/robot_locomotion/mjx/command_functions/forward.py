import numpy as np
import jax
import jax.numpy as jnp


class ForwardCommands:
    """Random non-negative X commands with zero lateral and yaw rates."""

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

        keep_nominal = np.zeros(env.nr_actuator_joints, dtype=bool)
        keep_nominal[
            env.robot_config["actuator_joints_to_stay_near_nominal"]
        ] = True
        self.default_actuator_joint_keep_nominal = jnp.asarray(keep_nominal)

    def init(self, internal_state):
        internal_state[
            "actuator_joint_keep_nominal"
        ] = self.default_actuator_joint_keep_nominal

    def get_next_command(
        self, internal_state, should_sample_commands, subkey
    ):
        velocity_sampling_key, all_zeroing_key, single_zeroing_key = (
            jax.random.split(subkey, 3)
        )

        max_velocities = internal_state["max_command_velocities"]
        min_x_velocity = (
            self.zero_clip_threshold_percentage * max_velocities[0]
        )
        goal_x_velocity = jax.random.uniform(
            velocity_sampling_key,
            (),
            minval=min_x_velocity,
            maxval=max_velocities[0],
        )
        goal_velocities = jnp.array([goal_x_velocity, 0.0, 0.0])
        goal_velocities = jnp.where(
            jax.random.bernoulli(all_zeroing_key, self.all_zero_chance),
            jnp.zeros(3),
            goal_velocities,
        )
        goal_velocities = jnp.where(
            jax.random.bernoulli(
                single_zeroing_key, self.single_zero_chance
            ),
            jnp.zeros(3),
            goal_velocities,
        )

        internal_state["goal_velocities"] = jnp.where(
            should_sample_commands,
            goal_velocities,
            internal_state["goal_velocities"],
        )

        actuator_joint_keep_nominal = jnp.where(
            jnp.all(goal_velocities == 0.0),
            jnp.ones(self.env.nr_actuator_joints, dtype=bool),
            self.default_actuator_joint_keep_nominal,
        )
        internal_state["actuator_joint_keep_nominal"] = jnp.where(
            should_sample_commands,
            actuator_joint_keep_nominal,
            internal_state["actuator_joint_keep_nominal"],
        )
