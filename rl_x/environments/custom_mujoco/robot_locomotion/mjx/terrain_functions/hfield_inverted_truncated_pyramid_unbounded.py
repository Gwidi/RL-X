import numpy as np
import jax.numpy as jnp

from rl_x.environments.custom_mujoco.robot_locomotion.mjx.terrain_functions.hfield_inverted_truncated_pyramid import (
    HFieldInvertedTruncatedPyramidTerrainGeneration,
)


class HFieldInvertedTruncatedPyramidUnboundedTerrainGeneration(
    HFieldInvertedTruncatedPyramidTerrainGeneration
):
    """Inverted square frustum with a non-negative, unbounded curriculum.

    Difficulty 1.0 produces the same terrain as
    ``HFieldInvertedTruncatedPyramidTerrainGeneration`` at
    ``env_curriculum_coeff == 1.0``. Values above 1.0 linearly increase the
    frustum height and therefore its slope, without changing the environment
    curriculum used by rewards, commands, domain randomization or termination.
    """

    terrain_label = "inverted truncated pyramid (unbounded)"
    uses_unbounded_curriculum = True

    def __init__(self, env):
        super().__init__(env)

        terrain_config = self.env.env_config["terrain"]
        self.initial_difficulty = terrain_config.get(
            "curriculum_initial_difficulty",
            0.0,
        )
        self.eval_difficulty = terrain_config.get(
            "curriculum_eval_difficulty",
            1.0,
        )
        self.curriculum_step_scale = terrain_config.get(
            "curriculum_step_scale",
            1.0,
        )

        if (
            not np.isfinite(self.initial_difficulty)
            or self.initial_difficulty < 0.0
        ):
            raise ValueError(
                "terrain.curriculum_initial_difficulty must be finite and "
                "non-negative."
            )
        if (
            not np.isfinite(self.eval_difficulty)
            or self.eval_difficulty < 0.0
        ):
            raise ValueError(
                "terrain.curriculum_eval_difficulty must be finite and "
                "non-negative."
            )
        if (
            not np.isfinite(self.curriculum_step_scale)
            or self.curriculum_step_scale <= 0.0
        ):
            raise ValueError(
                "terrain.curriculum_step_scale must be finite and positive."
            )

    @property
    def slope_angle_state_key(self):
        return f"terrain/{self.config_prefix}_slope_angle_deg"

    def init(self, internal_state):
        super().init(internal_state)

        difficulty = jnp.where(
            internal_state["in_eval_mode"],
            self.eval_difficulty,
            self.initial_difficulty,
        )
        internal_state["terrain_curriculum_coeff"] = difficulty
        internal_state["terrain_curriculum_applied_coeff"] = difficulty
        internal_state[self.slope_angle_state_key] = jnp.asarray(0.0)

    def update_curriculum(self, internal_state, curriculum_delta):
        next_difficulty = jnp.maximum(
            internal_state["terrain_curriculum_coeff"]
            + self.curriculum_step_scale * curriculum_delta,
            0.0,
        )
        internal_state["terrain_curriculum_coeff"] = jnp.where(
            internal_state["in_eval_mode"],
            self.eval_difficulty,
            next_difficulty,
        )

    def add_curriculum_info(self, internal_state, info):
        info["terrain_curriculum/applied_difficulty"] = internal_state[
            "terrain_curriculum_applied_coeff"
        ]
        info["terrain_curriculum/next_difficulty"] = internal_state[
            "terrain_curriculum_coeff"
        ]
        info["terrain_curriculum/applied_height_m"] = internal_state[
            self.height_state_key
        ]
        info["terrain_curriculum/slope_angle_deg"] = internal_state[
            self.slope_angle_state_key
        ]

    def sample(self, mjx_model, internal_state, key):
        difficulty = jnp.maximum(
            internal_state["terrain_curriculum_coeff"],
            0.0,
        )
        internal_state["terrain_curriculum_applied_coeff"] = difficulty

        height_field = self.pyramid_terrain(difficulty)
        new_height_field_data = self.isaac_hf_to_mujoco_hf(height_field)
        new_mjx_model = mjx_model.replace(
            hfield_data=new_height_field_data,
        )

        center_idx = (
            self.hfield_half_length * self.hfield_length
            + self.hfield_half_length
        )
        internal_state["center_height"] = (
            new_height_field_data[center_idx] * self.mujoco_height_scaling
        )
        internal_state["current_height_field_data"] = (
            new_height_field_data.reshape(
                self.hfield_length,
                self.hfield_length,
            )
        )

        applied_height_m = difficulty * self.height_m
        slope_run_m = self.outer_half_width_m - self.center_half_width_m
        slope_angle_deg = jnp.degrees(
            jnp.arctan2(applied_height_m, slope_run_m)
        )
        internal_state[self.height_state_key] = applied_height_m
        internal_state[self.max_height_state_key] = jnp.max(height_field)
        internal_state[self.slope_angle_state_key] = slope_angle_deg

        return new_mjx_model
