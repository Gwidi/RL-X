import numpy as np

from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.terrain_functions.hfield_inverted_truncated_pyramid import (
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

    def init(self):
        super().init()

        difficulty = (
            self.eval_difficulty
            if self.uses_fixed_difficulty()
            else self.initial_difficulty
        )
        self.env.internal_state["terrain_curriculum_coeff"] = difficulty
        self.env.internal_state["terrain_curriculum_applied_coeff"] = difficulty
        self.env.internal_state[self.slope_angle_state_key] = 0.0

    def uses_fixed_difficulty(self):
        return (
            self.env.internal_state["in_eval_mode"]
            or self.env.runner_mode == "test"
        )

    def get_difficulty(self):
        if self.uses_fixed_difficulty():
            return self.eval_difficulty
        return self.env.internal_state["terrain_curriculum_coeff"]

    def update_curriculum(self, curriculum_delta):
        if self.uses_fixed_difficulty():
            self.env.internal_state["terrain_curriculum_coeff"] = (
                self.eval_difficulty
            )
            return

        next_difficulty = max(
            self.env.internal_state["terrain_curriculum_coeff"]
            + self.curriculum_step_scale * curriculum_delta,
            0.0,
        )
        self.env.internal_state["terrain_curriculum_coeff"] = next_difficulty

    def add_curriculum_info(self):
        info = self.env.internal_state["info"]
        info["terrain_curriculum/applied_difficulty"] = self.env.internal_state[
            "terrain_curriculum_applied_coeff"
        ]
        info["terrain_curriculum/next_difficulty"] = self.env.internal_state[
            "terrain_curriculum_coeff"
        ]
        info["terrain_curriculum/applied_height_m"] = self.env.internal_state[
            self.height_state_key
        ]
        info["terrain_curriculum/slope_angle_deg"] = self.env.internal_state[
            self.slope_angle_state_key
        ]

    def sample(self):
        difficulty = max(self.get_difficulty(), 0.0)
        self.env.internal_state["terrain_curriculum_applied_coeff"] = difficulty

        height_field = self.pyramid_terrain(difficulty)
        new_height_field_data = self.isaac_hf_to_mujoco_hf(height_field)
        self.env.internal_state["mj_model"].hfield_data = new_height_field_data

        center_idx = (
            self.hfield_half_length * self.hfield_length
            + self.hfield_half_length
        )
        self.env.internal_state["center_height"] = (
            new_height_field_data[center_idx] * self.mujoco_height_scaling
        )
        self.env.internal_state["current_height_field_data"] = (
            new_height_field_data.reshape(
                self.hfield_length,
                self.hfield_length,
            )
        )

        applied_height_m = difficulty * self.height_m
        slope_run_m = self.outer_half_width_m - self.center_half_width_m
        slope_angle_deg = np.degrees(
            np.arctan2(applied_height_m, slope_run_m)
        )
        self.env.internal_state[self.height_state_key] = applied_height_m
        self.env.internal_state[self.max_height_state_key] = np.max(height_field)
        self.env.internal_state[self.slope_angle_state_key] = slope_angle_deg

    def get_debug_overlay(self):
        overlay = super().get_debug_overlay()
        overlay.extend(
            [
                (
                    "Difficulty",
                    f'{self.env.internal_state["terrain_curriculum_applied_coeff"]:.3f}',
                ),
                (
                    "Slope angle",
                    f'{self.env.internal_state[self.slope_angle_state_key]:.2f} deg',
                ),
            ]
        )
        return overlay
