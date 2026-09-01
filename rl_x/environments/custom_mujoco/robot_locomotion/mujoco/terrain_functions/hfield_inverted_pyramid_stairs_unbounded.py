import numpy as np

from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.terrain_functions.hfield_inverted_pyramid_stairs import (
    HFieldInvertedPyramidStairsTerrainGeneration,
)


class HFieldInvertedPyramidStairsUnboundedTerrainGeneration(
    HFieldInvertedPyramidStairsTerrainGeneration
):
    """Box stairs driven by a separate, non-negative unbounded curriculum.

    Difficulty linearly increases step heights and reduces mean tread depth
    down to a configurable lower scale. Tread-depth randomization reaches its
    fully configured range at difficulty 1.0. This curriculum does not affect
    rewards, commands, domain randomization, or termination.
    """

    uses_terrain_curriculum = False
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
        self.tread_depth_difficulty_decay = terrain_config.get(
            "inverted_pyramid_tread_depth_difficulty_decay",
            0.1,
        )
        self.tread_depth_difficulty_min_scale = terrain_config.get(
            "inverted_pyramid_tread_depth_difficulty_min_scale",
            0.6,
        )
        # Selecting the unbounded terrain explicitly enables its independent
        # curriculum; the bounded terrain's opt-out flag does not apply here.
        self.use_curriculum = True

        for name, value in (
            ("curriculum_initial_difficulty", self.initial_difficulty),
            ("curriculum_eval_difficulty", self.eval_difficulty),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"terrain.{name} must be finite and non-negative."
                )
        if (
            not np.isfinite(self.curriculum_step_scale)
            or self.curriculum_step_scale <= 0.0
        ):
            raise ValueError(
                "terrain.curriculum_step_scale must be finite and positive."
            )
        if (
            not np.isfinite(self.tread_depth_difficulty_decay)
            or self.tread_depth_difficulty_decay < 0.0
        ):
            raise ValueError(
                "terrain.inverted_pyramid_tread_depth_difficulty_decay must "
                "be finite and non-negative."
            )
        if (
            not np.isfinite(self.tread_depth_difficulty_min_scale)
            or not 0.0 < self.tread_depth_difficulty_min_scale <= 1.0
        ):
            raise ValueError(
                "terrain.inverted_pyramid_tread_depth_difficulty_min_scale "
                "must be finite and between 0.0 (exclusive) and 1.0."
            )

    def update_curriculum(self, curriculum_delta):
        if self.uses_fixed_difficulty():
            self.env.internal_state["terrain_curriculum_coeff"] = (
                self.eval_difficulty
            )
            return

        self.env.internal_state["terrain_curriculum_coeff"] = max(
            self.env.internal_state["terrain_curriculum_coeff"]
            + self.curriculum_step_scale * curriculum_delta,
            0.0,
        )

    def sample_tread_depths(self, curriculum_coeff):
        difficulty_scale = max(
            self.tread_depth_difficulty_min_scale,
            1.0
            - self.tread_depth_difficulty_decay * float(curriculum_coeff),
        )
        return difficulty_scale * super().sample_tread_depths(
            curriculum_coeff
        )

    def get_debug_overlay(self):
        overlay = super().get_debug_overlay()
        overlay[0] = ("Terrain", "inverted pyramid stairs (unbounded)")
        overlay.append(
            (
                "Difficulty",
                f'{self.env.internal_state["terrain_curriculum_applied_coeff"]:.3f}',
            )
        )
        return overlay
