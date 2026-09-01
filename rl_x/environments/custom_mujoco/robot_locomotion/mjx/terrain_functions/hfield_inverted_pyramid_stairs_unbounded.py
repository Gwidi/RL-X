import jax.numpy as jnp
import numpy as np

from rl_x.environments.custom_mujoco.robot_locomotion.mjx.terrain_functions.hfield_inverted_pyramid_stairs import (
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

    def sample_tread_depths(self, curriculum_coeff, key):
        difficulty_scale = jnp.maximum(
            self.tread_depth_difficulty_min_scale,
            1.0
            - self.tread_depth_difficulty_decay * curriculum_coeff,
        )
        return difficulty_scale * super().sample_tread_depths(
            curriculum_coeff,
            key,
        )
