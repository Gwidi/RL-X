import jax.numpy as jnp
import numpy as np

from rl_x.environments.custom_mujoco.robot_locomotion.mjx.terrain_functions.hfield_hurdles import (
    HFieldHurdlesTerrainGeneration,
)


class HFieldHurdlesUnboundedTerrainGeneration(
    HFieldHurdlesTerrainGeneration
):
    """Box hurdles driven by a separate non-negative unbounded curriculum.

    Difficulty 1.0 matches ``HFieldHurdlesTerrainGeneration`` at terrain
    curriculum 1.0. Larger values linearly increase hurdle height without
    affecting rewards, commands, domain randomization, or termination.
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
