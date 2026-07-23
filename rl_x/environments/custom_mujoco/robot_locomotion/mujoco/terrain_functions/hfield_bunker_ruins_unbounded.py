from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.terrain_functions.hfield_bunker_ruins import (
    HFieldBunkerRuinsTerrainGeneration,
)


class HFieldBunkerRuinsUnboundedTerrainGeneration(HFieldBunkerRuinsTerrainGeneration):
    """Bunker ruins driven by a separate, non-negative unbounded curriculum.

    Difficulty 1.0 produces the same obstacle amplitudes as
    ``HFieldBunkerRuinsTerrainGeneration`` at ``env_curriculum_coeff == 1.0``.
    Values above 1.0 scale obstacle height, slab slope and roughness linearly,
    without affecting reward scaling, domain randomization or termination.
    """

    uses_unbounded_curriculum = True

    def __init__(self, env):
        super().__init__(env)

        terrain_config = self.env.env_config["terrain"]
        self.initial_difficulty = terrain_config.get("curriculum_initial_difficulty", 0.0)
        self.eval_difficulty = terrain_config.get("curriculum_eval_difficulty", 1.0)
        self.curriculum_step_scale = terrain_config.get("curriculum_step_scale", 1.0)

        if self.initial_difficulty < 0.0:
            raise ValueError("terrain.curriculum_initial_difficulty must be non-negative.")
        if self.eval_difficulty < 0.0:
            raise ValueError("terrain.curriculum_eval_difficulty must be non-negative.")
        if self.curriculum_step_scale <= 0.0:
            raise ValueError("terrain.curriculum_step_scale must be positive.")

    def init(self):
        super().init()

        difficulty = (
            self.eval_difficulty
            if self.env.internal_state["in_eval_mode"]
            else self.initial_difficulty
        )
        self.env.internal_state["terrain_curriculum_coeff"] = difficulty
        self.env.internal_state["terrain_curriculum_applied_coeff"] = difficulty
        self.env.internal_state["terrain_max_obstacle_height"] = 0.0
        self.env.internal_state["terrain_max_slope_height"] = 0.0
        self.env.internal_state["terrain_roughness_height"] = 0.0

    def update_curriculum(self, curriculum_delta):
        next_difficulty = max(
            self.env.internal_state["terrain_curriculum_coeff"]
            + self.curriculum_step_scale * curriculum_delta,
            0.0,
        )
        self.env.internal_state["terrain_curriculum_coeff"] = (
            self.eval_difficulty
            if self.env.internal_state["in_eval_mode"]
            else next_difficulty
        )

    def add_curriculum_info(self):
        info = self.env.internal_state["info"]
        info["terrain_curriculum/applied_difficulty"] = self.env.internal_state[
            "terrain_curriculum_applied_coeff"
        ]
        info["terrain_curriculum/next_difficulty"] = self.env.internal_state[
            "terrain_curriculum_coeff"
        ]
        info["terrain_curriculum/max_obstacle_height"] = self.env.internal_state[
            "terrain_max_obstacle_height"
        ]
        info["terrain_curriculum/max_slope_height"] = self.env.internal_state[
            "terrain_max_slope_height"
        ]
        info["terrain_curriculum/roughness_height"] = self.env.internal_state[
            "terrain_roughness_height"
        ]

    def sample(self):
        difficulty = max(self.env.internal_state["terrain_curriculum_coeff"], 0.0)
        self.env.internal_state["terrain_curriculum_applied_coeff"] = difficulty

        max_obstacle_height = (
            difficulty
            * self.env.internal_state["robot_dimensions_mean"]
            * self.block_height_max_per_m_factor
        )
        max_slope_height = (
            difficulty
            * self.env.internal_state["robot_dimensions_mean"]
            * self.block_slope_height_max_per_m_factor
        )
        noise_height = difficulty * self.rng().uniform(
            low=0.0,
            high=(
                self.env.internal_state["robot_dimensions_mean"]
                * self.random_height_max_per_m_factor
            ),
        )

        self.env.internal_state["terrain_max_obstacle_height"] = max_obstacle_height
        self.env.internal_state["terrain_max_slope_height"] = max_slope_height
        self.env.internal_state["terrain_roughness_height"] = noise_height

        isaac_height_field = self.bunker_ruins_terrain(
            max_obstacle_height=max_obstacle_height,
            max_slope_height=max_slope_height,
            noise_height=noise_height,
        )

        new_height_field_data = self.isaac_hf_to_mujoco_hf(isaac_height_field)
        self.env.internal_state["mj_model"].hfield_data = new_height_field_data

        self.env.internal_state["center_height"] = (
            new_height_field_data[
                self.hfield_half_length * self.hfield_length
                + self.hfield_half_length
            ]
            * self.mujoco_height_scaling
        )
        self.env.internal_state["current_height_field_data"] = (
            new_height_field_data.reshape(self.hfield_length, self.hfield_length)
        )
