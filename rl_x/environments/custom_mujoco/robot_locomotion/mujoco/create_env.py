import importlib
from pathlib import Path
import gymnasium as gym

from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.environment import LocomotionEnv
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.wrappers import RLXInfo, RecordEpisodeStatistics
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.async_vectorized_wrapper import AsyncVectorEnvWithSkipping
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.general_properties import GeneralProperties
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_evaluation import LandingEvaluationWrapper, evaluation_heights


def create_train_and_eval_env(config):
    paired_evaluation = config.environment.get("landing_evaluation", {}).get("enabled", False)
    if paired_evaluation:
        evaluation_heights(config.environment.landing_evaluation)
        if (config.environment.terrain.type != "plane"
                or config.environment.reward.type != "simplified_landing"
                or config.environment.domain_randomization.initial_state.type != "energy_dissipation_curriculum"):
            raise ValueError("Paired landing evaluation requires plane, simplified_landing and energy_dissipation_curriculum")
    robot_config = importlib.import_module(f"rl_x.environments.custom_mujoco.robot_locomotion.robots.{config.environment.train_robot}.robot_config").robot_config
    robot_config["directory_path"] = Path(__file__).parent.parent / "robots" / config.environment.train_robot
    robot_config["spine_locked"] = config.environment.spine_locked
    
    def make_env(seed, runner_mode=None, evaluation_index=None):
        def thunk():
            env = LocomotionEnv(
                robot_config=robot_config,
                runner_mode=config.runner.mode if runner_mode is None else runner_mode,
                seed=seed,
                render=config.environment.render,
                env_config=config.environment,
                nr_envs=config.environment.nr_envs,
            )
            if evaluation_index is not None:
                env = LandingEvaluationWrapper(env, evaluation_index)
            env = RecordEpisodeStatistics(env)
            env.action_space.seed(config.environment.seed)
            env.observation_space.seed(config.environment.seed)
            return env
        return thunk

    make_env_functions = [make_env(config.environment.seed + i) for i in range(config.environment.nr_envs)]
    
    if config.environment.nr_envs == 1:
        train_env = gym.vector.SyncVectorEnv(make_env_functions)
    else:
        train_env = AsyncVectorEnvWithSkipping(make_env_functions, config.environment.async_skip_percentage)
    train_env = RLXInfo(train_env)
    train_env.horizon = train_env.call("horizon")[0]
    train_env.general_properties = GeneralProperties

    if config.environment.copy_train_env_for_eval and not paired_evaluation:
        return train_env, train_env

    make_eval_env_functions = [
        make_env(config.environment.seed + i, runner_mode="test",
                 evaluation_index=i if paired_evaluation else None)
        for i in range(config.environment.nr_envs)
    ]
    if config.environment.nr_envs == 1:
        eval_env = gym.vector.SyncVectorEnv(make_eval_env_functions)
    else:
        eval_env = AsyncVectorEnvWithSkipping(make_eval_env_functions,
                                             0.0 if paired_evaluation else config.environment.async_skip_percentage)
    eval_env = RLXInfo(eval_env)
    eval_env.horizon = eval_env.call("horizon")[0]
    eval_env.general_properties = GeneralProperties

    return train_env, eval_env
