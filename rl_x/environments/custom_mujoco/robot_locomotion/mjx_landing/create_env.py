import importlib
from pathlib import Path

from rl_x.environments.custom_mujoco.robot_locomotion.mjx_landing.environment import LandingMjxEnv
from rl_x.environments.custom_mujoco.robot_locomotion.mjx_landing.general_properties import GeneralProperties


def create_train_and_eval_env(config):
    robot_config = dict(importlib.import_module(
        f"rl_x.environments.custom_mujoco.robot_locomotion.robots.{config.environment.train_robot}.robot_config").robot_config)
    robot_config["directory_path"] = Path(__file__).parent.parent / "robots" / config.environment.train_robot
    robot_config["spine_locked"] = config.environment.spine_locked
    env = LandingMjxEnv(robot_config, config.environment.to_dict())
    env.general_properties = GeneralProperties
    # Evaluation is stateless: it only needs the same compiled model.
    return env, env
