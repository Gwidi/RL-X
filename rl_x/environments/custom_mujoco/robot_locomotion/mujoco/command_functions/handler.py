from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.command_functions.random import RandomCommands
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.command_functions.forward import ForwardCommands


def get_command_function(name, env, **kwargs):
    if name == "random":
        return RandomCommands(env, **kwargs)
    elif name == "forward":
        return ForwardCommands(env, **kwargs)
    else:
        raise NotImplementedError
