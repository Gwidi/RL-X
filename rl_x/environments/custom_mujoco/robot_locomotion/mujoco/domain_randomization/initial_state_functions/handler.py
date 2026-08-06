from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.domain_randomization.initial_state_functions.default import DefaultDRInitialState
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.domain_randomization.initial_state_functions.random import RandomDRInitialState
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.domain_randomization.initial_state_functions.energy_dissipation_curriculum import EnergyDissipationCurriculumInitialState


def get_initial_state_function(name, env, **kwargs):
    if name == "default":
        return DefaultDRInitialState(env, **kwargs)
    elif name == "random":
        return RandomDRInitialState(env, **kwargs)
    elif name == "energy_dissipation_curriculum":
        return EnergyDissipationCurriculumInitialState(env, **kwargs)
    else:
        raise NotImplementedError
