from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.reward_functions.default import DefaultReward
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.reward_functions.energy_dissipation_reward import EnergyDissipationReward
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.reward_functions.simplified_landing_reward import SimplifiedLandingReward


def get_reward_function(name, env, **kwargs):
    if name == "default":
        return DefaultReward(env, **kwargs)
    elif name == "energy_dissipation":
        return EnergyDissipationReward(env, **kwargs)
    elif name == "simplified_landing":
        return SimplifiedLandingReward(env, **kwargs)
    else:
        raise NotImplementedError
