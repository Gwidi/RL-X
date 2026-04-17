from rl_x.environments.custom_mujoco.robot_locomotion.mjx.terrain_functions.plane import PlaneTerrainGeneration
from rl_x.environments.custom_mujoco.robot_locomotion.mjx.terrain_functions.hfield_diverse import HFieldDiverseTerrainGeneration
from rl_x.environments.custom_mujoco.robot_locomotion.mjx.terrain_functions.hfield_curbs import HFieldCurbsTerrainGeneration
from rl_x.environments.custom_mujoco.robot_locomotion.mjx.terrain_functions.hfield_bunker_ruins import HFieldBunkerRuinsTerrainGeneration


def get_terrain_function(name, env, **kwargs):
    if name == "plane":
        return PlaneTerrainGeneration(env, **kwargs)
    elif name == "hfield_diverse":
        return HFieldDiverseTerrainGeneration(env, **kwargs)
    elif name == "hfield_curbs":
        return HFieldCurbsTerrainGeneration(env, **kwargs)
    elif name == "hfield_bunker_ruins":
        return HFieldBunkerRuinsTerrainGeneration(env, **kwargs)
    else:
        raise NotImplementedError
