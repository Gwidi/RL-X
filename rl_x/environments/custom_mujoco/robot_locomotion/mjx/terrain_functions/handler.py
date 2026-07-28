from rl_x.environments.custom_mujoco.robot_locomotion.mjx.terrain_functions.plane import PlaneTerrainGeneration
from rl_x.environments.custom_mujoco.robot_locomotion.mjx.terrain_functions.hfield_diverse import HFieldDiverseTerrainGeneration
from rl_x.environments.custom_mujoco.robot_locomotion.mjx.terrain_functions.hfield_curbs import HFieldCurbsTerrainGeneration
from rl_x.environments.custom_mujoco.robot_locomotion.mjx.terrain_functions.hfield_bunker_ruins import HFieldBunkerRuinsTerrainGeneration
from rl_x.environments.custom_mujoco.robot_locomotion.mjx.terrain_functions.hfield_bunker_ruins_unbounded import HFieldBunkerRuinsUnboundedTerrainGeneration
from rl_x.environments.custom_mujoco.robot_locomotion.mjx.terrain_functions.hfield_bunker_stairs import HFieldBunkerStairsTerrainGeneration
from rl_x.environments.custom_mujoco.robot_locomotion.mjx.terrain_functions.hfield_inverted_pyramid_stairs import HFieldInvertedPyramidStairsTerrainGeneration


def get_terrain_function(name, env, **kwargs):
    if name == "plane":
        return PlaneTerrainGeneration(env, **kwargs)
    elif name == "hfield_diverse":
        return HFieldDiverseTerrainGeneration(env, **kwargs)
    elif name == "hfield_curbs":
        return HFieldCurbsTerrainGeneration(env, **kwargs)
    elif name == "hfield_bunker_ruins":
        return HFieldBunkerRuinsTerrainGeneration(env, **kwargs)
    elif name == "hfield_bunker_ruins_unbounded":
        return HFieldBunkerRuinsUnboundedTerrainGeneration(env, **kwargs)
    elif name == "hfield_bunker_stairs":
        return HFieldBunkerStairsTerrainGeneration(env, **kwargs)
    elif name == "hfield_inverted_pyramid_stairs":
        return HFieldInvertedPyramidStairsTerrainGeneration(env, **kwargs)
    else:
        raise NotImplementedError
