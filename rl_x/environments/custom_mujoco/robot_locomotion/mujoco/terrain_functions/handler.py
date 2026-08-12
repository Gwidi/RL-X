from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.terrain_functions.plane import PlaneTerrainGeneration
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.terrain_functions.hfield_diverse import HFieldDiverseTerrainGeneration
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.terrain_functions.hfield_curbs import HFieldCurbsTerrainGeneration
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.terrain_functions.hfield_curb_course import HFieldCurbCourseTerrainGeneration
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.terrain_functions.hfield_bunker_ruins import HFieldBunkerRuinsTerrainGeneration
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.terrain_functions.hfield_bunker_ruins_unbounded import HFieldBunkerRuinsUnboundedTerrainGeneration
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.terrain_functions.hfield_bunker_stairs import HFieldBunkerStairsTerrainGeneration
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.terrain_functions.hfield_inverted_pyramid_stairs import HFieldInvertedPyramidStairsTerrainGeneration
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.terrain_functions.hfield_bowl import HFieldBowlTerrainGeneration
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.terrain_functions.hfield_truncated_pyramid import HFieldTruncatedPyramidTerrainGeneration
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.terrain_functions.hfield_inverted_truncated_pyramid import HFieldInvertedTruncatedPyramidTerrainGeneration
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.terrain_functions.hfield_inverted_truncated_pyramid_unbounded import HFieldInvertedTruncatedPyramidUnboundedTerrainGeneration
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.terrain_functions.hfield_hurdles import HFieldHurdlesTerrainGeneration

def get_terrain_function(name, env, **kwargs):
    if name == "plane":
        return PlaneTerrainGeneration(env, **kwargs)
    elif name == "hfield_diverse":
        return HFieldDiverseTerrainGeneration(env, **kwargs)
    elif name == "hfield_curbs":
        return HFieldCurbsTerrainGeneration(env, **kwargs)
    elif name == "hfield_curb_course":
        return HFieldCurbCourseTerrainGeneration(env, **kwargs)
    elif name == "hfield_bunker_ruins":
        return HFieldBunkerRuinsTerrainGeneration(env, **kwargs)
    elif name == "hfield_bunker_ruins_unbounded":
        return HFieldBunkerRuinsUnboundedTerrainGeneration(env, **kwargs)
    elif name == "hfield_bunker_stairs":
        return HFieldBunkerStairsTerrainGeneration(env, **kwargs)
    elif name == "hfield_inverted_pyramid_stairs":
        return HFieldInvertedPyramidStairsTerrainGeneration(env, **kwargs)
    elif name == "hfield_bowl":
        return HFieldBowlTerrainGeneration(env, **kwargs)
    elif name == "hfield_truncated_pyramid":
        return HFieldTruncatedPyramidTerrainGeneration(env, **kwargs)
    elif name == "hfield_inverted_truncated_pyramid":
        return HFieldInvertedTruncatedPyramidTerrainGeneration(env, **kwargs)
    elif name == "hfield_inverted_truncated_pyramid_unbounded":
        return HFieldInvertedTruncatedPyramidUnboundedTerrainGeneration(env, **kwargs)
    elif name == "hfield_hurdles":
        return HFieldHurdlesTerrainGeneration(env, **kwargs)
    else:
        raise NotImplementedError
