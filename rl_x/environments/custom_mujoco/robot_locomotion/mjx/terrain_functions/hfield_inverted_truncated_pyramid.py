from rl_x.environments.custom_mujoco.robot_locomotion.mjx.terrain_functions.hfield_truncated_pyramid import (
    HFieldTruncatedPyramidTerrainGeneration,
)


class HFieldInvertedTruncatedPyramidTerrainGeneration(
    HFieldTruncatedPyramidTerrainGeneration
):
    """Inverted square frustum with a flat center basin."""

    config_prefix = "inverted_truncated_pyramid"
    terrain_label = "inverted truncated pyramid"
    inverted = True
