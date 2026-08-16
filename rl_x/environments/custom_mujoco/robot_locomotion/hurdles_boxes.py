import math

from rl_x.environments.custom_mujoco.robot_locomotion.inverted_pyramid_boxes import (
    config_value,
)


TERRAIN_TYPE = "hfield_hurdles"
TERRAIN_GEOM_PREFIX = "hurdle_wall_"
DEFAULT_HALF_WIDTH_M = 4.0

THIGH_COLLIDER_FROMTO = {
    "RL_thigh": (0.053, 0.0, 0.0, 0.053, 0.2, 0.0),
    "RR_thigh": (0.053, 0.0, 0.0, 0.053, -0.2, 0.0),
    "FR_thigh": (0.053, 0.0, 0.0, 0.053, -0.2, 0.0),
    "FL_thigh": (0.053, 0.0, 0.0, 0.053, 0.2, 0.0),
}


def number_of_walls(terrain_config):
    wall_count = config_value(
        terrain_config,
        "hurdles_wall_count",
        4,
    )
    if isinstance(wall_count, bool):
        raise ValueError(
            "terrain.hurdles_wall_count must be a positive integer."
        )
    try:
        wall_count_as_float = float(wall_count)
        wall_count_as_int = int(wall_count)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "terrain.hurdles_wall_count must be a positive integer."
        ) from exc
    if (
        not math.isfinite(wall_count_as_float)
        or wall_count_as_float != wall_count_as_int
        or wall_count_as_int <= 0
    ):
        raise ValueError(
            "terrain.hurdles_wall_count must be a positive integer."
        )
    return wall_count_as_int


def add_hurdle_box_geoms(xml_handle, terrain_config):
    """Allocates four boxes for each concentric square hurdle wall."""
    uses_calf_colliders = bool(
        config_value(terrain_config, "uses_calf_colliders", True)
    )
    uses_thigh_colliders = bool(
        config_value(terrain_config, "uses_thigh_colliders", False)
    )
    wall_count = number_of_walls(terrain_config)
    first_wall_distance_m = float(
        config_value(
            terrain_config,
            "hurdles_first_wall_distance_m",
            1.0,
        )
    )
    wall_spacing_m = float(
        config_value(
            terrain_config,
            "hurdles_wall_spacing_m",
            0.8,
        )
    )
    wall_thickness_m = float(
        config_value(
            terrain_config,
            "hurdles_wall_thickness_m",
            0.025,
        )
    )
    wall_height_m = float(
        config_value(
            terrain_config,
            "hurdles_wall_height_m",
            0.20,
        )
    )
    values = (
        first_wall_distance_m,
        wall_spacing_m,
        wall_thickness_m,
        wall_height_m,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Hurdle dimensions must be finite.")
    if first_wall_distance_m <= 0.0:
        raise ValueError(
            "terrain.hurdles_first_wall_distance_m must be positive."
        )
    if wall_spacing_m <= 0.0:
        raise ValueError(
            "terrain.hurdles_wall_spacing_m must be positive."
        )
    if wall_thickness_m <= 0.0:
        raise ValueError(
            "terrain.hurdles_wall_thickness_m must be positive."
        )
    if wall_height_m < 0.0:
        raise ValueError(
            "terrain.hurdles_wall_height_m must be non-negative."
        )

    outer_edge_m = (
        first_wall_distance_m
        + (wall_count - 1) * wall_spacing_m
        + wall_thickness_m / 2.0
    )
    max_half_height_m = max(0.001, wall_height_m / 2.0)
    broad_phase_center_z_m = max_half_height_m

    collision_geoms = [
        geom
        for geom in xml_handle.find_all("geom")
        if geom.name
        and (
            "foot" in geom.name
            or (
                uses_calf_colliders
                and geom.name.endswith("_calf")
            )
            or (
                uses_thigh_colliders
                and geom.name.endswith("_thigh")
            )
        )
    ]
    for geom in collision_geoms:
        if geom.name.endswith("_calf"):
            geom.type = "capsule"
            geom.size = (0.015, 0.085)
            geom.rgba = "0 1 0 1"  # green
        if geom.name.endswith("_thigh"):
            geom.type = "capsule"
            geom.size = (0.015, 0.085)
            thigh_fromto = THIGH_COLLIDER_FROMTO.get(geom.name)
            if thigh_fromto is not None:
                # Follow the orange thigh link from the hip axis to the
                # calf-joint attachment instead of retaining the cylinder's
                # original orientation.
                geom.pos = None
                geom.quat = None
                geom.fromto = thigh_fromto
            geom.rgba = "0 1 0 1"  # green
        # Use a terrain-specific collision bit so the selected limb
        # colliders collide with the boxes but not with the plane below.
        geom.conaffinity = 2

    geom_names = []
    for wall_idx in range(wall_count):
        for side in ("north", "south", "east", "west"):
            name = f"{TERRAIN_GEOM_PREFIX}{wall_idx}_{side}"
            xml_handle.worldbody.add(
                "geom",
                name=name,
                type="box",
                pos=f"0 0 {broad_phase_center_z_m}",
                size=(
                    f"{outer_edge_m} {outer_edge_m} "
                    f"{max_half_height_m}"
                ),
                contype="2",
                conaffinity="0",
                group="1",
                rgba="0.38 0.42 0.46 1",
            )
            geom_names.append(name)

    return tuple(geom_names)
