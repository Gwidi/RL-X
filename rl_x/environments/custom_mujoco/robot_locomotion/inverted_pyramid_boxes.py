import math

import numpy as np


TERRAIN_TYPE = "hfield_inverted_pyramid_stairs"
TERRAIN_GEOM_PREFIX = "inverted_pyramid_step_"
DEFAULT_HALF_WIDTH_M = 4.0


def config_value(config, name, default):
    try:
        return config.get(name, default)
    except AttributeError:
        try:
            return config[name]
        except (KeyError, TypeError):
            return default


def maximum_number_of_steps(terrain_config):
    half_width_m = float(
        config_value(
            terrain_config,
            "inverted_pyramid_half_width_m",
            DEFAULT_HALF_WIDTH_M,
        )
    )
    center_half_width_m = float(
        config_value(
            terrain_config,
            "inverted_pyramid_center_half_width_m",
            0.8,
        )
    )
    tread_depth_m = float(
        config_value(
            terrain_config,
            "inverted_pyramid_tread_depth_m",
            0.4,
        )
    )
    randomize_tread_depth = bool(
        config_value(
            terrain_config,
            "inverted_pyramid_randomize_tread_depth",
            True,
        )
    )
    min_scale = float(
        config_value(
            terrain_config,
            "inverted_pyramid_tread_depth_scale_min",
            0.85,
        )
    )

    # Curriculum interpolation starts at scale 1.0, which can be smaller
    # than a user-provided scale_min greater than one.
    minimum_tread_depth_m = (
        tread_depth_m * min(1.0, min_scale)
        if randomize_tread_depth
        else tread_depth_m
    )
    if half_width_m <= 0.0:
        raise ValueError(
            "terrain.inverted_pyramid_half_width_m must be positive."
        )
    if not 0.0 <= center_half_width_m < half_width_m:
        raise ValueError(
            "terrain.inverted_pyramid_center_half_width_m must be "
            "non-negative and smaller than inverted_pyramid_half_width_m."
        )
    if minimum_tread_depth_m <= 0.0:
        raise ValueError(
            "The minimum inverted-pyramid tread depth must be positive."
        )

    return int(
        math.ceil(
            (half_width_m - center_half_width_m)
            / minimum_tread_depth_m
        )
    )


def update_mujoco_static_box_bvh(model, geom_ids):
    """Update MuJoCo's static-world BVH after moving axis-aligned boxes.

    MuJoCo compiles world-body BVH nodes from the initial geom positions.
    Changing ``geom_pos`` and ``geom_size`` at reset does not rebuild those
    nodes, so mask-filtered contacts can be rejected during broad phase.
    """
    geom_ids = np.asarray(geom_ids, dtype=np.int32)
    if geom_ids.size == 0:
        return

    geom_id_set = set(int(geom_id) for geom_id in geom_ids)
    world_bvh_adr = int(model.body_bvhadr[0])
    world_bvh_num = int(model.body_bvhnum[0])
    if world_bvh_adr < 0 or world_bvh_num == 0:
        raise RuntimeError("The MuJoCo model has no static-world BVH.")
    world_bvh_end = world_bvh_adr + world_bvh_num

    updated_geom_ids = set()
    for node_id in range(world_bvh_adr, world_bvh_end):
        geom_id = int(model.bvh_nodeid[node_id])
        if geom_id not in geom_id_set:
            continue
        # The generated terrain boxes have identity orientation, so their
        # local and world-aligned half-extents are both geom_size.
        model.geom_aabb[geom_id, :3] = 0.0
        model.geom_aabb[geom_id, 3:] = model.geom_size[geom_id]
        model.bvh_aabb[node_id, :3] = model.geom_pos[geom_id]
        model.bvh_aabb[node_id, 3:] = model.geom_size[geom_id]
        updated_geom_ids.add(geom_id)

    if updated_geom_ids != geom_id_set:
        missing_ids = sorted(geom_id_set - updated_geom_ids)
        raise RuntimeError(
            "Could not find terrain geom BVH nodes for IDs "
            f"{missing_ids}."
        )

    # Children have greater indices than their parents in MuJoCo's BVH, so a
    # reverse traversal propagates every changed leaf to the root.
    for node_id in range(world_bvh_end - 1, world_bvh_adr - 1, -1):
        child_a, child_b = model.bvh_child[node_id]
        child_a = int(child_a)
        child_b = int(child_b)
        if child_a < 0:
            continue
        lower = np.minimum(
            model.bvh_aabb[child_a, :3] - model.bvh_aabb[child_a, 3:],
            model.bvh_aabb[child_b, :3] - model.bvh_aabb[child_b, 3:],
        )
        upper = np.maximum(
            model.bvh_aabb[child_a, :3] + model.bvh_aabb[child_a, 3:],
            model.bvh_aabb[child_b, :3] + model.bvh_aabb[child_b, 3:],
        )
        model.bvh_aabb[node_id, :3] = (lower + upper) / 2.0
        model.bvh_aabb[node_id, 3:] = (upper - lower) / 2.0


def add_inverted_pyramid_box_geoms(xml_handle, terrain_config):
    """Allocates a fixed set of boxes whose dimensions are sampled at reset."""
    uses_calf_colliders = bool(
        config_value(terrain_config, "uses_calf_colliders", True)
    )
    nr_steps = maximum_number_of_steps(terrain_config)
    half_width_m = float(
        config_value(
            terrain_config,
            "inverted_pyramid_half_width_m",
            DEFAULT_HALF_WIDTH_M,
        )
    )
    step_height_m = float(
        config_value(
            terrain_config,
            "inverted_pyramid_step_height_m",
            0.15,
        )
    )
    if step_height_m < 0.0:
        raise ValueError(
            "terrain.inverted_pyramid_step_height_m must be non-negative."
        )

    # Compile with an upper bound on every box size. MJX uses the compiled
    # bounding radius during broad phase collision detection, so shrinking
    # these geoms later is safe while compiling them as tiny boxes is not.
    max_half_height_m = max(0.001, nr_steps * step_height_m / 2.0)
    hidden_z_m = -(2.0 * max_half_height_m + 1.0)
    foot_geoms = [
        geom
        for geom in xml_handle.find_all("geom")
        if geom.name and "foot" in geom.name
    ]
    calf_geoms = [
        geom
        for geom in xml_handle.find_all("geom")
        if uses_calf_colliders
        and geom.name
        and geom.name.endswith("_calf")
    ]

    # Bit 1 is reserved for the floor.  Use a separate bit for the boxes so
    # contacts are generated from geom properties rather than explicit pairs;
    # this lets runtime friction/solref/solimp randomization reach the contact.
    for geom in foot_geoms:
        geom.conaffinity = 2

    for geom in calf_geoms:
        geom.type = "capsule"
        geom.size = (0.015, 0.06)
        geom.conaffinity = 2
        # Calves only use collision bit 2, so these parameters affect their
        # contacts with the terrain boxes without changing foot-floor contact.
        # The higher priority is important: the boxes inherit the root geom's
        # direct-format solref (-1000, -80), which would otherwise be mixed
        # with this positive time-constant format.
        geom.condim = 3
        geom.priority = 1
        geom.solref = (0.03, 1.0)
        geom.solimp = (0.8, 0.95, 0.003, 0.5, 2.0)
        geom.friction = (0.5, 0.005, 0.0001)
        geom.rgba = [0.0, 1.0, 0.0, 1.0]  # green

    geom_names = []
    for step_idx in range(nr_steps):
        for side in ("north", "south", "east", "west"):
            name = f"{TERRAIN_GEOM_PREFIX}{step_idx}_{side}"
            xml_handle.worldbody.add(
                "geom",
                name=name,
                type="box",
                pos=f"0 0 {hidden_z_m}",
                size=(
                    f"{half_width_m} {half_width_m} "
                    f"{max_half_height_m}"
                ),
                group="1",
                contype="2",
                conaffinity="0",
                rgba="0.38 0.42 0.46 1",
                friction="1.0 0.005 0.0001",
            )
            geom_names.append(name)

    return tuple(geom_names)
