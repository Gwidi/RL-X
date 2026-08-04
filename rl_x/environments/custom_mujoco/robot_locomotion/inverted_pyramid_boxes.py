import math


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
            0.65,
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


def add_inverted_pyramid_box_geoms(xml_handle, terrain_config):
    """Allocates a fixed set of boxes whose dimensions are sampled at reset."""
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

    collision_geom_names = [
        geom.name
        for geom in xml_handle.find_all("geom")
        if geom.name
        and (
            "foot" in geom.name
            or geom.name.endswith("_calf")
        )
    ]

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
                rgba="0.38 0.42 0.46 1",
            )
            geom_names.append(name)
            for collision_geom_name in collision_geom_names:
                xml_handle.contact.add(
                    "pair",
                    geom1=collision_geom_name,
                    geom2=name,
                )

    return tuple(geom_names)
