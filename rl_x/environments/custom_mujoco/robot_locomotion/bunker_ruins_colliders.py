BUNKER_RUINS_TERRAIN_TYPES = frozenset(
    {
        "hfield_bunker_ruins",
        "hfield_bunker_ruins_unbounded",
    }
)

CALF_FLOOR_COLLIDER_TERRAIN_TYPES = (
    BUNKER_RUINS_TERRAIN_TYPES | {"plane"}
)

THIGH_FLOOR_COLLIDER_TERRAIN_TYPES = BUNKER_RUINS_TERRAIN_TYPES


def add_calf_floor_colliders(xml_handle):
    """Enable calf collisions with the floor geom."""
    calf_geoms = [
        geom
        for geom in xml_handle.find_all("geom")
        if geom.name and geom.name.endswith("_calf")
    ]

    for geom in calf_geoms:
        # Match the collider used by the inverted-pyramid box terrain.
        geom.type = "capsule"
        geom.size = (0.015, 0.06)
        xml_handle.contact.add(
            "pair",
            geom1=geom.name,
            geom2="floor",
        )

    return tuple(geom.name for geom in calf_geoms)


def add_thigh_floor_colliders(xml_handle):
    """Enable thigh collisions with the floor geom."""
    thigh_geoms = [
        geom
        for geom in xml_handle.find_all("geom")
        if geom.name and geom.name.endswith("_thigh")
    ]

    for geom in thigh_geoms:
        # MJX does not implement heightfield-to-cylinder contacts. Preserve
        # the cylinder's outer dimensions while representing it as a capsule.
        geom.type = "capsule"
        geom.size = (0.05, 0.005)
        xml_handle.contact.add(
            "pair",
            geom1=geom.name,
            geom2="floor",
        )

    return tuple(geom.name for geom in thigh_geoms)


def add_bunker_ruins_calf_colliders(xml_handle):
    """Backward-compatible alias for calf-to-floor colliders."""
    return add_calf_floor_colliders(xml_handle)
