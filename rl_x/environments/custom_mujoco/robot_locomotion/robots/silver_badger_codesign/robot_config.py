SPINE_DESIGN_PARAMETER_NAMES = (
    "armature",
    "frictionloss",
    "ctrlrange_lower",
    "ctrlrange_upper",
    "maxtorque",
    "maxvelocity",
    "positionpercentage",
    "rotationaxisx",
    "rotationaxisy",
    "rotationaxisz",
)


robot_config = {
    "short_name": "sbg_codesign",
    "spine_locked": False,
    "actuator_joint_max_velocities": [
        3.15,
        25.0, 25.0, 25.0,
        25.0, 25.0, 25.0,
        25.0, 25.0, 25.0,
        25.0, 25.0, 25.0,
    ],
    "scaling_factor": 0.25,
    "actuator_joints_to_stay_near_nominal": [],
    "spine_design": {
        "parameter_names": SPINE_DESIGN_PARAMETER_NAMES,
        # The fixed Silver Badger spine from plane.xml. The position percentage
        # maps back to the original rear-body position x=-0.141 m.
        "default": (
            0.013122, 0.48, -1.57, 1.57, 48.0, 3.15,
            0.19572953736654805, 1.0, 0.0, 0.0,
        ),
        # Full sampling range used by loco_mjx/co_design.
        "randomization_min": (
            0.0, 0.0, -3.14, 0.0, 0.0, 0.0,
            0.0, -0.99999, -1.0, -1.0,
        ),
        "randomization_max": (
            0.026244, 0.48, 0.0, 3.14, 96.0, 6.3,
            1.0, 1.0, 1.0, 1.0,
        ),
        "physical_min": (
            0.0, 0.0, -3.14, -3.14, 0.0, 0.0,
            0.0, -1.0, -1.0, -1.0,
        ),
        "physical_max": (
            None, None, 3.14, 3.14, None, None,
            1.0, 1.0, 1.0, 1.0,
        ),
    },
    # These primitives are resized when the spine position changes.
    "training_geom_names_to_keep": (
        "trunk_1", "trunk_2", "trunk_3", "trunk_4",
        "rear_1", "rear_2", "rear_3", "rear_4",
    ),
}
