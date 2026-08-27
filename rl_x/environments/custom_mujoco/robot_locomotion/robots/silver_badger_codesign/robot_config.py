SPINE_DESIGN_PARAMETER_NAMES = (
    "armature",
    "frictionloss",
    "ctrlrange_lower",
    "ctrlrange_upper",
    "maxtorque",
    "maxvelocity",
    "positionpercentage",
    "axis_tangent_y",
    "axis_tangent_z",
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
        # The hinge axis is exp([0, tangent_y, tangent_z]) applied to +x.
        # Restricting the tangent-vector norm to pi/2 gives one representative
        # of every unoriented hinge line (apart from the measure-zero equator).
        "axis_max_tilt_rad": 1.5707963267948966,
        # The fixed Silver Badger spine from plane.xml. The position percentage
        # maps back to the original rear-body position x=-0.141 m.
        "default": (
            0.013122, 0.48, -1.57, 1.57, 48.0, 3.15,
            0.19572953736654805, 0.0, 0.0,
        ),
        # Scalar bounds. The final two values additionally obey the circular
        # tangent-plane constraint defined by axis_max_tilt_rad.
        "randomization_min": (
            0.0, 0.0, -3.14, 0.0, 0.0, 0.0,
            0.0, -1.5707963267948966, -1.5707963267948966,
        ),
        "randomization_max": (
            0.026244, 0.48, 0.0, 3.14, 96.0, 6.3,
            1.0, 1.5707963267948966, 1.5707963267948966,
        ),
        "physical_min": (
            0.0, 0.0, -3.14, -3.14, 0.0, 0.0,
            0.0, -1.5707963267948966, -1.5707963267948966,
        ),
        "physical_max": (
            None, None, 3.14, 3.14, None, None,
            1.0, 1.5707963267948966, 1.5707963267948966,
        ),
    },
    # These primitives are resized when the spine position changes.
    "training_geom_names_to_keep": (
        "trunk_1", "trunk_2", "trunk_3", "trunk_4",
        "rear_1", "rear_2", "rear_3", "rear_4",
    ),
}
