robot_config = {
    "short_name": "sbg",
    "spine_locked": True,

    "actuator_joint_max_velocities": [
        3.15,
        25.0, 25.0, 25.0,
        25.0, 25.0, 25.0,
        25.0, 25.0, 25.0,
        25.0, 25.0, 25.0
    ],

    # Policy actions are joint-position offsets around the XML home keyframe.
    "scaling_factor": 0.6,

    "actuator_joints_to_stay_near_nominal": []
}
