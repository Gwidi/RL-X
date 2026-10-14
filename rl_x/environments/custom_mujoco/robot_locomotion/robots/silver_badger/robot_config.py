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

    # The 3 m landing pose from search/replay_target.py needs up to 0.576 rad
    # of offset from home. Keep it reachable with normalized [-1, 1] actions.
    "scaling_factor": 0.6,

    "actuator_joints_to_stay_near_nominal": []
}
