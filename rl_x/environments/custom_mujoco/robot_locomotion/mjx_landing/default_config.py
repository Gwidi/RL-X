from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.default_config import get_config as get_cpu_config


CPU_ENVIRONMENT = "custom_mujoco.robot_locomotion.mujoco"


def get_config(environment_name):
    """The CPU environment's config, so checkpoints stay interchangeable."""
    config = get_cpu_config(environment_name)
    config.device = "gpu"
    config.nr_envs = 1024
    config.mjx = dict(
        reset_pool_size=4096,       # released states per rollout; reused only when exhausted
        contacts_per_env=16,        # average over all worlds of one contact buffer
        constraints_per_env=160,    # per world: 13 friction + 13 limits + 4 per contact
        # Imperfect release: roll and pitch ~ U(-tilt, tilt), body angular rate with a random
        # direction and magnitude ~ U(0, rate). Zero disables it. The magnitude ramps from 0 to
        # its maximum over ramp_steps training steps; evaluation always uses the maximum.
        release_disturbance=dict(max_tilt_deg=0., max_rate_dps=0., ramp_steps=0),
        # Release pose: home + f * (standing - home). f = scale for a share 1 - home_probability of the episodes and 0 for the
        # rest. stand_fraction 0 disables it. scale ramps from 0 to stand_fraction over ramp_steps training steps.
        release_pose=dict(stand_fraction=0., home_probability=.2, ramp_steps=0),
        # soft: thigh and calf touches are allowed while their normal force, averaged over 10 ms, stays below force_limit [N];
        # above it the episode ends like any forbidden contact. Hip links, trunk and rear body stay forbidden.
        limb_contact=dict(soft=False, force_limit=200.),
        # Braking term of the reward: track = follow the decaying reference speed vector, upper = penalise only speeds above
        # the reference speed (braking faster is free, direction is free), off = no braking term.
        braking_reference='track',
        # Highest release speed of the training mixture, m/s. 5 is the V22 mixture; above it a 5..max_speed group is added.
        max_speed=5.,
        # Multiplier of the loaded-slip penalty (the CPU config keeps its weight, so checkpoints stay compatible).
        # 0 removes the penalty; loaded_slip_integral is still measured and logged.
        slip_penalty_scale=1.,
        # Left legs mirror the right ones in the nominal pose: release pose, action origin and zero of the joint
        # observations. The CPU model and its home keyframe are unchanged.
        symmetric_pose=False,
    )
    return config
