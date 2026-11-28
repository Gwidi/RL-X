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
        # Right front hip of the symmetric pose, rad (the left one is its mirror). 0.2687 is the home value; smaller
        # values move the front feet apart (0.009: +-14 cm instead of +-5 cm). Used only with symmetric_pose.
        symmetric_front_hip=0.2687,
        # Penalty before the first touchdown: -coefficient * sum over the four hip actions of (a_t - a_{t-1})^2.
        # Discourages a sudden hip movement when the actor takes over from the release pose. 0 disables it.
        flight_hip_rate_penalty=0.,
        # Domain randomization, drawn per episode (per released state) and kept until the next release. All zero/16 =
        # the nominal model. trunk_mass_kg: uniform +-, added to the trunk (inertia scaled with it). com_shift_m: shift of
        # the robot's total centre of mass, uniform +- in x and y (moved through the trunk). torque_*_nm: drive limit,
        # uniform per joint. pd_gain: kp and kd each scaled uniformly in 1 +- pd_gain per joint. friction_*: sliding
        # friction between the floor and every robot geom, one value per episode (nominal 1).
        randomization=dict(trunk_mass_kg=0., com_shift_m=0., torque_min_nm=16., torque_max_nm=16., pd_gain=0.,
                           friction_min=1., friction_max=1.),
        # Reward per step from 0.3 s after the first touchdown: coefficient * mean of exp(-(tilt / 0.5 rad)^2) and
        # exp(-(rms left-right joint mismatch / 0.4 rad)^2). 0 disables it.
        posture_reward=0.,
    )
    return config
