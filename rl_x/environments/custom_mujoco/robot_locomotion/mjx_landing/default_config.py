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
    )
    return config
