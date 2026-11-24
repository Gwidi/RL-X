from rl_x.algorithms.ppo.flax.default_config import get_config as get_cpu_config


def get_config(algorithm_name):
    """Same PPO as ppo.flax. Only the rollout is compiled together with the environment."""
    return get_cpu_config(algorithm_name)
