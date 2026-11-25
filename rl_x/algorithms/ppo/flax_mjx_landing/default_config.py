from rl_x.algorithms.ppo.flax.default_config import get_config as get_cpu_config


def get_config(algorithm_name):
    """Same PPO as ppo.flax. Only the rollout is compiled together with the environment."""
    config = get_cpu_config(algorithm_name)
    config.final_learning_rate = -1.0  # linear decay to this value over the run; negative keeps it constant
    config.min_std = 0.0  # floor on the learned std after every optimizer step; 0 disables it
    return config
