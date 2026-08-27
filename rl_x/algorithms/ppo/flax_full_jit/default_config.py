from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name

    config.device = "gpu"  # cpu, gpu
    config.nr_parallel_seeds = 1
    config.total_timesteps = 2e9
    config.learning_rate = 4e-4
    config.anneal_learning_rate = True
    config.nr_steps = 128
    config.nr_epochs = 10
    config.minibatch_size = 32768
    config.gamma = 0.99
    config.gae_lambda = 0.9
    config.clip_range = 0.1
    config.entropy_coef = 0.0
    config.critic_coef = 1.0
    config.max_grad_norm = 5.0
    config.std_dev = 1.0
    config.action_clipping_and_rescaling = False
    config.evaluation_and_save_frequency = 17301504  # -1 to disable
    config.evaluation_active = True

    # Offline Silver Badger spine co-design. These operations run in test mode
    # after policy training and use the PPO critic as a differentiable design
    # surrogate over a separately collected nominal-reference state bank.
    config.spine_codesign_operation = ""  # "collect_state_bank" or "search"
    config.spine_state_bank_path = ""
    config.spine_state_bank_size = 4096
    config.spine_search_output_path = ""
    config.spine_search_steps = 200
    config.spine_search_minibatch_size = 1024
    config.spine_search_learning_rate = 0.05
    config.spine_search_max_step = 0.05
    config.spine_search_l2_weight = 0.01
    config.spine_search_seed = 0

    return config
