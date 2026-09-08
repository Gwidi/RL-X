import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from rl_x.algorithms.ppo.flax.critic import get_critic as get_flax_critic
from rl_x.algorithms.ppo.flax.policy import get_policy as get_flax_policy
from rl_x.algorithms.ppo.flax_full_jit.critic import (
    get_critic as get_flax_full_jit_critic,
)
from rl_x.algorithms.ppo.flax_full_jit.policy import (
    get_policy as get_flax_full_jit_policy,
)
from rl_x.algorithms.ppo.height_map import HEIGHT_MAP_SHAPE
from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


class PPOHeightMapCNNTest(unittest.TestCase):
    def setUp(self):
        self.proprioception_dim = 20
        self.critic_state_dim = 27
        self.height_map_size = int(np.prod(HEIGHT_MAP_SHAPE))
        observation_dim = (
            self.proprioception_dim
            + self.height_map_size
            + (self.critic_state_dim - self.proprioception_dim)
            + self.height_map_size
        )

        policy_indices = np.arange(
            self.proprioception_dim + self.height_map_size
        )
        critic_indices = np.concatenate(
            (
                np.arange(self.proprioception_dim),
                np.arange(
                    self.proprioception_dim + self.height_map_size,
                    observation_dim,
                ),
            )
        )
        self.env = SimpleNamespace(
            general_properties=SimpleNamespace(
                action_space_type=ActionSpaceType.CONTINUOUS,
                observation_space_type=ObservationSpaceType.FLAT_VALUES,
            ),
            single_observation_space=SimpleNamespace(shape=(observation_dim,)),
            single_action_space=SimpleNamespace(
                shape=(4,),
                low=-np.ones(4),
                high=np.ones(4),
            ),
            policy_observation_indices=policy_indices,
            critic_observation_indices=critic_indices,
        )
        self.config = SimpleNamespace(
            algorithm=SimpleNamespace(
                std_dev=1.0,
                nr_hidden_units=256,
                action_clipping_and_rescaling=False,
                height_map_cnn_enabled=True,
                height_map_cnn_output_dim=8,
            ),
            environment=SimpleNamespace(
                policy_exteroceptive_observation_type="height_samples",
                critic_exteroceptive_observation_type="height_samples",
            ),
        )

    def test_flax_variants_replace_each_height_map_with_8d_embedding(self):
        observation = jnp.zeros(
            (3, self.env.single_observation_space.shape[0]),
            dtype=jnp.float32,
        )
        variants = (
            ("flax", get_flax_policy, get_flax_critic),
            (
                "flax_full_jit",
                get_flax_full_jit_policy,
                get_flax_full_jit_critic,
            ),
        )

        for name, get_policy, get_critic in variants:
            with self.subTest(variant=name):
                policy, _ = get_policy(self.config, self.env)
                critic = get_critic(self.config, self.env)
                policy_params = policy.init(jax.random.PRNGKey(0), observation)
                critic_params = critic.init(jax.random.PRNGKey(1), observation)

                mean, log_std = jax.jit(policy.apply)(
                    policy_params,
                    observation,
                )
                value = jax.jit(critic.apply)(critic_params, observation)

                self.assertEqual(mean.shape, (3, 4))
                self.assertEqual(log_std.shape, (1, 4))
                self.assertEqual(value.shape, (3, 1))
                self.assertEqual(
                    policy_params["params"]["Dense_0"]["kernel"].shape[0],
                    self.proprioception_dim + 8,
                )
                self.assertEqual(
                    critic_params["params"]["Dense_0"]["kernel"].shape[0],
                    self.critic_state_dim + 8,
                )

    def test_non_height_observation_keeps_the_flat_mlp_path(self):
        self.config.environment.policy_exteroceptive_observation_type = "none"
        policy, _ = get_flax_policy(self.config, self.env)
        observation = jnp.zeros(
            (1, self.env.single_observation_space.shape[0]),
            dtype=jnp.float32,
        )
        params = policy.init(jax.random.PRNGKey(2), observation)

        self.assertEqual(len(policy.height_map_indices), 0)
        self.assertEqual(
            params["params"]["Dense_0"]["kernel"].shape[0],
            len(self.env.policy_observation_indices),
        )


if __name__ == "__main__":
    unittest.main()
