"""The fixed seed606 training setup must match the filmed baseline."""
import unittest

import numpy as np

from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_seed606 import (
    CONTROLLER_ID, VIDEO_HEIGHTS, environment_config, make_report_env,
    training_profile, verify_controller,
)
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_evaluation import evaluation_heights


class Seed606TrainingTest(unittest.TestCase):
    def test_four_nominal_base_heights_succeed_with_fixed_gains_and_zero_actions(self):
        wrapper = make_report_env()
        self.addCleanup(wrapper.close)
        env = wrapper.env
        config = environment_config()["landing_evaluation"]
        for height in VIDEO_HEIGHTS:
            with self.subTest(height=height):
                wrapper.configure_landing_evaluation(dict(config, height=height, perturbed=False))
                wrapper.reset()
                model, data = env.internal_state["mj_model"], env.internal_state["data"]
                self.assertEqual(env.action_space.shape, (13,))
                self.assertAlmostEqual(data.qpos[2], height, places=12)
                self.assertEqual(wrapper.case_info["height_reference"], "base")
                verify_controller(model)
                gains, biases = model.actuator_gainprm.copy(), model.actuator_biasprm.copy()
                for _ in range(env.horizon):
                    _, _, terminated, truncated, _ = wrapper.step(np.zeros(13))
                    np.testing.assert_array_equal(model.actuator_gainprm, gains)
                    np.testing.assert_array_equal(model.actuator_biasprm, biases)
                    if terminated or truncated:
                        break
                self.assertTrue(env.internal_state["landing_success"])
                self.assertFalse(env.internal_state["landing_safety"].disallowed_contact)

    def test_training_curriculum_uses_base_height_without_home_offset(self):
        wrapper = make_report_env()
        self.addCleanup(wrapper.close)
        env = wrapper.env
        env.runner_mode = "train"
        env.initial_state_function.replay_easy_probability = 0.
        curriculum = env.initial_state_function._get_curriculum_state()
        for difficulty, expected in ((0., 1.), (1., 2.5)):
            curriculum.update(difficulty=difficulty, fixed_difficulty=False)
            env.reset()
            self.assertAlmostEqual(env.internal_state["data"].qpos[2], expected, places=12)

    def test_profile_declares_fixed_controller_and_paired_absolute_evaluation(self):
        profile = training_profile("test")
        env = profile["environment"]
        self.assertEqual(env["landing_controller_id"], CONTROLLER_ID)
        self.assertEqual(env["env_curriculum_disabled_coeff"], 0.)
        self.assertEqual(evaluation_heights(env["landing_evaluation"]), [1., 1.5, 2., 2.5])
        self.assertEqual(env["landing_evaluation"]["height_reference"], "base")
        self.assertEqual(env["domain_randomization"]["initial_state"]["energy_dissipation_curriculum"]["height_reference"], "base")
        with self.assertRaises(ValueError):
            evaluation_heights(dict(env["landing_evaluation"], height_reference="unknown"))


if __name__ == "__main__":
    unittest.main()
