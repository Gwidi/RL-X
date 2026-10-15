"""Compare search dynamics against the actual nominal RL environment.

Run from the repository root:
    rlx/bin/python -m unittest discover -s search -p 'test_physics.py' -v
"""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from search import main as search
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.default_config import get_config
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.environment import LocomotionEnv
from rl_x.environments.custom_mujoco.robot_locomotion.robots.silver_badger.robot_config import robot_config


class SearchPhysicsTest(unittest.TestCase):
    def make_env(self, locked=False):
        cfg = get_config("custom_mujoco.robot_locomotion.mujoco").to_dict()
        cfg.update(terrain={"type": "plane"}, env_curriculum_enabled=False,
                   env_curriculum_disabled_coeff=0.0, episode_length_in_seconds=5)
        robot = deepcopy(robot_config)
        robot.update(spine_locked=locked, directory_path=(
            ROOT / "rl_x/environments/custom_mujoco/robot_locomotion/robots/silver_badger"
        ))
        env = LocomotionEnv(robot, "test", 1, False, cfg, 1)
        self.addCleanup(env.close)
        env.handle_domain_randomization = lambda **kwargs: None
        env.reset()
        return env

    def search_model(self, locked=False):
        model = mujoco.MjModel.from_xml_path(str(search.XML_PATH))
        search.configure_spine(model, locked)
        return model

    def test_compiled_physics_matches_runtime_rl_settings(self):
        for locked in (False, True):
            with self.subTest(locked=locked):
                model = self.search_model(locked)
                reference = self.make_env(locked).initial_mj_model
                for field in (
                    "body_mass", "body_inertia", "body_ipos", "body_iquat",
                    "body_pos", "body_quat", "dof_damping", "dof_armature",
                    "dof_frictionloss", "dof_invweight0", "body_invweight0",
                    "jnt_pos", "jnt_axis", "jnt_range",
                    "jnt_stiffness", "geom_type", "geom_size", "geom_pos",
                    "geom_quat", "geom_contype", "geom_conaffinity",
                    "pair_geom1", "pair_geom2", "pair_dim", "pair_solref",
                    "pair_solimp", "pair_friction",
                ):
                    np.testing.assert_allclose(getattr(model, field),
                                               getattr(reference, field), atol=1e-12,
                                               err_msg=field)
                for field in ("timestep", "iterations", "ls_iterations", "cone",
                              "integrator", "disableflags", "density", "viscosity"):
                    self.assertEqual(getattr(model.opt, field),
                                     getattr(reference.opt, field), field)

    def test_drop_trajectory_matches_rl_in_both_spine_modes(self):
        for locked in (False, True):
            with self.subTest(locked=locked):
                env = self.make_env(locked)
                model = self.search_model(locked)
                data = mujoco.MjData(model)
                jmap = search.JointMap(model, search.NOMINAL_POSE)
                kp, kd, targets = {}, {}, {}
                for name in search.NOMINAL_POSE:
                    act = jmap.act_id[name]
                    gear = jmap.gear[name]
                    kp[name] = model.actuator_gainprm[act, 0] * gear**2
                    kd[name] = -model.actuator_biasprm[act, 2] * gear**2
                    targets[name] = model.key("home").qpos[jmap.qpos_adr[name]]
                targets["sp_j0"] = 0.0 if locked else -0.35
                targets["rr_j2"] -= 0.3
                if locked:
                    kp["sp_j0"] = kd["sp_j0"] = 0.0
                search.reset_drop(data, model, jmap, 2.0, targets)
                state = env.internal_state
                state["data"].qpos[:] = data.qpos
                # Exercise the velocity clamp at the same 50 Hz boundaries.
                data.qvel[6:] = 30.0
                state["data"].qvel[:] = data.qvel
                mujoco.mj_forward(state["mj_model"], state["data"])
                reverse_names = {v: k for k, v in search.JOINT_NAMES.items()}
                action = np.array([
                    (targets[reverse_names[name]]
                     - state["actuator_joint_nominal_positions"][i])
                    / robot_config["scaling_factor"]
                    for i, name in enumerate(env.actuator_joint_names)
                ])
                for _ in range(100):
                    for _ in range(env.nr_substeps):
                        search.controlled_step(model, data, jmap, targets, kp, kd)
                    env.step(action)
                    np.testing.assert_allclose(data.qpos, state["data"].qpos,
                                               rtol=1e-7, atol=1e-8)
                    np.testing.assert_allclose(data.qvel, state["data"].qvel,
                                               rtol=1e-7, atol=1e-8)

    def test_safety_factor_limits_actual_motor_torque(self):
        model = self.search_model()
        search.configure_safety_factor(model, 0.4)
        data = mujoco.MjData(model)
        jmap = search.JointMap(model, search.NOMINAL_POSE)
        pose = dict(search.NOMINAL_POSE)
        search.reset_drop(data, model, jmap, 2.0, pose)
        targets = {name: value + 0.5 for name, value in pose.items()}
        kp = dict.fromkeys(pose, 1000.0)
        kd = dict.fromkeys(pose, 0.0)
        search.controlled_step(model, data, jmap, targets, kp, kd)
        np.testing.assert_allclose(np.abs(data.actuator_force),
                                   model.actuator_forcerange[:, 1], atol=1e-12)

    def test_contact_map_sees_explicit_feet_and_rear_contacts(self):
        model = self.search_model()
        contacts = search.ContactMap(model)
        self.assertEqual(len(contacts.foot_geom_ids), 4)
        self.assertTrue(all(model.geom_contype[i] == 0
                            for i in contacts.foot_geom_ids))
        rear = model.geom("rear_mesh").id
        self.assertTrue(any({int(a), int(b)} == {rear, contacts.ground_geom_id}
                            for a, b in zip(model.pair_geom1, model.pair_geom2)))


if __name__ == "__main__":
    unittest.main()
