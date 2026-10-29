"""Compare search dynamics against the actual nominal RL environment.

Run from the repository root:
    rlx/bin/python -m unittest discover -s search -p 'test_physics.py' -v
"""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from search import main as search
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.default_config import get_config
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.environment import LocomotionEnv
from rl_x.environments.custom_mujoco.robot_locomotion.robots.silver_badger.robot_config import robot_config


class SearchPhysicsTest(unittest.TestCase):
    @staticmethod
    def mesh_vertices_in_body(model, data, geom):
        mesh = model.geom_dataid[geom]
        start = model.mesh_vertadr[mesh]
        vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
        body = model.geom_bodyid[geom]
        world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        return (world - data.xpos[body]) @ data.xmat[body].reshape(3, 3)

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
                safety = search.LandingSafetyMonitor(model)
                for _ in range(100):
                    for _ in range(env.nr_substeps):
                        search.controlled_step(model, data, jmap, targets, kp, kd)
                        safety.update(model, data)
                    env.step(action)
                    reference_safety = state["landing_safety"]
                    np.testing.assert_allclose(safety.peak_calf, reference_safety.peak_calf, rtol=1e-7, atol=1e-7)
                    np.testing.assert_allclose(safety.peak_foot, reference_safety.peak_foot, rtol=1e-7, atol=1e-7)
                    self.assertAlmostEqual(safety.peak_acceleration, reference_safety.peak_acceleration, places=6)
                    self.assertEqual(safety.violations(), reference_safety.violations())
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

    def test_training_reset_and_zero_action_match_active_xml(self):
        reference = self.search_model()
        pose, gains = {}, {}
        for name, joint_name in search.JOINT_NAMES.items():
            pose[name] = reference.key("home").qpos[reference.joint(joint_name).qposadr[0]]
            act = reference.actuator(search.ACTUATOR_NAMES[name]).id
            gear = reference.actuator_gear[act, 0]
            kp_key, kd_key = ("kp_spine", "kd_spine") if name == "sp_j0" else (f"kp_{name}", f"kd_{name}")
            gains[kp_key] = reference.actuator_gainprm[act, 0] * gear**2
            gains[kd_key] = -reference.actuator_biasprm[act, 2] * gear**2
        for locked in (False, True):
            with self.subTest(locked=locked):
                env = self.make_env(locked)
                state = env.internal_state
                model = env.initial_mj_model
                for name, joint_name in search.JOINT_NAMES.items():
                    qpos_adr = model.joint(joint_name).qposadr[0]
                    target = 0.0 if name == "sp_j0" and locked else pose[name]
                    self.assertAlmostEqual(state["data"].qpos[qpos_adr], target)
                    if name == "sp_j0" and locked:
                        continue
                    act = model.actuator(search.ACTUATOR_NAMES[name]).id
                    gear = model.actuator_gear[act, 0]
                    kp_key, kd_key = ("kp_spine", "kd_spine") if name == "sp_j0" else (f"kp_{name}", f"kd_{name}")
                    self.assertAlmostEqual(model.actuator_gainprm[act, 0] * gear**2, gains[kp_key])
                    self.assertAlmostEqual(-model.actuator_biasprm[act, 2] * gear**2, gains[kd_key])
                control = env.control_function.process_action(np.zeros(env.nr_actuator_joints))
                np.testing.assert_allclose(control / model.actuator_gear[:, 0],
                                           state["data"].qpos[env.actuator_joint_mask_qpos], atol=1e-12)

    def test_each_calf_generates_measurable_forbidden_floor_contact(self):
        model = self.search_model()
        contacts = search.ContactMap(model)
        for leg in ("RL", "RR", "FR", "FL"):
            with self.subTest(leg=leg):
                data = mujoco.MjData(model)
                mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
                mujoco.mj_forward(model, data)
                calf = model.geom(f"{leg}_calf").id
                # Isolate the calf so another collider cannot carry all load.
                model.pair_margin[:] = -1000.0
                selected = (model.pair_geom1 == calf) | (model.pair_geom2 == calf)
                model.pair_margin[selected] = 0.0
                mesh = model.geom_dataid[calf]
                start = model.mesh_vertadr[mesh]
                vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
                bottom = np.min(vertices @ data.geom_xmat[calf].reshape(3, 3)[2]
                                + data.geom_xpos[calf, 2])
                data.qpos[2] -= bottom + 0.005
                data.qvel[2] = -1.0
                data.ctrl[:] = (data.qpos[model.jnt_qposadr[model.actuator_trnid[:, 0]]]
                                * model.actuator_gear[:, 0])
                mujoco.mj_forward(model, data)
                self.assertTrue(any(
                    {int(c.geom1), int(c.geom2)} == {calf, contacts.ground_geom_id}
                    for c in data.contact
                ))
                *_, shin_force, disallowed_contact = contacts.support_contact_forces(model, data)
                self.assertGreater(shin_force, 0.0)
                # Other geometries may also touch in this deliberately lowered
                # pose. Isolate the calf contact to check its classification.
                calf_contact = next(c for c in data.contact
                                    if {int(c.geom1), int(c.geom2)}
                                    == {calf, contacts.ground_geom_id})
                with patch.object(search.mujoco, "mj_contactForce",
                                                 side_effect=lambda m, d, i, f: f.fill(1.0)):
                    *_, isolated_force, forbidden = contacts.support_contact_forces(
                        model, SimpleNamespace(ncon=1, contact=[calf_contact]))
                self.assertGreater(isolated_force, 0.0)
                self.assertTrue(forbidden)

    def test_body_contact_is_still_disallowed(self):
        model = self.search_model()
        contacts = search.ContactMap(model)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        data.qpos[2] = 0.0
        mujoco.mj_forward(model, data)
        *_, disallowed_contact = contacts.support_contact_forces(model, data)
        self.assertTrue(disallowed_contact)

    def test_lower_third_has_real_force_but_no_calf_penalty_or_foot_support(self):
        for leg in ("RL", "RR", "FR", "FL"):
            with self.subTest(leg=leg):
                model = self.search_model()
                data = mujoco.MjData(model)
                mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
                lower = model.geom(f"{leg}_calf_lower").id
                model.pair_margin[:] = -1000.0
                model.pair_margin[(model.pair_geom1 == lower) | (model.pair_geom2 == lower)] = 0.0
                mujoco.mj_forward(model, data)
                mesh = model.geom_dataid[lower]
                start = model.mesh_vertadr[mesh]
                vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
                bottom = np.min(vertices @ data.geom_xmat[lower].reshape(3, 3)[2]
                                + data.geom_xpos[lower, 2])
                data.qpos[2] -= bottom + .005
                data.qvel[2] = -1.0
                data.ctrl[:] = (data.qpos[model.jnt_qposadr[model.actuator_trnid[:, 0]]]
                                * model.actuator_gear[:, 0])
                mujoco.mj_forward(model, data)
                force = np.zeros(6)
                normal = 0.0
                for i, contact in enumerate(data.contact):
                    if lower in contact.geom:
                        mujoco.mj_contactForce(model, data, i, force)
                        normal += abs(force[0])
                self.assertGreater(normal, 0.0)
                total, _, feet, shin, forbidden, calves = search.ContactMap(model).support_contact_forces(
                    model, data, include_calf_forces=True)
                self.assertEqual(total, 0.0)
                self.assertEqual(shin, 0.0)
                self.assertFalse(forbidden)
                np.testing.assert_array_equal(feet, np.zeros(4))
                np.testing.assert_array_equal(calves, np.zeros(4))

    def test_split_hulls_cover_previous_hull_and_respect_two_thirds_boundary(self):
        model = self.search_model()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        normals = np.random.default_rng(43).normal(size=(512, 3))
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        for leg in ("RL", "RR", "FR", "FL"):
            with self.subTest(leg=leg):
                upper = self.mesh_vertices_in_body(model, data, model.geom(f"{leg}_calf").id)
                lower = self.mesh_vertices_in_body(model, data, model.geom(f"{leg}_calf_lower").id)
                foot = model.geom_pos[model.geom(f"{leg}_foot").id]
                sign = np.sign(foot[1])
                union = np.vstack((upper, lower))
                along = sign * union[:, 1]
                boundary = along.min() + np.ptp(along) * 2 / 3
                self.assertLessEqual(np.max(sign * upper[:, 1]), boundary + 1e-6)
                self.assertGreaterEqual(np.min(sign * lower[:, 1]), boundary - 1e-6)
                self.assertAlmostEqual(np.max(sign * lower[:, 1]), .180, places=6)
                side = "l" if sign > 0 else "r"
                source = (ROOT / "rl_x/environments/custom_mujoco/robot_locomotion/robots/"
                          f"silver_badger/data/assets/l2{side}_calf_collision.obj")
                old = np.array([[float(v) for v in line.split()[1:]]
                                for line in source.read_text().splitlines() if line.startswith("v ")])
                np.testing.assert_allclose((union @ normals.T).max(axis=0),
                                           (old @ normals.T).max(axis=0), atol=1e-6, rtol=0)

    def test_calf_contact_mesh_does_not_protrude_outside_visual_hull(self):
        model = self.search_model()
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        mujoco.mj_forward(model, data)
        normals = np.random.default_rng(42).normal(size=(512, 3))
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        for leg in ("RL", "RR", "FR", "FL"):
            with self.subTest(leg=leg):
                calf = model.geom(f"{leg}_calf").id
                body = model.geom_bodyid[calf]
                visual = next(g for g in range(model.ngeom)
                              if model.geom_bodyid[g] == body and g != calf
                              and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH)
                contact_vertices = np.vstack([
                    self.mesh_vertices_in_body(model, data, calf),
                    self.mesh_vertices_in_body(model, data, model.geom(f"{leg}_calf_lower").id)])
                visual_vertices = self.mesh_vertices_in_body(model, data, visual)
                # A floor must not touch an enlarged proxy while the full
                # visual leg is still clear, in any of these orientations.
                self.assertTrue(np.all((contact_vertices @ normals.T).min(axis=0)
                                       >= (visual_vertices @ normals.T).min(axis=0) - 1e-6))
                foot = model.geom_pos[model.geom(f"{leg}_foot").id]
                sign = np.sign(foot[1])
                self.assertLessEqual(np.max(sign * contact_vertices[:, 1]),
                                     abs(foot[1]) - 0.02 + 1e-6)

    def test_foot_support_has_no_phantom_calf_contact(self):
        for leg in ("RL", "RR", "FR", "FL"):
            with self.subTest(leg=leg):
                model = self.search_model()
                data = mujoco.MjData(model)
                calf, foot = model.geom(f"{leg}_calf").id, model.geom(f"{leg}_foot").id
                lower = model.geom(f"{leg}_calf_lower").id
                model.pair_margin[:] = -1000.0
                selected = (np.isin(model.pair_geom1, [calf, lower, foot])
                            | np.isin(model.pair_geom2, [calf, lower, foot]))
                model.pair_margin[selected] = 0.0
                mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
                mujoco.mj_forward(model, data)
                normal = np.array([0.9767571743890189, -0.013714816641995463,
                                   -0.21390962129847305])
                normal[1] *= np.sign(model.geom_pos[foot, 1])
                body = model.geom_bodyid[calf]
                current = data.xmat[body].reshape(3, 3) @ normal
                rotation = Rotation.align_vectors([[0., 0., 1.]], [current])[0]
                data.qpos[3:7] = rotation.as_quat(scalar_first=True)
                mujoco.mj_forward(model, data)
                # One millimetre of foot compression. The old cylinder
                # penetrated about 6.6 mm here despite the real calf clearing.
                data.qpos[2] += 0.02 - data.geom_xpos[foot, 2] - 0.001
                data.qvel[2] = -1.0
                data.ctrl[:] = (data.qpos[model.jnt_qposadr[model.actuator_trnid[:, 0]]]
                                * model.actuator_gear[:, 0])
                mujoco.mj_forward(model, data)
                contacts = search.ContactMap(model)
                self.assertTrue(any(foot in c.geom for c in data.contact))
                self.assertFalse(any(calf in c.geom for c in data.contact))
                self.assertFalse(any(lower in c.geom for c in data.contact))
                total, _, _, shin, forbidden, calves = contacts.support_contact_forces(
                    model, data, include_calf_forces=True)
                self.assertGreater(total, 0.)
                self.assertEqual(shin, 0.)
                self.assertFalse(forbidden)
                np.testing.assert_array_equal(calves, np.zeros(4))

    def test_calf_load_cost_penalizes_force_and_duration(self):
        weight, dt = 100.0, 0.005
        forces = np.array([weight, 0.0, 0.0, 0.0])
        contact = forces > search.CALF_CONTACT_FORCE_EPS
        force_cost, support_cost = search.calf_cost_increment(forces, contact, weight, dt)
        doubled_force, _ = search.calf_cost_increment(2 * forces, contact, weight, dt)
        longer_force, longer_support = search.calf_cost_increment(forces, contact, weight, 2 * dt)
        self.assertAlmostEqual(doubled_force, 4 * force_cost)
        self.assertAlmostEqual(longer_force, 2 * force_cost)
        self.assertAlmostEqual(longer_support, 2 * support_cost)
        self.assertEqual(search.calf_cost_increment(np.zeros(4), np.zeros(4, dtype=bool), weight, dt), (0, 0))

    def test_calf_cost_changes_objective_without_changing_trajectory(self):
        model = self.search_model()
        data = mujoco.MjData(model)
        jmap = search.JointMap(model, search.NOMINAL_POSE)
        contacts = search.ContactMap(model)
        # Import through search's CLI module path to use the saved replay pose.
        import importlib.util
        spec = importlib.util.spec_from_file_location("calf_replay", ROOT / "search/replay_target.py")
        replay = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"main": search}):
            spec.loader.exec_module(replay)
        args = (model, data, jmap, contacts, replay.target_params(),
                search.episode_steps(model, 3.5), 3.5, True, False)
        result = search.result_from_params(*args, individual_gains=True)
        final_qpos = data.qpos.copy()
        with patch.object(search, "CALF_FORCE_COST_WEIGHT", 0.0), \
                patch.object(search, "CALF_SUPPORT_COST_PER_SECOND", 0.0):
            baseline = search.result_from_params(*args, individual_gains=True)
        np.testing.assert_array_equal(data.qpos, final_qpos)
        self.assertAlmostEqual(result["cost"] - baseline["cost"],
                               result["calf_force_cost"] + result["calf_support_cost"])
        self.assertGreater(result["calf_force_cost"], 0.0)
        self.assertGreater(max(result[f"calf_contact_duration_{leg}"] for leg in search.CALF_LEGS), 0.0)
        self.assertTrue(any(result[f"peak_calf_force_raw_{leg}"] > result[f"peak_calf_force_{leg}"]
                            for leg in search.CALF_LEGS))
        self.assertEqual(search.is_safe(result), search.is_safe(baseline))

    def test_calf_force_limit_is_per_leg_and_inclusive(self):
        result = dict(foot_contact=True, crashed=False, disallowed_contact=False,
                      peak_foot_force=0.0, peak_body_acceleration=0.0,
                      shin_contact=True)
        result.update({f"peak_calf_force_{leg}": search.MAX_LANDING_CALF_FORCE
                       for leg in search.CALF_LEGS})
        # The sum may exceed the limit; each individual calf is what matters.
        self.assertTrue(search.is_safe(result))
        for leg in search.CALF_LEGS:
            with self.subTest(leg=leg):
                result[f"peak_calf_force_{leg}"] += 0.01
                self.assertFalse(search.is_safe(result))
                result[f"peak_calf_force_{leg}"] -= 0.01

    def test_calf_limit_adds_violation_penalty_to_real_drop(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("calf_limit_replay", ROOT / "search/replay_target.py")
        replay = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"main": search}):
            spec.loader.exec_module(replay)
        with patch.object(search, "MAX_LANDING_CALF_FORCE", float("inf")):
            *_, unlimited = replay.run(3.5)
        peak = max(unlimited[f"peak_calf_force_{leg}"] for leg in search.CALF_LEGS)
        self.assertGreater(peak, 0.0)
        # The saved controller changes; exercise an actual violation without
        # assuming that every future replay pose exceeds the production limit.
        limit = peak / 2.0
        with patch.object(search, "MAX_LANDING_CALF_FORCE", limit):
            *_, result = replay.run(3.5)
            self.assertFalse(search.is_safe(result))
        self.assertAlmostEqual(result["cost"] - unlimited["cost"],
                               search.LANDING_QUALITY_PENALTY * peak / limit)


if __name__ == "__main__":
    unittest.main()
