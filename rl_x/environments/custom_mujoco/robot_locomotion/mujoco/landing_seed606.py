"""Fixed seed-606 controller, training profile and reproducibility checks."""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ROBOT_DIR = Path(__file__).resolve().parent.parent / "robots" / "silver_badger"
PRESET_PATH = ROBOT_DIR / "data" / "landing_seed606.json"
XML_PATH = ROBOT_DIR / "data" / "plane.xml"
CONTROLLER_ID = "seed606_2p5m_rounded_fixed_pd"
VIDEO_HEIGHTS = (2.5, 2.0, 1.5, 1.0)


def load_preset():
    return json.loads(PRESET_PATH.read_text())


def model_fingerprint():
    """Include the source XML and mesh bytes, not workstation-dependent paths."""
    blob = XML_PATH.read_bytes()
    xml = ET.fromstring(blob)
    asset_dir = XML_PATH.parent / xml.find("compiler").get("assetdir", ".")
    assets = {}
    combined = hashlib.sha256(blob)
    for node in sorted(xml.findall("asset/mesh"), key=lambda n: n.get("file")):
        name = node.get("file")
        data = (asset_dir / name).read_bytes()
        assets[name] = hashlib.sha256(data).hexdigest()
        combined.update(name.encode())
        combined.update(data)
    return dict(plane_sha256=hashlib.sha256(blob).hexdigest(), assets_sha256=assets,
                combined_sha256=combined.hexdigest())


def verify_controller(model=None):
    """Fail before training if the copied XML is not the declared baseline."""
    import mujoco
    from .landing_safety import LandingContactMap
    model = model or mujoco.MjModel.from_xml_path(str(XML_PATH))
    preset = load_preset()
    for leg in ("RL", "RR", "FR", "FL"):
        for index, group in enumerate(("hip", "thigh", "calf")):
            actuator = model.actuator(f"{leg}_{group}")
            gear = float(actuator.gear[0])
            np.testing.assert_allclose(actuator.gainprm[0] * gear**2, preset["gains"]["kp_" + group], atol=1e-10)
            np.testing.assert_allclose(-actuator.biasprm[2] * gear**2, preset["gains"]["kd_" + group], atol=1e-10)
            joint = model.joint(f"{leg}_{group}_joint")
            np.testing.assert_allclose(model.key("home").qpos[joint.qposadr[0]], preset["pose"][f"{leg.lower()}_j{index}"], atol=1e-12)
    spine = model.actuator("spine")
    np.testing.assert_allclose(spine.gainprm[0] * spine.gear[0]**2, preset["gains"]["kp_spine"], atol=1e-10)
    np.testing.assert_allclose(-spine.biasprm[2] * spine.gear[0]**2, preset["gains"]["kd_spine"], atol=1e-10)
    np.testing.assert_allclose(model.key("home").qpos[model.joint("spine_joint").qposadr[0]], preset["pose"]["sp_j0"], atol=1e-12)
    contacts = LandingContactMap(model)
    if len(contacts.lower_calf_geom_ids) != 4 or len(contacts.allowed_support_geom_ids) != 8:
        raise ValueError("The seed606 experiment requires all four split calf colliders")
    return model_fingerprint()


def training_profile(run_name=None):
    """Only declared experiment overrides; general PPO/environment defaults stay."""
    run_name = run_name or os.environ.get("RUN_NAME") or ("seed606_fixed_pd_" + os.environ.get(
        "SLURM_JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S")))
    fingerprint = model_fingerprint()
    return {
        "algorithm": {"name": "ppo.flax", "total_timesteps": 30015488,
            "learning_rate": 3e-5, "std_dev": .15, "nr_steps": 2048, "nr_epochs": 10,
            "minibatch_size": 64, "evaluation_frequency": 131072, "evaluation_episodes": 64},
        "environment": {"name": "custom_mujoco.robot_locomotion.mujoco", "train_robot": "silver_badger",
            "spine_locked": False, "terrain": {"type": "plane"}, "render": False,
            "nr_envs": 64, "seed": 1, "async_skip_percentage": 0., "imu_history_length": 5,
            "add_goal_arrow": False, "copy_train_env_for_eval": False,
            "episode_length_in_seconds": 5, "env_curriculum_enabled": False,
            "env_curriculum_disabled_coeff": 0., "landing_controller_id": CONTROLLER_ID,
            "landing_controller_sha256": hashlib.sha256(PRESET_PATH.read_bytes()).hexdigest(),
            "landing_model_sha256": fingerprint["combined_sha256"],
            "landing_evaluation": {"enabled": True, "heights": "1.0,1.5,2.0,2.5",
                "height_reference": "base", "seed": 20261023, "frequency_multiplier": 8,
                "max_angle_deg": 3., "max_lateral_velocity": 3., "randomize_physics": False,
                "randomization_coefficient": 0.},
            "domain_randomization": {"initial_state": {"type": "energy_dissipation_curriculum",
                "energy_dissipation_curriculum": {"height_reference": "base", "start_height": 1.,
                    "target_height": 2.5, "min_angle_deg": 0., "max_angle_deg": 3.,
                    "min_lateral_velocity": 0., "max_lateral_velocity": 3.,
                    "success_threshold_up": .80, "success_threshold_down": .60,
                    "difficulty_step_up": .002, "difficulty_step_down": .01,
                    "replay_easy_probability": .15}}},
            "reward": {"type": "simplified_landing", "alive_coeff": .5, "base_height_coeff": 0.,
                "base_vel_coeff": 2., "joint_vel_coeff": 0., "roll_pitch_pos_coeff": 0.,
                "action_rate_coeff": 0., "joint_pos_coeff": 0., "collision_coeff": 5.,
                "floor_collision_coeff": 0., "body_support_penalty_coeff": 10.,
                "body_support_grace_time": .5, "post_touchdown_motion_grace_time": .5,
                "nominal_landing_height": .35, "leg_motor_torque_limit": 3.,
                "spine_motor_torque_limit": .785, "leg_gear_ratio": 9., "spine_gear_ratio": 36.,
                "thermal_limit_enabled": False, "thermal_penalty_coeff": 0.,
                "saturation_penalty_coeff": 0., "success_confirmation_time": 2.,
                "success_stability_time": .5, "success_max_angular_velocity": 2.,
                "base_impact_acceleration_limit": 98.1, "body_acceleration_limit": 60.,
                "body_acceleration_coeff": 500., "landing_success_coeff": 100., "landing_failure_coeff": 100.}},
        "runner": {"mode": "train", "track_console": True, "track_tb": True, "track_wandb": True,
            "wandb_entity": "filip-michalski-1-poznan-supercomputing-and-networking-c",
            "project_name": "soft_landing", "exp_name": "landing_seed606_fixed_pd", "run_name": run_name,
            "save_model": True, "load_model": "", "notes": "Seed606 2.5m rounded stdout; fixed PD; base heights; phase 1 nominal physics"},
    }


def cli_flags(mapping, prefix=""):
    flags = []
    for key, value in mapping.items():
        name = prefix + key
        if isinstance(value, dict):
            flags.extend(cli_flags(value, name + "."))
        else:
            flags.append(f"--{name}={value}")
    return flags


def environment_config():
    from .default_config import get_config
    config = get_config("custom_mujoco.robot_locomotion.mujoco").to_dict()
    def merge(target, updates):
        for key, value in updates.items():
            if isinstance(value, dict):
                merge(target[key], value)
            else:
                target[key] = value
    merge(config, training_profile("baseline_report")["environment"])
    return config


def make_report_env():
    from .environment import LocomotionEnv
    from .landing_evaluation import LandingEvaluationWrapper
    from ..robots.silver_badger.robot_config import robot_config
    robot = deepcopy(robot_config)
    robot.update(directory_path=ROBOT_DIR, spine_locked=False)
    config = environment_config()
    config["nr_envs"] = 1
    env = LocomotionEnv(robot, "test", 20261023, False, config, 1)
    verify_controller(env.initial_mj_model)
    return LandingEvaluationWrapper(env, 0)
