"""Measured drive limits, applied without changing plane.xml or the PD gains."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .landing_physics import PROJECT_ROOT


def read_landing_actuators(path):
    blob = (PROJECT_ROOT / Path(path)).read_bytes()
    profile = json.loads(blob)
    for key in ('leg_joint_torque_nm', 'spine_joint_torque_nm'):
        if not np.isfinite(profile[key]) or profile[key] <= 0.:
            raise ValueError(f'{key} must be positive and finite')
    low, high = profile['spine_range_rad']
    if not np.isfinite([low, high]).all() or low >= high:
        raise ValueError('spine_range_rad must be an increasing finite pair')
    return profile, hashlib.sha256(blob).hexdigest()


def motor_torque_limits(profile, leg_gear, spine_gear):
    """MuJoCo force ranges act before the gear, unlike the joint limits here."""
    return dict(leg_motor_torque_limit=profile['leg_joint_torque_nm'] / leg_gear,
                spine_motor_torque_limit=profile['spine_joint_torque_nm'] / spine_gear)


def apply_landing_actuators(model, config, reward):
    profile, sha256 = read_landing_actuators(config['profile'])
    if sha256 != config['sha256']:
        raise ValueError('Landing actuator profile SHA256 mismatch')
    expected = motor_torque_limits(profile, reward['leg_gear_ratio'], reward['spine_gear_ratio'])
    for key, value in expected.items():
        if not np.isclose(reward[key], value, rtol=1e-6):
            raise ValueError(f'reward.{key} differs from the landing actuator profile')
    for index in range(model.nu):
        spine = model.actuator(index).name == 'spine'
        gear = float(model.actuator_gear[index, 0])
        if not np.isclose(gear, reward['spine_gear_ratio' if spine else 'leg_gear_ratio']):
            raise ValueError('Model gear differs from the reward gear ratio')
        torque = profile['spine_joint_torque_nm' if spine else 'leg_joint_torque_nm']
        model.actuator_forcerange[index] = np.array([-torque, torque]) / gear
        model.actuator_forcelimited[index] = 1
    joint = model.joint('spine_joint').id
    low, high = profile['spine_range_rad']
    home = model.key('home').qpos[model.jnt_qposadr[joint]]
    if not low < home < high:
        raise ValueError('The spine range must contain the landing home pose')
    model.jnt_range[joint] = low, high
    return sha256
