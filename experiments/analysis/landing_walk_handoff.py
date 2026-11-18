"""Evaluate downloaded MJX walkers on the Silver Badger landing simulation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import orbax.checkpoint
from scipy.spatial.transform import Rotation

from rl_x.algorithms.ppo.flax_full_jit.policy import Policy
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_safety import LandingContactMap

# Original Silver Badger home, from plane.xml at 363e4ce, in actuator order.
WALK_HOME = np.array([0., -.1, -.8, 1.5, .1, .8, -1.5, -.1, .8, -1.5, .1, -.8, 1.5])
JOINT_ORDER = ('spine_joint',) + tuple(
    f'{leg}_{joint}_joint' for leg in ('RL', 'RR', 'FR', 'FL')
    for joint in ('hip', 'thigh', 'calf')
)


def read_walker_checkpoint(path: Path):
    with tempfile.TemporaryDirectory(prefix='silver_badger_walk_') as folder:
        destination = Path(folder).resolve()
        with zipfile.ZipFile(path) as archive:
            cfg = json.loads(archive.read('config_algorithm.json'))
            if cfg['name'] != 'ppo.flax_full_jit' or cfg['action_clipping_and_rescaling']:
                raise ValueError('Expected an unscaled MJX PPO actor')
            for member in archive.infolist():
                if not (destination / member.filename).resolve().is_relative_to(destination):
                    raise ValueError('Unsafe checkpoint member')
            archive.extractall(destination)
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        metadata = checkpointer.metadata(str(destination))
        tree = getattr(metadata, 'item_metadata', metadata)
        tree = getattr(tree, 'tree', tree)
        args = jax.tree_util.tree_map(
            lambda _: orbax.checkpoint.RestoreArgs(restore_type=np.ndarray), tree)
        restored = checkpointer.restore(str(destination), restore_args=args)
    params = restored['policy']['params']
    if params['params']['Dense_0']['kernel'].shape != (48, 512):
        raise ValueError('Expected 48 actor inputs')
    if params['params']['Dense_3']['kernel'].shape != (128, 13):
        raise ValueError('Expected 13 joint actions')
    critic_inputs = restored['critic']['params']['params']['Dense_0']['kernel'].shape[0]
    return params, cfg, dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             actor_inputs=48, critic_inputs=critic_inputs, actions=13)


def load_walker(path: Path):
    params, cfg, metadata = read_walker_checkpoint(path)
    policy = Policy((13,), cfg['std_dev'], tuple(range(48)))

    @jax.jit
    def infer(obs):
        return policy.apply(params, obs[None, :])[0][0]

    return infer, metadata


class WalkingController:
    """Drive the same MjData after landing, preserving pose, velocities and time."""

    def __init__(self, env, actor, home, scale, command, blend_s=.4):
        if tuple(env.actuator_joint_names) != JOINT_ORDER:
            raise ValueError('Silver Badger actuator order differs from the walking adapter')
        self.env = env
        self.actor = actor
        self.model = env.internal_state['mj_model']
        self.data = env.internal_state['data']
        self.home, self.scale = home, scale
        self.command = np.asarray(command)
        self.blend_s = blend_s
        self.gear = self.model.actuator_gear[:, 0].copy()
        self.initial_target = self.data.ctrl.copy() / self.gear
        self.previous = (self.initial_target - home) / scale
        self.start_time = float(self.data.time)
        self.contact_map = LandingContactMap(self.model)
        self.limits = self.model.jnt_range[env.actuator_joint_mask_joints]

    def observation(self):
        env, data = self.env, self.data
        matrix = data.site_xmat[env.imu_site_id].reshape(3, 3)
        gyro = data.sensordata[env.imu_angular_velocity_sensor_adr:
                              env.imu_angular_velocity_sensor_adr + 3]
        return np.asarray(np.concatenate([
            (data.qpos[env.actuator_joint_mask_qpos] - self.home) / 3.14,
            data.qvel[env.actuator_joint_mask_qvel] / 100.,
            self.previous / 10.,
            np.clip(gyro / 50., -1., 1.),
            self.command,
            -matrix[2],
        ]).clip(-10., 10.), dtype=np.float32)

    def step(self):
        action = np.asarray(self.actor(jnp.asarray(self.observation())))
        if not np.all(np.isfinite(action)):
            raise ValueError('Walker returned nonfinite commands')
        elapsed = self.data.time - self.start_time
        position = self.home + self.scale * action
        fraction = min(1., elapsed / self.blend_s) if self.blend_s else 1.
        fraction = fraction**2 * (3. - 2. * fraction)
        target = np.clip((1. - fraction) * self.initial_target + fraction * position,
                         self.limits[:, 0], self.limits[:, 1])
        self.data.ctrl[:] = self.gear * target
        # Match the training observation: previous raw policy action. During
        # the blend the effective target remains an interpolation in joint space.
        self.previous = action
        forbidden = False
        support = np.zeros(4)
        for _ in range(self.env.nr_substeps):
            mujoco.mj_step(self.model, self.data)
            total, _, forces, _, hit = self.contact_map.support_contact_forces(self.model, self.data)
            forbidden |= hit
            support = np.maximum(support, forces)
        return forbidden, support


def yaw_matrix(data):
    yaw = Rotation.from_quat(data.qpos[3:7], scalar_first=True).as_euler('xyz')[2]
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])
