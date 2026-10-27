"""Two horizontal velocity features and the shared external-tracking convention.

World axes must be calibrated to MuJoCo's right-handed, Z-up world. The
reference point is the base/trunk origin, not an arbitrary marker or IMU site.
"""
from collections import deque
import hashlib
import json
import numpy as np
from scipy.spatial.transform import Rotation


def heading_velocity_xy(velocity_world, world_from_base):
    """Forward/left components in the gravity-aligned heading frame, in m/s."""
    velocity = np.asarray(velocity_world, dtype=float)
    rotation = np.asarray(world_from_base, dtype=float)
    forward = rotation[:, 0].copy()
    forward[2] = 0.0
    length = np.linalg.norm(forward)
    if length < 1e-8:
        # A fall may put body X upright even if the training starts near level.
        left = rotation[:, 1].copy()
        left[2] = 0.0
        left /= np.linalg.norm(left)
        forward = np.cross(left, np.array([0.0, 0.0, 1.0]))
    else:
        forward /= length
    left = np.cross(np.array([0.0, 0.0, 1.0]), forward)
    return np.array([np.dot(velocity, forward), np.dot(velocity, left)])


def tracking_velocity_features(velocity_tracking, world_from_tracking,
                               world_from_base, angular_velocity_world,
                               marker_offset_base, scale_mps=4.0):
    """Map calibrated OptiTrack-compatible marker velocity to policy features.

    Inputs: SI units, rotation matrices mapping coordinates to world, angular
    velocity in world, and vector base-origin -> tracked-marker in base frame.
    The caller supplies the aligned base attitude, timestamps and calibration;
    this function does not connect to NatNet or estimate velocity from poses.
    """
    if not np.isfinite(scale_mps) or scale_mps <= 0:
        raise ValueError('scale_mps must be finite and positive')
    rotation = np.asarray(world_from_base, dtype=float)
    marker_world = np.asarray(world_from_tracking, dtype=float) @ np.asarray(velocity_tracking, dtype=float)
    offset_world = rotation @ np.asarray(marker_offset_base, dtype=float)
    base_world = marker_world - np.cross(np.asarray(angular_velocity_world, dtype=float), offset_world)
    return heading_velocity_xy(base_world, rotation) / scale_mps


def velocity_observation_contract(config):
    mode = config.get('mode', 'none')
    if mode not in ('none', 'masked', 'heading_xy'):
        raise ValueError('policy_velocity.mode must be none, masked or heading_xy')
    schema = config.get('schema', 'base_heading_xy_v1')
    if schema != 'base_heading_xy_v1':
        raise ValueError('Unsupported policy velocity observation schema')
    scale = float(config.get('scale_mps', 4.0))
    noise = float(config.get('noise_std_mps', 0.0))
    bias = float(config.get('bias_std_mps', 0.0))
    delay = config.get('delay_steps', 0)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('policy_velocity.scale_mps must be finite and positive')
    if not np.isfinite(noise) or noise < 0 or not np.isfinite(bias) or bias < 0:
        raise ValueError('Velocity noise and bias must be finite and nonnegative')
    if isinstance(delay, bool) or not isinstance(delay, (int, np.integer)) or delay < 0:
        raise ValueError('policy_velocity.delay_steps must be a nonnegative integer')
    if mode == 'none':
        return {'mode': 'none'}  # Compatible with checkpoints predating the feature.
    return dict(version=schema, mode=mode, scale_mps=scale,
                noise_std_mps=noise, bias_std_mps=bias, delay_steps=int(delay),
                frame='gravity_aligned_heading', reference_point='base_origin')


class PolicyVelocityObservation:
    def __init__(self, config):
        self.contract = velocity_observation_contract(config)
        self.mode = self.contract['mode']
        self.dimension = 0 if self.mode == 'none' else 2

    def reset(self, env):
        if not self.dimension:
            return
        # A dedicated sensor RNG must not advance the dynamics/reset RNG.
        seed_bytes = json.dumps(env.np_rng.bit_generator.state, sort_keys=True,
                                default=lambda x: np.asarray(x).tolist()).encode()
        sensor_seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:16], 'little')
        rng = np.random.default_rng(sensor_seed)
        bias = np.zeros(2)
        if self.contract['bias_std_mps']:
            bias = rng.normal(0.0, self.contract['bias_std_mps'], 2)
        env.internal_state['policy_velocity_sensor'] = dict(
            history=deque(maxlen=self.contract['delay_steps'] + 1),
            rng=rng, bias_world_xy=bias, last_time=None, last_features=np.zeros(2))

    def observe(self, env):
        if not self.dimension:
            return np.empty(0)
        if 'policy_velocity_sensor' not in env.internal_state:
            self.reset(env)
        sensor = env.internal_state['policy_velocity_sensor']
        data = env.internal_state['data']
        if sensor['last_time'] == float(data.time):
            return sensor['last_features'].copy()
        rotation = Rotation.from_quat(data.qpos[3:7], scalar_first=True).as_matrix()
        true_velocity = np.asarray(data.qvel[:3]).copy()
        measurement = true_velocity.copy()
        measurement[:2] += sensor['bias_world_xy']
        if self.contract['noise_std_mps']:
            measurement[:2] += sensor['rng'].normal(0.0, self.contract['noise_std_mps'], 2)
        # At reset no pre-episode samples are assumed; hold the first reading.
        if not sensor['history']:
            for _ in range(self.contract['delay_steps'] + 1):
                sensor['history'].append(measurement.copy())
        else:
            sensor['history'].append(measurement)
        true_xy = heading_velocity_xy(true_velocity, rotation)
        measured_xy = heading_velocity_xy(sensor['history'][0], rotation)
        exposed = np.zeros(2) if self.mode == 'masked' else measured_xy
        features = exposed / self.contract['scale_mps']  # No hidden speed clipping.
        sensor['last_time'] = float(data.time)
        sensor['last_features'] = features.copy()
        info = env.internal_state['info']
        info.update({'observation/velocity_forward_true': float(true_xy[0]),
                     'observation/velocity_lateral_true': float(true_xy[1]),
                     'observation/velocity_forward_observed': float(exposed[0]),
                     'observation/velocity_lateral_observed': float(exposed[1]),
                     'observation/velocity_speed_true': float(np.linalg.norm(true_xy)),
                     'observation/velocity_speed_observed': float(np.linalg.norm(exposed)),
                     'observation/velocity_error_norm': float(np.linalg.norm(exposed - true_xy))})
        return features
