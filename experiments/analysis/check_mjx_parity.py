"""Step-by-step comparison of the MJX-Warp landing environment with the CPU one.

Both run the same conditions, launch durations and actions, without IMU noise.
The actions come from the CPU closed loop, so later errors include chaotic
divergence after contact; the first steps are the port check.
"""

import argparse
import os
from pathlib import Path
import sys
import zipfile
import tempfile

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import numpy as np

from experiments.train_landing_inertial_v16 import inertial_environment_config, make_inertial_env
from experiments.train_landing_real_drives_fresh_v22 import fresh_profile
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_seed606 import ROBOT_DIR
from rl_x.environments.custom_mujoco.robot_locomotion.robots.silver_badger.robot_config import robot_config


def merge(target, updates):
    for key, value in updates.items():
        if isinstance(value, dict):
            merge(target.setdefault(key, {}), value)
        else:
            target[key] = value
    return target


def v22_config(**inertial):
    config = merge(inertial_environment_config(), fresh_profile()['environment'])
    config['policy_inertial'].update(inertial)
    return config


def actor_params(path):
    """Actor of a ppo.flax checkpoint as {layer: {kernel, bias, scale}}."""
    import orbax.checkpoint
    with tempfile.TemporaryDirectory() as directory:
        with zipfile.ZipFile(path) as archive:
            archive.extractall(directory)
        with orbax.checkpoint.PyTreeCheckpointer() as checkpointer:
            params = checkpointer.restore(directory)['policy']['params']['params']
    return jax.tree.map(np.asarray, params)


def actor(params, x, xp=np):
    """The policy network mean, clipped like the PPO action processing."""
    elu = lambda v: xp.where(v > 0, v, xp.expm1(xp.minimum(v, 0)))
    dense = lambda v, p: v @ p['kernel'] + p['bias']
    x = dense(x, params['Dense_0'])
    mean = x.mean(-1, keepdims=True)
    variance = ((x - mean) ** 2).mean(-1, keepdims=True)
    norm = params['LayerNorm_0']
    x = elu((x - mean) / xp.sqrt(variance + 1e-6) * norm['scale'] + norm['bias'])
    x = elu(dense(x, params['Dense_1']))
    x = elu(dense(x, params['Dense_2']))
    return xp.clip(dense(x, params['Dense_3']), -1., 1.)


def make_mjx_env(config):
    from rl_x.environments.custom_mujoco.robot_locomotion.mjx_landing.environment import LandingMjxEnv
    from rl_x.environments.custom_mujoco.robot_locomotion.mjx_landing.default_config import get_config
    full = merge(get_config('mjx_landing').to_dict(), config)
    return LandingMjxEnv({**robot_config, 'directory_path': ROBOT_DIR, 'spine_locked': False}, full)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='modelsv19/latest.model')
    parser.add_argument('--episodes', type=int, default=48)
    parser.add_argument('--noise', type=float, default=.075)
    args = parser.parse_args()
    params = actor_params(args.model)
    silent = dict(acceleration_noise_std=0., acceleration_bias_std=0., gyro_noise_std=0., gyro_bias_std=0.)
    config = v22_config(**silent)
    cpu = make_inertial_env('train', config, seed=7)
    rng = np.random.default_rng(3)
    episodes = []
    for _ in range(args.episodes):
        obs, info = cpu.reset()
        cpu.internal_state['robot_imu_height_over_ground'] = cpu.internal_state['data'].site_xpos[cpu.imu_site_id, 2]
        data = cpu.internal_state['data']
        start = dict(height=info['curriculum/v22_height'], duration=info['inertial/launch_duration_s'],
                     velocity=[info['curriculum/v22_speed'] * np.cos(a) for a in (0.,)], qpos=data.qpos.copy(), qvel=data.qvel.copy())
        # Requested direction: the release velocity is the requested one up to 1e-3.
        speed, v = info['curriculum/v22_speed'], data.qvel[:2]
        start['velocity'] = v / max(np.linalg.norm(v), 1e-9) * speed
        rows, done = [], False
        first = obs
        while not done:
            action = np.clip(actor(params, obs[cpu.policy_observation_indices]) + args.noise * rng.normal(size=13), -1, 1)
            obs, reward, terminated, truncated, info = cpu.step(action)
            done = terminated or truncated
            rows.append(dict(action=action, obs=obs, reward=reward, done=done, info={k: v for k, v in info.items() if np.isscalar(v)}))
        episodes.append((start, first, rows))
    env = make_mjx_env(config)
    n = args.episodes
    height = jnp.array([s['height'] for s, _, _ in episodes], jnp.float32)
    velocity = jnp.array([s['velocity'] for s, _, _ in episodes], jnp.float32)
    controls = jnp.array([round(s['duration'] / env.dt) for s, _, _ in episodes])
    pool = jax.jit(env.launch)(jax.random.split(jax.random.PRNGKey(0), n), height, velocity, controls)
    assert not bool(pool['invalid'])
    first = np.array([f for _, f, _ in episodes])
    delta = np.abs(np.asarray(pool['observation']) - first)
    delta[:, np.asarray(cpu.critic_exteroception_obs_idx)] = 0  # the CPU reset keeps the previous episode's height here
    print(f"release: |dqpos| {np.abs(np.asarray(pool['release']['qpos']) - [s['qpos'] for s, _, _ in episodes]).max():.1e} "
          f"|dqvel| {np.abs(np.asarray(pool['release']['qvel']) - [s['qvel'] for s, _, _ in episodes]).max():.1e} "
          f"|dobs| {delta.max():.1e}")
    state = jax.jit(lambda p: env.reset(p, n))(pool)
    step = jax.jit(env.step)
    horizon = max(len(r) for _, _, r in episodes)
    reward_keys = ('alive', 'braking_tracking', 'loaded_slip', 'collision', 'body_acceleration', 'body_support', 'landing_event', 'total')
    records = []
    alive = np.ones(n, bool)
    for t in range(horizon):
        action = np.array([r[min(t, len(r) - 1)]['action'] for _, _, r in episodes], np.float32)
        state, _ = step(state, jnp.asarray(action), pool, jnp.asarray(0))
        obs, done = np.asarray(state.actual_observation), np.asarray(state.terminated | state.truncated)
        info = {k: np.asarray(v) for k, v in state.info.items()}
        for i in np.flatnonzero(alive):
            row = episodes[i][2][t]
            record = dict(i=i, t=t, obs=np.abs(obs[i] - row['obs']), reward=abs(float(state.reward[i]) - row['reward']),
                          done=bool(done[i]) == row['done'], cpu_done=row['done'], mjx_done=bool(done[i]),
                          touched='metrics/braking_elapsed' in row['info'] and bool(row['info'].get('reward/braking_tracking', 0)),
                          terms={k: abs(float(info['reward/' + k][i]) - float(row['info'].get('reward/' + k, 0.))) for k in reward_keys})
            if row['done'] or done[i]:
                alive[i] = False
                record['success'] = (float(row['info']['curriculum/landing_success']), float(info['curriculum/landing_success'][i]))
                record['reason'] = ([k for k, v in row['info'].items() if k.startswith('outcome/') and v],
                                    [k for k, v in info.items() if k.startswith('outcome/') and v[i]])
            records.append(record)
        if not alive.any():
            break
    touch = {r['i']: r['t'] for r in reversed(records) if r['touched']}
    for label, low, high in (('flight', -1000, 0), ('touch +0..1', 0, 2), ('touch +2..4', 2, 5), ('touch +5..19', 5, 20), ('touch +20..', 20, 1000)):
        part = [r for r in records if low <= r['t'] - touch.get(r['i'], 10 ** 6) < high] if label != 'flight' else [
            r for r in records if r['t'] < touch.get(r['i'], 10 ** 6)]
        if not part:
            continue
        obs = np.array([r['obs'].max() for r in part])
        rew = np.array([r['reward'] for r in part])
        terms = {k: f"{np.max([r['terms'][k] for r in part]):.0e}/{np.median([r['terms'][k] for r in part]):.0e}" for k in reward_keys}
        print(f"{label:13s} n={len(part):4d} obs med {np.median(obs):.1e} p95 {np.percentile(obs, 95):.1e} max {obs.max():.1e} | "
              f"reward med {np.median(rew):.1e} p95 {np.percentile(rew, 95):.1e} | done agrees {np.mean([r['done'] for r in part]):.3f}")
        print('   term max/median', terms)
    ends = [r for r in records if 'success' in r]
    print('outcome agrees (open loop)', np.mean([a == b for a, b in (r['success'] for r in ends)]), 'reason agrees',
          np.mean([a == b for a, b in (r['reason'] for r in ends)]), 'same final step', np.mean([r['done'] for r in ends]),
          '| successes CPU', sum(r['success'][0] for r in ends), 'MJX', sum(r['success'][1] for r in ends))
    worst = max((r for r in records if r['t'] < touch.get(r['i'], 10 ** 6)), key=lambda r: r['obs'].max())
    print('largest flight observation error at index', int(worst['obs'].argmax()), 'step', worst['t'])
    print('overflow', bool(state.overflow))


if __name__ == '__main__':
    main()
