"""Success of one checkpoint over the V22 grid on MJX-Warp and on CPU MuJoCo.

Use it as the port gate and to validate a model trained on MJX before any
deployment. Scenarios are independent draws from the same distribution.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import sys

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

_env = None


def cpu_cell(job):
    """Runs in worker processes, which must not import JAX on the GPU."""
    global _env
    from experiments.analysis.check_mjx_parity import actor, v22_config
    from experiments.train_landing_inertial_v16 import make_inertial_env
    from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_evaluation import LandingEvaluationWrapper
    height, speed, seeds, params = job
    if _env is None:
        _env = LandingEvaluationWrapper(make_inertial_env('test', v22_config()), 0)
    base, results = _env.env, []
    for seed in seeds:
        _env.configure_landing_evaluation(dict(base.env_config['landing_evaluation'], height=height, perturbed=speed > 0,
                                               max_lateral_velocity=speed, velocity_sampling='speed_ring', seed=seed))
        obs, info = _env.reset()
        done = False
        while not done:
            obs, _, terminated, truncated, info = _env.step(actor(params, obs[base.policy_observation_indices]))
            done = terminated or truncated
        results.append((info['curriculum/landing_success'], info['metrics/peak_body_acceleration'],
                        info['metrics/peak_foot_force_10ms'], info['metrics/safety_min_base_height'],
                        info['metrics/inertial_velocity_error'], info['rollout/episode_length'],
                        info['rollout/episode_return']))
    return height, speed, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--episodes', type=int, default=128)
    parser.add_argument('--workers', type=int, default=min(12, len(os.sched_getaffinity(0))))
    parser.add_argument('--backend', choices=('both', 'cpu', 'mjx'), default='both')
    parser.add_argument('--heights', help='default: the V22 grid')
    parser.add_argument('--speeds', help='default: the V22 grid; must start at 0')
    parser.add_argument('--seed', type=int, default=0, help='nonzero selects independent scenarios on both backends')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    from experiments.analysis.check_mjx_parity import actor, actor_params, make_mjx_env, v22_config
    params = actor_params(args.model)
    config = v22_config()
    evaluation = config['landing_evaluation']
    evaluation['heights'] = args.heights or evaluation['heights']
    evaluation['velocity_grid'] = args.speeds or evaluation['velocity_grid']
    evaluation['seed'] += args.seed
    speeds = [float(v) for v in evaluation['velocity_grid'].split(',')]
    heights = [float(h) for h in evaluation['heights'].split(',')]
    names = ('success_rate', 'peak_body_acceleration', 'peak_foot_force_10ms', 'safety_min_base_height',
             'inertial_velocity_error', 'episode_length', 'episode_return')
    cpu, mjx = {}, {}
    if args.backend != 'mjx':
        chunk = 8
        first = 50_000 + 100_000 * args.seed
        jobs = [(h, v, list(range(first + s, first + min(s + chunk, args.episodes))), params)
                for h in heights for v in speeds for s in range(0, args.episodes, chunk)]
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            for height, speed, part in pool.map(cpu_cell, jobs):
                cpu.setdefault((height, speed), []).extend(part)
    if args.backend != 'cpu':
        import jax
        from rl_x.environments.custom_mujoco.robot_locomotion.mjx_landing.evaluation import LandingGrid
        env = make_mjx_env(config)
        device_params = jax.tree.map(jax.numpy.asarray, params)
        grid = LandingGrid(env, evaluation, args.episodes, lambda p, obs: actor(
            p, obs[:, env.policy_observation_indices], jax.numpy))
        record = grid._run(grid.policy_rollout, device_params)
        for i, cell in enumerate(grid.cells):
            mjx[cell] = np.array([record[name][i] for name in names]).T
    rows, z = [], []
    for height in heights:
        for speed in speeds:
            key, row = (height, speed), dict(height=height, speed=speed, episodes=args.episodes)
            for label, source in (('cpu', cpu), ('mjx', mjx)):
                if key in source:
                    row.update({f'{label}_{name}': float(v) for name, v in zip(names, np.mean(source[key], axis=0))})
            if key in cpu and key in mjx:
                a, b = row['cpu_success_rate'], row['mjx_success_rate']
                mean = (a + b) / 2
                deviation = np.sqrt(2 * mean * (1 - mean) / args.episodes)
                row['z'] = float((b - a) / deviation) if deviation > 0 else 0.
                z.append(row['z'])
            rows.append(row)
    for height in heights:
        for label in ('cpu', 'mjx'):
            values = [r.get(f'{label}_success_rate') for r in rows if r['height'] == height]
            if values[0] is not None:
                print(f'h {height:<5g} {label}', ' '.join(f'{100 * v:5.0f}' for v in values))
    if z:
        z = np.array(z)
        print(f'speeds {speeds}; success difference z: mean {z.mean():+.2f}, rms over informative cells '
              f'{np.sqrt((z[z != 0] ** 2).mean()) if (z != 0).any() else 0:.2f}, max |z| {np.abs(z).max():.2f}')
        for name in names[1:]:
            relative = [(r[f'mjx_{name}'] - r[f'cpu_{name}']) / max(abs(r[f'cpu_{name}']), 1e-6) for r in rows]
            print(f'{name:26s} mean relative difference {100 * np.mean(relative):+.2f}%, largest cell {100 * np.max(np.abs(relative)):.1f}%')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=1) + '\n')


if __name__ == '__main__':
    main()
