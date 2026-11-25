"""Velocity-grid evaluation on MJX, with the metric names of landing_velocity_grid.

All cells run as one batch. The scenario of episode i in a cell depends only on
the evaluation seed, so every checkpoint and the zero-action baseline are paired.
"""

import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from rl_x.environments.custom_mujoco.robot_locomotion.mjx_landing.environment import FAILURES, STAND_POSE
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_evaluation import evaluation_heights
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_velocity_grid import boundary_summary, wilson


MEASUREMENTS = {
    'success_rate': 'curriculum/landing_success', 'episode_return': 'episode_return', 'episode_length': 'episode_length',
    'unresolved_rate': 'outcome/unresolved',
    **{f'failure_{name}_rate': f'outcome/failure_{name}' for name in FAILURES if name != 'unresolved'},
    **{name: f'metrics/{name}' for name in (
        'peak_foot_force_10ms', 'peak_calf_force_10ms', 'peak_body_acceleration', 'peak_body_acceleration_raw',
        'peak_base_impact_acceleration', 'peak_leg_motor_torque', 'peak_spine_motor_torque', 'leg_tau_squared_integral',
        'spine_tau_squared_integral', 'safety_min_base_height', 'braking_tracking_mse', 'loaded_slip_integral',
        'inertial_velocity_error', 'inertial_estimated_speed', 'inertial_clipped_fraction')},
    **{f'inertial_{name}': f'inertial/{name}' for name in ('launch_duration_s', 'requested_speed_xy', 'release_speed_xy')},
}


class LandingGrid:
    def __init__(self, env, config, episodes, deterministic_action):
        """deterministic_action(policy_params, observation) -> processed action."""
        self.env, self.config, self.episodes = env, config, episodes
        self.heights = evaluation_heights(config)
        self.speeds = [float(v) for v in config['velocity_grid'].split(',')]
        self.threshold = float(config['success_boundary_threshold'])
        if (not self.speeds or self.speeds[0] != 0 or any(b <= a for a, b in zip(self.speeds, self.speeds[1:]))
                or config['velocity_sampling'] != 'speed_ring' or config['max_angle_deg'] != 0 or config['randomize_physics']):
            raise ValueError('MJX evaluation implements an increasing speed_ring grid without angles or randomized physics')
        self.cells = [(h, v) for v in self.speeds for h in self.heights]
        count = len(self.cells) * episodes
        height = jnp.repeat(jnp.asarray([h for h, _ in self.cells]), episodes)
        speed = jnp.repeat(jnp.asarray([v for _, v in self.cells]), episodes)
        launch_key, azimuth_key = jax.random.split(jax.random.PRNGKey(config['seed']))
        azimuth = jax.random.uniform(azimuth_key, (count,), maxval=2 * jnp.pi)
        velocity = speed[:, None] * jnp.stack([jnp.cos(azimuth), jnp.sin(azimuth)], -1)
        tilt = rate = None
        if env.disturbed:  # every cell, the zero-action baseline included, gets the same imperfect release
            tilt, rate = env.sample_disturbance(jax.random.fold_in(launch_key, 7), count)
        pose = None
        if env.stand_fraction:  # evaluation always releases from the full standing pose
            pose = env.nominal + env.stand_fraction * (jnp.asarray(STAND_POSE, jnp.float32) - env.nominal)
        self.pool = jax.jit(env.launch)(jax.random.split(launch_key, count), height, velocity, tilt=tilt, rate=rate, pose=pose)
        if bool(self.pool['invalid']):
            raise RuntimeError('Evaluation launch touched the terrain or overflowed a contact buffer')
        zeros = jnp.zeros((count,) + env.single_action_space.shape)

        def rollout(action_function):
            state = env.reset(self.pool, count)

            def step(carry, _):
                state, finished, record = carry
                # Finished worlds restart from pool entries and are ignored.
                state, _ = env.step(state, action_function(state.observation), self.pool, 0)
                done = state.terminated | state.truncated
                new = done & ~finished
                record = {k: jnp.where(new, state.info[source], record[k]) for k, source in MEASUREMENTS.items()}
                return (state, finished | done, record), None

            initial = (state, jnp.zeros(count, bool), {k: jnp.zeros(count) for k in MEASUREMENTS})
            (state, finished, record), _ = jax.lax.scan(step, initial, None, env.horizon)
            return record, finished.all(), state.overflow

        self.policy_rollout = jax.jit(lambda params: rollout(lambda obs: deterministic_action(params, obs)))
        self.zero_rollout = jax.jit(lambda: rollout(lambda obs: zeros))
        self.zero = None

    def _run(self, function, *arguments):
        record, finished, overflow = function(*arguments)
        if not bool(finished) or bool(overflow):
            raise RuntimeError(f'MJX evaluation failed: finished={bool(finished)}, contact overflow={bool(overflow)}')
        return {k: np.asarray(v, dtype=float).reshape(len(self.cells), self.episodes) for k, v in record.items()}

    def evaluate(self, params, output_dir, global_step):
        config, threshold = self.config, self.threshold
        policy = self._run(self.policy_rollout, params)
        if self.zero is None:  # The baseline does not depend on the policy.
            self.zero = self._run(self.zero_rollout)
        metrics, cells = {}, []
        for controller, results in (('policy', policy), ('zero', self.zero)):
            for i, (height, speed) in enumerate(self.cells):
                tag = lambda prefix, value: f'{prefix}_{value:g}'.replace('.', 'p')
                condition = 'nominal' if speed == 0 else 'perturbed'
                base = f"eval/grid/{tag('v', speed)}/{condition}"
                prefix = f"{base}/{controller}/{tag('h', height)}"
                metrics.update({f'{prefix}/{k}': float(v[i].mean()) for k, v in results.items()})
                metrics[f'{prefix}/episode_count'] = self.episodes
                for name in ('peak_calf_force_10ms', 'peak_foot_force_10ms', 'peak_body_acceleration'):
                    metrics[f'{prefix}/{name}_p95'] = float(np.percentile(results[name][i], 95))
                    metrics[f'{prefix}/{name}_max'] = float(results[name][i].max())
                if controller == 'policy':
                    gain = policy['success_rate'][i] - self.zero['success_rate'][i]
                    metrics[f"{base}/{tag('h', height)}/success_gain"] = float(gain.mean())
                wins = int(results['success_rate'][i].sum())
                rate, interval = wins / self.episodes, wilson(wins, self.episodes)
                cells.append(dict(height=height, speed=speed, controller=controller, successes=wins, episodes=self.episodes,
                                  success_rate=rate, wilson95=interval, passes=rate > threshold,
                                  confident_above_threshold=interval[0] > threshold))
        own = [c for c in cells if c['controller'] == 'policy']
        nominal, velocity = [c for c in own if c['speed'] == 0], [c for c in own if c['speed'] > 0]
        low, high = config['selection_speed_min'], config['selection_speed_max']
        selection = [c for c in velocity if c['speed'] >= low and (high < 0 or c['speed'] <= high)]
        eligible = float(config['nominal_success_threshold'])
        returns = policy['episode_return'][[i for i, (_, v) in enumerate(self.cells) if v > 0]]
        metrics.update({
            'eval/landing_success_rate': float(np.mean([c['success_rate'] for c in velocity])),
            'eval/episode_return': float(returns.mean()),
            'eval/landing_episode_count': self.episodes * len(velocity),
            'eval/worst_cell_success_rate': min(c['success_rate'] for c in selection),
            'eval/nominal_eligible': float(all(
                c['success_rate'] >= 1. if eligible == 1. else c['success_rate'] > eligible for c in nominal)),
            'eval/nominal/policy/success_rate': float(np.mean([c['success_rate'] for c in nominal])),
            'eval/worst_height_success_rate': min(
                float(np.mean([c['success_rate'] for c in selection if c['height'] == h])) for h in self.heights)})
        boundary = boundary_summary(cells, self.heights, self.speeds, threshold)
        for height, value in boundary.items():
            metrics[f"eval/boundary/h_{height.replace('.', 'p')}/max_speed_over_80"] = -1. if value is None else value
        self._write(Path(output_dir) / 'landing_evaluation', global_step, cells, boundary, policy)
        return metrics

    def _write(self, directory, step, cells, boundary, policy):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / 'boundary_history.json'
        history = json.loads(path.read_text()) if path.exists() else dict(threshold=self.threshold, first_pass={}, evaluations=[])
        for cell in cells:
            if cell['controller'] == 'policy' and cell['passes']:
                history['first_pass'].setdefault(f"h_{cell['height']:g}_v_{cell['speed']:g}", dict(step=step, **cell))
        summary = dict(step=step, threshold=self.threshold, heights=self.heights, speeds=self.speeds, cells=cells,
                       contiguous_speed_boundary=boundary, physics='MJX-Warp float32; validate the final model on CPU MuJoCo')
        history['evaluations'] = [e for e in history['evaluations'] if e['step'] != step] + [summary]
        path.write_text(json.dumps(history, indent=2))
        (directory / f'{step}_boundaries.json').write_text(json.dumps(summary, indent=2))
        (directory / 'config.json').write_text(json.dumps(dict(self.config, episodes_per_cell_per_controller=self.episodes), indent=2))
        with (directory / f'{step}_summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(cells[0]))
            writer.writeheader()
            writer.writerows(cells)
        with (directory / f'{step}_episodes.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['height', 'speed', 'episode', *policy])
            for i, (height, speed) in enumerate(self.cells):
                for j in range(self.episodes):
                    writer.writerow([height, speed, j, *(values[i, j] for values in policy.values())])
