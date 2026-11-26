"""Rewrites a V22 checkpoint for mjx.symmetric_pose, so that it commands the same joint targets as before.

The nominal pose is the action origin (target = nominal + 0.6 * action) and the zero of the joint observations. Moving it
is absorbed exactly by the biases of the first layer (actor, critic) and of the actor's output layer. What cannot be
kept: actions are still clipped to +-1, now around the new origin. Optimizer state, std and step count are unchanged.

    python experiments/convert_parent_symmetric_pose.py --input latest.model --output symmetric.model [--front-hip 0.009]
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import orbax.checkpoint

from experiments.analysis.check_mjx_parity import make_mjx_env, merge, v22_config
from rl_x.environments.custom_mujoco.robot_locomotion.mjx_landing.environment import symmetric_pose

EXTRA = ('config_algorithm.json', 'config_environment.json', 'checkpoint_metrics.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--front-hip', type=float, default=.2687, help='right front hip of the symmetric pose, rad')
    args = parser.parse_args()
    with zipfile.ZipFile(args.input) as archive:
        texts = {name: archive.read(name).decode() for name in EXTRA}
    metrics = json.loads(texts['checkpoint_metrics.json'])
    if metrics['mjx'].get('symmetric_pose'):
        raise SystemExit('This checkpoint already uses the symmetric pose')
    env = make_mjx_env(merge(v22_config(), {'mjx': {k: v for k, v in metrics['mjx'].items()
                                                    if k not in ('constraints_per_env', 'contacts_per_env')}}))
    old = np.asarray(env.nominal, np.float64)
    new = symmetric_pose(old, args.front_hip)
    size = env.single_observation_space.shape[0]
    # Old observation = new observation + shift.
    shift = np.zeros(size)
    shift[np.asarray(env.index.joint_positions_obs_idx)] = (new - old) / 3.14
    shift[np.asarray(env.index.joint_previous_actions_obs_idx)] = (new - old) / env.action_scale / 10.
    with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as target:
        with zipfile.ZipFile(args.input) as archive:
            archive.extractall(directory)
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        tree = checkpointer.restore(directory)
        for network, indices in (('policy', env.policy_observation_indices), ('critic', env.critic_observation_indices)):
            layers = tree[network]['params']['params']
            first = layers['Dense_0']
            kernel = np.asarray(first['kernel'])
            assert kernel.shape[0] == len(indices), (network, kernel.shape, len(indices))
            first['bias'] = np.asarray(first['bias']) + (shift[np.asarray(indices)] @ kernel).astype(kernel.dtype)
        output = tree['policy']['params']['params']['Dense_3']
        output['bias'] = np.asarray(output['bias']) + ((old - new) / env.action_scale).astype(np.float32)
        checkpointer.save(f'{target}/tree', tree)
        metrics['mjx']['symmetric_pose'] = True
        metrics['mjx']['symmetric_front_hip'] = args.front_hip
        metrics['converted_to_symmetric_pose_from'] = Path(args.input).name
        texts['checkpoint_metrics.json'] = json.dumps(metrics)
        for name, text in texts.items():
            Path(f'{target}/tree/{name}').write_text(text)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(Path(f'{target}/tree').rglob('*')):
                archive.write(path, path.relative_to(f'{target}/tree'))
    print('written', args.output, '| largest output shift (action units):', round(float(np.abs((old - new) / env.action_scale).max()), 3))


if __name__ == '__main__':
    main()
