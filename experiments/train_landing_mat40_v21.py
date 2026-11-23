"""Continue V20 on a mat calibrated for 40 mm of impactor deflection."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import train_landing_handoff_v20 as v20
from experiments.train_landing_inertial_v16 import inertial_environment_config, make_inertial_env
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_mat import read_landing_mat
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_seed606 import cli_flags, verify_controller

PARENT_RUNS = {
    'b6cf8114cd1f50eaa8333745172d2e491f4576497b3901612e3cb2861733d60c': '9684297',
    '48c00d297d5d103f477b5e84258549e6ce8252412ba5f7b9ff201b184caee77c': '9697297',
}
MAT_PROFILE = 'rl_x/environments/custom_mujoco/robot_locomotion/robots/silver_badger/data/landing_mat_40mm.json'
PARENT_STEPS = 30_015_488


def training_profile(parent, weights):
    mat, digest = read_landing_mat(MAT_PROFILE)
    profile = v20.handoff_profile(parent, weights)
    profile['environment']['landing_mat'] = dict(profile=MAT_PROFILE, sha256=digest, calf_contact=False)
    profile['runner'].update(
        exp_name='landing_walk_mat40_v21',
        run_name=os.environ.get('RUN_NAME') or 'landing_walk_mat40_v21_' + os.environ.get('SLURM_JOB_ID', 'local'),
        notes=('V21 continues final V20: actor, critic, learned handoff and Adam. '
               '40 mm impactor deflection with the same calibrated surface in training and evaluation. '
               'Both calf-floor pairs per leg removed. Frozen E19 and the current V20 handoff gate. '
               '30,015,488 new transitions.'),
    )
    return profile, mat


def verify_parent(parent, profile):
    with zipfile.ZipFile(parent) as archive:
        saved = json.loads(archive.read('config_environment.json'))
        metrics = json.loads(archive.read('checkpoint_metrics.json'))
    expected = inertial_environment_config()
    for key in ('landing_braking', 'reward', 'policy_inertial', 'imu_history_length',
                'landing_physics_sha256', 'landing_controller_sha256', 'landing_model_sha256'):
        if saved.get(key) != expected.get(key):
            raise ValueError(f'Parent task mismatch: {key}')
    for key in ('enabled', 'walk_sha256', 'walk_speed_limit'):
        if saved['landing_handoff'].get(key) != profile['environment']['landing_handoff'][key]:
            raise ValueError(f'Parent handoff mismatch: {key}')
    if metrics['nr_env_steps'] != PARENT_STEPS:
        raise ValueError('V21 requires the final V20 latest checkpoint')


def check_model(profile, mat):
    import numpy as np
    from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_safety import LandingContactMap

    config = inertial_environment_config()
    for key in ('landing_handoff', 'landing_mat'):
        config[key] = profile['environment'][key]
    env = make_inertial_env(config=config)
    try:
        model = env.initial_mj_model
        floor = model.geom('floor').id
        contacts = LandingContactMap(model)
        calves = sorted(contacts.calf_geom_ids | contacts.lower_calf_geom_ids)
        floor_pairs = (model.pair_geom1 == floor) | (model.pair_geom2 == floor)
        paired = set(model.pair_geom1[floor_pairs]) | set(model.pair_geom2[floor_pairs])
        body = {model.geom(name).id for name in ('trunk_box', 'rear_mesh')}
        body.update(model.geom(f'{leg}_{part}_housing').id
                    for leg in ('RL', 'RR', 'FR', 'FL') for part in ('hip', 'thigh'))
        assert paired - {floor} == contacts.foot_geom_ids | body
        assert not np.any(model.geom_contype[calves]) and not np.any(model.geom_conaffinity[calves])
        for array, key in ((model.pair_solref, 'solref'), (model.pair_solimp, 'solimp')):
            np.testing.assert_allclose(array[floor_pairs], np.broadcast_to(mat[key], array[floor_pairs].shape))
    finally:
        env.close()


def main():
    default_parent = 'modelsv20/models/latest.model'
    if not (v20.ROOT / default_parent).is_file():
        default_parent = 'modelsv20/latest.model'
    parent = v20.input_path('LANDING_V21_CHECKPOINT', default_parent, *PARENT_RUNS)
    parent_sha = hashlib.sha256(parent.read_bytes()).hexdigest()
    parent_job = PARENT_RUNS[parent_sha]
    walker = v20.input_path('LANDING_V20_WALKER', 'E19SBGppo_hfield_bunker_ruins_foot_velocity.model', v20.WALKER_SHA)
    weights = v20.ROOT / 'runs' / 'handoff_assets' / f'e19_{v20.WALKER_SHA[:12]}.npz'
    profile, mat = training_profile(parent, weights)
    profile['runner']['notes'] += f' V20 source job {parent_job}, latest SHA256 {parent_sha}.'
    verify_parent(parent, profile)
    verify_controller()
    weights.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, v20.__file__, '--export-e19', str(walker), str(weights)],
                   check=True, env={**os.environ, 'JAX_PLATFORMS': 'cpu'})
    if '--check-only' in sys.argv:
        check_model(profile, mat)
        print(json.dumps(dict(profile=profile, parent_job=parent_job, parent_sha256=parent_sha, mat=mat), indent=2))
        return
    if any(arg.startswith(('--runner.load_model', '--environment.landing_mat', '--environment.landing_handoff'))
           for arg in sys.argv[1:]):
        raise ValueError('Use the V21 profile and export LANDING_V21_CHECKPOINT and LANDING_V20_WALKER')
    sys.argv = [sys.argv[0], *cli_flags(profile), *sys.argv[1:]]
    from rl_x.runner.runner import Runner
    Runner().run()


if __name__ == '__main__':
    main()
