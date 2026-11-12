"""V18: V17 distribution plus noisy, delayed OptiTrack-compatible height."""
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.train_landing_inertial_v17 import continuation_profile as v17_profile, verify_parent
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_seed606 import cli_flags, verify_controller

ROOT = Path(__file__).resolve().parents[1]


def height_profile():
    p = v17_profile()
    # Separate parent selection from V17 so cluster exports cannot select it implicitly.
    path = os.environ.get('LANDING_V18_CHECKPOINT') or str(ROOT/'models/best.model')
    p['runner'].update(load_model=str(Path(path).expanduser().resolve()),
        exp_name='landing_inertial_height_v18',
        run_name=os.environ.get('RUN_NAME') or 'inertial_height_v18_'+os.environ.get('SLURM_JOB_ID','local'),
        notes=p['runner']['notes']+';V18 adds actor-only final height channel:base origin above fixed calibrated plane,scale3m;provisional noise sigma5mm,bias sigma5mm/episode,delay20ms;not measured OptiTrack calibration;351 actor inputs,critic unchanged;zero initial new weights/Adam moments preserve initial actor;not continuing a trained V17')
    p['environment']['policy_height'] = dict(enabled=True, noise_std_m=.005,
        bias_std_m=.005, delay_steps=1, plane_z_m=0.)
    return p


if __name__ == '__main__':
    p = height_profile()
    parent = verify_parent(p['runner']['load_model'])
    fingerprint = verify_controller()
    if '--check-only' in sys.argv:
        print(json.dumps(dict(profile=p, parent=parent, fingerprint=fingerprint,
            migration='append zero actor input weights and Adam moments; critic unchanged'), indent=2))
    else:
        if any(a.startswith('--runner.load_model') for a in sys.argv[1:]):
            raise ValueError('Select V16 best with export LANDING_V18_CHECKPOINT="/full/path/to/v16/best.model"')
        sys.argv = [sys.argv[0], *cli_flags(p), *sys.argv[1:]]
        from rl_x.runner.runner import Runner
        Runner().run()
