"""V22 from scratch: the V22 task and drive limits without V19 weights."""

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.train_landing_real_drives_v22 import check_model, real_drive_profile
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_seed606 import cli_flags, verify_controller

TOTAL_STEPS = 90_046_464


def fresh_profile():
    profile = real_drive_profile('')
    profile['algorithm']['total_timesteps'] = TOTAL_STEPS
    profile['runner'].update(
        load_model='', exp_name='landing_real_drives_fresh_v22',
        run_name=os.environ.get('RUN_NAME') or 'landing_real_drives_fresh_v22_' + os.environ.get('SLURM_JOB_ID', 'local'),
        notes=('V22 from scratch: new actor, critic, Adam and std 0.15; 90,046,464 transitions. '
               'Same 16 Nm joint limits, spine +-0.785 rad, heights, speed mixture, reward, IMU contract and '
               'evaluation as the V22 warm start, so only initialisation and budget differ.'))
    return profile


def main():
    profile = fresh_profile()
    fingerprint = verify_controller()
    if '--check-only' in sys.argv:
        print(json.dumps(dict(profile=profile, fresh_initialization=True, fingerprint=fingerprint,
                              model=check_model(profile)), indent=2))
        return
    if any(arg.startswith(('--runner.load_model', '--environment.landing_actuators')) for arg in sys.argv[1:]):
        raise ValueError('This V22 arm trains from scratch with the fixed actuator profile')
    sys.argv = [sys.argv[0], *cli_flags(profile), *sys.argv[1:]]
    from rl_x.runner.runner import Runner
    Runner().run()


if __name__ == '__main__':
    main()
