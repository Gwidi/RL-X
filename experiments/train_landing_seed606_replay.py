"""Second experiment: fixed PD, curriculum plus nominal full-height replay."""
from pathlib import Path
import json
import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_seed606 import training_profile, cli_flags, verify_controller

def replay_profile():
    profile = training_profile()
    profile['runner']['exp_name'] = 'landing_seed606_nominal_replay'
    profile['runner']['run_name'] = os.environ.get('RUN_NAME') or profile['runner']['run_name'].replace('seed606_fixed_pd_', 'seed606_nominal_replay_')
    profile['runner']['notes'] = 'Fresh weights; 30% nominal replay (half top height, half uniform range); 70% original curriculum; unchanged PD/physics/reward/PPO'
    curriculum = profile['environment']['domain_randomization']['initial_state']['energy_dissipation_curriculum']
    curriculum.update(nominal_replay_probability=.30, nominal_replay_top_probability=.50)
    return profile

if __name__ == '__main__':
    fingerprint = verify_controller()
    profile = replay_profile()
    if '--check-only' in sys.argv:
        print(json.dumps(dict(profile=profile, fingerprint=fingerprint), indent=2))
    else:
        sys.argv = [sys.argv[0], *cli_flags(profile), *sys.argv[1:]]
        print('Seed606 nominal replay v2; fresh weights; verified model:', fingerprint['combined_sha256'], flush=True)
        from rl_x.runner.runner import Runner
        Runner().run()
