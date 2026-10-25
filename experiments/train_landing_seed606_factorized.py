"""V3: independent height sampling, separate angle/velocity curricula."""
from pathlib import Path
import json
import os
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_seed606 import training_profile,cli_flags,verify_controller

def factorized_profile():
    profile = training_profile()
    profile['runner']['exp_name'] = 'landing_seed606_factorized'
    profile['runner']['run_name'] = os.environ.get('RUN_NAME') or profile['runner']['run_name'].replace('seed606_fixed_pd_', 'seed606_factorized_')
    profile['runner']['notes'] = 'V3 fresh weights; h independent; groups 20/20/20/40; separate angle/velocity EMA; fixed PD/physics/reward/observations'
    c = profile['environment']['domain_randomization']['initial_state']['energy_dissipation_curriculum']
    c.update(factorized_enabled=True, replay_easy_probability=0.,nominal_replay_probability=0.,
             factorized_top_probability=.25,factorized_start_angle_deg=1.,factorized_start_velocity=.5,
             factorized_nominal_probability=.2,factorized_angles_probability=.2,
             factorized_velocity_probability=.2,factorized_combined_probability=.4)
    profile['environment']['landing_evaluation'].update(factorized_conditions=True,require_nominal_for_best=True,evaluate_final=True)
    return profile

if __name__=='__main__':
    fingerprint=verify_controller();profile=factorized_profile()
    if '--check-only' in sys.argv:
        print(json.dumps(dict(profile=profile,fingerprint=fingerprint),indent=2))
    else:
        sys.argv=[sys.argv[0],*cli_flags(profile),*sys.argv[1:]]
        print('Seed606 V3 factorized: fresh weights, fixed controller verified:',fingerprint['combined_sha256'],flush=True)
        from rl_x.runner.runner import Runner
        Runner().run()
