"""V4: stable integration; revised task h<=2.2m and roll/pitch curriculum to 10deg."""
import hashlib,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.train_landing_seed606_factorized import factorized_profile
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_seed606 import verify_controller,cli_flags
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_physics import read_landing_physics
ROOT=Path(__file__).resolve().parents[1]
PHYSICS='rl_x/environments/custom_mujoco/robot_locomotion/robots/silver_badger/data/landing_physics_v4.json'

def stable_profile():
    profile=factorized_profile();physics,digest=read_landing_physics(PHYSICS)
    profile['environment'].update(timestep=physics['timestep'],landing_physics_profile=PHYSICS,landing_physics_sha256=digest,landing_motor_diagnostics=True)
    curriculum=profile['environment']['domain_randomization']['initial_state']['energy_dissipation_curriculum']
    curriculum.update(target_height=2.2,max_angle_deg=10.)
    profile['environment']['landing_evaluation'].update(heights='1.0,1.5,2.0,2.2',max_angle_deg=10.)
    profile['runner'].update(exp_name='landing_seed606_stable_v4',run_name=os.environ.get('RUN_NAME') or ('seed606_stable_v4_'+os.environ.get('SLURM_JOB_ID','local')),load_model='',notes='V4 revised task 2026-10-27: implicitfast 2.5ms, control50Hz, originalPD606; fresh weights; v3 factorization, height1to2.2m, maxangles10deg, vxy3peraxis, std0.15 unchanged; motor diagnostics enabled')
    return profile

if __name__=='__main__':
    fingerprint=verify_controller();profile=stable_profile()
    if '--check-only' in sys.argv:
        print(json.dumps(dict(profile=profile,fingerprint=fingerprint),indent=2))
    else:
        sys.argv=[sys.argv[0],*cli_flags(profile),*sys.argv[1:]]
        print('V4 fixed controller verified; physics profile SHA256:',profile['environment']['landing_physics_sha256'],flush=True)
        from rl_x.runner.runner import Runner
        Runner().run()
