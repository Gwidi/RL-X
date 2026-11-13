"""Continue V17 best: equal heights and horizontal speed up to 10 m/s."""
import hashlib,json,os,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.train_landing_inertial_v17 import continuation_profile
from experiments.train_landing_inertial_v16 import inertial_environment_config
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_seed606 import cli_flags,verify_controller
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.policy_velocity import velocity_observation_contract
ROOT=Path(__file__).resolve().parents[1]
PARENT_SHA='623fb041dea9cce63687c16ced46b575207b2f83d91055e17a587b1365a8b35c'


def speed10_profile():
    p=continuation_profile();e=p['environment']
    e.update(landing_balanced_replay=False,landing_uniform_speed10=True)
    e['domain_randomization']['initial_state']['energy_dissipation_curriculum']['max_lateral_velocity']=10.
    e['landing_evaluation'].update(max_lateral_velocity=10.,
        velocity_grid='0,1,2,3,4,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10',
        selection_speed_min=6.,selection_speed_max=10.,best_by_worst_cell=True)
    path=os.environ.get('LANDING_V19_CHECKPOINT') or str(ROOT/'modelsv17/best.model')
    p['runner'].update(load_model=str(Path(path).expanduser().resolve()),
        exp_name='landing_inertial_speed10_v19',
        run_name=os.environ.get('RUN_NAME') or 'inertial_speed10_v19_'+os.environ.get('SLURM_JOB_ID','local'),
        notes='V19 continues V17 best20971520(job9573646),aggregate109051904;30015488 extra;actor+critic+Adam+learned std;IMU only,no height;unchanged sensor/launch/PD/physics/reward/PPO;equal discrete heights0.75/1/1.25/1.5 independently of speed;40pct uniform speed5to8;30pct extension(20pct all uniform8to10,10pct all exact10);20pct uniform0to8;10pct nominal;all azimuths;best worst cell6to10+0.001global subject nominal eligibility;new episodes and counters,not exact simulation resume')
    return p


def verify_parent(path):
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(f'V17 best not found: {path}. Export LANDING_V19_CHECKPOINT="/full/path/to/v17/best.model" before sbatch.')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=PARENT_SHA:raise ValueError('V19 requires final V17 best (job9573646), not latest or V18')
    with zipfile.ZipFile(path) as z:
        saved=json.loads(z.read('config_environment.json'));m=json.loads(z.read('checkpoint_metrics.json'))
    expected=inertial_environment_config()
    for k in ('policy_inertial','imu_history_length','landing_braking','reward','timestep','control_type','spine_locked','landing_physics_sha256','landing_controller_sha256','landing_model_sha256'):
        if saved.get(k)!=expected.get(k):raise ValueError(f'Parent contract mismatch: {k}')
    if velocity_observation_contract(saved['policy_velocity'])!={'mode':'none'} or saved.get('policy_height',{}).get('enabled',False):raise ValueError('V19 requires IMU-only actor')
    if m['nr_env_steps']!=20971520 or not saved.get('landing_balanced_replay'):raise ValueError('Unexpected V17 parent')
    return dict(path=str(path.resolve()),sha256=digest,parent_run_steps=20971520,
        parent_aggregate_steps=109051904,additional_steps=30015488,aggregate_steps=139067392)


if __name__=='__main__':
    p=speed10_profile();parent=verify_parent(p['runner']['load_model']);fingerprint=verify_controller()
    if '--check-only' in sys.argv:print(json.dumps(dict(profile=p,parent=parent,fingerprint=fingerprint),indent=2))
    else:
        if any(a.startswith('--runner.load_model') for a in sys.argv[1:]):raise ValueError('Use LANDING_V19_CHECKPOINT to select V17 best')
        sys.argv=[sys.argv[0],*cli_flags(p),*sys.argv[1:]]
        from rl_x.runner.runner import Runner
        Runner().run()
