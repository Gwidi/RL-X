"""V15: extend the learned V14 safety envelope to horizontal norm8m/s."""
import hashlib,json,os,sys,zipfile
from pathlib import Path
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.train_landing_braking_v14 import fresh_moving_profile,fresh_moving_environment_config
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_seed606 import cli_flags,verify_controller,ROBOT_DIR
ROOT=Path(__file__).resolve().parents[1]
ALLOWED_PARENTS={'03c9483f01f6a6bb3bdd1d7507222b865fa93cc1c77a392704611887e0c854fb': {'checkpoint': 'best', 'parent_steps': 59375616}, '47a587180bd4456d536e368cac201050154e8f7789899e022078c8ba12bea6ca': {'checkpoint': 'latest', 'parent_steps': 60030976}}

def speed_limits_profile():
    p=fresh_moving_profile()
    c=p['environment']['domain_randomization']['initial_state']['energy_dissipation_curriculum']
    c.update(max_lateral_velocity=8.,focused_speed_min=5.,focused_speed_max=8.)
    p['environment']['landing_evaluation'].update(velocity_grid='0,1,2,3,4,5,5.5,6,6.5,7,7.5,8',selection_speed_min=1.,selection_speed_max=8.)
    path=os.environ.get('LANDING_V15_CHECKPOINT') or str(ROOT/'docs/artifacts/v14_continue_completed_9465726_2026-11-06/best.model')
    p['runner'].update(load_model=str(Path(path).resolve()),exp_name='landing_speed_limits_v15',
        run_name=os.environ.get('RUN_NAME') or 'speed_limits_v15_'+os.environ.get('SLURM_JOB_ID','local'),
        notes='V15 continues completed extended V14 best by default, policy+critic+Adam+learned std;30M extra;ONLY training domain/eval speed range changes;70pct target5to8(17.5pct all episodes exact8,52.5pct uniform5to8),20pct disk<=5,10pct zero;heights0.75to1.5,angles0;unchanged reward/PD/physics/sensors/PPO/safety criterion;fixed mixture,not adaptive frontier;eval0to8,0.5 increments above5;best selection1to8;no flip task')
    return p

def verify_parent(path):
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(f'V15 parent not found: {path}. Export LANDING_V15_CHECKPOINT="/full/path/to/completed/v14/continuation/models/best.model"')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest not in ALLOWED_PARENTS:raise ValueError('V15 requires the downloaded final best/latest of V14 continuation (job9465726), not the first30M or a partial model')
    with zipfile.ZipFile(path) as z:saved=json.loads(z.read('config_environment.json'));metrics=json.loads(z.read('checkpoint_metrics.json'))
    expected=fresh_moving_environment_config()
    for key in ('landing_braking','reward','policy_velocity','domain_randomization','landing_physics_sha256','landing_controller_sha256','landing_model_sha256','imu_history_length','timestep','control_type','spine_locked'):
        if saved.get(key)!=expected.get(key):raise ValueError(f'Parent differs from V14 in {key}')
    return dict(path=str(path.resolve()),sha256=digest,**ALLOWED_PARENTS[digest],additional_steps=30015488,
                aggregate_steps=ALLOWED_PARENTS[digest]['parent_steps']+30015488)

def speed_limits_environment_config():
    cfg=fresh_moving_environment_config()
    def merge(a,b):
        for k,v in b.items():
            if isinstance(v,dict):merge(a.setdefault(k,{}),v)
            else:a[k]=v
    merge(cfg,speed_limits_profile()['environment']);return cfg

def make_speed_limits_env(mode='test'):
    from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.environment import LocomotionEnv
    from rl_x.environments.custom_mujoco.robot_locomotion.robots.silver_badger.robot_config import robot_config
    cfg=speed_limits_environment_config();cfg['nr_envs']=1;robot=deepcopy(robot_config);robot.update(directory_path=ROBOT_DIR,spine_locked=False)
    env=LocomotionEnv(robot,mode,20261106,False,cfg,1);verify_controller(env.initial_mj_model);return env

if __name__=='__main__':
    p=speed_limits_profile();parent=verify_parent(p['runner']['load_model']);fingerprint=verify_controller()
    if '--check-only' in sys.argv:print(json.dumps(dict(profile=p,parent=parent,fingerprint=fingerprint),indent=2))
    else:
        if any(a.startswith('--runner.load_model') for a in sys.argv[1:]):raise ValueError('Select the validated parent using LANDING_V15_CHECKPOINT')
        sys.argv=[sys.argv[0],*cli_flags(p),*sys.argv[1:]]
        from rl_x.runner.runner import Runner
        Runner().run()
