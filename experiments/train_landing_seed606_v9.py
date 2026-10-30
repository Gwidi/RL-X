"""V9: continue v8, fixed 70/20/10 speed-band/easy/nominal mixture."""
import json,os,sys,hashlib,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.train_landing_seed606_v8 import velocity_only_profile
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_seed606 import cli_flags,verify_controller
ROOT=Path(__file__).resolve().parents[1]
def focused_profile():
    p=velocity_only_profile();c=p['environment']['domain_randomization']['initial_state']['energy_dissipation_curriculum']
    c.update(height_then_velocity_enabled=False,focused_velocity_enabled=True,max_lateral_velocity=4.,focused_speed_min=3.,focused_speed_max=4.,focused_target_probability=.7,focused_nominal_probability=.1,focused_top_probability=.25,focused_boundary_probability=.5)
    p['environment']['landing_evaluation'].update(velocity_grid='0,0.5,1,2,3,3.25,3.5,3.75,4,5',frequency_multiplier=64,selection_speed_min=3.,selection_speed_max=4.)
    checkpoint=os.environ.get('LANDING_V9_CHECKPOINT',str(ROOT/'models/best.model'))
    p['runner'].update(load_model=str(Path(checkpoint).resolve()),exp_name='landing_seed606_focused_velocity_v9',run_name=os.environ.get('RUN_NAME') or 'seed606_focused_velocity_v9_'+os.environ.get('SLURM_JOB_ID','local'),notes='V9 continues v8 policy+critic+Adam state; new run counters and environments; unchanged reward,PD,physics,sensor,PPO; 35pctexact4+35pct speedU[3,4],20pct disk<=3,10pctzero; heights1to3 with25pctexact3; 5mps evaluationonly; checkpoint selection focuses3to4')
    return p

def focused_environment_config():
    from experiments.train_landing_seed606_v8 import velocity_only_environment_config
    cfg=velocity_only_environment_config()
    def merge(a,b):
        for k,v in b.items():
            if isinstance(v,dict):merge(a[k],v)
            else:a[k]=v
    merge(cfg,focused_profile()["environment"]);return cfg

def make_focused_env(mode="test"):
    from copy import deepcopy
    from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_seed606 import ROBOT_DIR
    from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.environment import LocomotionEnv
    from rl_x.environments.custom_mujoco.robot_locomotion.robots.silver_badger.robot_config import robot_config
    cfg=focused_environment_config();cfg["nr_envs"]=1
    robot=deepcopy(robot_config);robot.update(directory_path=ROBOT_DIR,spine_locked=False)
    env=LocomotionEnv(robot,mode,20261030,False,cfg,1);verify_controller(env.initial_mj_model);return env

def verify_parent(path):
    path=Path(path)
    if not path.is_file():raise FileNotFoundError('Set LANDING_V9_CHECKPOINT to the downloaded v8 best.model or original cluster v8 checkpoint')
    with zipfile.ZipFile(path) as z:
        e=json.loads(z.read('config_environment.json'));metrics=json.loads(z.read('checkpoint_metrics.json'))
    if not e['domain_randomization']['initial_state']['energy_dissipation_curriculum'].get('height_then_velocity_enabled') or metrics['nr_env_steps']!=30015488:raise ValueError('V9 requires the completed v8 checkpoint, not a smoke model or another version')
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),parent_steps=metrics['nr_env_steps'])
if __name__=='__main__':
    p=focused_profile();parent=verify_parent(p['runner']['load_model']);fingerprint=verify_controller()
    if '--check-only' in sys.argv:print(json.dumps(dict(profile=p,parent=parent,fingerprint=fingerprint),indent=2))
    else:
        sys.argv=[sys.argv[0],*cli_flags(p),*sys.argv[1:]]
        from rl_x.runner.runner import Runner
        Runner().run()
