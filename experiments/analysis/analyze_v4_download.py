"""Archive and analyze the completed v4 run and its downloaded checkpoints."""
import sys,json,re,shutil,zipfile
from pathlib import Path
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.analysis.report_landing_seed606 import load_policy,sha
from experiments.train_landing_seed606_v4 import stable_profile
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_seed606 import environment_config,ROBOT_DIR,verify_controller
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.environment import LocomotionEnv
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_evaluation import LandingEvaluationWrapper
from rl_x.environments.custom_mujoco.robot_locomotion.robots.silver_badger.robot_config import robot_config
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/artifacts/v4_completed_9376691_2026-10-27'
LOG=OUT/'out_and_err_landing_seed606_stable_v4_9376691.txt'

def merge(target,updates):
    for k,v in updates.items():
        if isinstance(v,dict):merge(target[k],v)
        else:target[k]=v

def make_v4_env():
    config=environment_config();merge(config,stable_profile()['environment']);config['nr_envs']=1
    robot=deepcopy(robot_config);robot.update(directory_path=ROBOT_DIR,spine_locked=False)
    env=LocomotionEnv(robot,'test',20261027,False,config,1);verify_controller(env.initial_mj_model)
    return LandingEvaluationWrapper(env,0)

def main():
    OUT.mkdir(parents=True,exist_ok=True);text=LOG.read_text(errors='replace');rows=[];current={};timestamp=None
    for line in text.splitlines():
        stamp=re.search(r'\[(\d\d-\d\d \d\d:\d\d:\d\d)\]',line)
        if stamp:timestamp=stamp.group(1)
        parts=line.split('│')
        if len(parts)>=4:
            k=parts[1].strip();value=parts[2].strip()
            try:current[k]=float(value)
            except ValueError:pass
        if '└' in line and current:
            if 'steps/nr_env_steps' in current:rows.append(dict(time=timestamp,metrics=current))
            current={}
    assert rows and rows[-1]['metrics']['steps/nr_env_steps']==30015488
    parsed=dict(source=LOG.name,source_sha256=sha(LOG),rows=rows,evaluations=[r for r in rows if 'eval/landing_success_rate' in r['metrics']],total_steps=30015488,first_log_time=re.search(r'\[(\d\d-\d\d \d\d:\d\d:\d\d)\]',text).group(1),last_metrics_time=rows[-1]['time'],errors={term:text.count(term) for term in ['Traceback (most recent call last)','CUDA out of memory','NaN']})
    (OUT/'parsed.json').write_text(json.dumps(parsed,indent=2))
    wrapper=make_v4_env();metadata={}
    try:
        for name in ['best','latest']:
            source=OUT/f'{name}.model';dest=source
            _,info=load_policy(dest,wrapper.env)
            with zipfile.ZipFile(dest) as z:info['metrics']=json.loads(z.read('checkpoint_metrics.json'))
            metadata[name]=info
    finally:wrapper.close()
    metadata['same_policy_weights']=metadata['best']['policy_weights_sha256']==metadata['latest']['policy_weights_sha256']
    (OUT/'checkpoints.json').write_text(json.dumps(metadata,indent=2));print('Rows',len(rows),'evaluations',len(parsed['evaluations']),'same policy weights',metadata['same_policy_weights'],flush=True)
    print('FINAL',json.dumps({k:v for k,v in rows[-1]['metrics'].items() if k.startswith('env_info/curriculum/') or 'flight_calf' in k or k in ['policy/std_dev','policy_ratio/approx_kl','policy_ratio/clip_fraction','eval/landing_success_rate','eval/worst_height_success_rate']}),flush=True)
if __name__=='__main__':main()
