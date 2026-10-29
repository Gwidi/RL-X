"""Archive v7 cluster checkpoints and parse its completed training log."""
import sys,json,re,shutil,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.analysis.report_landing_seed606 import load_policy,sha
from experiments.train_landing_seed606_v7 import make_noisy_env
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_evaluation import LandingEvaluationWrapper
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/artifacts/v7_completed_9400825_2026-10-29'
LOG=OUT/'out_and_err_landing_seed606_noisy_velocity_v7_9400825.txt'
def main():
 OUT.mkdir(parents=True,exist_ok=True);text=LOG.read_text(errors='replace');rows=[];current={};timestamp=None
 for line in text.splitlines():
  stamp=re.search(r'\[(\d\d-\d\d \d\d:\d\d:\d\d)\]',line)
  if stamp:timestamp=stamp.group(1)
  parts=line.split('│')
  if len(parts)>=4:
   try:current[parts[1].strip()]=float(parts[2].strip())
   except ValueError:pass
  if '└' in line and current:
   if 'steps/nr_env_steps' in current:rows.append(dict(time=timestamp,metrics=current))
   current={}
 assert rows[-1]['metrics']['steps/nr_env_steps']==30015488
 parsed=dict(source=LOG.name,source_sha256=sha(LOG),rows=rows,evaluations=[r for r in rows if 'eval/landing_success_rate' in r['metrics']],total_steps=30015488,first_log_time=re.search(r'\[(\d\d-\d\d \d\d:\d\d:\d\d)\]',text).group(1),last_metrics_time=rows[-1]['time'],errors={term:text.count(term) for term in ['Traceback (most recent call last)','CUDA out of memory','NaN']})
 (OUT/'parsed.json').write_text(json.dumps(parsed,indent=2))
 w=LandingEvaluationWrapper(make_noisy_env(),0);metadata={}
 try:
  for name in ['best','latest']:
   src=OUT/f'{name}.model';dst=src
   _,info=load_policy(dst,w.env)
   with zipfile.ZipFile(dst) as z:info['metrics']=json.loads(z.read('checkpoint_metrics.json'))
   metadata[name]=info
 finally:w.close()
 metadata['same_policy_weights']=metadata['best']['policy_weights_sha256']==metadata['latest']['policy_weights_sha256']
 (OUT/'checkpoints.json').write_text(json.dumps(metadata,indent=2))
 print('Log rows',len(rows),'same policy',metadata['same_policy_weights'],'errors',parsed['errors'])
 for name in ['best','latest']:
  m=metadata[name]['metrics'];print(name,{k:m.get(k) for k in ['nr_env_steps','eval/landing_success_rate','eval/worst_height_success_rate','eval/nominal/policy/success_rate','eval/angles/policy/success_rate','eval/velocity/policy/success_rate']})
if __name__=='__main__':main()
