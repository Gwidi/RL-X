"""Completed grace1s: checkpoint grids and paired re-evaluation on frozen v8 cases."""
import json,re,shutil,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.analysis.analyze_v8_download import grid,V8Experiment
from experiments.analysis.evaluate_seed606_v3 import Experiment
from experiments.train_landing_seed606_v10_grace1s import make_grace1s_env
from experiments.analysis.report_landing_seed606 import load_policy,sha
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_evaluation import LandingEvaluationWrapper
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'docs/artifacts/v10_grace1s_completed_9439587_2026-11-03';OLD=ROOT/'docs/artifacts/v9_completed_9412383_2026-11-02'
class Grace1sExperiment(V8Experiment):
 def __init__(self):Experiment.__init__(self,LandingEvaluationWrapper(make_grace1s_env(),0))
def main():
 OUT.mkdir(exist_ok=True)
 for name in ['best','latest']:
  src=ROOT/'models'/f'{name}.model';dst=OUT/src.name
  if dst.exists():assert sha(dst)==sha(src)
  else:shutil.copy2(src,dst)
 log=ROOT/'out_and_err_landing_seed606_grace1s_v10_9439587.txt';shutil.copy2(log,OUT/log.name);text=log.read_text();rows=[];current={}
 for line in text.splitlines():
  parts=line.split('│')
  if len(parts)>=4:
   try:current[parts[1].strip()]=float(parts[2].strip())
   except ValueError:pass
  if '└' in line and current:
   if 'steps/nr_env_steps' in current:rows.append(dict(step=round(current['steps/nr_env_steps']),metrics=current))
   current={}
 meta={};actions={};ex=Grace1sExperiment()
 try:
  for name in ['best','latest']:
   actions[name],meta[name]=load_policy(OUT/f'{name}.model',ex.wrapper.env)
   with zipfile.ZipFile(OUT/f'{name}.model') as z:meta[name]['metrics']=json.loads(z.read('checkpoint_metrics.json'))
 finally:ex.close()
 meta['same_policy_weights']=meta['best']['policy_weights_sha256']==meta['latest']['policy_weights_sha256']
 (OUT/'checkpoints.json').write_text(json.dumps(meta,indent=2))
 evaluations=[grid(r['metrics'],r['step']) for r in rows if 'eval/landing_success_rate' in r['metrics']]
 for name in ['best','latest']:
  step=meta[name]['metrics']['evaluation_nr_env_steps'];evaluations=[e for e in evaluations if e['step']!=step]+[grid(meta[name]['metrics'],step)]
 evaluations.sort(key=lambda e:e['step'])
 (OUT/'history.json').write_text(json.dumps(dict(evaluations=evaluations,checkpoint_grids={n:grid(meta[n]['metrics'],meta[n]['metrics']['evaluation_nr_env_steps']) for n in ['best','latest']},source='Intermediate grids reconstructed from printed log; best/latest exact checkpoint metrics; no original evaluation CSV downloaded'),indent=2))
 (OUT/'parsed.json').write_text(json.dumps(dict(rows=rows,total_steps=rows[-1]['step'],errors={s:text.count(s) for s in ['Traceback (most recent call last)','CUDA out of memory','NaN']}),indent=2))
 print('Restored both; equal weights:',meta['same_policy_weights'],flush=True)
 cases=json.loads((OLD/'paired_protocol.json').read_text())['cases'];baseline={r['case']['id']:r for r in json.loads((OLD/'paired_results.json').read_text()) if r['controller']=='best'};results=[]
 (OUT/'paired_protocol.json').write_text(json.dumps(dict(cases=cases,source=str(OLD/'paired_protocol.json'),note='Frozen diagnostic cases reused across variants; paired comparison, not a new held-out set. Grace1s best/latest re-evaluated once if weights identical.'),indent=2))
 for name in (['best'] if meta['same_policy_weights'] else ['best','latest']):
  ex=Grace1sExperiment()
  try:
   for i,c in enumerate(cases):
    r=ex.trial(c,actions[name],trace=c['height']==3. and c['speed']==4.);r['controller']=name;
    if 'trajectory' in r:
     trajectory=r.pop('trajectory');(OUT/f'trace_{c["id"]}.json').write_text(json.dumps(trajectory))
     import numpy as np
     r['base_horizontal_travel']=float(np.linalg.norm(np.array(trajectory[-1]['qpos'][:2])-trajectory[0]['qpos'][:2]))
    r['time_since_first_support']=r['end_time']-r['first_support'] if r['first_support'] is not None else None; assert r['scenario_hash']==baseline[c['id']]['scenario_hash'];results.append(r)
    if (i+1)%64==0:print(name,i+1,flush=True)
  finally:ex.close()
 (OUT/'paired_results.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
