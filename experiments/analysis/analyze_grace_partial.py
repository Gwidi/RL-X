"""Versioned partial2s/3s snapshots; frozen paired protocol shared with1s/v9."""
import argparse,json,re,shutil,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.analysis.analyze_v8_download import grid,V8Experiment
from experiments.analysis.evaluate_seed606_v3 import Experiment
from experiments.analysis.report_landing_seed606 import load_policy,sha
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_evaluation import LandingEvaluationWrapper
ROOT=Path(__file__).resolve().parents[2];OLD=ROOT/'docs/artifacts/v9_completed_9412383_2026-11-02'
def factory(grace):
 if grace==2:
  from experiments.train_landing_seed606_v10_grace2s import make_grace2s_env
  env=make_grace2s_env()
 else:
  from experiments.train_landing_seed606_v10_grace3s import make_grace3s_env
  env=make_grace3s_env()
 class PartialExperiment(V8Experiment):
  def __init__(self):Experiment.__init__(self,LandingEvaluationWrapper(env,0))
 return PartialExperiment()
def main():
 p=argparse.ArgumentParser();p.add_argument('--grace',type=int,choices=[2,3],required=True);g=p.parse_args().grace;job={2:9441591,3:9441491}[g]
 source=ROOT/f'models{g}s'
 with zipfile.ZipFile(source/'latest.model') as z:step=json.loads(z.read('checkpoint_metrics.json'))['nr_env_steps']
 out=ROOT/f'docs/artifacts/v10_grace{g}s_partial_{job}_step{step}_2026-11-03';out.mkdir(exist_ok=True)
 for n in ['best','latest']:
  dest=out/f'{n}.model'
  if dest.exists():assert sha(dest)==sha(source/f'{n}.model')
  else:shutil.copy2(source/f'{n}.model',dest)
 log=ROOT/f'out_and_err_landing_seed606_grace{g}s_v10_{job}.txt';shutil.copy2(log,out/log.name);rows=[];current={}
 for line in log.read_text().splitlines():
  parts=line.split('│')
  if len(parts)>=4:
   try:current[parts[1].strip()]=float(parts[2].strip())
   except ValueError:pass
  if '└' in line and current:
   if 'steps/nr_env_steps' in current:rows.append(dict(step=round(current['steps/nr_env_steps']),metrics=current))
   current={}
 meta={};actions={};ex=factory(g)
 try:
  for n in ['best','latest']:
   actions[n],meta[n]=load_policy(out/f'{n}.model',ex.wrapper.env)
   with zipfile.ZipFile(out/f'{n}.model') as z:meta[n]['metrics']=json.loads(z.read('checkpoint_metrics.json'))
   assert meta[n]['saved_environment']['reward']['post_touchdown_motion_grace_time']==g
 finally:ex.close()
 same=meta['best']['policy_weights_sha256']==meta['latest']['policy_weights_sha256'];meta['same_policy_weights']=same
 (out/'checkpoints.json').write_text(json.dumps(meta,indent=2));(out/'history.json').write_text(json.dumps(dict(evaluations=[grid(r['metrics'],r['step']) for r in rows if 'eval/landing_success_rate' in r['metrics']],checkpoint_grids={n:grid(meta[n]['metrics'],meta[n]['metrics']['evaluation_nr_env_steps']) for n in ['best','latest']}),indent=2))
 errors={s:log.read_text().count(s) for s in ['Traceback (most recent call last)','CUDA out of memory','NaN']}
 (out/'parsed.json').write_text(json.dumps(dict(rows=rows,last_log_step=rows[-1]['step'],checkpoint_step=step,partial=True,errors=errors),indent=2))
 cases=json.loads((OLD/'paired_protocol.json').read_text())['cases'];reference={r['case']['id']:r for r in json.loads((OLD/'paired_results.json').read_text()) if r['controller']=='best'};results=[]
 (out/'protocol.json').write_text(json.dumps(dict(cases=cases,source=str(OLD/'paired_protocol.json'),partial=True),indent=2));print('grace',g,'log',rows[-1]['step'],'checkpoint',step,'same',same,flush=True)
 for n in (['best'] if same else ['best','latest']):
  ex=factory(g)
  try:
   for i,c in enumerate(cases):
    r=ex.trial(c,actions[n],trace=c['height']==3 and c['speed']==4);r['controller']=n;assert r['scenario_hash']==reference[c['id']]['scenario_hash']
    if 'trajectory' in r:
     trajectory=r.pop('trajectory');(out/f'{n}_trace_{c["id"]}.json').write_text(json.dumps(trajectory))
    results.append(r)
    if (i+1)%64==0:print(g,n,i+1,flush=True)
  finally:ex.close()
 (out/'paired_results.json').write_text(json.dumps(results,indent=2))
 print(out,flush=True)
if __name__=='__main__':main()
