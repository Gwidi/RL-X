"""Archive completed v8, reconstruct logged grid history, and test new seeds."""
import json,re,shutil,zipfile,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from experiments.train_landing_seed606_v8 import make_velocity_only_env
from experiments.analysis.report_landing_seed606 import load_policy,sha
from experiments.analysis.evaluate_seed606_v3 import Experiment
from experiments.analysis.evaluate_v6_download import V6Experiment
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_evaluation import LandingEvaluationWrapper
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_velocity_grid import wilson,boundary_summary
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'docs/artifacts/v8_completed_9404358_2026-10-30'
class V8Experiment(V6Experiment):
 def __init__(self):Experiment.__init__(self,LandingEvaluationWrapper(make_velocity_only_env(),0))
def grid(metrics,step):
 cells=[]
 for k,v in metrics.items():
  match=re.fullmatch(r'eval/grid/v_([\dp]+)/(?P<condition>nominal|perturbed)/(?P<controller>policy|zero)/h_([\dp]+)/success_rate',k)
  if not match:continue
  speed=float(match[1].replace('p','.'));height=float(match[4].replace('p','.'));n=round(metrics[k.replace('success_rate','episode_count')]);wins=round(v*n)
  cells.append(dict(height=height,speed=speed,controller=match['controller'],successes=wins,episodes=n,success_rate=wins/n,wilson95=wilson(wins,n),passes=wins/n>.8))
 heights=sorted(set(c['height'] for c in cells));speeds=sorted(set(c['speed'] for c in cells))
 return dict(step=step,cells=cells,heights=heights,speeds=speeds,contiguous_speed_boundary=boundary_summary(cells,heights,speeds,.8))
def main():
 OUT.mkdir(exist_ok=True)
 for name in ['best','latest']:
  src=ROOT/'models'/f'{name}.model';dst=OUT/src.name
  if dst.exists():assert sha(src)==sha(dst)
  else:shutil.copy2(src,dst)
 log=ROOT/'out_and_err_landing_seed606_velocity_only_v8_9404358.txt';shutil.copy2(log,OUT/log.name);text=log.read_text();rows=[];current={}
 for line in text.splitlines():
  parts=line.split('│')
  if len(parts)>=4:
   try:current[parts[1].strip()]=float(parts[2].strip())
   except ValueError:pass
  if '└' in line and current:
   if 'steps/nr_env_steps' in current:rows.append(dict(step=round(current['steps/nr_env_steps']),metrics=current))
   current={}
 ex=V8Experiment();meta={};policies={}
 try:
  for name in ['best','latest']:
   policies[name],meta[name]=load_policy(OUT/f'{name}.model',ex.wrapper.env)
   with zipfile.ZipFile(OUT/f'{name}.model') as z:meta[name]['metrics']=json.loads(z.read('checkpoint_metrics.json'))
 finally:ex.close()
 same=meta['best']['policy_weights_sha256']==meta['latest']['policy_weights_sha256'];meta['same_policy_weights']=same
 (OUT/'checkpoints.json').write_text(json.dumps(meta,indent=2))
 evaluations=[grid(r['metrics'],r['step']) for r in rows if 'eval/landing_success_rate' in r['metrics']]
 exact=grid(meta['latest']['metrics'],meta['latest']['metrics']['evaluation_nr_env_steps']);evaluations=[e for e in evaluations if e['step']!=exact['step']]+[exact]
 first={}
 for e in evaluations:
  for c in e['cells']:
   if c['controller']=='policy' and c['passes']:first.setdefault(f'h_{c["height"]:g}_v_{c["speed"]:g}',dict(step=e['step'],**c))
 result=dict(source=str(OUT),latest=exact,first_pass=first,evaluations=evaluations,checkpoints={n:meta[n]['metrics'] for n in ['best','latest']},history_source='Reconstructed from printed log rates/counts; final grid exact checkpoint metrics. Original episode CSV/boundary_history not downloaded.')
 (OUT/'reconstructed_history.json').write_text(json.dumps(result,indent=2))
 (OUT/'parsed.json').write_text(json.dumps(dict(rows=rows,total_steps=rows[-1]['step'],errors={s:text.count(s) for s in ['Traceback (most recent call last)','CUDA out of memory','NaN']}),indent=2))
 print('Same weights',same,'evaluations',len(evaluations),'last steps',rows[-1]['step'],flush=True)
 rng=np.random.default_rng(2026103001);cases=[]
 for h,v in [(3.,4.),(2.5,4.),(3.,5.)]:
  for episode in range(64):
   phi=rng.uniform(0,2*np.pi);cases.append(dict(id=1300000+len(cases),kind='v8_fresh',height=h,speed=v,direction=float(np.degrees(phi)),vx=float(v*np.cos(phi)),vy=float(v*np.sin(phi)),roll=0.,pitch=0.,episode=episode))
 (OUT/'fresh_protocol.json').write_text(json.dumps(dict(cases=cases,direction_seed=2026103001,notes='Fixed before testing; 64 fresh directions/noise realizations per cell; native noisy sensor; shared case IDs and physical hashes for policy and zero.'),indent=2))
 results=[];hashes={}
 for name,action in [('zero',None),('best',policies['best'])]+([] if same else [('latest',policies['latest'])]):
  ex=V8Experiment()
  try:
   for i,c in enumerate(cases):
    r=ex.trial(c,action);r['controller']=name
    if name=='zero':hashes[c['id']]=r['scenario_hash']
    else:assert hashes[c['id']]==r['scenario_hash']
    results.append(r)
    if (i+1)%64==0:print(name,i+1,flush=True)
  finally:ex.close()
 (OUT/'fresh_results.json').write_text(json.dumps(results,indent=2))
 p=ROOT/'docs/report/report-data.json';data=json.loads(p.read_text());data['v8_results']=result;data['v8_completed']=dict(artifact='../artifacts/'+OUT.name+'/',same_weights=same);data['project']['updated']='2026-10-30';p.write_text(json.dumps(data,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
