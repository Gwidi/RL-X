"""V7 on the fixed v6 benchmark, plus explicit sensor interventions."""
import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from experiments.analysis.analyze_v7_download import OUT
from experiments.analysis.evaluate_v6_download import V6Experiment,scenarios
from experiments.train_landing_seed606_v6 import make_extended_env
from experiments.train_landing_seed606_v7 import make_noisy_env
from experiments.analysis.evaluate_seed606_v3 import Experiment
from experiments.analysis.report_landing_seed606 import load_policy
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_evaluation import LandingEvaluationWrapper
ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'docs/artifacts/v6_completed_9398616_2026-10-29'
class SensorExperiment(V6Experiment):
 def __init__(self,sensor):Experiment.__init__(self,LandingEvaluationWrapper(make_noisy_env() if sensor=='noisy' else make_extended_env(),0))
def policy_for(training,checkpoint):
 # Load in matching native sensor env; intervention changes evaluation input only.
 native=make_noisy_env() if training=='v7' else make_extended_env()
 try:
  action,info=load_policy((OUT if training=='v7' else OLD)/f'{checkpoint}.model',native)
  indices=native.policy_observation_indices.copy()
 finally:native.close()
 return action,info,indices

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['native','v6_noisy','v7_clean'],required=True);args=p.parse_args();cases=scenarios();reference={r['case']['id']:r for r in json.loads((OLD/'zero_results.json').read_text())};meta=json.loads((OUT/'checkpoints.json').read_text());rows=[]
 if args.mode=='native':
  models=[('v7','best','noisy')];cases_to_use=cases
 else:
  training,sensor=('v6','noisy') if args.mode=='v6_noisy' else ('v7','clean');models=[(training,n,sensor) for n in ['best','latest'] if not(training=='v7' and n=='latest' and meta['same_policy_weights'])];cases_to_use=[c for c in cases if c['kind']=='random_combined']
 for training,checkpoint,sensor in models:
  action,info,indices=policy_for(training,checkpoint);ex=SensorExperiment(sensor);assert np.array_equal(indices,ex.wrapper.env.policy_observation_indices)
  try:
   for i,c in enumerate(cases_to_use):
    r=ex.trial(c,action);r.update(training=training,checkpoint=checkpoint,sensor=sensor);assert r['scenario_hash']==reference[c['id']]['scenario_hash'];rows.append(r)
    if (i+1)%80==0:print(args.mode,checkpoint,i+1,'/',len(cases_to_use),flush=True)
  finally:ex.close()
 (OUT/f'{args.mode}_results.json').write_text(json.dumps(rows,indent=2));print(args.mode,'complete',sum(r['success'] for r in rows),'/',len(rows),flush=True)
 if args.mode=='native':
  assert meta['same_policy_weights'],'Use separate full latest evaluation if weights differ'
  # Same checkpoint arrays, replay representative cases rather than duplicate all episodes.
  action,_,_=policy_for('v7','latest');ex=SensorExperiment('noisy');by={r['case']['id']:r for r in rows};verified=[]
  try:
   for kind in ['nominal','angle_grid','speed_grid','random_angles','random_velocity','random_combined','combined_lower']:
    c=next(c for c in cases if c['kind']==kind);r=ex.trial(c,action)
    for key in ['scenario_hash','success','reason','end_time','max_action','peak_foot_force','peak_acceleration']:assert r[key]==by[c['id']][key]
    verified.append(dict(case_id=c['id'],identical=True))
  finally:ex.close()
  (OUT/'latest_verification.json').write_text(json.dumps(verified,indent=2))
if __name__=='__main__':main()
