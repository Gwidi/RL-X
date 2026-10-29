"""Evaluation-time velocity masking probe; not a separately retrained ablation."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from experiments.analysis.evaluate_v5_download import V5Experiment,OUT,OLD
from experiments.analysis.evaluate_v4_download import V4Experiment
from experiments.analysis.report_landing_seed606 import load_policy

def main():
 old=json.loads((OLD/'test_results.json').read_text());cases=json.loads((OLD/'test_protocol.json').read_text())['cases'];by={(r['controller'],r['case']['id']):r for r in old}
 ex=V4Experiment();policy,_=load_policy(OLD/'best.model',ex.wrapper.env);verification=[]
 try:
  for name,act in [('zero',None),('v4',policy)]:
   for h in [1.,1.5,2.,2.2]:
    for kind in ['nominal','random_angles','random_velocity','random_combined']:
     c=next(c for c in cases if c['height']==h and c['kind']==kind);r=ex.trial(c,act);expected=by[name,c['id']]
     for field in ['scenario_hash','success','reason','end_time','max_action','peak_foot_force','peak_acceleration']:assert r[field]==expected[field],(name,c,field)
     verification.append(dict(controller=name,case_id=c['id'],identical=True))
 finally:ex.close()
 (OUT/'baseline_replay_verification.json').write_text(json.dumps(verification,indent=2));print('Historical baselines exactly reproduced:',len(verification),flush=True)
 ex=V5Experiment();policy,_=load_policy(OUT/'best.model',ex.wrapper.env);indices=ex.wrapper.env.policy_velocity_obs_idx;rows=[]
 def hidden(obs):
  copied=obs.copy();copied[indices]=0;return policy(copied)
 try:
  for c in [c for c in cases if c['kind']=='random_combined']:
   r=ex.trial(c,hidden);r['controller']='v5_best_input_zeroed';rows.append(r)
   if len(rows)%64==0:print('Zeroed velocity probe',len(rows),flush=True)
 finally:ex.close()
 (OUT/'velocity_zeroed_probe.json').write_text(json.dumps(dict(note='Evaluation-time distribution shift: zero two features only. Not a retrained masked policy, not causal attribution of entire v4/v5 gap.',results=rows),indent=2))
 print('Probe success',sum(r['success'] for r in rows),'/',len(rows),flush=True)
if __name__=='__main__':main()
