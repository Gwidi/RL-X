"""Evaluate both v5 checkpoints on the exact archived v4 held-out protocol."""
import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.analysis.analyze_v5_download import OUT
from experiments.train_landing_seed606_v5 import make_velocity_report_env
from experiments.analysis.evaluate_v4_download import V4Experiment
from experiments.analysis.report_landing_seed606 import load_policy
ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'docs/artifacts/v4_completed_9376691_2026-10-27'
class V5Experiment(V4Experiment):
 def __init__(self):
  from experiments.analysis.evaluate_seed606_v3 import Experiment
  Experiment.__init__(self,make_velocity_report_env())
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--controller',choices=['best','latest'],required=True);args=parser.parse_args();name=args.controller
 protocol=json.loads((OLD/'test_protocol.json').read_text());cases=protocol['cases'];old=json.loads((OLD/'test_results.json').read_text());reference={r['case']['id']:r for r in old if r['controller']=='zero'}
 if not (OUT/'test_protocol.json').exists():
  (OUT/'test_protocol.json').write_text(json.dumps(dict(**{k:v for k,v in protocol.items() if k not in ['controllers','policy_label','same_policy_weights','checkpoint_sha256','notes']},controllers=['zero','v4','v5_best','v5_latest'],notes='Exact 925 archived v4 scenarios, same seeds and physics; v4/zero replay validated separately. Fixed PD, deterministic policy; external velocity ideal. One training seed, no masked retraining.'),indent=2))
 ex=V5Experiment();action,_=load_policy(OUT/f'{name}.model',ex.wrapper.env);rows=[]
 try:
  for i,c in enumerate(cases):
   r=ex.trial(c,action);r['controller']='v5_'+name;r['telemetry']=ex.wrapper.env.internal_state['landing_safety'].metrics()
   assert r['scenario_hash']==reference[c['id']]['scenario_hash'],c
   rows.append(r)
   if (i+1)%100==0:print(name,i+1,'/',len(cases),flush=True)
 finally:ex.close()
 (OUT/f'{name}_results.json').write_text(json.dumps(rows,indent=2));print(name,'complete',sum(r['success'] for r in rows),'/',len(cases),flush=True)
if __name__=='__main__':main()
