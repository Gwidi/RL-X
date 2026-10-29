"""Check retention on a fixed 64-case subset of the archived v5 benchmark."""
import sys,json,types
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.analysis.evaluate_v6_download import V6Experiment,OUT
from experiments.analysis.evaluate_v4_download import V4Experiment
from experiments.analysis.report_landing_seed606 import load_policy
ROOT=Path(__file__).resolve().parents[2]
def main():
 old=ROOT/'docs/artifacts/v5_completed_9379363_2026-10-28';protocol=json.loads((old/'test_protocol.json').read_text());cases=[c for c in protocol['cases'] if c['kind']=='random_combined' and c['episode']<16];assert len(cases)==64;ref={r['case']['id']:r for r in json.loads((old/'latest_results.json').read_text())};rows=[]
 for name in ['best','latest']:
  ex=V6Experiment();ex.reset=types.MethodType(V4Experiment.reset,ex);act,_=load_policy(OUT/f'{name}.model',ex.wrapper.env)
  try:
   for c in cases:
    r=ex.trial(c,act);r['controller']=name;assert r['scenario_hash']==ref[c['id']]['scenario_hash'];rows.append(r)
  finally:ex.close()
  print(name,'retention',sum(r['success'] for r in rows if r['controller']==name),'/64',flush=True)
 result=dict(cases=cases,results=rows,v5_latest_reference_successes=sum(ref[c['id']]['success'] for c in cases),notes='First16 archived random_combined cases per height; subset fixed before evaluation, retrospective diagnostic not new final held-out set.')
 (OUT/'previous_domain_probe.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
