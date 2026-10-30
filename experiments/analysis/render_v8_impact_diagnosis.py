"""Illustrate worst acceleration failure and nearest-direction success."""
import json
import numpy as np
from experiments.analysis.analyze_v8_download import OUT,V8Experiment
from experiments.analysis.report_landing_seed606 import load_policy
from experiments.analysis.render_v8_completed import render_montage

def main():
 diag=json.loads((OUT/'impact_diagnostics.json').read_text())['records'];fresh=json.loads((OUT/'fresh_results.json').read_text())
 failed=max((r for r in diag if not r['success']),key=lambda r:r['peak_acc'])
 distance=lambda r:abs((r['case']['direction']-failed['case']['direction']+180)%360-180)
 success=min((r for r in diag if r['success']),key=distance);cases=[failed['case'],success['case']]
 ex=V8Experiment()
 try:action=load_policy(OUT/'best.model',ex.wrapper.env)[0]
 finally:ex.close()
 movie=render_montage(cases,'v8-impact-diagnosis',[None,action],['zero','v8'])
 movie['selection']='Worst registered acceleration failure, then nearest-direction success among the same64cases; retrospective illustrations, not a success estimate.'
 for clip in movie['clips']:
  for name,callback,label in [('zero',None,'zero'),('best',action,'v8')]:
   ex=V8Experiment()
   try:r=ex.trial(clip['case'],callback)
   finally:ex.close()
   expected=next(row for row in fresh if row['controller']==name and row['case']['id']==clip['case']['id']);shown=next(o for o in clip['outcomes'] if o['controller']==label)
   for key in ['success','reason','end_time']:assert r[key]==expected[key]==shown[key]
   assert r['scenario_hash']==expected['scenario_hash']==clip['scenario_hash']
 movie['verified']=True;(OUT/'impact_video.json').write_text(json.dumps(movie,indent=2))
if __name__=='__main__':main()
