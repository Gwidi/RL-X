"""Matched v4/v5 latest videos for pure left/right initial velocity."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.analysis.render_v5_download import render_montage,OUT,OLD,V4Experiment,V5Experiment,HEIGHTS,load_policy

def main():
 cases=json.loads((OLD/'test_protocol.json').read_text())['cases']
 loaders=[V4Experiment(),V5Experiment()]
 try:actions=[load_policy(path,ex.wrapper.env)[0] for path,ex in zip([OLD/'best.model',OUT/'latest.model'],loaders)]
 finally:
  for ex in loaders:ex.close()
 reference=json.loads((OLD/'test_results.json').read_text())+json.loads((OUT/'latest_results.json').read_text());by={(r['controller'],r['case']['id']):r for r in reference};movies=[]
 for direction,label in [(90,'bok-lewo-3ms'),(270,'bok-prawo-3ms')]:
  selected=[next(c for c in cases if c['kind']=='speed_grid' and c['height']==h and c['speed']==3 and c['direction']==direction) for h in HEIGHTS]
  movie=render_montage(selected,label,actions,'latest');checks=0
  for clip in movie['clips']:
   for result in clip['outcomes']:
    expected=by[result['controller'],clip['case']['id']]
    assert clip['scenario_hash']==expected['scenario_hash']
    for key in ['success','reason','end_time']:assert result[key]==expected[key],(clip['case']['id'],key)
    checks+=1
  movie.update(selection='Fixed archived speed_grid cases: magnitude 3m/s, direction '+str(direction)+'deg, roll=pitch=0, all four heights; no selection by outcome.',verified_outcomes=checks)
  movies.append(movie)
 (OUT/'lateral_videos.json').write_text(json.dumps(movies,indent=2));print('Verified all lateral video outcomes against held-out evaluation',flush=True)
if __name__=='__main__':main()
