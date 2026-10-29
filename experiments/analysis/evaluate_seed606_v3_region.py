"""Exploratory post-grid validation of the limited |v_xy| <= 1 m/s region."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.analysis.evaluate_seed606_v3 import Experiment,OUT
from experiments.analysis.report_landing_seed606 import make_report_env,load_policy
import numpy as np


def main():
    rng=np.random.default_rng(2026102602);cases=[]
    for height in [1.,1.5,2.,2.5]:
        for episode in range(32):
            speed=float(rng.uniform(0,1));phi=float(rng.uniform(0,2*np.pi));roll,pitch=rng.uniform(-3,3,2)
            cases.append(dict(id=60000+len(cases),kind='limited_region',height=height,roll=float(roll),pitch=float(pitch),vx=float(speed*np.cos(phi)),vy=float(speed*np.sin(phi)),speed=speed,direction=float(np.rad2deg(phi)),episode=episode))
    (OUT/'region_protocol.json').write_text(json.dumps(dict(seed=2026102602,exploratory_post_grid=True,speed_distribution='magnitude uniform [0,1] m/s; azimuth uniform [0,2pi]',angles='independent roll/pitch uniform [-3,3] deg',cases=cases),indent=2))
    w=make_report_env();actions={n:load_policy(OUT/f'{n}.model',w.env)[0] for n in ['best','latest']};w.close()
    rows=[];hashes={}
    for name,action in [('zero',None),*actions.items()]:
        ex=Experiment()
        try:
            for case in cases:
                result=ex.trial(case,action);result['controller']=name;rows.append(result)
                if name=='zero':hashes[case['id']]=result['scenario_hash']
                assert hashes[case['id']]==result['scenario_hash']
                if (case['episode']+1)%32==0:print(name,case['height'],sum(r['success'] for r in rows if r['controller']==name and r['case']['height']==case['height']),'/32',flush=True)
        finally:ex.close()
        (OUT/'region_results.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':main()
