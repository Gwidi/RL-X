"""Find and render an illustrative exact 3m/s, 2m, |roll|=|pitch|=10deg landing."""
import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from experiments.analysis.evaluate_v4_download import V4Experiment
from experiments.analysis.analyze_v4_download import OUT,make_v4_env
from experiments.analysis.report_landing_seed606 import load_policy
from experiments.analysis.render_v4_download import render_montage
DEST=OUT/'exact_3ms_2m_10deg'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--refine',action='store_true');args=parser.parse_args()
    DEST.mkdir(exist_ok=True);w=make_v4_env();policy,_=load_policy(OUT/'best.model',w.env);w.close();ex=V4Experiment();rows=json.loads((DEST/'screening.json').read_text()) if args.refine else [];chosen=None
    try:
        for roll,pitch in [(10,10),(10,-10),(-10,10),(-10,-10)]:
            directions=list(range(0,360,45))
            if args.refine:
                previous=[r for r in rows if r['case']['roll']==roll and r['case']['pitch']==pitch]
                center=max(previous,key=lambda r:r['end_time'])['case']['direction']
                directions=sorted([d for d in range(0,360,5) if d%45!=0],key=lambda d:min((d-center)%360,(center-d)%360))
            for direction in directions:
                phi=np.deg2rad(direction);case=dict(id=1000000+len(rows),kind='exact_3ms_demo',height=2.,roll=float(roll),pitch=float(pitch),vx=float(3*np.cos(phi)),vy=float(3*np.sin(phi)),speed=3.,direction=direction)
                r=ex.trial(case,policy);r['controller']='v4';rows.append(r);print(direction,roll,pitch,r['success'],r['reason'],flush=True)
                if len(rows)%16==0:(DEST/'screening.json').write_text(json.dumps(rows,indent=2))
                if r['success']:
                    chosen=r;break
            if chosen:break
    finally:ex.close()
    (DEST/'screening.json').write_text(json.dumps(rows,indent=2))
    if chosen is None:print('No success in tested direction grid',flush=True);return
    assert abs(np.hypot(chosen['case']['vx'],chosen['case']['vy'])-3)<1e-12
    video=render_montage([chosen['case']],'3ms-2m-10deg',policy)
    result=next(r for r in video['clips'][0]['outcomes'] if r['controller']=='v4')
    assert result['success'] and result['end_time']==chosen['end_time']
    assert video['clips'][0]['scenario_hash']==chosen['scenario_hash']
    (DEST/'video.json').write_text(json.dumps(video,indent=2));print('VIDEO',video['src'],chosen['case'],flush=True)
if __name__=='__main__':main()
