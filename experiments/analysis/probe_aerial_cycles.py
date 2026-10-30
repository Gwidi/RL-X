"""Closed joint cycles: retained body angle with actual position/velocity closure."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from experiments.analysis.probe_aerial_reorientation import Probe,OUT

def trial(p,c,dt=.0025,horizon=.8,height=20.):
    r,t,q=p.trial(c,physics_dt=dt,duration=horizon,height=height)
    r['joint_closure_max_deg']=float(np.max(np.abs(q[-1,7:]-p.q0[7:]))*180/np.pi)
    r['joint_speed_max_rad_s']=float(np.max(np.abs(p.data.qvel[6:])))
    r['closed']=bool(r['joint_closure_max_deg']<.5 and r['joint_speed_max_rad_s']<.05 and r['final_angular_speed']<.1 and not r['contacts'] and r['within_nominal_motor_limits'])
    return r,t,q

def main():
    p=Probe();rng=np.random.default_rng(20261030);rows=[]
    old=json.loads((OUT/'results.json').read_text())
    candidates=[r['candidate'] for r in old['rows'] if r['candidate'].get('kind')!='hold']
    for i in range(1000):
        candidates.append(dict(name=f'cycle_extended_{i}',duration=float(rng.choice([.2,.3,.4,.5,.6])),amplitude=1.,a=rng.uniform(-1,1,13).tolist(),b=rng.uniform(-1,1,13).tolist()))
    try:
        for i,c in enumerate(candidates):
            r,_,_=trial(p,c);rows.append(r)
            if i%200==0:print(i,flush=True)
        chosen={};fine=[]
        for axis,name in enumerate(['roll','pitch','yaw']):
            eligible=sorted([r for r in rows if r['closed']],key=lambda r:abs(r['rotation_vector_deg'][axis]),reverse=True)[:8]
            verified=[]
            for r in eligible:
                f,_,_=trial(p,r['candidate'],dt=.00125);f['axis']=name;fine.append(f)
                if f['closed']:verified.append((r,f))
            if not verified:continue
            coarse,f=max(verified,key=lambda pair:abs(pair[1]['rotation_vector_deg'][axis]));finer,t,q=trial(p,f['candidate'],dt=.000625)
            longer,_,_=trial(p,f['candidate'],dt=.00125,horizon=1.2)
            tails=[trial(p,f['candidate'],dt=.000625,horizon=h,height=100.)[0] for h in [1.2,2.,3.]]
            chosen[name]=dict(coarse=coarse,fine=f,finer=finer,hold_until_1p2=longer,settling_tail=tails)
            np.savez_compressed(OUT/f'cycle_{name}_trace.npz',trace=t,qpos=q)
        data=dict(candidate_count=len(rows),closed_count=sum(r['closed'] for r in rows),horizon=.8,closure_criterion='max actual joint position error <0.5deg; max joint speed <0.05rad/s; base speed <0.1rad/s; no contacts; nominal motor torque limits',rows=rows,selected=chosen,fine_screen=fine,limitation='Finite open-loop search, not global optimum; vacuum; no landing evaluation')
        (OUT/'cycles.json').write_text(json.dumps(data,indent=2))
        print(json.dumps({k:{'rotation':v['finer']['rotation_vector_deg'],'closure_deg':v['finer']['joint_closure_max_deg'],'joint_speed':v['finer']['joint_speed_max_rad_s'],'body_speed':v['finer']['final_angular_speed'],'rotation_1p2':v['hold_until_1p2']['rotation_vector_deg']} for k,v in chosen.items()},indent=2))
    finally:p.close()
if __name__=='__main__':main()
