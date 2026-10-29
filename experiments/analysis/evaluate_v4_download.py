"""New paired v4 tests: separate factors, envelopes, OOD tilts and matched videos."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from experiments.analysis.analyze_v4_download import OUT,make_v4_env
from experiments.analysis.evaluate_seed606_v3 import Experiment
from experiments.analysis.report_landing_seed606 import load_policy,sha
SEED=2026102701
HEIGHTS=[1.,1.5,2.,2.2]

class V4Experiment(Experiment):
    def __init__(self):super().__init__(make_v4_env())
    def reset(self,case):
        self.current=case;cfg=self.wrapper.env.env_config['landing_evaluation']
        self.wrapper.configure_landing_evaluation(dict(cfg,height=case['height'],perturbed=False,seed=SEED+case['id']))
        return self.wrapper.reset()[0]

def scenarios():
    out=[]
    def add(kind,h,vx=0.,vy=0.,roll=0.,pitch=0.,**extra):
        out.append(dict(id=900000+len(out),kind=kind,height=float(h),vx=float(vx),vy=float(vy),roll=float(roll),pitch=float(pitch),**extra))
    for h in np.round(np.arange(1,2.201,.1),2):add('nominal',h)
    for h in HEIGHTS:
        for axis in ['roll','pitch']:
            for angle in [-30,-20,-10,-3,3,10,20,30]:add('angle_grid',h,**{axis:angle},axis=axis,angle=angle,outside_training=abs(angle)>10)
        for r,p in [(-10,-10),(-10,10),(10,-10),(10,10)]:add('angle_corner',h,roll=r,pitch=p)
        for speed in [.5,1.,1.5,2.,2.5,3.]:
            for direction in range(0,360,45):
                phi=np.deg2rad(direction);add('speed_grid',h,speed*np.cos(phi),speed*np.sin(phi),speed=speed,direction=direction)
    rng=np.random.default_rng(SEED)
    for h in HEIGHTS:
        for kind,count in [('random_angles',32),('random_velocity',32),('random_combined',64),('limited_combined',32)]:
            for episode in range(count):
                roll,pitch=rng.uniform(-10,10,2) if kind!='random_velocity' else [0,0]
                vx,vy=rng.uniform(-3,3,2) if kind in ['random_velocity','random_combined'] else [0,0]
                if kind=='limited_combined':
                    speed=rng.uniform(0,1);phi=rng.uniform(0,2*np.pi);vx,vy=speed*np.cos(phi),speed*np.sin(phi)
                add(kind,h,vx,vy,roll,pitch,episode=episode)
    return out

def main():
    cases=scenarios();meta=json.loads((OUT/'checkpoints.json').read_text());assert meta['same_policy_weights']
    loader=make_v4_env();action,info=load_policy(OUT/'best.model',loader.env);latest,_=load_policy(OUT/'latest.model',loader.env);loader.close()
    (OUT/'test_protocol.json').write_text(json.dumps(dict(seed=SEED,cases=cases,controllers=['zero','v4'],policy_label='best=latest (identical restored policy arrays)',same_policy_weights=True,checkpoint_sha256={n:meta[n]['sha256'] for n in ['best','latest']},physics_profile_sha256=info['saved_environment']['landing_physics_sha256'],physics_dt=.0025,control_dt=.02,angle_distribution='independent roll/pitch uniform [-10,10] degrees',full_velocity_distribution='independent vx/vy uniform [-3,3] m/s',limited_velocity_distribution='magnitude uniform [0,1], independent azimuth uniform [0,2pi]',notes='All tests planned before new evaluation; limited region is exploratory based on prior v3 work, not a replacement for the full domain. One training seed.'),indent=2))
    rows=[];hashes={}
    for name,policy in [('zero',None),('v4',action)]:
        ex=V4Experiment()
        try:
            for i,c in enumerate(cases):
                r=ex.trial(c,policy);r['controller']=name;r['telemetry']=ex.wrapper.env.internal_state['landing_safety'].metrics()
                if name=='zero':hashes[c['id']]=r['scenario_hash']
                assert r['scenario_hash']==hashes[c['id']];rows.append(r)
                if (i+1)%100==0:print(name,i+1,'/',len(cases),flush=True)
        finally:ex.close()
        (OUT/'test_results.json').write_text(json.dumps(rows,indent=2));print(name,'complete',sum(r['success'] for r in rows if r['controller']==name),'/',len(cases),flush=True)
    # Verify paired deterministic replay from latest for four kinds at each height.
    verify_cases=[]
    for h in HEIGHTS:
        for kind in ['nominal','random_angles','random_velocity','random_combined']:
            verify_cases.append(next(c for c in cases if c['height']==h and c['kind']==kind))
    ex=V4Experiment();verification=[];by={r['case']['id']:r for r in rows if r['controller']=='v4'}
    try:
        for c in verify_cases:
            r=ex.trial(c,latest);b=by[c['id']]
            for field in ['scenario_hash','success','reason','end_time','max_action','peak_foot_force','peak_acceleration']:assert r[field]==b[field],field
            verification.append(dict(case=c,identical_outcome_and_metrics=True))
    finally:ex.close()
    (OUT/'latest_verification.json').write_text(json.dumps(verification,indent=2));print('Latest replay verified',len(verification),flush=True)
if __name__=='__main__':main()
