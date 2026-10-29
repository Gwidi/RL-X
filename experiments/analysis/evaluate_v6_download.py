"""Fixed diagnostic v6 benchmark: separate factors, height, angles and speed."""
import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from experiments.analysis.analyze_v6_download import OUT
from experiments.train_landing_seed606_v6 import make_extended_env
from experiments.analysis.evaluate_v4_download import V4Experiment
from experiments.analysis.evaluate_seed606_v3 import Experiment
from experiments.analysis.report_landing_seed606 import load_policy
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_evaluation import LandingEvaluationWrapper
SEED=2026102901
class V6Experiment(V4Experiment):
 def __init__(self):Experiment.__init__(self,LandingEvaluationWrapper(make_extended_env(),0))
 def reset(self,case):
  self.current=case;cfg=self.wrapper.env.env_config['landing_evaluation'];self.wrapper.configure_landing_evaluation(dict(cfg,height=case['height'],perturbed=False,seed=SEED+case['id']));return self.wrapper.reset()[0]
def scenarios():
 cases=[]
 def add(kind,h=3.,vx=0.,vy=0.,roll=0.,pitch=0.,**extra):cases.append(dict(id=980000+len(cases),kind=kind,height=float(h),vx=float(vx),vy=float(vy),roll=float(roll),pitch=float(pitch),**extra))
 for h in [1.,1.5,2.,2.2,2.5,3.]:add('nominal',h)
 for axis in ['roll','pitch']:
  for angle in [-30,-20,-10,10,20,30]:add('angle_grid',**{axis:angle},axis=axis,angle=angle)
 for speed in [1.,2.,3.,4.,5.]:
  for direction in range(0,360,45):
   phi=np.deg2rad(direction);add('speed_grid',vx=speed*np.cos(phi),vy=speed*np.sin(phi),speed=speed,direction=direction)
 rng=np.random.default_rng(SEED)
 for kind in ['random_angles','random_velocity','random_combined']:
  for episode in range(64):
   roll,pitch=rng.uniform(-30,30,2) if kind!='random_velocity' else [0,0]
   radius=5*np.sqrt(rng.uniform());phi=rng.uniform(0,2*np.pi)
   vx,vy=(radius*np.cos(phi),radius*np.sin(phi)) if kind!='random_angles' else [0,0]
   add(kind,vx=vx,vy=vy,roll=roll,pitch=pitch,episode=episode)
   if kind=='random_combined':add('combined_lower',h=2.2,vx=vx,vy=vy,roll=roll,pitch=pitch,episode=episode)
 return cases

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--controller',choices=['zero','best','latest'],required=True);name=parser.parse_args().controller;cases=scenarios();ex=V6Experiment();policy=None if name=='zero' else load_policy(OUT/f'{name}.model',ex.wrapper.env)[0];rows=[]
 (OUT/f'{name}_protocol.json').write_text(json.dumps(dict(seed=SEED,cases=cases,notes='Protocol frozen before new evaluation. Ideal velocity, frozen physics, deterministic policy. Paired height cases reuse same vx/vy/angles. No seed replication.'),indent=2))
 try:
  for i,c in enumerate(cases):
   r=ex.trial(c,policy);r['controller']=name;rows.append(r)
   if (i+1)%80==0:print(name,i+1,'/',len(cases),flush=True)
 finally:ex.close()
 (OUT/f'{name}_results.json').write_text(json.dumps(rows,indent=2));print(name,'complete',sum(r['success'] for r in rows),'/',len(rows),flush=True)
if __name__=='__main__':main()
