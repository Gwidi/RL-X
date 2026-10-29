"""Closed-loop controller/integrator ablations of the same tilt cases."""
import argparse,json,sys,copy,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from experiments import diagnose_tilt_control as diagnostic
from experiments.analysis.diagnose_tilt_control import cases,trial
from experiments.analysis.evaluate_seed606_v3 import Experiment,OUT as MODELS
from experiments.analysis.report_landing_seed606 import load_policy
from experiments.analysis.check_pd_stability import OUT,VARIANTS,apply_variant
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_seed606 import environment_config,ROBOT_DIR
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.environment import LocomotionEnv
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_evaluation import LandingEvaluationWrapper
from rl_x.environments.custom_mujoco.robot_locomotion.robots.silver_badger.robot_config import robot_config

class VariantExperiment(Experiment):
    def __init__(self,variant):
        self.variant=variant;cfg=environment_config();cfg['nr_envs']=1;cfg['timestep']=VARIANTS[variant]['dt'];robot=copy.deepcopy(robot_config);robot.update(directory_path=ROBOT_DIR,spine_locked=False)
        env=LocomotionEnv(robot,'test',20261023,False,cfg,1)
        super().__init__(LandingEvaluationWrapper(env,0))
    def reset(self,case):
        obs=super().reset(case);e=self.wrapper.env;m=e.internal_state['mj_model'];apply_variant(m,self.variant)
        # hash retains baseline controller; record actual controller separately.
        state_hash=hashlib.sha256()
        for x in [e.internal_state['data'].qpos,e.internal_state['data'].qvel]:state_hash.update(x.tobytes())
        self.initial_state_hash=state_hash.hexdigest()
        return obs

def main():
    p=argparse.ArgumentParser();p.add_argument('--variants',nargs='+',required=True);p.add_argument('--screen',action='store_true');args=p.parse_args()
    loader=Experiment();actions={'zero':None};metadata={}
    for n in ['best','latest']:actions[n],metadata[n]=load_policy(MODELS/f'{n}.model',loader.wrapper.env)
    loader.close();selected=cases()
    if args.screen:selected=[c for c in selected if c['axis']=='nominal' or c['height']==2.5 and abs(c['angle']) in [10,30]]
    out=OUT/('screen' if args.screen else 'closed_loop');out.mkdir(parents=True,exist_ok=True)
    (out/'protocol.json').write_text(json.dumps(dict(variants={n:VARIANTS[n] for n in args.variants},cases=selected,checkpoint_sha256={n:metadata[n]['sha256'] for n in metadata},notes='Old-policy transfer diagnostic, not retraining. Variant gains applied after reset; no per-step physics randomization. Geometry, home, limits, action mapping and outcome unchanged.'),indent=2))
    for variant in args.variants:
        dest=out/variant;dest.mkdir(exist_ok=True);(dest/'traces').mkdir(exist_ok=True)
        if (dest/'results.json').exists():print('Already completed',variant,flush=True);continue
        diagnostic.OUT=dest;results=[];hashes={};states={}
        for name in ['zero','best','latest']:
            ex=VariantExperiment(variant)
            try:
                for c in selected:
                    r=trial(ex,c,actions[name],name);r['variant']=variant;r['initial_state_hash']=ex.initial_state_hash
                    if name=='zero':hashes[c['id']]=r['scenario_hash'];states[c['id']]=r['initial_state_hash']
                    assert r['scenario_hash']==hashes[c['id']] and r['initial_state_hash']==states[c['id']]
                    results.append(r)
            finally:ex.close()
            print(variant,name,sum(r['success'] for r in results if r['controller']==name),'/',len(selected),flush=True)
        (dest/'results.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
