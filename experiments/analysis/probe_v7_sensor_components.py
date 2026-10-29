"""Latest-checkpoint interventions: noise alone versus delay alone, no retraining."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.analysis.evaluate_v7_download import OUT,OLD,policy_for
from experiments.analysis.evaluate_v6_download import V6Experiment,scenarios
from experiments.analysis.evaluate_seed606_v3 import Experiment
from experiments.train_landing_seed606_v6 import make_extended_env
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.policy_velocity import PolicyVelocityObservation
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_evaluation import LandingEvaluationWrapper
class PartialSensorExperiment(V6Experiment):
 def __init__(self,sensor):
  env=make_extended_env();cfg=env.env_config['policy_velocity'];cfg.update(noise_std_mps=.1 if sensor=='noise_only' else 0.,delay_steps=1 if sensor=='delay_only' else 0)
  env.policy_velocity_observation=PolicyVelocityObservation(cfg);Experiment.__init__(self,LandingEvaluationWrapper(env,0))
def main():
 cases=[c for c in scenarios() if c['kind']=='random_combined'];ref={r['case']['id']:r for r in json.loads((OLD/'zero_results.json').read_text())};rows=[]
 for training in ['v6','v7']:
  action,_,_=policy_for(training,'latest')
  for sensor in ['noise_only','delay_only']:
   ex=PartialSensorExperiment(sensor);group=[]
   try:
    for c in cases:
     r=ex.trial(c,action);r.update(training=training,checkpoint='latest',sensor=sensor);assert r['scenario_hash']==ref[c['id']]['scenario_hash'];rows.append(r);group.append(r)
   finally:ex.close()
   print(training,sensor,sum(r['success'] for r in group),'/64',flush=True)
 (OUT/'sensor_components_results.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
