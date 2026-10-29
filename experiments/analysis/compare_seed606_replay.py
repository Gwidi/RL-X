"""Fresh paired, deterministic evaluation of downloaded best/latest and zero."""
import json
import zipfile
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.analysis.report_landing_seed606 import make_report_env, environment_config, load_policy, sha
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/artifacts/seed606_replay_comparison_2026-10-24'

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir",type=Path,required=True,help="Directory containing the original best.model and latest.model for this historical experiment")
    checkpoints=parser.parse_args().checkpoint_dir
    for name in ["best","latest"]:
        if not (checkpoints/f"{name}.model").is_file():parser.error(f"Missing {name}.model in {checkpoints}")
    OUT.mkdir(parents=True,exist_ok=True)
    w=make_report_env()
    actions={name:load_policy(checkpoints/f'{name}.model',w.env)[0] for name in ['best','latest']}
    w.close()
    metadata={}
    for name in actions:
        with zipfile.ZipFile(checkpoints/f'{name}.model') as z:
            metadata[name]=dict(sha256=sha(checkpoints/f'{name}.model'),metrics=json.loads(z.read('checkpoint_metrics.json')),environment=json.loads(z.read('config_environment.json')))
    (OUT/'checkpoints.json').write_text(json.dumps(metadata,indent=2))
    results=[];hashes={}
    for name,action in [('zero',None),*actions.items()]:
        w=make_report_env()
        for height in [1.,1.5,2.,2.5]:
            w.configure_landing_evaluation(dict(environment_config()['landing_evaluation'],height=height,perturbed=True,seed=20261024))
            for episode in range(32):
                obs,_=w.reset();case=dict(w.case_info)
                key=(height,episode)
                if name=='zero':hashes[key]=case['scenario_hash']
                assert hashes[key]==case['scenario_hash']
                for _ in range(w.env.horizon):
                    obs,_,done,truncated,_=w.step(np.zeros(13) if action is None else action(obs))
                    if done or truncated:break
                state=w.env.internal_state;monitor=state['landing_safety']
                results.append(dict(controller=name,height=height,episode=episode,scenario_hash=case['scenario_hash'],success=bool(state['landing_success']),reason=state['landing_failure_reason'],peak_foot=float(monitor.peak_foot.max()),peak_acceleration=float(monitor.peak_acceleration)))
            print(name,height,sum(r['success'] for r in results if r['controller']==name and r['height']==height),'/32',flush=True)
        w.close()
        (OUT/'heldout_results.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()
