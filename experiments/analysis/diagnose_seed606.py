"""Nominal height sweep and action interventions for the seed606 checkpoints."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.analysis.report_landing_seed606 import make_report_env, environment_config, load_policy
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/artifacts/seed606_diagnosis_2026-10-23'

def trial(height, action=None, mode='full', trace=False):
    wrapper = make_report_env()
    env = wrapper.env
    wrapper.configure_landing_evaluation(dict(environment_config()['landing_evaluation'], height=float(height), perturbed=False))
    obs, _ = wrapper.reset()
    state = env.internal_state
    model, data = state['mj_model'], state['data']
    monitor = state['landing_safety']
    mapping = monitor.contacts
    original = monitor.update
    events, trajectory = [], []
    first_support = None
    def update(m, d):
        nonlocal first_support
        for c in d.contact:
            pair = {int(c.geom1), int(c.geom2)}
            robot = {g for g in pair if m.geom_bodyid[g] != 0}
            forbidden = bool(robot and not (pair & mapping.support_geom_ids and robot <= mapping.allowed_support_geom_ids))
            if pair & mapping.support_geom_ids and first_support is None:
                first_support = float(d.time)
            if forbidden and not events:
                events.append(dict(time=float(d.time), geoms=[m.geom(int(g)).name for g in (c.geom1,c.geom2)], distance=float(c.dist), base_z=float(d.qpos[2])))
        return original(m,d)
    monitor.update = update
    names = [model.joint(int(j)).name for j in env.actuator_joint_mask_joints]
    for _ in range(env.horizon):
        a = np.zeros(13) if action is None else action(obs).copy()
        if mode.startswith('scale_'): a *= float(mode.split('_')[1])
        if mode == 'flight_only' and first_support is not None: a[:] = 0
        if mode == 'contact_only' and first_support is None: a[:] = 0
        if mode.startswith('zero_'):
            part = mode[5:]
            for i,n in enumerate(names):
                if n == 'spine_joint' and part == 'spine' or f'_{part}_joint' in n: a[i] = 0
        if trace:
            trajectory.append(dict(t=float(data.time), qpos=data.qpos.tolist(), qvel=data.qvel.tolist(), action=a.tolist()))
        obs, _, done, truncated, _ = wrapper.step(a)
        if done or truncated: break
    result = dict(height=float(height),mode=mode,success=bool(state['landing_success']),reason=state['landing_failure_reason'],end=float(data.time),first_support=first_support,forbidden=events,names=names)
    if trace: result['trajectory'] = trajectory
    wrapper.close()
    return result

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir",type=Path,required=True,help="Directory containing the original best.model and latest.model for this historical experiment")
    checkpoints=parser.parse_args().checkpoint_dir
    for name in ["best","latest"]:
        if not (checkpoints/f"{name}.model").is_file():parser.error(f"Missing {name}.model in {checkpoints}")
    OUT.mkdir(parents=True,exist_ok=True)
    wrapper=make_report_env()
    actions={name:load_policy(checkpoints/f'{name}.model',wrapper.env)[0] for name in ['best','latest']}
    wrapper.close()
    rows=[]
    for name,action in [('pd',None),*actions.items()]:
        for height in np.round(np.arange(1,2.501,.1),2):
            r=trial(height,action,trace=height==2.5);r['checkpoint']=name;rows.append(r)
            print(name,height,r['success'],r['forbidden'],flush=True)
    for name,action in actions.items():
        for mode in ['scale_0.25','scale_0.5','scale_0.75','flight_only','contact_only','zero_spine','zero_hip','zero_thigh','zero_calf']:
            r=trial(2.5,action,mode,True);r['checkpoint']=name;rows.append(r)
            print(name,mode,r['success'],r['forbidden'],flush=True)
    (OUT/'results.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__': main()
