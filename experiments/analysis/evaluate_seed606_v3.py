"""Controlled, paired experiments on downloaded v3 best/latest and zero PD."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.analysis.report_landing_seed606 import make_report_env, environment_config, load_policy, sha
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/artifacts/v3_checkpoint_experiments_2026-10-26'
SEED=2026102601


def scenarios():
    rows=[]
    def add(kind,h,vx=0.,vy=0.,roll=0.,pitch=0.,**extra):
        rows.append(dict(id=len(rows),kind=kind,height=float(h),vx=float(vx),vy=float(vy),roll=float(roll),pitch=float(pitch),**extra))
    for h in np.round(np.arange(1,2.501,.1),2):add('nominal',h)
    for h in [1.,1.5,2.,2.5]:
        for speed in [.5,1.,1.5,2.,2.5,3.]:
            for direction in range(0,360,45):
                phi=np.deg2rad(direction)
                add('speed',h,speed*np.cos(phi),speed*np.sin(phi),speed=speed,direction=direction)
        for roll,pitch in [(3,0),(-3,0),(0,3),(0,-3),(3,3),(3,-3),(-3,3),(-3,-3)]:
            add('angle',h,roll=roll,pitch=pitch)
    rng=np.random.default_rng(SEED)
    for h in [1.,1.5,2.,2.5]:
        for index in range(64):
            roll,pitch=rng.uniform(-3,3,2);vx,vy=rng.uniform(-3,3,2)
            add('heldout',h,vx,vy,roll,pitch,episode=index)
    return rows


class Experiment:
    def __init__(self,wrapper=None):
        self.wrapper=wrapper if wrapper is not None else make_report_env()
        self.initial=self.wrapper.env.initial_state_function
        self.original_setup=self.initial.setup
        self.current=None
        def setup():
            qpos,qvel=self.original_setup()
            c=self.current
            qpos[3:7]=Rotation.from_euler('xyz',[c['roll'],c['pitch'],0.],degrees=True).as_quat(scalar_first=True)
            qvel[:3]=[c['vx'],c['vy'],0.]
            return qpos,qvel
        self.initial.setup=setup

    def reset(self,case):
        self.current=case
        self.wrapper.configure_landing_evaluation(dict(environment_config()['landing_evaluation'],height=case['height'],perturbed=False,seed=SEED+case['id']))
        obs,_=self.wrapper.reset()
        return obs

    def trial(self,case,action=None,trace=False):
        obs=self.reset(case);w=self.wrapper;e=w.env;s=e.internal_state
        model,data=s['mj_model'],s['data'];monitor=s['landing_safety'];mapping=monitor.contacts
        gains=model.actuator_gainprm.copy();bias=model.actuator_biasprm.copy()
        first_support=None;forbidden=[];trajectory=[];original=monitor.update
        def update(m,d):
            nonlocal first_support
            for index,c in enumerate(d.contact):
                pair={int(c.geom1),int(c.geom2)}
                robot={g for g in pair if m.geom_bodyid[g]!=0}
                if pair & mapping.support_geom_ids and first_support is None:first_support=float(d.time)
                if robot and not(pair & mapping.support_geom_ids and robot<=mapping.allowed_support_geom_ids) and not forbidden:
                    force=np.zeros(6);mujoco.mj_contactForce(m,d,index,force)
                    forbidden.append(dict(time=float(d.time),geoms=[m.geom(int(g)).name for g in (c.geom1,c.geom2)],depth_mm=max(0.,-float(c.dist)*1000),normal_force=float(abs(force[0]))))
            return original(m,d)
        monitor.update=update
        maximum=0.;reward_sum=0.;precontact=[]
        for _ in range(e.horizon):
            a=np.zeros(13) if action is None else action(obs)
            maximum=max(maximum,float(np.abs(a).max()))
            if first_support is None:precontact.append(a.copy())
            if trace:trajectory.append(dict(t=float(data.time),qpos=data.qpos.tolist(),qvel=data.qvel.tolist(),action=a.tolist()))
            obs,reward,done,truncated,_=w.step(a);reward_sum+=float(reward)
            if done or truncated:break
        np.testing.assert_array_equal(gains,model.actuator_gainprm)
        np.testing.assert_array_equal(bias,model.actuator_biasprm)
        result=dict(case=case,scenario_hash=w.case_info['scenario_hash'],success=bool(s['landing_success']),reason=s['landing_failure_reason'],end_time=float(data.time),first_support=first_support,forbidden=forbidden,max_action=maximum,episode_return=reward_sum,peak_foot_force=float(monitor.peak_foot.max()),peak_calf_force=float(monitor.peak_calf.max()),peak_acceleration=float(monitor.peak_acceleration),min_height=float(monitor.min_height),precontact_mean_action=np.mean(precontact,axis=0).tolist())
        if trace:result['trajectory']=trajectory
        return result

    def close(self):self.wrapper.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    cases=scenarios();cases=cases if args.limit is None else cases[:args.limit]
    metadata={}
    # Read only the checkpoints archived for this experiment.
    for name in ['best','latest']:
        source=OUT/f'{name}.model';target=OUT/f'{name}.model'
        with zipfile.ZipFile(target) as z:
            metrics=json.loads(z.read('checkpoint_metrics.json'));env=json.loads(z.read('config_environment.json'))
        assert env['domain_randomization']['initial_state']['energy_dissipation_curriculum']['factorized_enabled']
        metadata[name]=dict(sha256=sha(target),metrics=metrics,environment=env)
    (OUT/'checkpoints.json').write_text(json.dumps(metadata,indent=2))
    (OUT/'protocol.json').write_text(json.dumps(dict(seed=SEED,controlled_scenarios=cases,controllers=['zero','best','latest'],fixed_pd=True,physics_randomization=False,deterministic_policy=True,dt=.005,control_dt=.02),indent=2))
    loader=make_report_env()
    actions={name:load_policy(OUT/f'{name}.model',loader.env)[0] for name in ['best','latest']};loader.close()
    results=[];hashes={}
    for name,action in [('zero',None),*actions.items()]:
        ex=Experiment()
        try:
            for i,case in enumerate(cases):
                trace=(case['height']==2.5 and (case['kind']=='nominal' or case['kind']=='speed' and case['speed'] in [1.5,3.] and case['direction']%90==0))
                result=ex.trial(case,action,trace);result['controller']=name
                if name=='zero':hashes[case['id']]=result['scenario_hash']
                assert result['scenario_hash']==hashes[case['id']]
                results.append(result)
                if (i+1)%32==0:print(name,i+1,'/',len(cases),flush=True)
        finally:ex.close()
        (OUT/'results.json').write_text(json.dumps(results,indent=2))
        print(name,'completed',sum(r['success'] for r in results if r['controller']==name),flush=True)


if __name__=='__main__':main()
