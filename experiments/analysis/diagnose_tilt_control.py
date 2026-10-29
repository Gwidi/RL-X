"""Paired deterministic tilt sweep, with read-only physics-rate diagnostics."""
from pathlib import Path
import argparse
import json
import sys
import hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.analysis.evaluate_seed606_v3 import Experiment,OUT as CHECKPOINTS
from experiments.analysis.report_landing_seed606 import load_policy,sha
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/artifacts/tilt_control_diagnostic'
NAMES=['zero','best','latest']


def cases():
    result=[]
    for h in [1.,1.5,2.,2.5]:
        for axis,angle in [('nominal',0)]+[(axis,sign*a) for axis in ['roll','pitch'] for a in [3,10,20,30] for sign in [-1,1]]:
            result.append(dict(id=80000+len(result),kind='tilt_diagnostic',height=h,roll=float(angle if axis=='roll' else 0),pitch=float(angle if axis=='pitch' else 0),vx=0.,vy=0.,axis=axis,angle=angle))
    return result


def trial(ex,case,action,name):
    obs=ex.reset(case);e=ex.wrapper.env;s=e.internal_state;m=s['mj_model'];d=s['data'];mon=s['landing_safety'];original=mon.update
    joints=np.asarray(e.actuator_joint_mask_joints);qa=m.jnt_qposadr[joints];va=m.jnt_dofadr[joints];gear=m.actuator_gear[:,0].copy()
    gains=m.actuator_gainprm.copy();bias=m.actuator_biasprm.copy();ranges=m.actuator_forcerange.copy();home=s['actuator_joint_nominal_positions'].copy()
    fields={k:[] for k in ['t','q','qd','target','motor_torque','joint_torque','motor_limit','rpy','action','qpos','qvel']}
    control={k:[] for k in ['t','action','target']};current=np.zeros(13)
    first_contact=None;first_loaded=None;first_forbidden=None
    def snapshot(torque=True):
        fields['t'].append(float(d.time));fields['q'].append(d.qpos[qa].copy());fields['qd'].append(d.qvel[va].copy());fields['target'].append(d.ctrl.copy()/gear)
        fields['motor_torque'].append(d.actuator_force.copy() if torque else np.zeros(13));fields['joint_torque'].append(d.actuator_force.copy()*gear if torque else np.zeros(13));fields['motor_limit'].append(np.max(np.abs(m.actuator_forcerange),axis=1))
        fields['rpy'].append(Rotation.from_quat(d.qpos[3:7],scalar_first=True).as_euler('xyz',degrees=True));fields['action'].append(current.copy());fields['qpos'].append(d.qpos.copy());fields['qvel'].append(d.qvel.copy())
    def update(model,data):
        nonlocal first_contact,first_loaded,first_forbidden
        force=np.zeros(6)
        for idx,c in enumerate(data.contact):
            pair={int(c.geom1),int(c.geom2)};robot={g for g in pair if model.geom_bodyid[g]!=0}
            if not robot:continue
            mujoco.mj_contactForce(model,data,idx,force)
            item=dict(time=float(data.time),geoms=[model.geom(int(g)).name for g in (c.geom1,c.geom2)],distance_m=float(c.dist),normal_force_n=float(abs(force[0])))
            if pair & mon.contacts.support_geom_ids:
                if first_contact is None and c.dist<=0:first_contact=item
                if first_loaded is None and abs(force[0])>=1.:first_loaded=item
            if first_forbidden is None and not(pair & mon.contacts.support_geom_ids and robot<=mon.contacts.allowed_support_geom_ids):first_forbidden=item
        snapshot();return original(model,data)
    # No state changes beyond those made by the normal environment.
    mon.update=update
    d.ctrl[:]=e.control_function.process_action(np.zeros(13));snapshot(torque=False)
    try:
        for _ in range(e.horizon):
            current=np.zeros(13) if action is None else action(obs)
            control['t'].append(float(d.time));control['action'].append(current.copy());control['target'].append(e.control_function.process_action(current)/gear)
            obs,_,done,truncated,_=ex.wrapper.step(current)
            if done or truncated:break
    finally:mon.update=original
    for arr,expected in [(m.actuator_gainprm,gains),(m.actuator_biasprm,bias),(m.actuator_forcerange,ranges)]:np.testing.assert_array_equal(arr,expected)
    fields={k:np.asarray(v) for k,v in fields.items()};control={k:np.asarray(v) for k,v in control.items()}
    arrays={**fields,**{'control_'+k:v for k,v in control.items()},'home':home}
    filename=f'{name}_{case["id"]}.npz';np.savez_compressed(OUT/'traces'/filename,**arrays)
    contact=first_contact['time'] if first_contact else float(d.time)
    pre=(fields['t']>0)&(fields['t']<contact);post=fields['t']>=contact
    metrics={}
    for phase,mask in [('flight',pre),('after_contact',post)]:
        if not np.any(mask):metrics[phase]=None;continue
        err=fields['target'][mask]-fields['q'][mask];util=np.abs(fields['motor_torque'][mask])/fields['motor_limit'][mask]
        metrics[phase]=dict(tracking_rms_deg=float(np.degrees(np.sqrt(np.mean(err**2)))),tracking_p95_abs_deg=float(np.degrees(np.quantile(np.abs(err),.95))),per_joint_rms_deg=np.degrees(np.sqrt(np.mean(err**2,axis=0))).tolist(),motor_saturation_fraction=float(np.mean(util>=.95)),motor_saturation_any_fraction=float(np.mean(np.any(util>=.95,axis=1))),per_joint_saturation_fraction=np.mean(util>=.95,axis=0).tolist(),max_motor_utilization=float(util.max()),max_abs_action=float(np.abs(fields['action'][mask]).max()),action_bound_fraction=float(np.mean(np.abs(fields['action'][mask])>=.99)))
    last=np.where(fields['t']<contact)[0][-1];idx=0 if case['axis']=='roll' else 1
    return dict(controller=name,case=case,scenario_hash=ex.wrapper.case_info['scenario_hash'],success=bool(s['landing_success']),reason=s['landing_failure_reason'],end_time=float(d.time),first_contact=first_contact,first_loaded_contact=first_loaded,first_forbidden=first_forbidden,rpy_before_contact=fields['rpy'][last].tolist(),angle_change_before_contact=float(fields['rpy'][last,idx]-case['angle']) if case['axis']!='nominal' else None,metrics=metrics,trace='traces/'+filename)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'traces').mkdir(exist_ok=True)
    scenarios=cases();loader=Experiment();metadata={};actions={'zero':None}
    for name in ['best','latest']:actions[name],metadata[name]=load_policy(CHECKPOINTS/f'{name}.model',loader.wrapper.env)
    loader.close()
    if args.smoke:
        ex=Experiment();c=scenarios[0];r=trial(ex,c,actions['best'],'smoke');baseline=ex.trial(c,actions['best']);ex.close()
        assert r['success']==baseline['success'] and r['reason']==baseline['reason'] and abs(r['end_time']-baseline['end_time'])<1e-9 and r['scenario_hash']==baseline['scenario_hash']
        print('Read-only instrumentation preserves episode outcome/time/hash',flush=True);return
    protocol=dict(cases=scenarios,controllers=NAMES,physics_dt=.005,control_dt=.02,checkpoint_sha256={n:metadata[n]['sha256'] for n in metadata},deterministic=True,notes=['Pure roll/pitch, both signs, initial vx=vy=vz=0; nominal angular velocities, no added perturbation.','Joint torque = actuator_force * gear; actuator_force is motor-side.','Physics samples after mj_step; force/contact dynamics buffers correspond to integration step (timing uncertainty <=5ms).','Flight ends at first ground/support contact with distance <=0; first >=1N load recorded separately.','No training or reward changes; large tilts are out-of-training-distribution.'],source_sha256=sha(Path(__file__)))
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));results=[];hashes={}
    for name in NAMES:
        ex=Experiment()
        try:
            for i,c in enumerate(scenarios):
                r=trial(ex,c,actions[name],name)
                if name=='zero':hashes[c['id']]=r['scenario_hash']
                assert r['scenario_hash']==hashes[c['id']]
                results.append(r)
                if (i+1)%17==0:print(name,i+1,'/',len(scenarios),flush=True)
        finally:ex.close()
        (OUT/'results.json').write_text(json.dumps(results,indent=2))
    print('COMPLETE',len(results),'episodes',flush=True)

if __name__=='__main__':main()
