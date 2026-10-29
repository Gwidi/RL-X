"""Read-only ablations: smaller PD gains and integration variants."""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.analysis.diagnose_tilt_control import OUT as SOURCE
from experiments.analysis.report_landing_seed606 import make_report_env,sha
import mujoco
import numpy as np
OUT=Path(__file__).resolve().parents[2]/'docs/artifacts/pd_stability_v4'

VARIANTS={
 'original':dict(kp=1.,kd=1.,dt=.005,integrator='Euler'),
 'kp_half':dict(kp=.5,kd=1.,dt=.005,integrator='Euler'),
 'kd_075':dict(kp=1.,kd=.75,dt=.005,integrator='Euler'),
 'kd_half':dict(kp=1.,kd=.5,dt=.005,integrator='Euler'),
 'kd_quarter':dict(kp=1.,kd=.25,dt=.005,integrator='Euler'),
 'kd_tenth':dict(kp=1.,kd=.1,dt=.005,integrator='Euler'),
 'both_half':dict(kp=.5,kd=.5,dt=.005,integrator='Euler'),
 'both_quarter':dict(kp=.25,kd=.25,dt=.005,integrator='Euler'),
 'calf_kd_half':dict(kp=1.,kd=1.,calf_kd=.5,dt=.005,integrator='Euler'),
 'calf_kd_quarter':dict(kp=1.,kd=1.,calf_kd=.25,dt=.005,integrator='Euler'),
 'implicit_5ms':dict(kp=1.,kd=1.,dt=.005,integrator='implicitfast'),
 'implicit_2p5ms':dict(kp=1.,kd=1.,dt=.0025,integrator='implicitfast'),
 'implicit_1p25ms':dict(kp=1.,kd=1.,dt=.00125,integrator='implicitfast'),
 'euler_2p5ms':dict(kp=1.,kd=1.,dt=.0025,integrator='Euler'),
 'euler_1p25ms':dict(kp=1.,kd=1.,dt=.00125,integrator='Euler'),
}

def apply_variant(model,variant):
    v=VARIANTS[variant];model.opt.timestep=v['dt'];model.opt.integrator=getattr(mujoco.mjtIntegrator,'mjINT_'+v['integrator'].upper())
    model.actuator_gainprm[:,0]*=v['kp'];model.actuator_biasprm[:,1]*=v['kp'];model.actuator_biasprm[:,2]*=v['kd']
    if 'calf_kd' in v:
        for i in range(model.nu):
            if 'calf' in model.actuator(i).name:model.actuator_biasprm[i,2]*=v['calf_kd']

def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'replays').mkdir(exist_ok=True)
    rows=json.loads((SOURCE/'results.json').read_text());w=make_report_env();original=w.env.initial_mj_model
    results=[]
    try:
        for variant,v in VARIANTS.items():
            for name in ['best','latest']:
                for axis,angle in [('nominal',0),('roll',30)]:
                    r=next(r for r in rows if r['controller']==name and r['case']['height']==2.5 and r['case']['axis']==axis and r['case']['angle']==angle);z=np.load(SOURCE/r['trace'])
                    import copy
                    m=copy.deepcopy(original);apply_variant(m,variant);d=mujoco.MjData(m);d.qpos[:]=z['qpos'][0];d.qvel[:]=z['qvel'][0];mujoco.mj_forward(m,d);qa=m.jnt_qposadr[w.env.actuator_joint_mask_joints];trace=[]
                    for k,t in enumerate(z['control_t']):
                        if t>=.3-1e-9:break
                        d.ctrl[:]=z['control_target'][k]*m.actuator_gear[:,0]
                        for _ in range(round(.02/v['dt'])):
                            mujoco.mj_step(m,d);trace.append([float(d.time),*d.actuator_force.tolist(),*d.qpos[qa].tolist()])
                        limit=np.ones(m.nv)*100;limit[w.env.actuator_joint_mask_qvel]=w.env.actuator_joint_max_velocities;d.qvel[:]=np.clip(d.qvel,-limit,limit)
                    a=np.array(trace);mask=a[:,0]>.05;tau=a[mask,1:14];q=a[mask,14:];limits=np.max(np.abs(m.actuator_forcerange),axis=1);util=np.abs(tau)/limits;target=np.stack([np.interp(a[mask,0],z['t'],z['target'][:,j]) for j in range(13)],axis=1)
                    filename=f'replays/{variant}_{name}_{axis}.npz';np.savez_compressed(OUT/filename,t=a[:,0],tau=a[:,1:14],q=a[:,14:])
                    results.append(dict(variant=variant,controller=name,axis=axis,calf_saturation_fraction=float(np.mean(util[:,[3,6,9,12]]>=.95)),FL_sign_flip_fraction=float(np.mean(tau[1:,12]*tau[:-1,12]<0)),tracking_rms_deg=float(np.degrees(np.sqrt(np.mean((target-q)**2)))),trace=filename))
            print(variant,'calf saturation',np.mean([r['calf_saturation_fraction'] for r in results if r['variant']==variant]),flush=True)
    finally:w.close()
    (OUT/'replay_results.json').write_text(json.dumps(dict(variants=VARIANTS,rows=results),indent=2))
if __name__=='__main__':main()
