"""Open-loop flight replay isolates timestep sensitivity from policy decisions."""
import json
import numpy as np
import mujoco
from experiments.analysis.diagnose_tilt_control import Experiment,OUT

def main():
    results=json.loads((OUT/'results.json').read_text());out=[]
    ex=Experiment()
    try:
        for name in ['best','latest']:
            for axis,angle in [('nominal',0),('roll',30)]:
                r=next(r for r in results if r['controller']==name and r['case']['height']==2.5 and r['case']['axis']==axis and r['case']['angle']==angle);z=np.load(OUT/r['trace'])
                for dt in [.005,.0025,.00125]:
                    ex.reset(r['case']);e=ex.wrapper.env;m=e.internal_state['mj_model'];m.opt.timestep=dt
                    d=mujoco.MjData(m);d.qpos[:]=z['qpos'][0];d.qvel[:]=z['qvel'][0];mujoco.mj_forward(m,d)
                    qa=m.jnt_qposadr[e.actuator_joint_mask_joints];trace=[];original_error=[]
                    for k,t in enumerate(z['control_t']):
                        if t>=.3-1e-9:break
                        d.ctrl[:]=z['control_target'][k]*m.actuator_gear[:,0]
                        for _ in range(round(.02/dt)):
                            mujoco.mj_step(m,d)
                            trace.append([float(d.time),*d.actuator_force.tolist(),*d.qpos[qa].tolist()])
                            if dt==.005:
                                idx=int(round(d.time/.005));original_error.append(float(np.max(np.abs(d.qpos-z['qpos'][idx]))))
                        limit=np.ones(m.nv)*100;limit[e.actuator_joint_mask_qvel]=e.internal_state['actuator_joint_max_velocities'];d.qvel[:]=np.clip(d.qvel,-limit,limit)
                    a=np.array(trace);mask=a[:,0]>.05;tau=a[mask,1:14];sat=np.abs(tau)/np.max(np.abs(m.actuator_forcerange),axis=1)
                    filename=f'replay_{name}_{axis}_{dt:g}.npz';np.savez_compressed(OUT/filename,t=a[:,0],motor_torque=a[:,1:14],q=a[:,14:])
                    out.append(dict(controller=name,axis=axis,angle=angle,dt=dt,trace=filename,integrator=int(m.opt.integrator),baseline_max_qpos_difference=max(original_error) if original_error else None,FL_calf_sign_flip_fraction=float(np.mean(tau[1:,12]*tau[:-1,12]<0)),FL_calf_saturated_fraction=float(np.mean(sat[:,12]>=.95)),calf_saturated_fraction=float(np.mean(sat[:,[3,6,9,12]]>=.95))))
                    m.opt.timestep=.005
        assert max(r['baseline_max_qpos_difference'] for r in out if r['dt']==.005)<1e-8
    finally:ex.close()
    (OUT/'timestep_probe.json').write_text(json.dumps(dict(mode='open-loop replay of recorded targets for first 0.3s, h=2.5m; not a policy success evaluation',rows=out),indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
