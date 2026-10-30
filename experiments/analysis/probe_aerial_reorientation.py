"""Free-flight authority probe, using actual v8 PD mapping and unconstrained base.
No policy training. Actions are deterministic candidate trajectories, not a bound
on the robot's reachable set. Gravity and all robot collisions remain enabled.
"""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from experiments.train_landing_seed606_v8 import make_velocity_only_env
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/artifacts/aerial_reorientation_feasibility'

def rotation(q):return Rotation.from_quat(q,scalar_first=True)

def candidates(names):
    yield dict(name='zero',duration=.4,amplitude=0.,a=[0.]*13,b=[0.]*13)
    patterns={}
    for group in ['hip','thigh','calf']:
        for symmetry in ['all','left_right','front_rear']:
            a=np.zeros(13)
            for i,name in enumerate(names):
                if group not in name:continue
                a[i]=1 if symmetry=='all' else (1 if (name[:2] in (['RL','FL'] if symmetry=='left_right' else ['FR','FL'])) else -1)
            patterns[f'{group}_{symmetry}']=a
    a=np.zeros(13);a[0]=1;patterns['spine']=a
    for name,a in patterns.items():
        for duration in [.2,.4,.6]:
            for amp in [.5,1.]:
                for sign in [-1,1]:
                    for kind in ['cycle','hold']:
                        yield dict(name=f'{name}_{duration}_{amp}_{sign}_{kind}',kind=kind,duration=duration,amplitude=amp,a=(a*amp*sign).tolist(),b=(-a*amp*sign).tolist())
    rng=np.random.default_rng(20261029)
    for k in range(120):
        yield dict(name=f'coordinated_{k:03d}',duration=float(rng.choice([.2,.4,.6])),amplitude=1.,a=rng.uniform(-1,1,13).tolist(),b=rng.uniform(-1,1,13).tolist())

def action_at(c,t):
    # Three smooth segments: home -> A -> B -> home; then hold home.
    if c.get('kind')=='hold':
        f=(1-np.cos(np.pi*min(1.,t/c['duration'])))/2
        return np.array(c['a'])*f
    u=t/c['duration']*3
    if u>=3:return np.zeros(13)
    knots=[np.zeros(13),np.array(c['a']),np.array(c['b']),np.zeros(13)]
    k=int(u);f=(1-np.cos(np.pi*(u-k)))/2
    return knots[k]*(1-f)+knots[k+1]*f

class Probe:
    def __init__(self):
        self.env=make_velocity_only_env();self.env.reset()
        self.model=self.env.internal_state['mj_model'];self.data=self.env.internal_state['data']
        self.q0=self.data.qpos.copy();self.q0[:3]=[0,0,20.];self.q0[3:7]=[1,0,0,0]
        self.names=[self.model.joint(int(j)).name for j in self.env.actuator_joint_mask_joints]
        self.base=int(self.model.jnt_bodyid[0]);self.dt=self.model.opt.timestep
        self.air=(self.model.opt.density,self.model.opt.viscosity)
        assert int(self.model.jnt_type[0])==int(mujoco.mjtJoint.mjJNT_FREE)
        assert np.all(self.model.dof_damping[:6]==0)
        assert np.allclose(self.model.opt.gravity,[0,0,-9.81])
    def trial(self,c,height=20.,duration=.8,physics_dt=None,native_air=False,initial_angles=(0.,0.,0.)):
        m,d=self.model,self.data;m.opt.timestep=self.dt if physics_dt is None else physics_dt
        m.opt.density,m.opt.viscosity=self.air if native_air else (0.,0.)
        mujoco.mj_resetData(m,d);d.qpos[:]=self.q0;d.qpos[2]=height;d.qpos[3:7]=Rotation.from_euler("xyz",initial_angles,degrees=True).as_quat(scalar_first=True);d.qvel[:]=0
        d.ctrl[:]=self.env.control_function.process_action(np.zeros(13));mujoco.mj_forward(m,d);mujoco.mj_subtreeVel(m,d)
        com0=d.subtree_com[self.base].copy();mom0=d.subtree_angmom[self.base].copy()
        gravity=m.opt.gravity.copy();trace=[];qpos=[];mom_error=0.;com_error=0.;contacts=set();peak=np.zeros(13);first_contact=None
        steps=round(duration/m.opt.timestep);stride=round(.02/m.opt.timestep)
        for step in range(steps+1):
            mujoco.mj_forward(m,d);mujoco.mj_subtreeVel(m,d)
            for contact in d.contact:
                if contact.dist<=0:
                    contacts.add(tuple(sorted([m.geom(int(contact.geom1)).name or str(contact.geom1),m.geom(int(contact.geom2)).name or str(contact.geom2)])))
                    if 0 in [m.geom_bodyid[contact.geom1],m.geom_bodyid[contact.geom2]]:first_contact=float(d.time)
            if first_contact is not None:break # retain strictly pre-contact samples
            mom_error=max(mom_error,float(np.linalg.norm(d.subtree_angmom[self.base]-mom0)))
            com_error=max(com_error,float(np.linalg.norm(d.subtree_com[self.base]-(com0+.5*gravity*d.time*d.time))))
            peak=np.maximum(peak,np.abs(d.actuator_force))
            if step%stride==0 or step==steps:
                angles=rotation(d.qpos[3:7]).as_rotvec()*180/np.pi
                trace.append([float(d.time),*angles,float(np.linalg.norm(d.qvel[3:6]))]);qpos.append(d.qpos.copy())
            if step==steps:break
            if step%stride==0:d.ctrl[:]=self.env.control_function.process_action(action_at(c,d.time))
            mujoco.mj_step(m,d)
            peak=np.maximum(peak,np.abs(d.actuator_force))
        trace=np.array(trace)
        result=dict(candidate=c,initial_angles=list(initial_angles),final_tilt_deg=float(np.degrees(np.arccos(np.clip(rotation(qpos[-1][3:7]).apply([0,0,1])[2],-1,1)))),native_air=native_air,height=height,duration=duration,physics_dt=float(m.opt.timestep),final_time=float(trace[-1,0]),
            rotation_vector_deg=trace[-1,1:4].tolist(),final_angular_speed=float(trace[-1,4]),
            peak_rotation_deg=float(np.max(np.linalg.norm(trace[:,1:4],axis=1))),
            peak_motor_torque=peak.tolist(),within_nominal_motor_limits=bool(peak[0]<=.785+1e-6 and np.max(peak[1:])<=3.+1e-6),
            contacts=sorted(contacts),first_ground_contact=first_contact,angular_momentum_drift=mom_error,com_ballistic_error_m=com_error)
        return result,trace,np.array(qpos)
    def close(self):self.env.close()

def main():
    OUT.mkdir(parents=True,exist_ok=True);probe=Probe();rows=[]
    try:
        for i,c in enumerate(candidates(probe.names)):
            r,_,_=probe.trial(c);rows.append(r)
            if i%60==0:print('Candidates',i,flush=True)
        chosen={}
        for axis,label in enumerate(['roll','pitch','yaw']):
            feasible=[r for r in rows if not r['contacts'] and r['within_nominal_motor_limits'] and r['final_angular_speed']<2.]
            closed=[r for r in feasible if r['candidate'].get('kind')!='hold']
            held=[r for r in feasible if r['candidate'].get('kind')=='hold']
            chosen[label+'_hold']=max(held,key=lambda r:abs(r['rotation_vector_deg'][axis])) if held else rows[0]
            chosen[label]=max(closed,key=lambda r:abs(r['rotation_vector_deg'][axis])) if feasible else rows[0]
        validations=[]
        for label,r in chosen.items():
            _,trace,qpos=probe.trial(r['candidate']);np.savez_compressed(OUT/f'{label}_trace.npz',trace=trace,qpos=qpos)
            for height in [1.,2.,3.]:
                result,_,_=probe.trial(r['candidate'],height=height,native_air=True);result['axis']=label;validations.append(result)
            fine,_,_=probe.trial(r['candidate'],physics_dt=.00125);fine['axis']=label;validations.append(fine)
        for label,angles in [('roll_hold',(-15.,0.,0.)),('pitch_hold',(0.,-10.,0.))]:
            result,_,_=probe.trial(chosen[label]['candidate'],height=2.,native_air=True,initial_angles=angles)
            result['axis']=label;validations.append(result)
        (OUT/'results.json').write_text(json.dumps(dict(protocol=dict(candidates=len(rows),flight_seconds=.8,start_height=20.,gravity=[0,0,-9.81],native_air_density=probe.air[0],native_air_viscosity=probe.air[1],main_air='vacuum to isolate internal inertial authority; native air retained for 1/2/3m validation',control_dt=.02,physics_dt=.0025,actions='actual residual PD, smooth home-A-B-home or home-A-hold, within [-1,1]',seed=20261029,selection='maximum absolute final rotation component, no contacts, motor limits respected, angular speed <2 rad/s',limitation='Finite candidate search; no landing success claim. Rotation vectors, not Euler angles. 20m isolates aerial dynamics; 1/2/3m stop at first ground contact.'),joints=probe.names,rows=rows,selected=chosen,validation=validations,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
        print(json.dumps({k:{x:v[x] for x in ['rotation_vector_deg','final_angular_speed','angular_momentum_drift','com_ballistic_error_m']} for k,v in chosen.items()},indent=2))
    finally:probe.close()
if __name__=='__main__':main()
