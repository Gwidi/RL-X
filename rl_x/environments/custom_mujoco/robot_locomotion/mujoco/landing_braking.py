"""Opt-in smooth horizontal braking goal, driven by actor-visible measurement.
All episode state is in internal_state, preserving reset/deepcopy pairing.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .policy_velocity import heading_velocity_xy

class LandingBraking:
    def __init__(self,env):
        self.env=env;cfg=env.env_config.get('landing_braking',{})
        self.enabled=bool(cfg.get('enabled',False));self.duration=float(cfg.get('duration',2.5));self.weight=float(cfg.get('tracking_weight',5.));self.sigma=float(cfg.get('tracking_sigma',.5))
        if self.enabled:
            if not all(np.isfinite(v) and v>0 for v in [self.duration,self.weight,self.sigma]):raise ValueError('Braking parameters must be finite and positive')
            if env.env_config.get('policy_velocity',{}).get('mode')!='heading_xy':raise ValueError('Braking requires heading_xy measured velocity')
            if env.env_config['reward'].get('base_vel_coeff',0)!=0:raise ValueError('Braking tracking and absolute body-speed cost are mutually exclusive')
            if self.duration+env.env_config['reward'].get('success_stability_time',.5)>=env.env_config['episode_length_in_seconds']:raise ValueError('Braking and final stability do not fit the episode')
    def state(self):
        return self.env.internal_state.setdefault('landing_braking',dict(last_precontact_world=np.zeros(2),initial_world=None,touch_time=None,error_integral=0.,tracked_time=0.,path=0.,last_xy=None))
    def elapsed(self):
        s=self.state();return 0. if s['touch_time'] is None else max(0.,float(self.env.internal_state['data'].time)-s['touch_time'])
    def reference_world(self):
        s=self.state()
        if s['initial_world'] is None:return np.array(s['last_precontact_world'])
        u=np.clip(self.elapsed()/self.duration,0.,1.);return np.array(s['initial_world'])*(1-3*u*u+2*u*u*u)
    def goal(self):
        e=self.env;s=self.state()
        if s['initial_world'] is None:
            e.policy_velocity_observation.observe(e)
            s['last_precontact_world']=np.array(e.internal_state['policy_velocity_sensor']['history'][0][:2]).copy()
        world=np.r_[self.reference_world(),0.];rotation=Rotation.from_quat(e.internal_state['data'].qpos[3:7],scalar_first=True).as_matrix()
        return np.r_[heading_velocity_xy(world,rotation),0.] # Existing goal channels are raw m/s, not /4.
    def ready(self):return self.state()['initial_world'] is not None and self.elapsed()+1e-9>=self.duration
    def step(self):
        e=self.env;s=self.state();data=e.internal_state['data'];info=e.internal_state['info'];reward=0.
        if e.internal_state['has_touched_ground']:
            if s['initial_world'] is None:
                s['initial_world']=np.array(s['last_precontact_world']).copy();s['touch_time']=float(data.time);s['last_xy']=data.qpos[:2].copy()
            error=float(np.sum((data.qvel[:2]-self.reference_world())**2));reward=self.weight*e.dt*np.exp(-error/self.sigma**2)
            s['error_integral']+=error*e.dt;s['tracked_time']+=e.dt;s['path']+=float(np.linalg.norm(data.qpos[:2]-s['last_xy']));s['last_xy']=data.qpos[:2].copy()
        ref=self.reference_world();info.update({'reward/braking_tracking':float(reward),'braking/reference_vx':float(ref[0]),'braking/reference_vy':float(ref[1]),'braking/elapsed':self.elapsed(),'metrics/braking_tracking_mse':s['error_integral']/max(e.dt,s['tracked_time']),'metrics/braking_postcontact_path':s['path'],'metrics/braking_elapsed':self.elapsed()})
        return float(reward)

def check_braking_contract(saved_env,current_env,allow_legacy_parent=False):
    old=saved_env.get('landing_braking',{});new=current_env.get('landing_braking',{})
    a=bool(old.get('enabled',False));b=bool(new.get('enabled',False))
    if a!=b:
        if allow_legacy_parent and not a and b:return
        raise ValueError('Checkpoint braking task differs: use its native goal-reference environment')
    if a and float(old['duration'])!=float(new['duration']):
        raise ValueError('Checkpoint braking-reference duration differs')
