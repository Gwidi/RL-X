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
        self.final_speed=float(cfg.get('final_speed',0.));self.terminal_tolerance=float(cfg.get('terminal_velocity_tolerance',.25))
        if not np.isfinite(self.final_speed) or self.final_speed<0 or self.terminal_tolerance<=0:raise ValueError('Invalid terminal velocity settings')
        from .landing_slip import LandingSlip
        self.slip = LandingSlip(env, cfg['slip']) if cfg.get('slip', {}).get('weight', 0) > 0 else None
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
        u=np.clip(self.elapsed()/self.duration,0.,1.)
        initial=np.array(s['initial_world']);speed=float(np.linalg.norm(initial))
        if self.final_speed==0:return initial*(1-3*u*u+2*u*u*u)
        direction=initial/speed if speed>=.2 else np.array([1.,0.])
        magnitude=self.final_speed+(max(speed,self.final_speed)-self.final_speed)*(1-3*u*u+2*u*u*u)
        return direction*magnitude
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
        if self.slip is not None and s['touch_time'] is not None:
            reward -= self.slip.step(self.elapsed())
        return float(reward)

def check_braking_contract(saved_env,current_env,allow_legacy_parent=False):
    from .landing_frontflip import check_frontflip_contract
    check_frontflip_contract(saved_env, current_env)
    old=saved_env.get('landing_braking',{});new=current_env.get('landing_braking',{})
    a=bool(old.get('enabled',False));b=bool(new.get('enabled',False))
    if a!=b:
        if allow_legacy_parent and not a and b:return
        raise ValueError('Checkpoint braking task differs: use its native goal-reference environment')
    explicit_v11_warmstart = (allow_legacy_parent and a and b
        and new.get('warm_start_from_v11', False)
        and float(old.get('duration', 0)) == 2.5
        and float(old.get('tracking_weight', 0)) == 5.
        and not old.get('slip', {}).get('weight', 0))
    explicit_v12_warmstart=(allow_legacy_parent and a and b and new.get('warm_start_from_v12',False)
        and float(old.get('duration',0))==2. and float(old.get('tracking_weight',0))==6.
        and float(old.get('final_speed',0))==0 and old.get('slip',{}).get('weight')==5.)
    if explicit_v11_warmstart or explicit_v12_warmstart:return
    if a and (old.get('final_speed',0)!=new.get('final_speed',0) or old.get('terminal_velocity_tolerance',.25)!=new.get('terminal_velocity_tolerance',.25)):
        raise ValueError('Checkpoint terminal velocity task differs')
    if a and (old.get('success_tracks_reference',True)!=new.get('success_tracks_reference',True) or old.get('success_max_horizontal_speed',1.)!=new.get('success_max_horizontal_speed',1.)):
        raise ValueError('Checkpoint landing success contract differs')
    if a and float(old['duration'])!=float(new['duration']):
        raise ValueError('Checkpoint braking-reference duration differs')
    old_slip = old.get('slip', {}) if old.get('slip', {}).get('weight', 0) else {}
    new_slip = new.get('slip', {}) if new.get('slip', {}).get('weight', 0) else {}
    # Runner uses ConfigDict, native replay uses dict. Compare values rather
    # than treating these equivalent representations as different tasks.
    old_slip = old_slip.to_dict() if hasattr(old_slip, 'to_dict') else old_slip
    new_slip = new_slip.to_dict() if hasattr(new_slip, 'to_dict') else new_slip
    if a and old_slip != new_slip:
        raise ValueError('Checkpoint loaded-slip task differs')
