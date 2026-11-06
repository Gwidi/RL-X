"""Opt-in contact -> aerial forward flip -> second landing reward.
Physics-rate phase, positive actuator work and peak costs; control-rate signed
mechanical-energy shaping. All episode state is deepcopy/reset compatible.
"""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from .landing_slip import LandingSlip

DEFAULTS = dict(enabled=False, energy_weight=10., work_weight=10., force_weight=20.,
                acceleration_weight=30., time_weight=1., takeoff_bonus=2., half_bonus=3.,
                landing_bonus=5., load_threshold_n=10., takeoff_clear_s=.02,
                min_air_s=.08, min_takeoff_vz=.3, min_pitch_rate=.5,
                lateral_tolerance_deg=30., upright_tolerance_deg=20.,
                rotation_tolerance_rad=.2, max_rotation_rad=2*np.pi+.5,
                ready_com_height=.25, energy_floor_j=1., slip_weight=5.)

class LandingFrontflip:
    def __init__(self, env):
        self.env=env
        self.cfg={**DEFAULTS, **env.env_config.get('landing_frontflip',{})}
        self.enabled=bool(self.cfg['enabled'])
        for k,v in self.cfg.items():
            if k!='enabled' and (not np.isfinite(v) or v<0):raise ValueError(f'Invalid frontflip {k}')
        if self.cfg['energy_floor_j']<=0 or self.cfg['min_air_s']<=self.cfg['takeoff_clear_s']:
            raise ValueError('Invalid frontflip energy floor or airborne debounce')
        if self.enabled and env.landing_braking.enabled:
            raise ValueError('Frontflip and braking tracking are mutually exclusive')
        self.slip=LandingSlip(env,dict(weight=self.cfg['slip_weight']))

    def state(self):
        # LocomotionEnv queries reward once during construction, before reset.
        if 'landing_frontflip' not in self.env.internal_state:self.reset()
        return self.env.internal_state['landing_frontflip']

    def energy(self):
        e=self.env;m=e.internal_state['mj_model'];d=e.internal_state['data']
        mujoco.mj_energyPos(m,d);mujoco.mj_energyVel(m,d)
        return float(np.sum(d.energy))

    def reset(self):
        e=self.env;m=e.internal_state['mj_model'];d=e.internal_state['data']
        mujoco.mj_forward(m,d)
        R=Rotation.from_quat(d.qpos[3:7],scalar_first=True)
        lateral=R.apply([0,1,0]);lateral[2]=0;lateral/=max(np.linalg.norm(lateral),1e-12)
        energy=self.energy();scale=max(energy-float(m.body_mass.sum())*float(np.linalg.norm(m.opt.gravity))*self.cfg['ready_com_height'],self.cfg['energy_floor_j'])
        e.internal_state['landing_frontflip']=dict(previous_rotation=R.as_quat(),axis=lateral,
            angle=0.,touch=None,clear_since=None,air_start=None,return_time=None,landed=False,half=False,invalid=False,
            work=0.,reported_work=0.,peak_cost=0.,reported_peak=0.,pending_bonus=0.,
            previous_energy=energy,energy_scale=scale,loaded_feet=0,upright=False)

    def physics_step(self):
        e=self.env;m=e.internal_state['mj_model'];d=e.internal_state['data'];s=self.state();c=self.cfg
        # Actuator velocity is gear*joint velocity, matching motor-side force.
        # Sum positive work per actuator; do not cancel one motor's absorption
        # against another motor's positive mechanical work.
        s['work']+=float(np.maximum(d.actuator_force*d.actuator_velocity,0).sum())*m.opt.timestep
        safety=e.internal_state['landing_safety'];mapping=safety.contacts
        forces=mapping.support_contact_forces(m,d)[2]
        s['loaded_feet']=int(np.sum(forces>=c['load_threshold_n']))
        loaded=bool(np.any(forces>=c['load_threshold_n']))
        # A lightly loaded foot (or an allowed distal calf) still touching
        # the floor is not an aerial phase. Debounce geometric separation too.
        separated = not any(contact.dist <= 0 and
            {int(contact.geom1),int(contact.geom2)} & mapping.support_geom_ids and
            {int(contact.geom1),int(contact.geom2)} & mapping.allowed_support_geom_ids
            for contact in d.contact)
        clear = not np.any(forces > 1.) and separated
        R=Rotation.from_quat(d.qpos[3:7],scalar_first=True)
        delta=(R*Rotation.from_quat(s['previous_rotation']).inv()).as_rotvec()
        s['previous_rotation']=R.as_quat()
        if s['touch'] is None and loaded:s['touch']=float(d.time)
        if s['touch'] is not None:
            s['angle']+=float(np.dot(delta,s['axis']))
            lateral=float(np.dot(R.apply([0,1,0]),s['axis']))
            if lateral<np.cos(np.deg2rad(c['lateral_tolerance_deg'])) or s['angle']>c['max_rotation_rad'] or s['angle']<-.5:
                s['invalid']=True
            if clear:
                if s['clear_since'] is None:s['clear_since']=float(d.time)
                if s['air_start'] is None and d.time-s['clear_since']+1e-9>=c['takeoff_clear_s']:
                    mujoco.mj_subtreeVel(m,d)
                    pitch_rate=float(np.dot(R.apply(d.qvel[3:6]),s['axis']))
                    if d.subtree_linvel[1,2]>=c['min_takeoff_vz'] and pitch_rate>=c['min_pitch_rate'] and not s['invalid']:
                        s['air_start']=s['clear_since'];s['pending_bonus']+=c['takeoff_bonus']
            else:s['clear_since']=None
            # A touch during the rotation cannot be hidden by a later hop.
            if s['air_start'] is not None and not clear and s['return_time'] is None:
                s['return_time']=float(d.time)
                if (d.time-s['air_start']<c['min_air_s'] or not s['half']
                        or s['angle']<2*np.pi-c['rotation_tolerance_rad']):s['invalid']=True
            airborne=s['air_start'] is not None and clear and s['return_time'] is None
            if airborne and not s['half'] and s['angle']>=np.pi and not s['invalid']:
                s['half']=True;s['pending_bonus']+=c['half_bonus']
            if loaded and s['air_start'] is not None and not s['landed']:
                if d.time-s['air_start']<c['min_air_s']:s['invalid']=True
                elif s['half'] and s['angle']>=2*np.pi-c['rotation_tolerance_rad'] and not s['invalid']:
                    s['landed']=True;s['pending_bonus']+=c['landing_bonus']
                else:s['invalid']=True
        s['upright']=bool(R.apply([0,0,1])[2]>=np.cos(np.deg2rad(c['upright_tolerance_deg'])))
        # Monotone episode peaks include both impacts; paid once per increase.
        s['peak_cost']=c['force_weight']*float((safety.peak_foot.max()/3000.)**2)+c['acceleration_weight']*float((safety.peak_acceleration/98.1)**2)

    def ready(self):
        s=self.state()
        return s['landed'] and not s['invalid'] and s['upright'] and s['loaded_feet']>=2

    def goal(self):
        s=self.state();phase=3 if s['landed'] else 2 if s['air_start'] is not None else 1 if s['touch'] is not None else 0
        t=float(self.env.internal_state['data'].time)
        return np.array([phase/3.,np.clip(s['angle']/(2*np.pi),-1.,1.5),0. if s['air_start'] is None else min((t-s['air_start'])/2.,1.)])

    def step(self):
        e=self.env;s=self.state();c=self.cfg;energy=self.energy()
        terms=dict(energy=c['energy_weight']*(s['previous_energy']-energy)/s['energy_scale'],
            positive_work=-c['work_weight']*(s['work']-s['reported_work'])/s['energy_scale'],
            peaks=-(s['peak_cost']-s['reported_peak']),milestones=s['pending_bonus'],time=-c['time_weight']*e.dt)
        s.update(previous_energy=energy,reported_work=s['work'],reported_peak=s['peak_cost'],pending_bonus=0.)
        if s['touch'] is not None:terms['slip']=-self.slip.step(float(e.internal_state['data'].time)-s['touch'])
        info=e.internal_state['info'];info.update({f'reward/frontflip_{k}':float(v) for k,v in terms.items()})
        info.update({'metrics/frontflip_angle':s['angle'],'metrics/frontflip_positive_work_j':s['work'],
            'metrics/frontflip_ready':float(self.ready()),'metrics/frontflip_invalid':float(s['invalid']),
            'metrics/frontflip_takeoff':float(s['air_start'] is not None),
            'metrics/frontflip_half':float(s['half']),'metrics/frontflip_landed':float(s['landed']),
            'metrics/frontflip_energy_scale_j':s['energy_scale'],'metrics/frontflip_energy_j':energy})
        return float(sum(terms.values()))

def check_frontflip_contract(old,new):
    a=old.get('landing_frontflip',{});b=new.get('landing_frontflip',{})
    if bool(a.get('enabled',False))!=bool(b.get('enabled',False)):
        raise ValueError('Checkpoint frontflip task differs; use its native observation/reward contract')
    if a.get('enabled',False):
        a=a.to_dict() if hasattr(a,'to_dict') else dict(a)
        b=b.to_dict() if hasattr(b,'to_dict') else dict(b)
        if {**DEFAULTS,**a}!={**DEFAULTS,**b}:raise ValueError('Checkpoint frontflip configuration differs')
