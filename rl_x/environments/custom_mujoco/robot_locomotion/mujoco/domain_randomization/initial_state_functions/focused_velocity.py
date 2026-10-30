"""V9 fixed mixture: target speed band, easy retention, nominal landings."""
import numpy as np
class FocusedVelocity:
    def __init__(self,initial,config):
        self.initial=initial
        self.low=float(config['focused_speed_min']);self.high=float(config['focused_speed_max'])
        self.target=float(config['focused_target_probability']);self.nominal=float(config['focused_nominal_probability']);self.top=float(config['focused_top_probability']);self.boundary=float(config['focused_boundary_probability'])
        if not 0<self.low<self.high<=initial.max_lateral_velocity or not 0<=self.nominal<1 or not 0<self.target<=1-self.nominal or not 0<=self.top<=1 or not 0<=self.boundary<=1 or initial.max_angle_deg!=0 or initial.min_angle_deg!=0:raise ValueError('Invalid focused velocity mixture')
    def sample(self,curriculum):
        rng=self.initial.env.np_rng
        height=self.initial.target_height if rng.random()<self.top else rng.uniform(self.initial.start_height,self.initial.target_height)
        u=rng.random()
        if u<self.target:group=0;speed=self.high if rng.random()<self.boundary else rng.uniform(self.low,self.high)
        elif u<1-self.nominal:group=1;speed=self.low*np.sqrt(rng.uniform())
        else:group=2;speed=0.
        self.initial.velocity_sampling='speed_ring'
        curriculum.update(difficulty=1.,applied_difficulty=1.,replay_easy=group==1,replay_nominal=group==2)
        info=self.initial.env.internal_state['info'];info.update({'curriculum/focused_target':float(group==0),'curriculum/focused_easy':float(group==1),'curriculum/focused_nominal':float(group==2),'curriculum/focused_sampled_speed':float(speed)})
        return float(height),0.,float(speed),1.
    def update(self,success):pass # fixed distribution, no success-dependent advancement
