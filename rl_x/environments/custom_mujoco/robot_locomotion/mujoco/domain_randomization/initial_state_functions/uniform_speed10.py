"""V19 fixed mixture: equal height allocation, retention and speed extension."""
import numpy as np


class UniformSpeed10:
    def __init__(self, initial):
        self.initial = initial
        if (initial.start_height != .75 or initial.target_height != 1.5
                or initial.max_lateral_velocity != 10.
                or initial.min_angle_deg != 0. or initial.max_angle_deg != 0.):
            raise ValueError('V19 requires .75–1.5 m, 10 m/s, zero sampled angles')

    def sample(self, curriculum):
        rng = self.initial.env.np_rng
        height = float(rng.choice([.75, 1., 1.25, 1.5]))
        u = rng.random()
        group = 0 if u < .4 else 1 if u < .7 else 2 if u < .9 else 3
        if group == 0:
            speed = float(rng.uniform(5., 8.))
        elif group == 1:
            speed = 10. if rng.random() < 1/3 else float(rng.uniform(8., 10.))
        elif group == 2:
            speed = float(rng.uniform(0., 8.))
        else:
            speed = 0.
        self.initial.velocity_sampling = 'speed_ring'
        curriculum.update(difficulty=1., applied_difficulty=1.,
                          replay_easy=group == 2, replay_nominal=group == 3)
        self.initial.env.internal_state['info'].update({
            'curriculum/v19_group': group, 'curriculum/v19_height': height,
            'curriculum/v19_speed': speed})
        return height, 0., speed, 1.

    def update(self, success):
        pass
