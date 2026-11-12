"""V17 fixed height/speed mixture, independent of success history."""
import numpy as np


class BalancedReplay:
    def __init__(self, initial):
        self.initial = initial
        if (initial.start_height != .75 or initial.target_height != 1.5
                or initial.max_lateral_velocity != 8.
                or initial.min_angle_deg != 0. or initial.max_angle_deg != 0.):
            raise ValueError('V17 replay requires heights .75–1.5, speed <=8, zero angles')

    def sample(self, curriculum):
        rng = self.initial.env.np_rng
        u = rng.random()
        group = 0 if u < .6 else 1 if u < .9 else 2
        heights = [.75, 1., 1.25] if group == 0 else [.75, 1., 1.25, 1.5]
        height = float(rng.choice(heights))
        speed = float(rng.uniform(5., 8.) if group == 0 else
                      rng.uniform(0., 8.) if group == 1 else 0.)
        # The downstream sampler draws an azimuth and preserves this speed.
        self.initial.velocity_sampling = 'speed_ring'
        curriculum.update(difficulty=1., applied_difficulty=1.,
                          replay_easy=group == 1, replay_nominal=group == 2)
        self.initial.env.internal_state['info'].update({
            'curriculum/balanced_hard': float(group == 0),
            'curriculum/balanced_full': float(group == 1),
            'curriculum/balanced_nominal': float(group == 2),
            'curriculum/balanced_height': height,
            'curriculum/balanced_speed': speed,
        })
        return height, 0., speed, 1.

    def update(self, success):
        pass
