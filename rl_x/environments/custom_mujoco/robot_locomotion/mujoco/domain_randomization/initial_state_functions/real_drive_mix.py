"""V22 fixed mixture for the measured drive limits: low drops, speed up to 5 m/s."""


class RealDriveMix:
    heights = (.5, .75, 1., 1.25)

    def __init__(self, initial):
        self.initial = initial
        if (initial.start_height != self.heights[0] or initial.target_height != self.heights[-1]
                or initial.max_lateral_velocity != 5.
                or initial.min_angle_deg != 0. or initial.max_angle_deg != 0.):
            raise ValueError('V22 requires 0.5–1.25 m, 5 m/s, zero sampled angles')

    def sample(self, curriculum):
        rng = self.initial.env.np_rng
        height = float(rng.choice(self.heights))
        u = rng.random()
        # 10% vertical, 40% where V19 starts to fail, 30% and 20% extension.
        group = 0 if u < .1 else 1 if u < .5 else 2 if u < .8 else 3
        low, high = ((0., 0.), (0., 2.), (2., 3.5), (3.5, 5.))[group]
        speed = float(rng.uniform(low, high)) if group else 0.
        self.initial.velocity_sampling = 'speed_ring'
        curriculum.update(difficulty=1., applied_difficulty=1., replay_easy=group == 1, replay_nominal=group == 0)
        self.initial.env.internal_state['info'].update({
            'curriculum/v22_group': group, 'curriculum/v22_height': height, 'curriculum/v22_speed': speed})
        return height, 0., speed, 1.

    def update(self, success):
        pass
