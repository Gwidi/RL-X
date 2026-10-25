"""Independent height sampling and angle/velocity curricula for landing v3."""
import numpy as np

GROUPS = ('nominal', 'angles', 'velocity', 'combined')


class FactorizedLanding:
    def __init__(self, initial, config):
        self.initial = initial
        self.weights = np.array([config.get(f'factorized_{g}_probability', p)
                                 for g,p in zip(GROUPS, (.2,.2,.2,.4))], dtype=float)
        self.top_probability = float(config.get('factorized_top_probability', .25))
        self.start_angle = float(config.get('factorized_start_angle_deg', 1.))
        self.start_velocity = float(config.get('factorized_start_velocity', .5))
        if (not np.all(np.isfinite(self.weights)) or np.any(self.weights < 0)
                or not np.isclose(self.weights.sum(), 1.)):
            raise ValueError('Factorized group probabilities must be nonnegative and sum to one')
        if not np.isfinite(self.top_probability) or not 0 <= self.top_probability <= 1:
            raise ValueError('factorized_top_probability must be between zero and one')
        if not 0 <= self.start_angle <= initial.max_angle_deg or not 0 <= self.start_velocity <= initial.max_lateral_velocity:
            raise ValueError('Factorized initial amplitudes must lie within target amplitudes')
        if not np.isfinite(initial.start_height) or not np.isfinite(initial.target_height) or initial.start_height <= 0 or initial.target_height < initial.start_height:
            raise ValueError('Invalid factorized height interval')

    def state(self):
        state = self.initial.env.internal_state
        if 'landing_factorized' not in state:
            state['landing_factorized'] = {
                'group': 0,
                **{f'{factor}_{key}': value for factor in ('angle','velocity')
                   for key,value in [('difficulty',0.),('ema',0.),('count',0)]},
            }
        return state['landing_factorized']

    def sample(self, curriculum):
        initial, state = self.initial, self.state()
        rng = initial.env.np_rng
        group = int(rng.choice(4, p=self.weights))
        state['group'] = group
        height = (initial.target_height if rng.random() < self.top_probability
                  else rng.uniform(initial.start_height, initial.target_height))
        angle = self.start_angle + state['angle_difficulty'] * (initial.max_angle_deg-self.start_angle)
        velocity = self.start_velocity + state['velocity_difficulty'] * (initial.max_lateral_velocity-self.start_velocity)
        angle = angle if group in (1,3) else 0.
        velocity = velocity if group in (2,3) else 0.
        curriculum.update(replay_easy=False, replay_nominal=group==0,
                          difficulty=(state['angle_difficulty']+state['velocity_difficulty'])/2,
                          applied_difficulty=(state['angle_difficulty']+state['velocity_difficulty'])/2)
        info = initial.env.internal_state['info']
        info['curriculum/factorized_group'] = group
        for i,g in enumerate(GROUPS):
            info[f'curriculum/group_{g}'] = float(group==i)
        for factor in ('angle','velocity'):
            for key in ('difficulty','ema','count'):
                info[f'curriculum/{factor}_{key}'] = state[f'{factor}_{key}']
        return float(height), np.deg2rad(angle), float(velocity), curriculum['difficulty']

    def update(self, success):
        initial, state = self.initial, self.state()
        # Pure-factor trials update their factor only; combined trials update
        # both from the joint outcome. Nominal trials never change either level.
        for factor, groups in [('angle',(1,3)), ('velocity',(2,3))]:
            if state['group'] not in groups:
                continue
            state[f'{factor}_count'] += 1
            ema = (1-initial.ema_alpha)*state[f'{factor}_ema'] + initial.ema_alpha*float(bool(success))
            state[f'{factor}_ema'] = ema
            change = (initial.difficulty_step_up if ema > initial.success_threshold_up else
                      -initial.difficulty_step_down if ema < initial.success_threshold_down else 0.)
            state[f'{factor}_difficulty'] = float(np.clip(state[f'{factor}_difficulty']+change,0,1))
