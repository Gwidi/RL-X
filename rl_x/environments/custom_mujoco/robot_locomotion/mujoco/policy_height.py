"""OptiTrack-compatible base height above a calibrated fixed landing plane."""
from collections import deque
import numpy as np


class PolicyHeight:
    def __init__(self, env):
        self.env = env
        self.config = dict(env.env_config.get('policy_height', {}))
        self.enabled = self.config.get('enabled', False)
        self.dimension = int(self.enabled)
        for key in ('noise_std_m', 'bias_std_m'):
            if not np.isfinite(self.config.get(key, 0.)) or self.config.get(key, 0.) < 0:
                raise ValueError('Height noise/bias must be finite and nonnegative')
        delay = self.config.get('delay_steps', 0)
        if not isinstance(delay, int) or isinstance(delay, bool) or delay < 0:
            raise ValueError('Height delay must be a nonnegative integer')
        if not np.isfinite(self.config.get('plane_z_m', 0.)):
            raise ValueError('Height plane must be finite')

    def reset(self):
        if not self.enabled:
            return
        rng = np.random.default_rng(int(self.env.np_rng.integers(0, 2**32)))
        self.env.internal_state['policy_height'] = dict(
            rng=rng, bias=rng.normal(0., self.config['bias_std_m']),
            queue=deque(maxlen=self.config['delay_steps'] + 1))
        self.publish()
        state = self.env.internal_state['policy_height']
        while len(state['queue']) < state['queue'].maxlen:
            state['queue'].append(state['queue'][0])

    def publish(self):
        if not self.enabled:
            return
        state = self.env.internal_state['policy_height']
        height = (float(self.env.internal_state['data'].qpos[2]) - self.config['plane_z_m']
                  + state['bias'] + state['rng'].normal(0., self.config['noise_std_m']))
        state['queue'].append(height)

    def observe(self):
        if not self.enabled:
            return np.empty(0)
        # Metres, scaled by 3 m; no velocity derived from privileged state.
        return np.array([self.env.internal_state['policy_height']['queue'][0] / 3.])
