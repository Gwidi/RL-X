"""Opt-in v8: certify height frontiers locally, then learn horizontal speed."""
import numpy as np


def levels(value):
    result = [float(x) for x in value.split(',')]
    if not result or any(not np.isfinite(x) or x <= 0 for x in result) or any(b <= a for a,b in zip(result,result[1:])):
        raise ValueError('Sequential levels must be positive, finite and strictly increasing')
    return result


class HeightThenVelocity:
    def __init__(self, initial, config):
        self.initial = initial
        self.heights = levels(config['sequential_heights'])
        self.speeds = levels(config['sequential_speeds'])
        self.window = int(config['sequential_window'])
        self.threshold = float(config['sequential_threshold'])
        self.frontier_probability = float(config['sequential_frontier_probability'])
        if self.window < 1 or self.window != config['sequential_window'] or not 0 < self.threshold < 1 or not 0 < self.frontier_probability <= 1:
            raise ValueError('Invalid sequential window, threshold or frontier probability')
        if initial.max_angle_deg != 0 or initial.min_angle_deg != 0:
            raise ValueError('Velocity-only curriculum requires zero initial angles')
        if initial.velocity_sampling != 'speed_disk':
            raise ValueError('Sequential training requires speed_disk sampling')
        if self.heights[0] != initial.start_height or self.heights[-1] != initial.target_height or self.speeds[-1] != initial.max_lateral_velocity:
            raise ValueError('Sequential levels must match configured height and velocity limits')

    def state(self):
        return self.initial.env.internal_state.setdefault('landing_sequential', dict(
            stage=0, history=[], frontier=False, mastered=False, last_rate=0., advances=0))

    def sample(self, curriculum):
        state=self.state();rng=self.initial.env.np_rng;stage=state['stage']
        height_phase=stage < len(self.heights)
        frontier=bool(rng.random() < self.frontier_probability)
        if height_phase:
            height=self.heights[stage] if frontier else float(rng.choice(self.heights[:stage+1]))
            speed=0.
        else:
            speed=self.speeds[stage-len(self.heights)]
            # Advance speed only on the joint upper height/speed frontier.
            height=self.heights[-1] if frontier else float(rng.choice(self.heights))
        state['frontier']=frontier
        curriculum.update(replay_easy=False,replay_nominal=height_phase,difficulty=stage/(len(self.heights)+len(self.speeds)-1),applied_difficulty=stage/(len(self.heights)+len(self.speeds)-1))
        info=self.initial.env.internal_state['info']
        info.update({'curriculum/sequential_stage':stage,'curriculum/sequential_velocity_phase':float(not height_phase),
            'curriculum/sequential_height_limit':self.heights[stage] if height_phase else self.heights[-1],
            'curriculum/sequential_speed_limit':speed,'curriculum/sequential_frontier':float(frontier),
            'curriculum/sequential_frontier_count':len(state['history']),
            'curriculum/sequential_frontier_success_rate':float(np.mean(state['history'])) if state['history'] else 0.,
            'curriculum/sequential_last_pass_rate':state['last_rate'],
            'curriculum/sequential_mastered':float(state['mastered'])})
        # Exact speed at frontier; disk sampling elsewhere supplies easier replay.
        self.initial.velocity_sampling='speed_ring' if frontier and not height_phase else 'speed_disk'
        return float(height),0.,float(speed),curriculum['difficulty']

    def update(self, success):
        state=self.state()
        if not state['frontier']:return
        state['history'].append(int(bool(success)))
        state['history']=state['history'][-self.window:]
        if len(state['history']) == self.window and np.mean(state['history']) > self.threshold:
            state['last_rate']=float(np.mean(state['history']))
            last=len(self.heights)+len(self.speeds)-1
            if state['stage'] < last:
                state['stage']+=1;state['advances']+=1;state['history']=[]
            else:state['mastered']=True
