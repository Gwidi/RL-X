"""V22 soft: statistics and videos of one checkpoint, with the environment it was trained in.

The MJX settings (soft thigh/calf contact, braking reference, slip penalty) are read from the checkpoint.

    python experiments/analysis/render_v22_soft_best.py --model models_soft/models/best.model --output DIR
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.analysis.check_mjx_parity import actor, actor_params, make_mjx_env, merge, v22_config

HEIGHTS, SPEEDS, AZIMUTHS, REPEATS = (.5, .75, 1., 1.25), (0., 2., 3., 4., 5.), 16, 4
REASONS = {'success': 'SUKCES', 'disallowed_contact': 'KONTAKT NIEDOZWOLONY', 'calf_force': 'SIŁA ŁYDKI > 200 N',
           'body_acceleration': 'PRZYSPIESZENIE KORPUSU', 'base_crash': 'UDERZENIE KORPUSU', 'base_height': 'KORPUS ZA NISKO',
           'foot_force': 'SIŁA STOPY', 'unresolved': 'BRAK STABILIZACJI', 'motor_hard_limit': 'LIMIT NAPĘDU',
           'overturned': 'PRZEWRÓCONY (test sukcesu go zalicza)'}


def checkpoint_mjx(path):
    with zipfile.ZipFile(path) as archive:
        mjx = json.loads(archive.read('checkpoint_metrics.json'))['mjx']
    return {k: v for k, v in mjx.items() if k not in ('constraints_per_env', 'contacts_per_env')}


def run(env, params, seed=5):
    grid = np.array([(h, v, a) for h in HEIGHTS for v in SPEEDS for a in range(AZIMUTHS) for _ in range(REPEATS)])
    n = len(grid)
    azimuth = grid[:, 2] / AZIMUTHS * 2 * np.pi
    velocity = jnp.asarray(grid[:, 1:2] * np.stack([np.cos(azimuth), np.sin(azimuth)], -1), jnp.float32)
    pool = jax.jit(env.launch)(jax.random.split(jax.random.PRNGKey(seed), n), jnp.asarray(grid[:, 0], jnp.float32), velocity)
    assert not bool(pool['invalid'])
    original = env._info
    env._info = lambda task, imu, data, *a: {**original(task, imu, data, *a), 'T/qpos': data.qpos}

    @jax.jit
    def one(state, finished, record):
        new_state, _ = env.step(state, actor(params, state.observation[:, env.policy_observation_indices], jnp), pool, 0)
        done = (new_state.terminated | new_state.truncated) & ~finished
        record = jax.tree.map(lambda old, x: jnp.where(done.reshape((-1,) + (1,) * (x.ndim - 1)), x, old), record, new_state.info)
        return new_state, finished | done, record, done, new_state.info['T/qpos']

    state = jax.jit(lambda p: env.reset(p, n))(pool)
    finished = jnp.zeros(n, bool)
    record = jax.tree.map(jnp.zeros_like, state.info)
    qpos, ends = [np.asarray(pool['release']['qpos'])], np.full(n, -1)
    for step in range(env.horizon):
        state, finished, record, done, q = one(state, finished, record)
        qpos.append(np.asarray(q))
        ends[np.asarray(done)] = step + 1
        if step % 20 == 19 and bool(finished.all()):
            break
    return grid, {k: np.asarray(v) for k, v in record.items()}, np.stack(qpos), ends


def tilt_of(q):
    return np.degrees(np.arccos(np.clip(1 - 2 * (q[..., 4] ** 2 + q[..., 5] ** 2), -1, 1)))


def reason(record, i):
    if record['curriculum/landing_success'][i] > .5:
        q = record['T/qpos'][i]
        return 'success' if tilt_of(q) < 45 and q[2] > .2 else 'overturned'
    for key, value in record.items():
        if key.startswith('outcome/') and value[i] > .5:
            return key.split('failure_')[-1].split('/')[-1]
    return 'unresolved'


def statistics(grid, record):
    ok = record['curriculum/landing_success'] > .5
    q = record['T/qpos']
    upright = ok & (tilt_of(q) < 45) & (q[:, 2] > .2)
    rows = {}
    for speed in SPEEDS:
        s = grid[:, 1] == speed
        reasons = {}
        for i in np.flatnonzero(s & ~upright):
            r = reason(record, i)
            reasons[r] = reasons.get(r, 0) + 1
        rows[f'{speed:g}'] = dict(
            episodes=int(s.sum()), success=round(100 * float(ok[s].mean()), 1), success_upright=round(100 * float(upright[s].mean()), 1),
            loaded_slip_integral=round(float(record['metrics/loaded_slip_integral'][s].mean()), 3),
            peak_body_acceleration=round(float(record['metrics/peak_body_acceleration'][s].mean()), 1),
            failures={k: round(100 * v / int(s.sum()), 1) for k, v in sorted(reasons.items())})
    return rows


class Renderer:
    def __init__(self, model):
        self.model, self.data = model, mujoco.MjData(model)
        self.renderer = mujoco.Renderer(model, height=360, width=640)
        self.camera = mujoco.MjvCamera()
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 2.6, 135, -20
        self.font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)

    def frame(self, q, title, text, color):
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)
        self.camera.lookat[:] = self.data.qpos[:3]
        self.renderer.update_scene(self.data, camera=self.camera)
        image = Image.new('RGB', (640, 420), '#101820')
        image.paste(Image.fromarray(self.renderer.render()), (0, 32))
        draw = ImageDraw.Draw(image)
        draw.text((8, 7), title, font=self.font, fill='#d8e4f0')
        draw.text((8, 396), text, font=self.font, fill=color)
        return image


def write_video(path, frames, size=(640, 420), fps=12.5):
    process = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{size[0]}x{size[1]}',
                                '-r', str(fps), '-i', '-', '-an', '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '24', str(path)],
                               stdin=subprocess.PIPE)
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    assert process.wait() == 0


def video(renderer, record, qpos, ends, index, title, path, stride=2, hold=14):
    end, outcome = int(ends[index]), reason(record, index)
    frames = []
    for step in range(end + 1):
        if step % stride and step != end:
            continue
        text, color = f't = {step * .02:4.2f} s', '#d8e4f0'
        if step == end:
            text = REASONS.get(outcome, outcome.upper()) + f'  (t = {step * .02:4.2f} s)'
            color = '#77dec4' if outcome == 'success' else '#ff7b72'
        frames.append(renderer.frame(qpos[step, index], title, text, color))
    frames += [frames[-1]] * hold
    write_video(path, frames)
    frames[min(len(frames) - 1, int(.45 * len(frames)))].save(path.with_suffix('.png'))
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--media', type=Path, required=True)
    parser.add_argument('--prefix', default='v22-soft', help='file name prefix of the videos')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.media.mkdir(parents=True, exist_ok=True)
    mjx = checkpoint_mjx(args.model)
    global SPEEDS
    SPEEDS += tuple(float(v) for v in range(6, int(mjx.get('max_speed', 5)) + 1))
    env = make_mjx_env(merge(v22_config(), {'mjx': mjx}))
    params = jax.tree.map(jnp.asarray, actor_params(args.model))
    grid, record, qpos, ends = run(env, params)
    stats = statistics(grid, record)
    for speed, row in stats.items():
        print(speed, row, flush=True)

    def pick(speed, height, want, azimuths=range(AZIMUTHS)):
        for a in azimuths:
            for i in np.flatnonzero((grid[:, 0] == height) & (grid[:, 1] == speed) & (grid[:, 2] == a)):
                if (reason(record, i) == 'success') == want:
                    return int(i)
        return None

    cases = [(3., 1., True, 'v22-soft-3ms-sukces', 'V22 soft · 3 m/s · 1 m'),
             (4., 1.25, True, 'v22-soft-4ms-sukces', 'V22 soft · 4 m/s · 1,25 m'),
             (5., .75, True, 'v22-soft-5ms-sukces', 'V22 soft · 5 m/s · 0,75 m'),
             (5., 1.25, False, 'v22-soft-5ms-porazka', 'V22 soft · 5 m/s · 1,25 m (porażka)'),
             (5., .5, False, 'v22-soft-5ms-porazka-niska', 'V22 soft · 5 m/s · 0,5 m (porażka)')]
    renderer, movies = Renderer(env.model), []
    for speed, height, want, slug, title in cases:
        index = pick(speed, height, want)
        if index is None:
            print('no scenario', slug, flush=True)
            continue
        slug = slug.replace('v22-soft', args.prefix, 1)
        path = args.media / f'{slug}.mp4'
        outcome = video(renderer, record, qpos, ends, index, title, path)
        movies.append(dict(title=title, src=f'media/{path.name}', speed=speed, height=height, index=index, outcome=outcome,
                           azimuth_deg=float(grid[index, 2] * 360 / AZIMUTHS),
                           note='Akcja deterministyczna (średnia polityki), 2× wolniej.'))
        print('video', path.name, outcome, flush=True)
    (args.output / 'soft_best.json').write_text(json.dumps(dict(
        model=str(args.model), mjx=mjx, episodes_per_speed=int((grid[:, 1] == SPEEDS[0]).sum()), statistics=stats, movies=movies), indent=1))


if __name__ == '__main__':
    main()
