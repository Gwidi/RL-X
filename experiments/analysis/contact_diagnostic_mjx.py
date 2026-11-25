"""Does a V22 policy survive a non-foot leg contact, and from which release pose?

Runs one checkpoint on the MJX landing grid in four conditions: release from the
landing home pose or from the symmetric standing pose, with a contact of the thigh or
upper calf ending the episode (the training rule) or merely being recorded. Scenarios
are paired across conditions. Writes statistics and side-by-side videos.

    python experiments/analysis/contact_diagnostic_mjx.py --model modelsv22/latest.model --output DIR
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.analysis.check_mjx_parity import actor, actor_params, make_mjx_env, v22_config
import rl_x.environments.custom_mujoco.robot_locomotion.mjx_landing.environment as E
from rl_x.environments.custom_mujoco.robot_locomotion.mjx_landing.contacts import active_contacts

# Original symmetric standing pose (plane.xml_backup): spine, then RL, RR, FR, FL hip/thigh/calf.
STAND = [0, -.1, -.8, 1.5, .1, .8, -1.5, -.1, .8, -1.5, .1, -.8, 1.5]
HEIGHTS, SPEEDS, AZIMUTHS, REPEATS = (.5, .75, 1., 1.25), (0., 2., 3., 4., 5.), 16, 8
LEG = {'rl': 'tylna lewa', 'rr': 'tylna prawa', 'fr': 'przednia prawa', 'fl': 'przednia lewa'}
PART = {'l0': 'biodro', 'l1': 'udo', 'l2': 'łydka'}
REASONS = {'success': 'SUKCES', 'disallowed_contact': 'KONTAKT NIEDOZWOLONY', 'calf_force': 'SIŁA ŁYDKI > 1000 N',
           'body_acceleration': 'PRZYSPIESZENIE KORPUSU', 'base_crash': 'UDERZENIE KORPUSU',
           'base_height': 'KORPUS ZA NISKO', 'foot_force': 'SIŁA STOPY', 'unresolved': 'BRAK STABILIZACJI', 'overturned': 'PRZEWRÓCONY (liczy się jako sukces)',
           'motor_hard_limit': 'LIMIT NAPĘDU'}


def body_label(name):
    if name == 'rear':
        return 'tylna część korpusu'
    return f"{PART[name[3:]]} {LEG[name[:2]]}"


class Diagnostic:
    def __init__(self):
        self.env = make_mjx_env(v22_config())
        self.model = self.env.model
        self.original = dict(contact_metrics=E.contact_metrics, initial_task=self.env._initial_task,
                             substep_task=self.env._substep_task, violations=E.landing_violations, info=self.env._info)
        env, model, original = self.env, self.model, self.original

        # Thighs and calves (links l1, l2). Hip links (l0), the trunk and the rear body stay forbidden.
        limb = np.array([name[3:] in ('l1', 'l2') and name[:2] in LEG for name in
                         (model.body(i).name for i in range(model.nbody))])
        self.limb, self.relaxed = jnp.asarray(limb), False

        def contact_metrics(data, tables, n, soft_limbs=False):
            result = original["contact_metrics"](data, tables, n, soft_limbs)
            active, world, geom, force = active_contacts(data, tables)
            bad = active & ~tables.allowed[geom]
            body = tables.geom_body[geom]
            # Normal force of thigh and calf contacts, summed per world (no feet: those are allowed supports).
            result['limb_force'] = jax.ops.segment_sum(jnp.where(bad & self.limb[body], force, 0.), world, n)
            index = world * model.nbody + body
            result['fbody'] = jax.ops.segment_max(bad.astype(jnp.int32), index, n * model.nbody).reshape(n, model.nbody) > 0
            if self.relaxed:  # only limb contacts are forgiven
                fatal = bad & ~self.limb[body]
                result['forbidden'] = jax.ops.segment_max(fatal.astype(jnp.int32), world, n) > 0
                result['limb_touch'] = jax.ops.segment_max((bad & self.limb[body]).astype(jnp.int32), world, n) > 0
            else:
                result['limb_touch'] = result['forbidden']
            return result

        def initial_task(data, height, velocity, duration):
            task = original['initial_task'](data, height, velocity, duration)
            n = height.shape[0]
            task.update(fbody=jnp.zeros((n, model.nbody), bool), tforb=jnp.zeros(n, jnp.int32), vforb=jnp.zeros(n),
                        limb=jnp.zeros(n, bool), tlimb=jnp.zeros(n, jnp.int32),
                        lf_history=jnp.zeros((n, env.force_window)), lf_peak=jnp.zeros(n))
            return task

        def substep_task(task, data, contacts, totals):
            new = contacts['limb_touch'] & ~task['limb']
            task, totals = original['substep_task'](task, data, contacts, totals)
            task.update(fbody=jnp.where(new[:, None], contacts['fbody'], task['fbody']),
                        tforb=jnp.where(new, task['step'], task['tforb']),
                        vforb=jnp.where(new, jnp.linalg.norm(data.qvel[:, :2], axis=-1), task['vforb']),
                        limb=task['limb'] | contacts['limb_touch'])
            history = jnp.concatenate([task['lf_history'][:, 1:], contacts['limb_force'][:, None]], 1)
            task.update(lf_history=history, lf_peak=jnp.maximum(task['lf_peak'], history.mean(1)))  # 10 ms mean, like the foot force
            return task, totals

        def info(task, imu, data, *args):
            result = original['info'](task, imu, data, *args)
            result.update({'T/forbidden': task['limb'], 'T/fbody': task['fbody'], 'T/tforb': task['tforb'],
                           'T/vforb': task['vforb'], 'T/qpos': data.qpos, 'T/lf_peak': task['lf_peak']})
            return result

        E.contact_metrics, env._initial_task, env._substep_task, env._info = contact_metrics, initial_task, substep_task, info
        self.stand = jnp.asarray(STAND, jnp.float32)

    def run(self, params, pose, relaxed, seed=5):
        env, original = self.env, self.original
        env.release_pose = env.nominal if pose == 'home' else self.stand
        self.relaxed = relaxed
        grid = np.array([(h, v, a) for h in HEIGHTS for v in SPEEDS for a in range(AZIMUTHS) for _ in range(REPEATS)])
        n = len(grid)
        azimuth = grid[:, 2] / AZIMUTHS * 2 * np.pi
        velocity = jnp.asarray(grid[:, 1:2] * np.stack([np.cos(azimuth), np.sin(azimuth)], -1), jnp.float32)
        pool = jax.jit(env.launch)(jax.random.split(jax.random.PRNGKey(seed), n), jnp.asarray(grid[:, 0], jnp.float32), velocity)
        assert not bool(pool['invalid'])

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


def reason(record, i):
    if record['curriculum/landing_success'][i] > .5:
        q = record['T/qpos'][i]
        tilt = np.degrees(np.arccos(np.clip(1 - 2 * (q[4] ** 2 + q[5] ** 2), -1, 1)))
        return 'success' if tilt < 45 and q[2] > .2 else 'overturned'  # the success test alone accepts a robot lying on its back
    for key, value in record.items():
        if key.startswith('outcome/') and value[i] > .5:
            return key.split('failure_')[-1].split('/')[-1]
    return 'unresolved'


def statistics(grid, record):
    ok = record['curriculum/landing_success'] > .5
    rows = {}
    for speed in SPEEDS:
        s = grid[:, 1] == speed
        touched = record['T/forbidden'][s] > .5
        q = record['T/qpos']
        tilt = np.degrees(np.arccos(np.clip(1 - 2 * (q[:, 4] ** 2 + q[:, 5] ** 2), -1, 1)))
        upright = ok & (tilt < 45) & (q[:, 2] > .2)
        reasons = {}
        for i in np.flatnonzero(s):
            if not ok[i]:
                reasons[reason(record, i)] = reasons.get(reason(record, i), 0) + 1
        rows[f'{speed:g}'] = dict(episodes=int(s.sum()), success=round(100 * float(ok[s].mean()), 1),
                                  forbidden_touch=round(100 * float(touched.mean()), 1),
                                  success_upright=round(100 * float(upright[s].mean()), 1),
                                  end_tilt_deg_median=round(float(np.median(tilt[s & ok])), 1) if (s & ok).any() else None,
                                  end_height_m_median=round(float(np.median(q[s & ok, 2])), 3) if (s & ok).any() else None,
                                  success_given_touch=round(100 * float(ok[s][touched].mean()), 1) if touched.any() else None,
                                  failures={k: round(100 * v / int(s.sum()), 1) for k, v in sorted(reasons.items())})
    return rows


class Renderer:
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.renderer = mujoco.Renderer(model, height=270, width=480)
        self.camera = mujoco.MjvCamera()
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 2.4, 135, -20
        self.rgba = model.geom_rgba.copy()
        self.font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)

    def panel(self, q, fbody, title, text, color):
        model, data = self.model, self.data
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        model.geom_rgba[:] = self.rgba
        for body in np.flatnonzero(fbody):
            model.geom_rgba[model.geom_bodyid == body] = (1., .15, .1, 1.)
        self.camera.lookat[:] = data.qpos[:3]
        self.renderer.update_scene(data, camera=self.camera)
        image = Image.new('RGB', (480, 330), '#101820')
        image.paste(Image.fromarray(self.renderer.render()), (0, 30))
        draw = ImageDraw.Draw(image)
        draw.text((8, 6), title, font=self.font, fill='#d8e4f0')
        draw.text((8, 306), text, font=self.font, fill=color)
        return image


def write_video(path, frames, size=(960, 330), fps=12.5):
    process = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{size[0]}x{size[1]}',
                                '-r', str(fps), '-i', '-', '-an', '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '24', str(path)],
                               stdin=subprocess.PIPE)
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    assert process.wait() == 0


def pair_video(renderer, names, runs, index, grid, pose, path, left_title, right_title, stride=2, hold=14):
    """runs = {'strict': (record, qpos, ends), 'relaxed': ...}"""
    panels = []
    for mode, title in (('strict', left_title), ('relaxed', right_title)):
        record, qpos, ends = runs[(pose, mode)]
        end = int(ends[index])
        outcome = reason(record, index)
        bodies = record['T/fbody'][index]
        touch_step = int(record['T/tforb'][index]) if record['T/forbidden'][index] > .5 else None
        frames = []
        for step in range(0, end + 1):
            if step % stride and step != end:
                continue
            touched = touch_step is not None and step >= touch_step
            text = f't = {step * .02:4.2f} s'
            color = '#d8e4f0'
            if touched:
                text += '   dotyk: ' + ', '.join(body_label(names[b]) for b in np.flatnonzero(bodies))
                color = '#ffb36b'
            if step == end:
                text = REASONS.get(outcome, outcome.upper()) + f'  (t = {step * .02:4.2f} s)'
                color = '#77dec4' if outcome == 'success' else '#ff7b72'
            frames.append(renderer.panel(qpos[step, index], bodies if touched else np.zeros_like(bodies), title, text, color))
        frames += [frames[-1]] * hold
        panels.append(frames)
    count = max(map(len, panels))
    composed = []
    for i in range(count):
        image = Image.new('RGB', (960, 330))
        for j, frames in enumerate(panels):
            image.paste(frames[min(i, len(frames) - 1)], (480 * j, 0))
        composed.append(image)
    write_video(path, composed)
    composed[min(len(composed) - 1, max(0, int(0.45 * count)))].save(path.with_suffix('.png'))
    return [reason(runs[(pose, m)][0], index) for m in ('strict', 'relaxed')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--media', type=Path, help='directory for the videos (default: OUTPUT/media)')
    args = parser.parse_args()
    media = args.media or args.output / 'media'
    media.mkdir(parents=True, exist_ok=True)
    diagnostic = Diagnostic()
    params = jax.tree.map(jnp.asarray, actor_params(args.model))
    names = [diagnostic.model.body(i).name for i in range(diagnostic.model.nbody)]
    runs, stats, grid = {}, {}, None
    for pose in ('home', 'stand'):
        for mode in ('strict', 'relaxed'):
            grid, record, qpos, ends = diagnostic.run(params, pose, mode == 'relaxed')
            runs[(pose, mode)] = (record, qpos, ends)
            stats[f'{pose}/{mode}'] = statistics(grid, record)
            print(pose, mode, {k: (v['success'], v['forbidden_touch']) for k, v in stats[f'{pose}/{mode}'].items()}, flush=True)

    def pick(pose, speed, height, azimuth, want):
        """First scenario with the requested (strict, relaxed) outcomes: 'ok' or 'fail'."""
        for i in range(len(grid)):
            if grid[i, 1] != speed or grid[i, 0] != height or grid[i, 2] != azimuth:
                continue
            outcome = [reason(runs[(pose, m)][0], i) == 'success' for m in ('strict', 'relaxed')]
            if outcome == [w == 'ok' for w in want]:
                return i
        raise LookupError((pose, speed, height, azimuth, want))

    cases = [
        ('home', 5., 1., 12, ('fail', 'ok'), 'start-z-pozy-ladowania-5ms-prawo', 'Poza lądowania · 5 m/s · 1 m · w prawo-tył'),
        ('home', 4., .75, 11, ('fail', 'ok'), 'start-z-pozy-ladowania-4ms', 'Poza lądowania · 4 m/s · 0,75 m'),
        ('stand', 0., .75, 0, ('fail', 'ok'), 'start-ze-stania-pion', 'Poza stania · pionowo · 0,75 m'),
        ('stand', 4., 1., 3, ('fail', 'ok'), 'start-ze-stania-4ms', 'Poza stania · 4 m/s · 1 m'),
        ('stand', 5., 1., 3, ('fail', 'fail'), 'start-ze-stania-5ms-porazka', 'Poza stania · 5 m/s · 1 m (porażka w obu)'),
    ]
    renderer, movies = Renderer(diagnostic.model), []
    for pose, speed, height, azimuth, want, slug, title in cases:
        try:
            index = pick(pose, speed, height, azimuth, want)
        except LookupError:
            candidates = [a for a in range(AZIMUTHS) if any(
                grid[i, 1] == speed and grid[i, 0] == height and grid[i, 2] == a for i in range(len(grid)))]
            index = next(pick(pose, speed, height, a, want) for a in candidates if _try(pick, pose, speed, height, a, want))
        path = media / f'v22-kontakt-{slug}.mp4'
        outcomes = pair_video(renderer, names, runs, index, grid, pose, path, 'Ścisłe: dotyk uda/łydki kończy próbę',
                              'Poluzowane: udo i łydka dozwolone', stride=2)
        movies.append(dict(title=title, src=f'media/{path.name}', slug=slug, pose=pose, speed=speed, height=height,
                           azimuth_deg=float(azimuth * 360 / AZIMUTHS), index=int(index), strict=outcomes[0], relaxed=outcomes[1],
                           note='Ścisłe po lewej, poluzowane po prawej. Ten sam scenariusz. Czerwone: pierwszy niedozwolony kontakt. 2× wolniej.'))
        print('video', path.name, outcomes, flush=True)
    (args.output / 'contact_diagnostic.json').write_text(json.dumps(dict(
        model=str(args.model), episodes_per_speed=len(grid) // len(SPEEDS), stand_pose=STAND, statistics=stats, movies=movies), indent=1))


def _try(function, *args):
    try:
        function(*args)
        return True
    except LookupError:
        return False


if __name__ == '__main__':
    main()
