"""The robot hanging in the air in the old and in the symmetric release pose. No physics is stepped.

    python experiments/analysis/view_release_pose.py                 # interactive viewer, SPACE switches old/new
    python experiments/analysis/view_release_pose.py --png out.png   # headless: front, top, side and rear views of both

The symmetric pose copies the wider (right) legs to the left side, front and rear; the spine is unchanged.
"""
import argparse
import os
from pathlib import Path
import time

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--png', type=Path, help='write a comparison image instead of opening the viewer')
parser.add_argument('--height', type=float, default=1.0, help='height of the trunk above the floor, m')
parser.add_argument('--front-hip', type=float, help='right front hip of the new pose, rad (left mirrors it); 0.159 widens the front')
args = parser.parse_args()
if args.png:
    os.environ.setdefault('MUJOCO_GL', 'egl')

import mujoco
import numpy as np

XML = Path(__file__).resolve().parents[2] / 'rl_x/environments/custom_mujoco/robot_locomotion/robots/silver_badger/data/plane.xml'
LEGS = ('RL', 'RR', 'FR', 'FL')
JOINTS = ['spine_joint'] + [f'{leg}_{part}_joint' for leg in LEGS for part in ('hip', 'thigh', 'calf')]

model = mujoco.MjModel.from_xml_path(str(XML))
data = mujoco.MjData(model)
address = [model.joint(name).qposadr[0] for name in JOINTS]
old = model.key('home').qpos[address].copy()

new = old.copy()
for target, source in (('RL', 'RR'), ('FL', 'FR')):  # mirrored leg joints change sign
    t, s = 1 + 3 * LEGS.index(target), 1 + 3 * LEGS.index(source)
    new[t:t + 3] = -old[s:s + 3]

if args.front_hip is not None:
    new[1 + 3 * LEGS.index('FR')], new[1 + 3 * LEGS.index('FL')] = args.front_hip, -args.front_hip


def show(pose):
    data.qpos[:] = 0
    data.qpos[2], data.qpos[3] = args.height, 1
    data.qpos[address] = pose
    mujoco.mj_forward(model, data)


def describe(label, pose):
    show(pose)
    print(f'\n{label}: foot position relative to the trunk, m (x forward, y left, z up)')
    for leg in LEGS:
        position = data.geom(f'{leg}_foot').xpos - data.qpos[:3]
        angles = pose[1 + 3 * LEGS.index(leg):4 + 3 * LEGS.index(leg)]
        print(f'  {leg}: x {position[0]:+.3f}  y {position[1]:+.3f}  z {position[2]:+.3f}   hip/thigh/calf {np.round(angles, 3).tolist()}')
    for joint, value in zip(JOINTS, pose):
        low, high = model.joint(joint).range
        assert low <= value <= high, f'{joint} = {value} is outside [{low}, {high}]'
    print(f'  contacts in the air (self-collisions): {data.ncon}')


describe('OLD pose', old)
describe('NEW symmetric pose', new)
print('\nNEW pose as the "home" joint values (spine, RL, RR, FR, FL):\n  ' + ' '.join(f'{v:.4f}' for v in new))

if args.png:
    from PIL import Image, ImageDraw, ImageFont
    renderer = mujoco.Renderer(model, height=420, width=480)
    camera = mujoco.MjvCamera()
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    # MuJoCo azimuth is the viewing direction: 180 looks at the robot from the front.
    views = (('z przodu', 180, 0), ('z góry', 0, -89), ('z boku (lewego)', 270, 0), ('z tyłu', 0, 0))
    sheet = Image.new('RGB', (480 * len(views), 420 * 2), 'white')
    for row, (label, pose) in enumerate((('STARA', old), ('NOWA', new))):
        show(pose)
        for column, (name, azimuth, elevation) in enumerate(views):
            camera.lookat[:] = data.qpos[:3] + (-.06, 0, -.1)
            camera.distance, camera.azimuth, camera.elevation = 1.5, azimuth, elevation
            renderer.update_scene(data, camera=camera)
            tile = Image.fromarray(renderer.render())
            ImageDraw.Draw(tile).text((10, 8), f'{label} · {name}', font=font, fill='black', stroke_width=3, stroke_fill='white')
            sheet.paste(tile, (480 * column, 420 * row))
    sheet.save(args.png)
    print('\nwritten', args.png)
else:
    import mujoco.viewer
    state = dict(new=True, changed=True)

    def key(code):
        if code == 32:  # space
            state['new'], state['changed'] = not state['new'], True

    with mujoco.viewer.launch_passive(model, data, key_callback=key) as viewer:
        viewer.cam.lookat[:] = (-.06, 0, args.height - .1)
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.6, 140, -15
        print('\nSPACE: switch between the old and the new pose. Close the window to quit.')
        while viewer.is_running():
            if state['changed']:
                with viewer.lock():
                    show(new if state['new'] else old)
                print('showing', 'NEW symmetric pose' if state['new'] else 'OLD pose')
                state['changed'] = False
            viewer.sync()
            time.sleep(.02)
