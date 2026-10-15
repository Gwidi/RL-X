import mujoco
import mujoco.viewer
import numpy as np
import time

import main as landing

# Nominal pose defined PER JOINT NAME - edit values here and re-run
# to quickly test which sign/direction is correct for each joint.
NOMINAL_POSE = {
    "rl_j0": -0.1, "rl_j1": -0.2, "rl_j2":  0.8,
    "rr_j0":  0.1, "rr_j1": -0.2, "rr_j2": -0.8,
    "fr_j0": -0.1, "fr_j1":  0.5, "fr_j2": -0.8,
    "fl_j0":  0.1, "fl_j1": -0.5, "fl_j2":  0.8,
    "sp_j0":  0.0,
}

# Reasonable fixed PD gains just to hold the pose steady for viewing.
KP = 40.0
KV = 3.0

# Body starting height - set high enough that legs don't clip the
# ground before you've had a chance to see the pose, but you can also
# set this to something like 0.3 if you just want to see the pose
# sitting near the floor without a drop.
START_HEIGHT = 0.6


def main():
    model = mujoco.MjModel.from_xml_path(str(landing.XML_PATH))
    data = mujoco.MjData(model)

    joint_names = list(NOMINAL_POSE.keys())
    jmap = landing.JointMap(model, joint_names)
    kp = dict.fromkeys(joint_names, KP)
    kd = dict.fromkeys(joint_names, KV)

    def reset_to_nominal():
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = [0.0, 0.0, START_HEIGHT]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for name, val in NOMINAL_POSE.items():
            data.qpos[jmap.qpos_adr[name]] = val
        landing.apply_position_pd(model, data, jmap, NOMINAL_POSE, kp, kd)
        mujoco.mj_forward(model, data)

    reset_to_nominal()

    print("Wizualizacja pozycji nominalnej.")
    print("Nacisnij spacje w oknie viewera aby zapauzowac/wznowic fizyke.")
    print("Zamknij okno viewera aby zakonczyc.")
    for name, val in NOMINAL_POSE.items():
        print(f"  {name}: {val}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_reset = time.time()
        while viewer.is_running():
            landing.controlled_step(model, data, jmap, NOMINAL_POSE, kp, kd)
            viewer.sync()
            time.sleep(model.opt.timestep)

            # Auto-reset every 5s so the pose doesn't slowly drift/fall
            # off screen if you leave it running.
            if time.time() - last_reset > 5.0:
                reset_to_nominal()
                last_reset = time.time()


if __name__ == "__main__":
    main()
