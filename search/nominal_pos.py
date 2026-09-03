import mujoco
import mujoco.viewer
import numpy as np
import time

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
    xml_path = "intention.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    joint_names = list(NOMINAL_POSE.keys())
    qpos_adr = {n: model.joint(n).qposadr[0] for n in joint_names}
    act_id = {n: model.actuator(n).id for n in joint_names}

    # IMPORTANT: the XML defines these as <motor> actuators, which have
    # biastype="none" hardcoded. That means biasprm is silently ignored
    # by the physics engine - ctrl acts as raw torque, NOT a position
    # target, no matter what you put in gainprm/biasprm.
    #
    # To actually get position-control (PD) behavior:
    #   force = gainprm[0]*(ctrl) + biasprm[1]*qpos + biasprm[2]*qvel
    # we need biastype = affine (mjBIAS_AFFINE = 1). This is safe to set
    # at runtime because actuator_biastype is just a mutable model array,
    # even though it's normally fixed by the XML element type at compile
    # time.
    mjBIAS_AFFINE = 1
    all_act_idx = np.array([act_id[n] for n in joint_names])
    model.actuator_biastype[all_act_idx] = mjBIAS_AFFINE
    model.actuator_gainprm[all_act_idx, 0] = KP
    model.actuator_biasprm[all_act_idx, 1] = -KP
    model.actuator_biasprm[all_act_idx, 2] = -KV

    def reset_to_nominal():
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = [0.0, 0.0, START_HEIGHT]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for name, val in NOMINAL_POSE.items():
            data.qpos[qpos_adr[name]] = val
            data.ctrl[act_id[name]] = val
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
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

            # Auto-reset every 5s so the pose doesn't slowly drift/fall
            # off screen if you leave it running.
            if time.time() - last_reset > 5.0:
                reset_to_nominal()
                last_reset = time.time()


if __name__ == "__main__":
    main()