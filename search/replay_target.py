"""Replay one selected controller and nominal pose without optimization."""

import argparse

import mujoco

import main


DEFAULT_SAFETY_FACTOR = 1.0
GAINS = {
    "kp_rl_j0": 47.08596923496373, "kp_rr_j0": 47.08596923496373,
    "kp_fr_j0": 47.08596923496373, "kp_fl_j0": 47.08596923496373,
    "kp_rl_j1": 27.917316744291625, "kp_rr_j1": 27.917316744291625,
    "kp_fr_j1": 27.917316744291625, "kp_fl_j1": 27.917316744291625,
    "kp_rl_j2": 35.12551071141482, "kp_rr_j2": 35.12551071141482,
    "kp_fr_j2": 35.12551071141482, "kp_fl_j2": 35.12551071141482,
    "kd_rl_j0": 4.982580586614543, "kd_rr_j0": 4.982580586614543,
    "kd_fr_j0": 4.982580586614543, "kd_fl_j0": 4.982580586614543,
    "kd_rl_j1": 4.097134814994861, "kd_rr_j1": 4.097134814994861,
    "kd_fr_j1": 4.097134814994861, "kd_fl_j1": 4.097134814994861,
    "kd_rl_j2": 4.736790692908366, "kd_rr_j2": 4.736790692908366,
    "kd_fr_j2": 4.736790692908366, "kd_fl_j2": 4.736790692908366,
    "kp_spine": 93.72874168562895, "kd_spine": 6.452430889459536,
}

TARGET_POSE = {
    "rl_j0": -0.35526737949810094, "rl_j1": -0.5414327949563804,
    "rl_j2": 1.7844124379552,
    "rr_j0": -0.305742365720427, "rr_j1": 0.8910097959948484,
    "rr_j2": -1.8914502524731234,
    "fr_j0": 0.20936793090716296, "fr_j1": 0.6683483071033249,
    "fr_j2": -1.511636207634632,
    "fl_j0": -0.01361010976020341, "fl_j1": -0.5962346129233117,
    "fl_j2": 1.3203758695715557,
    "sp_j0": 0.14655053593988143,
}


def target_params():
    """Return gains followed by pose deltas in main.py's expected order."""
    names = main.nominal_position_names(
        optimize_nominal_position=True,
        lock_spine=False,
    )
    deltas = tuple(TARGET_POSE[name] - main.NOMINAL_POSE[name] for name in names)
    gain_names = (*main.INDIVIDUAL_GAIN_NAMES, "kp_spine", "kd_spine")
    return tuple(GAINS[name] for name in gain_names) + deltas


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the saved target controller in the CPU MuJoCo simulation."
    )
    parser.add_argument(
        "--height",
        type=float,
        default=2.3,
        help="drop height in metres (default: 2.3 m)",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="only evaluate and print metrics; do not open the 3D viewer",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.25,
        help="3D playback speed relative to real time (default: 0.25)",
    )
    parser.add_argument(
        "--show-colliders",
        action="store_true",
        help="feet green, permitted lower calf third blue, upper two thirds orange (red on contact)",
    )
    parser.add_argument(
        "--contact-pause",
        type=float,
        default=1.0,
        help="pause at first forbidden contact (upper calf or other body part); default: 1 s, 0 disables",
    )
    parser.add_argument(
        "--safety-factor", type=float, default=DEFAULT_SAFETY_FACTOR, metavar="F",
        help=f"maksymalny uzywany moment jako czesc limitu aktuatora (0<F<=1; domyslnie {DEFAULT_SAFETY_FACTOR:g})",
    )
    args = parser.parse_args()
    if args.height <= 0.0:
        parser.error("--height must be positive")
    if args.speed <= 0.0:
        parser.error("--speed must be positive")
    if args.contact_pause < 0.0:
        parser.error("--contact-pause must be non-negative")
    if not 0.0 < args.safety_factor <= 1.0:
        parser.error("--safety-factor must be greater than 0 and at most 1")
    return args


def run(height, safety_factor=DEFAULT_SAFETY_FACTOR):
    model = mujoco.MjModel.from_xml_path(str(main.XML_PATH))
    main.configure_safety_factor(model, safety_factor)
    data = mujoco.MjData(model)
    jmap = main.JointMap(model, main.NOMINAL_POSE)
    contact_map = main.ContactMap(model)
    steps = main.episode_steps(model, height)
    params = target_params()
    result = main.result_from_params(
        model,
        data,
        jmap,
        contact_map,
        params,
        steps,
        height,
        optimize_nominal_position=True,
        lock_spine=False,
        individual_gains=True,
    )
    return model, data, jmap, steps, params, result


def cli():
    args = parse_args()
    model, data, jmap, steps, params, result = run(args.height, args.safety_factor)
    print(f"\nOdtwarzanie zapisanych parametrow z wysokosci {args.height:g} m")
    main.print_result(
        result,
        model,
        optimize_nominal_position=True,
        lock_spine=False,
        individual_gains=True,
    )

    if not args.no_viewer:
        if args.show_colliders:
            print("Collidery: zielone = stopy, niebieskie = dolna 1/3 lydki (dozwolona), "
                  "pomaranczowe = gorne 2/3, czerwone = ich niedozwolony kontakt.")
        print(
            f"\nOtwieram symulacje 3D z predkoscia {args.speed:g}x "
            "... Zamknij okno, by zakonczyc skrypt."
        )
        main.show_best(
            model,
            data,
            jmap,
            params,
            steps,
            args.height,
            optimize_nominal_position=True,
            lock_spine=False,
            playback_speed=args.speed,
            individual_gains=True,
            show_contacts=True,
            show_colliders=args.show_colliders,
            contact_pause=args.contact_pause,
        )


if __name__ == "__main__":
    cli()
