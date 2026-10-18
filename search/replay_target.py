"""Replay one selected controller and nominal pose without optimization."""

import argparse

import mujoco

import main


DEFAULT_SAFETY_FACTOR = 1.0
GAINS = {
    "kp_rl_j0": 58.09479648934883, "kp_rr_j0": 58.09479648934883,
    "kp_fr_j0": 58.09479648934883, "kp_fl_j0": 58.09479648934883,
    "kp_rl_j1": 52.80778600694903, "kp_rr_j1": 52.80778600694903,
    "kp_fr_j1": 52.80778600694903, "kp_fl_j1": 52.80778600694903,
    "kp_rl_j2": 26.47990980211776, "kp_rr_j2": 26.47990980211776,
    "kp_fr_j2": 26.47990980211776, "kp_fl_j2": 26.47990980211776,
    "kd_rl_j0": 0.8238884722767676, "kd_rr_j0": 0.8238884722767676,
    "kd_fr_j0": 0.8238884722767676, "kd_fl_j0": 0.8238884722767676,
    "kd_rl_j1": 2.9008560450560466, "kd_rr_j1": 2.9008560450560466,
    "kd_fr_j1": 2.9008560450560466, "kd_fl_j1": 2.9008560450560466,
    "kd_rl_j2": 5.51261940468796, "kd_rr_j2": 5.51261940468796,
    "kd_fr_j2": 5.51261940468796, "kd_fl_j2": 5.51261940468796,
    "kp_spine": 55.674070964772085, "kd_spine": 4.7500745112296885,
}

TARGET_POSE = {
    "rl_j0": -0.41171856408561114, "rl_j1": -0.5570486148551262,
    "rl_j2": 1.1690932541029424,
    "rr_j0": -0.21451776249894913, "rr_j1": 0.9546235042547561,
    "rr_j2": -1.3457963120954846,
    "fr_j0": -0.6786961214908629, "fr_j1": 0.9637337413006044,
    "fr_j2": -1.4839694192500061,
    "fl_j0": -0.15542555790314902, "fl_j1": -1.0489715078078325,
    "fl_j2": 1.8830434949818602,
    "sp_j0": 0.5356723828513866,
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
        default=2.5,
        help="drop height in metres (default: 2.5 m)",
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
        help="show calf collision cylinders in translucent orange",
    )
    parser.add_argument(
        "--contact-pause",
        type=float,
        default=1.0,
        help="pause at first non-foot contact for this many seconds (default: 1; 0 disables)",
    )
    parser.add_argument(
        "--safety-factor", type=float, default=DEFAULT_SAFETY_FACTOR, metavar="F",
        help="maksymalny uzywany moment jako czesc limitu aktuatora (0<F<=1; domyslnie 0.4)",
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
