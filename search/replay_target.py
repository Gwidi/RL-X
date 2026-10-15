"""Replay one selected controller and nominal pose without optimization."""

import argparse

import mujoco

import main


DEFAULT_SAFETY_FACTOR = 1.0
GAINS = {
    "kp_rl_j0": 78.29790238325148, "kp_rr_j0": 78.29790238325148,
    "kp_fr_j0": 78.29790238325148, "kp_fl_j0": 78.29790238325148,
    "kp_rl_j1": 34.55019222159159, "kp_rr_j1": 34.55019222159159,
    "kp_fr_j1": 34.55019222159159, "kp_fl_j1": 34.55019222159159,
    "kp_rl_j2": 70.23450465243452, "kp_rr_j2": 70.23450465243452,
    "kp_fr_j2": 70.23450465243452, "kp_fl_j2": 70.23450465243452,
    "kd_rl_j0": 4.120761713956751, "kd_rr_j0": 4.120761713956751,
    "kd_fr_j0": 4.120761713956751, "kd_fl_j0": 4.120761713956751,
    "kd_rl_j1": 3.8116563934277656, "kd_rr_j1": 3.8116563934277656,
    "kd_fr_j1": 3.8116563934277656, "kd_fl_j1": 3.8116563934277656,
    "kd_rl_j2": 1.8898610610930175, "kd_rr_j2": 1.8898610610930175,
    "kd_fr_j2": 1.8898610610930175, "kd_fl_j2": 1.8898610610930175,
    "kp_spine": 76.29336012180711, "kd_spine": 5.707560370178117,
}

TARGET_POSE = {
    "rl_j0": -0.13927784509755228, "rl_j1": -0.7970243693190545,
    "rl_j2": 1.2269909750004113,
    "rr_j0": 0.07883064076202409, "rr_j1": 0.871019917290016,
    "rr_j2": -2.076092219982915,
    "fr_j0": -0.038510704532694146, "fr_j1": 0.8785414591207148,
    "fr_j2": -2.0259944183220675,
    "fl_j0": -0.14699264218755745, "fl_j1": -0.609438804223659,
    "fl_j2": 1.4055736643922827,
    "sp_j0": -0.49934998209733805,
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
        default=3.0,
        help="drop height in metres (default: 3 m)",
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
        "--safety-factor", type=float, default=DEFAULT_SAFETY_FACTOR, metavar="F",
        help="maksymalny uzywany moment jako czesc limitu aktuatora (0<F<=1; domyslnie 0.4)",
    )
    args = parser.parse_args()
    if args.height <= 0.0:
        parser.error("--height must be positive")
    if args.speed <= 0.0:
        parser.error("--speed must be positive")
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
        )


if __name__ == "__main__":
    cli()
