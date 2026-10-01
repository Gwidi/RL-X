"""Evaluate one fixed controller and nominal pose at several drop heights.

Without --config, use the individual joint gains and pose below with
safety_factor=0.4. A custom JSON config can override those values.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

import main


GROUP_GAIN_NAMES = (*main.LEG_GAIN_PARAMETER_NAMES, "kp_spine", "kd_spine")
INDIVIDUAL_GAIN_NAMES = (*main.INDIVIDUAL_GAIN_NAMES, "kp_spine", "kd_spine")
DEFAULT_SAFETY_FACTOR = 0.4
DEFAULT_GAINS = {
    "kp_rl_j0": 34.69, "kp_rr_j0": 46.28, "kp_fr_j0": 34.96, "kp_fl_j0": 57.14,
    "kp_rl_j1": 72.03, "kp_rr_j1": 31.57, "kp_fr_j1": 34.15, "kp_fl_j1": 95.40,
    "kp_rl_j2": 79.89, "kp_rr_j2": 32.85, "kp_fr_j2": 46.39, "kp_fl_j2": 20.24,
    "kd_rl_j0": 2.04, "kd_rr_j0": 3.90, "kd_fr_j0": 1.46, "kd_fl_j0": 3.32,
    "kd_rl_j1": 2.10, "kd_rr_j1": 5.48, "kd_fr_j1": 6.98, "kd_fl_j1": 4.69,
    "kd_rl_j2": 0.86, "kd_rr_j2": 4.95, "kd_fr_j2": 2.34, "kd_fl_j2": 2.88,
    "kp_spine": 49.5, "kd_spine": 1.6,
}
DEFAULT_POSE = {
    "rl_j0": -0.2317, "rr_j0": -0.2311, "fr_j0": -0.2693, "fl_j0": 0.2561,
    "rl_j1": -0.6539, "rr_j1": 0.9465, "fr_j1": 0.7974, "fl_j1": -0.8094,
    "rl_j2": 1.4727, "rr_j2": -1.1603, "fr_j2": -1.9541, "fl_j2": 1.8656,
    "sp_j0": -0.3930,
}
METRICS = (
    "cost", "min_height", "peak_foot_force", "peak_foot_force_raw",
    "peak_total_foot_force", "peak_body_acceleration",
    "peak_body_acceleration_raw", "peak_leg_torque", "total_motor_effort",
    "total_leg_effort", "total_spine_effort", "peak_leg_current_proxy",
    "peak_spine_current_proxy", "crashed", "foot_contact",
    "non_foot_contact", "shin_contact",
)


def read_settings(path):
    if path is None:
        return dict(DEFAULT_GAINS), dict(DEFAULT_POSE), DEFAULT_SAFETY_FACTOR
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict) or not {"gains", "pose"} <= set(config) or set(config) - {"gains", "pose", "safety_factor"}:
        raise ValueError('Config must contain "gains" and "pose"; optional: "safety_factor"')
    gains = config["gains"]
    if not isinstance(gains, dict) or set(gains) not in (set(GROUP_GAIN_NAMES), set(INDIVIDUAL_GAIN_NAMES)):
        raise ValueError("gains must contain all group gains or all individual joint gains, plus spine gains")
    for label, values, expected in (
        ("gains", gains, set(gains)),
        ("pose", config["pose"], set(main.NOMINAL_POSE)),
    ):
        if not isinstance(values, dict) or set(values) != expected:
            raise ValueError(f"{label} must contain exactly: {', '.join(sorted(expected))}")
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{label}.{name} must be a finite number")
    factor = config.get("safety_factor", 1.0)
    if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not math.isfinite(factor) or not 0 < factor <= 1:
        raise ValueError("safety_factor must be in (0, 1]")
    return gains, config["pose"], factor


def plot_results(rows, output_dir):
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    heights = [row["height"] for row in rows]
    series = (
        ("cost", "Koszt", None),
        ("peak_foot_force", "Szczytowa średnia siła jednej stopy [N]", main.MAX_LANDING_FOOT_FORCE),
        ("peak_body_acceleration", "Szczytowe średnie przyspieszenie [m/s²]", main.MAX_LANDING_BODY_ACCELERATION),
        ("total_motor_effort", "Całkowity wysiłek silników", None),
        ("min_height", "Minimalna wysokość bazy [m]", None),
        ("safe", "Bezpieczne (1 = tak)", None),
    )
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    for axis, (key, title, limit) in zip(axes.flat, series):
        axis.plot(heights, [row[key] for row in rows], marker="o")
        if limit is not None:
            axis.axhline(limit, color="tab:red", linestyle="--", label="limit")
            axis.legend()
        axis.set_title(title)
        axis.set_xlabel("Wysokość zrzutu [m]")
        axis.grid(alpha=0.25)
    axes.flat[-1].set_yticks([0, 1], ["nie", "tak"])
    fig.suptitle("Jedna pozycja i jeden regulator — różne wysokości")
    fig.savefig(output_dir / "wyniki_wzgledem_wysokosci.png", dpi=150)
    plt.close(fig)


def validate_fixed_target(heights, params, individual_gains, safety_factor, args):
    """Reuse main.py's disturbances, safety checks, CI, CSV, and plot."""
    reference_model = mujoco.MjModel.from_xml_path(str(main.XML_PATH))
    validation = {}
    settings = SimpleNamespace(
        validation_trials=args.validation_trials,
        validation_height_jitter=args.validation_height_jitter,
        validation_position_jitter=args.validation_position_jitter,
        validation_angle_jitter_deg=args.validation_angle_jitter_deg,
        validation_linear_velocity_jitter=args.validation_linear_velocity_jitter,
        validation_angular_velocity_jitter_deg=args.validation_angular_velocity_jitter_deg,
        validation_joint_jitter=args.validation_joint_jitter,
        validation_mass_jitter=args.validation_mass_jitter,
        validation_friction_jitter=args.validation_friction_jitter,
    )
    print(f"Walidacja Monte Carlo: {args.validation_trials} prób na wysokość | seed={args.validation_seed}", flush=True)
    for height_index, height in enumerate(heights):
        scenario_seed = np.random.SeedSequence([
            args.validation_seed % (2**32), height_index,
        ])
        scenarios = main.validation_scenarios(settings, reference_model, scenario_seed)
        summary = main.validate_candidate(
            {"params": params}, height, False, True, scenarios,
            individual_gains=individual_gains, safety_factor=safety_factor,
        )
        validation[height] = {False: summary}
        print(
            f"{height:g} m | {summary['successes']}/{summary['trials']} bezpiecznych "
            f"({summary['success_rate']:.1%}; 95% CI "
            f"{summary['confidence_low']:.1%}–{summary['confidence_high']:.1%})",
            flush=True,
        )
    main.save_validation_results(validation, args.output_dir, settings, args.validation_seed)
    print(f"Raport walidacji zapisano w: {args.output_dir}")


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Przykład: python3 sweep_target.py --heights 0.8 0.9 1.0 1.1 "
                "--output-dir output/sweep_target\n"
                "Format --config: JSON z gains (grupowe lub indywidualne Kp/Kd), "
                "pose (13 stawów) i opcjonalnym safety_factor."))
    parser.add_argument("--heights", nargs="+", type=float, required=True, metavar="M")
    parser.add_argument("--config", type=Path, help="JSON z własnymi nastawami; domyślnie wartości w tym skrypcie")
    parser.add_argument("--output-dir", type=Path, default=Path("output/sweep_target"))
    parser.add_argument("--safety-factor", type=float, default=None,
                        help="nadpisuje wartość z konfiguracji (domyślnie 0.4)")
    parser.add_argument("--validation-trials", type=int, default=1000,
                        help="liczba zaburzonych prób na wysokość (domyślnie 1000)")
    parser.add_argument("--validation-seed", type=int, default=42,
                        help="ziarno losowe walidacji (domyślnie 42)")
    parser.add_argument("--no-validation", action="store_true",
                        help="uruchom tylko próby nominalne, bez walidacji Monte Carlo")
    parser.add_argument("--validation-height-jitter", type=float, default=0.02)
    parser.add_argument("--validation-position-jitter", type=float, default=0.005)
    parser.add_argument("--validation-angle-jitter-deg", type=float, default=2.0)
    parser.add_argument("--validation-linear-velocity-jitter", type=float, default=0.10)
    parser.add_argument("--validation-angular-velocity-jitter-deg", type=float, default=1.0)
    parser.add_argument("--validation-joint-jitter", type=float, default=0.02)
    parser.add_argument("--validation-mass-jitter", type=float, default=0.02)
    parser.add_argument("--validation-friction-jitter", type=float, default=0.01)
    args = parser.parse_args()
    if any(not math.isfinite(h) or h <= 0 for h in args.heights):
        parser.error("all heights must be finite and positive")
    try:
        gains, pose, config_safety_factor = read_settings(args.config)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    safety_factor = config_safety_factor if args.safety_factor is None else args.safety_factor
    if not math.isfinite(safety_factor) or not 0 < safety_factor <= 1:
        parser.error("--safety-factor must be in (0, 1]")
    if args.validation_trials < 1:
        parser.error("--validation-trials must be at least 1")
    for option in (
        "validation_height_jitter", "validation_position_jitter",
        "validation_angle_jitter_deg", "validation_linear_velocity_jitter",
        "validation_angular_velocity_jitter_deg", "validation_joint_jitter",
        "validation_mass_jitter", "validation_friction_jitter",
    ):
        value = getattr(args, option)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{option.replace('_', '-')} must be finite and non-negative")
    if args.validation_mass_jitter >= 1 or args.validation_friction_jitter >= 1:
        parser.error("mass and friction jitter must be less than 1")

    names = main.nominal_position_names(True, False)
    individual_gains = set(gains) == set(INDIVIDUAL_GAIN_NAMES)
    gain_names = INDIVIDUAL_GAIN_NAMES if individual_gains else GROUP_GAIN_NAMES
    params = tuple(gains[name] for name in gain_names) + tuple(
        pose[name] - main.NOMINAL_POSE[name] for name in names
    )
    model = mujoco.MjModel.from_xml_path(str(main.XML_PATH))
    main.configure_safety_factor(model, safety_factor)
    data = mujoco.MjData(model)
    jmap = main.JointMap(model, main.NOMINAL_POSE)
    contact_map = main.ContactMap(model)
    rows = []
    for height in sorted(set(args.heights)):
        result = main.result_from_params(
            model, data, jmap, contact_map, params, main.episode_steps(model, height),
            height, optimize_nominal_position=True, lock_spine=False,
            individual_gains=individual_gains,
        )
        row = {"height": height, "safety_factor": safety_factor,
               "safe": int(main.is_safe(result))}
        row.update((key, result[key]) for key in METRICS)
        rows.append(row)
        print(f"{height:g} m | safe={bool(row['safe'])} | cost={row['cost']:.2f} | "
              f"foot={row['peak_foot_force']:.1f} N | "
              f"acc={row['peak_body_acceleration']:.1f} m/s²")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "wyniki.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["height", "safety_factor", "safe", *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "ustawienia.json").open("w", encoding="utf-8") as stream:
        json.dump({"gains": gains, "pose": pose, "safety_factor": safety_factor},
                  stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    plot_results(rows, args.output_dir)
    print(f"CSV, wykres i nastawy zapisano w: {args.output_dir}")
    if not args.no_validation:
        validate_fixed_target(
            [row["height"] for row in rows], params, individual_gains,
            safety_factor, args,
        )


if __name__ == "__main__":
    main_cli()
