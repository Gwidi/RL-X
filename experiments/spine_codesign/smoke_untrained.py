"""End-to-end smoke test using freshly initialized policy and critic weights."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def run(command):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
    subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="runs/spine_codesign_smoke",
    )
    parser.add_argument("--nr-envs", type=int, default=2)
    parser.add_argument("--state-bank-size", type=int, default=4)
    parser.add_argument("--search-steps", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_bank_path = output_dir / "nominal_state_bank.npz"
    search_path = output_dir / "best_spine_design.npz"
    common = [
        f"--algorithm.device={args.device}",
        f"--environment.device={args.device}",
        "--environment.terrain.type=plane",
        "--environment.policy_exteroceptive_observation_type=none",
        "--environment.critic_exteroceptive_observation_type=none",
        "--environment.env_curriculum_enabled=False",
        "--runner.track_console=True",
    ]
    run([
        sys.executable,
        str(Path(__file__).with_name("collect_state_bank.py")),
        *common,
        f"--environment.nr_envs={args.nr_envs}",
        f"--algorithm.spine_state_bank_size={args.state_bank_size}",
        f"--algorithm.spine_state_bank_path={state_bank_path}",
        "--runner.run_name=untrained_state_bank_smoke",
    ])
    run([
        sys.executable,
        str(Path(__file__).with_name("search_design.py")),
        *common,
        f"--algorithm.spine_state_bank_path={state_bank_path}",
        f"--algorithm.spine_search_output_path={search_path}",
        f"--algorithm.spine_search_steps={args.search_steps}",
        f"--algorithm.spine_search_minibatch_size={args.state_bank_size}",
        "--runner.run_name=untrained_design_search_smoke",
    ])

    with np.load(state_bank_path) as bank:
        state_bank = bank["state_bank"]
        design_indices = bank["spine_design_obs_indices"]
        nominal_patch = bank["nominal_spine_design_observation"]
        assert state_bank.shape[0] == args.state_bank_size
        assert np.all(np.isfinite(state_bank))
        assert np.allclose(state_bank[:, design_indices], nominal_patch[None, :])
    with np.load(search_path) as result:
        axis = result["best_rotation_axis"]
        design = result["best_spine_design"]
        assert all(
            np.all(np.isfinite(result[name]))
            for name in result.files
            if result[name].dtype.kind in "fiu"
        )
        assert np.isclose(np.linalg.norm(axis), 1.0, atol=1e-5)
        assert np.linalg.norm(design[-2:]) <= np.pi / 2.0 + 1e-5
        assert result["best_objective"] >= result["nominal_objective"]
    print(f"Untrained spine co-design smoke passed; artifacts: {output_dir}")


if __name__ == "__main__":
    main()
