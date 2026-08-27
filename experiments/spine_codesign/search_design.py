"""Run nominal-referenced single-critic gradient search over spine designs."""

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from rl_x.runner.runner import Runner


def ensure_flag(name, value):
    prefix = f"--{name}="
    if not any(argument.startswith(prefix) for argument in sys.argv[1:]):
        sys.argv.append(f"{prefix}{value}")


if __name__ == "__main__":
    ensure_flag("algorithm.name", "ppo.flax_full_jit")
    ensure_flag("environment.name", "custom_mujoco.robot_locomotion.mjx")
    ensure_flag("runner.mode", "test")
    ensure_flag("runner.project_name", "spine_codesign")
    ensure_flag("runner.exp_name", "design_search")
    ensure_flag("environment.nr_envs", "1")
    ensure_flag("environment.train_robot", "silver_badger_codesign")
    ensure_flag("environment.spine_locked", "False")
    ensure_flag("environment.spine_design_randomization_enabled", "False")
    ensure_flag("algorithm.spine_codesign_operation", "search")
    Runner().run()
