"""Train the fixed seed606 baseline; --check-only prints the resolved profile."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.landing_seed606 import (
    training_profile, cli_flags, verify_controller,
)

if __name__ == "__main__":
    fingerprint = verify_controller()
    profile = training_profile()
    if "--check-only" in sys.argv:
        print(json.dumps(dict(profile=profile, fingerprint=fingerprint), indent=2))
    else:
        overrides = sys.argv[1:]
        sys.argv = [sys.argv[0], *cli_flags(profile), *overrides]
        print("Seed606 fixed PD verified; model fingerprint:", fingerprint["combined_sha256"], flush=True)
        from rl_x.runner.runner import Runner
        Runner().run()
