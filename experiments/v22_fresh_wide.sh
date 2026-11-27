#!/bin/bash
#SBATCH --job-name=v22_fresh_wide
#SBATCH --output=out_and_err_v22_fresh_wide_%j.txt
#SBATCH --error=out_and_err_v22_fresh_wide_%j.txt
#SBATCH --partition=proxima
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# From scratch in the current pose (symmetric, front feet +-14 cm): does the policy still pull its legs in when it does
# not inherit the habits of a parent trained in the old pose? No motion penalty. Same task as v22_sym_tilt: soft limb
# contact, speeds up to 7 m/s, roll and pitch each up to 30 deg and spin 30 deg/s (ramped over the first 40M steps).
# PPO as the earlier successful from-scratch V22 run (job 9732244): 1024 envs, minibatch 64, std 0.15; std floor 0.05.
# 149,946,368 steps. No parent file is needed.
# Submit from the repository root: sbatch experiments/v22_fresh_wide.sh --runner.track_wandb=False
# Every LANDING_V22_* setting is fixed here; variables left in the submitting shell are dropped.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
for variable in $(compgen -v LANDING_V22_ || true); do unset "$variable"; done
export LANDING_V22_ARM=soft
export LANDING_V22_FRESH=1
export LANDING_V22_TOTAL_STEPS=149946368
export LANDING_V22_NR_ENVS=1024
export LANDING_V22_MIN_STD=0.05
export LANDING_V22_SLIP_SCALE=0
export LANDING_V22_REFERENCE=upper
export LANDING_V22_MAX_SPEED=7
export LANDING_V22_SYMMETRIC_POSE=1
export LANDING_V22_FRONT_HIP=0.009
export LANDING_V22_RELEASE_TILT_DEG=30
export LANDING_V22_RELEASE_RATE_DPS=30
export LANDING_V22_RELEASE_RAMP_STEPS=40000000
grep -q LANDING_V22_FRESH "$ROOT/experiments/train_landing_real_drives_mjx_v22.py" || {
    echo "Old code: the train script does not know LANDING_V22_FRESH. Sync the repository first." >&2; exit 1; }
export RUN_NAME=v22_fresh_wide_${SLURM_JOB_ID:-local}
echo "From scratch  reference: $LANDING_V22_REFERENCE  max speed: $LANDING_V22_MAX_SPEED  symmetric pose, front hip 0.009  tilt: 30 deg  rate: 30 deg/s  no motion penalty"
exec bash "$ROOT/experiments/real_drives_mjx_v22.sh" "$@"
