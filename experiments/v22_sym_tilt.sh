#!/bin/bash
#SBATCH --job-name=v22_sym_tilt
#SBATCH --output=out_and_err_v22_sym_tilt_%j.txt
#SBATCH --error=out_and_err_v22_sym_tilt_%j.txt
#SBATCH --partition=proxima
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Symmetric nominal pose and imperfect release (roll and pitch each up to 30 deg, 15 deg/s, ramped over 20M steps),
# 60,030,976 more steps from job 9746307 with release speeds up to 7 m/s. Same budget and PPO as v22_speed_7ms_control.sh.
# Parent: job 9746307 latest converted to the new nominal pose, so that it commands the same joint targets as before:
#   python experiments/convert_parent_symmetric_pose.py --input modelsv22_upper/latest.model --output modelsv22_upper/symmetric.model
# Copy that file to the cluster (a repeated conversion has another checksum), or set PARENT_SYM=<path to it>.
# Submit from the repository root: sbatch experiments/v22_sym_tilt.sh --runner.track_wandb=False
# Every LANDING_V22_* setting is fixed here; variables left in the submitting shell are dropped.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
for variable in $(compgen -v LANDING_V22_ || true); do unset "$variable"; done
export LANDING_V22_ARM=soft
export LANDING_V22_NR_ENVS=2048
export LANDING_V22_MINIBATCH=8192
export LANDING_V22_ENTROPY_COEF=0.001
export LANDING_V22_MIN_STD=0.05
export LANDING_V22_SLIP_SCALE=0
export LANDING_V22_REFERENCE=upper
export LANDING_V22_MAX_SPEED=7
export LANDING_V22_SYMMETRIC_POSE=1
export LANDING_V22_RELEASE_TILT_DEG=30
export LANDING_V22_RELEASE_RATE_DPS=15
export LANDING_V22_CHECKPOINT="${PARENT_SYM:-$ROOT/modelsv22_upper/symmetric.model}"
export LANDING_V22_PARENT_STEPS=60030976
grep -q LANDING_V22_SYMMETRIC_POSE "$ROOT/experiments/train_landing_real_drives_mjx_v22.py" || {
    echo "Old code: the train script does not know LANDING_V22_SYMMETRIC_POSE. Sync the repository first." >&2; exit 1; }
echo "7f70903e83c9e704b098d64d4679d5997b6ee930ad5edac59256bae7ac2e3f84  $LANDING_V22_CHECKPOINT" | sha256sum --check --quiet
export RUN_NAME=v22_sym_tilt_${SLURM_JOB_ID:-local}
echo "Parent: $LANDING_V22_CHECKPOINT  reference: $LANDING_V22_REFERENCE  max speed: ${LANDING_V22_MAX_SPEED:-5}  symmetric pose  tilt: 30 deg  rate: 15 deg/s"
exec bash "$ROOT/experiments/real_drives_mjx_v22.sh" "$@"
