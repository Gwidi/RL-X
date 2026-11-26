#!/bin/bash
#SBATCH --job-name=v22_speed_7ms_control
#SBATCH --output=out_and_err_v22_speed_7ms_control_%j.txt
#SBATCH --error=out_and_err_v22_speed_7ms_control_%j.txt
#SBATCH --partition=proxima
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Speed test, arm A (control): 60,030,976 more steps with release speeds up to 7 m/s, as in the parent run.
# Compare with arm B (v22_speed_10ms.sh).
# Parent for both: job 9746307 (upper, up to 7 m/s). Copy the local modelsv22_upper/latest.model to the same path on the cluster,
# or set PARENT_UPPER7=<path to that file>. The checksum below pins the file.
# Submit from the repository root: sbatch experiments/v22_speed_7ms_control.sh --runner.track_wandb=False
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
export LANDING_V22_CHECKPOINT="${PARENT_UPPER7:-$ROOT/modelsv22_upper/latest.model}"
export LANDING_V22_PARENT_STEPS=60030976
grep -q LANDING_V22_MAX_SPEED "$ROOT/experiments/train_landing_real_drives_mjx_v22.py" || {
    echo "Old code: the train script does not know LANDING_V22_MAX_SPEED. Sync the repository first." >&2; exit 1; }
echo "c257fe187d4e3ff26fdf32a44f76e746411c1921088dd1339ba2a2b57cc30cd8  $LANDING_V22_CHECKPOINT" | sha256sum --check --quiet
export RUN_NAME=v22_speed_7ms_control_${SLURM_JOB_ID:-local}
echo "Parent: $LANDING_V22_CHECKPOINT  reference: $LANDING_V22_REFERENCE  max speed: ${LANDING_V22_MAX_SPEED:-5}"
exec bash "$ROOT/experiments/real_drives_mjx_v22.sh" "$@"
