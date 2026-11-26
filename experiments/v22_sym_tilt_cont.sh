#!/bin/bash
#SBATCH --job-name=v22_sym_tilt_cont
#SBATCH --output=out_and_err_v22_sym_tilt_cont_%j.txt
#SBATCH --error=out_and_err_v22_sym_tilt_cont_%j.txt
#SBATCH --partition=proxima
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Continuation of job 9755192 (v22_sym_tilt): 60,030,976 more steps from its latest.model with the same pose (front feet
# +-14 cm), tilt (roll and pitch each up to 30 deg), spin 30 deg/s and speeds up to 7 m/s. Changes: full disturbance from
# the first step (the parent already trained with it), learning rate again 3e-5 -> 1e-5, and a penalty on sudden hip
# movements before touchdown (the policy pulled its front feet from 27.5 to 15.5 cm apart right after release).
# The parent (db66b1b8...) is gitignored: copy it to the cluster, or set PARENT_SYM=<path to it>.
# Submit from the repository root: sbatch experiments/v22_sym_tilt_cont.sh --runner.track_wandb=False
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
export LANDING_V22_FRONT_HIP=0.009
export LANDING_V22_RELEASE_TILT_DEG=30
export LANDING_V22_RELEASE_RATE_DPS=30
export LANDING_V22_RELEASE_RAMP_STEPS=0
export LANDING_V22_FLIGHT_HIP_RATE=3
export LANDING_V22_CHECKPOINT="${PARENT_SYM:-$ROOT/modelsv22_tilt/latest.model}"
export LANDING_V22_PARENT_STEPS=60030976
grep -q LANDING_V22_FLIGHT_HIP_RATE "$ROOT/experiments/train_landing_real_drives_mjx_v22.py" || {
    echo "Old code: the train script does not know LANDING_V22_FLIGHT_HIP_RATE. Sync the repository first." >&2; exit 1; }
echo "db66b1b8b75345bc378428d38dd3530629a253c124c24d2d5528af9d63fddc37  $LANDING_V22_CHECKPOINT" | sha256sum --check --quiet
export RUN_NAME=v22_sym_tilt_cont_${SLURM_JOB_ID:-local}
echo "Parent: $LANDING_V22_CHECKPOINT  reference: $LANDING_V22_REFERENCE  max speed: ${LANDING_V22_MAX_SPEED:-5}  symmetric pose, front hip 0.009  tilt: 30 deg  rate: 30 deg/s, no ramp  flight hip rate penalty: 3"
exec bash "$ROOT/experiments/real_drives_mjx_v22.sh" "$@"
