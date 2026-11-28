#!/bin/bash
#SBATCH --job-name=v22_end_success
#SBATCH --output=out_and_err_v22_end_success_%j.txt
#SBATCH --error=out_and_err_v22_end_success_%j.txt
#SBATCH --partition=proxima
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# EXPERIMENT: no prescribed braking. From job 9761870, 200,015,872 steps. The braking reward and all braking time
# requirements are removed: success is judged once, 5 s after release (no violation, on its feet, almost still for the
# last 0.5 s, tilt < 45 deg, base above 0.2 m). The failure limits and small penalties are unchanged. gamma 0.995 instead
# of 0.99, because the success reward now arrives at the end of the episode. Randomization with floor friction as in
# v22_rand_friction; no hip-motion penalty, no posture reward. Success rates are NOT comparable with earlier runs.
# Success also requires a side roll of at most 15 deg at the end (LANDING_V22_END_ROLL_DEG below).
# The parent (48bbde43...) is gitignored: copy modelsv22_randomized/latest.model to the cluster, or set PARENT_RAND.
# Submit from the repository root: sbatch experiments/v22_end_success.sh --runner.track_wandb=False
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
export LANDING_V22_REFERENCE=off
export LANDING_V22_SUCCESS_AT_END=1
export LANDING_V22_END_ROLL_DEG=15
export LANDING_V22_GAMMA=0.995
export LANDING_V22_MAX_SPEED=7
export LANDING_V22_SYMMETRIC_POSE=1
export LANDING_V22_FRONT_HIP=0.009
export LANDING_V22_RELEASE_TILT_DEG=30
export LANDING_V22_RELEASE_RATE_DPS=30
export LANDING_V22_RELEASE_RAMP_STEPS=0
export LANDING_V22_RANDOM_MASS_KG=2
export LANDING_V22_RANDOM_COM_M=0.03
export LANDING_V22_TORQUE_MIN_NM=14
export LANDING_V22_TORQUE_MAX_NM=18
export LANDING_V22_RANDOM_PD=0.1
export LANDING_V22_FRICTION_MIN=0.5
export LANDING_V22_FRICTION_MAX=1.25
export LANDING_V22_TOTAL_STEPS=200015872
export LANDING_V22_CONSTRAINTS_PER_ENV=320
export LANDING_V22_CONTACTS_PER_ENV=32
export LANDING_V22_CHECKPOINT="${PARENT_RAND:-$ROOT/modelsv22_randomized/latest.model}"
export LANDING_V22_PARENT_STEPS=200015872
grep -q SUCCESS_AT_END "$ROOT/experiments/train_landing_real_drives_mjx_v22.py" || {
    echo "Old code: the train script does not know LANDING_V22_SUCCESS_AT_END. Sync the repository first." >&2; exit 1; }
echo "48bbde43f1833859f0b6e1b998c1b04829fa7628ec947b67aca85915f7882bcf  $LANDING_V22_CHECKPOINT" | sha256sum --check --quiet
export RUN_NAME=v22_end_success_${SLURM_JOB_ID:-local}
echo "Parent: $LANDING_V22_CHECKPOINT  randomized: mass +-2 kg, com +-3 cm, torque 14-18 Nm, PD +-10 %, friction 0.5-1.25  success at end (side roll <= 15 deg), no braking reward, gamma 0.995  steps: 200M"
exec bash "$ROOT/experiments/real_drives_mjx_v22.sh" "$@"
