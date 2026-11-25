#!/bin/bash
#SBATCH --job-name=v22_next_ablation_track
#SBATCH --output=out_and_err_v22_next_ablation_track_%j.txt
#SBATCH --error=out_and_err_v22_next_ablation_track_%j.txt
#SBATCH --partition=proxima
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Ablation of job 9743266: identical (same parent, PPO, 2048 envs, no slip penalty, up to 5 m/s) except
# the braking reference is "track". It separates the effect of the reference from the effect of the new PPO settings.
# Submit from the repository root: sbatch experiments/v22_next_ablation_track.sh --runner.track_wandb=False
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
export LANDING_V22_REFERENCE=track
export LANDING_V22_CHECKPOINT="$ROOT/runs/soft_landing/landing_real_drives_mjx_continue_v22/landing_real_drives_mjx_continue_v22_9738171/models/latest.model"
export LANDING_V22_PARENT_STEPS=39976960
export RUN_NAME=v22_2048_track_${SLURM_JOB_ID:-local}
echo "Parent: $LANDING_V22_CHECKPOINT  reference: $LANDING_V22_REFERENCE  max speed: ${LANDING_V22_MAX_SPEED:-5}"
exec bash "$ROOT/experiments/real_drives_mjx_v22.sh" "$@"
