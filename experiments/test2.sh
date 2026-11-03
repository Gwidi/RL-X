#!/bin/bash
#SBATCH --job-name=landing_braking_reference_v11
#SBATCH --output=out_and_err_landing_braking_reference_v11_%j.txt
#SBATCH --error=out_and_err_landing_braking_reference_v11_%j.txt
#SBATCH --partition=proxima
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

# Keep the established entry point and one shared training profile.
set -euo pipefail
if [[ -n "${RLX_PROJECT_ROOT:-}" ]]; then
    SCRIPT_DIR="$RLX_PROJECT_ROOT/experiments"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/experiments/train_landing_seed606.sh" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR/experiments"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/train_landing_seed606.sh" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fi
export LANDING_TRAIN_ENTRY="${LANDING_TRAIN_ENTRY:-train_landing_braking_v11.py}"
# Also forward a checkpoint assigned inside this script without an explicit export.
if [[ -n "${LANDING_V10_CHECKPOINT+x}" ]]; then
    export LANDING_V10_CHECKPOINT
fi
if [[ -n "${LANDING_BRAKING_CHECKPOINT+x}" ]]; then
    export LANDING_BRAKING_CHECKPOINT
fi
exec bash "$SCRIPT_DIR/train_landing_seed606.sh" "$@"
