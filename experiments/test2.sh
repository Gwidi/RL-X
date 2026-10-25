#!/bin/bash
#SBATCH --job-name=landing_seed606_factorized
#SBATCH --output=out_and_err_landing_seed606_factorized_%j.txt
#SBATCH --error=out_and_err_landing_seed606_factorized_%j.txt
#SBATCH --partition=proxima
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=7:30:00

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
export LANDING_TRAIN_ENTRY=train_landing_seed606_factorized.py
exec bash "$SCRIPT_DIR/train_landing_seed606.sh" "$@"
