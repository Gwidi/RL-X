#!/bin/bash
#SBATCH --job-name=landing_seed606
#SBATCH --output=out_and_err_landing_seed606_%j.txt
#SBATCH --error=out_and_err_landing_seed606_%j.txt
#SBATCH --partition=proxima
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=7:30:00

set -euo pipefail
if [[ -n "${RLX_PROJECT_ROOT:-}" ]]; then
    SCRIPT_DIR="$RLX_PROJECT_ROOT/experiments"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/experiments/train_landing_seed606.py" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR/experiments"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/train_landing_seed606.py" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fi
PROJECT_ROOT="$(dirname -- "$SCRIPT_DIR")"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -f /mnt/storage_6/project_data/pl0467-01/soft/miniconda3/bin/conda ]]; then
        # Conda compiler activation/deactivation hooks may read unset backup
        # variables. Run third-party hooks without nounset, then restore it.
        set +u
        CONDA_HOOK="$(/mnt/storage_6/project_data/pl0467-01/soft/miniconda3/bin/conda shell.bash hook)"
        eval "$CONDA_HOOK"
        unset CONDA_HOOK
        conda activate /mnt/storage_6/project_data/pl0467-01/conda_filip/conda_envs/rlx_filip
        set -u
    fi
    PYTHON_BIN=python
fi
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" "$SCRIPT_DIR/train_landing_seed606.py" "$@"
