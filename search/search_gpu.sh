#!/bin/bash

#SBATCH --job-name=optimize_experiment
#SBATCH --output=log/out_and_err_gpu.txt
#SBATCH --error=log/out_and_err_gpu.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4000
#SBATCH --time=3:30:00

eval "$(/mnt/storage_6/project_data/pl0467-01/soft/miniconda3/bin/conda shell.bash hook)"
conda activate /mnt/storage_6/project_data/pl0467-01/conda_filip/conda_envs/rlx_filip


export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python3 main_gpu.py \
    --compare-spine \
    --heights 1.21 1.22 1.23 1.24 1.25 1.26 1.27 1.28 1.29 \
    --trials 15000 \
    --optimize-nominal-position \
    --initial-trials 64 \
    --batch-size 1024 \
    --seed 42 \
    --no-viewer \
    --candidate-pool 8192
