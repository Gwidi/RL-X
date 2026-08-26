#!/bin/bash
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-23:59:59
#SBATCH --job-name=rlx_spine_bunker
#SBATCH --output=/home/bohlinger/rlx_spine_codesign_cluster/runs/slurm_%j.log
#SBATCH --error=/home/bohlinger/rlx_spine_codesign_cluster/runs/slurm_%j.log

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: sbatch [resource overrides] ias_spine_bunker.sbatch <run.args> <run-label>" >&2
    exit 2
fi

SPINE_ROOT=/home/bohlinger/rlx_spine_codesign_cluster
SPINE_COMMIT=abfbf8a171d8567b8d4c243308ef834d3e62ebf8
RLX_SOURCE="$SPINE_ROOT/software/RL-X-$SPINE_COMMIT"
PYTHON_BIN="$SPINE_ROOT/env/bin/python"
RUN_ARGS_FILE="$1"
RUN_LABEL="$2"
RUN_DIR="$SPINE_ROOT/runs/${RUN_LABEL}_${SLURM_JOB_ID}"
JAX_CACHE_DIR="$SPINE_ROOT/jax_cache/${RUN_LABEL}_${SLURM_JOB_ID}"

test "$(git -C "$RLX_SOURCE" rev-parse HEAD)" = "$SPINE_COMMIT"
test -z "$(git -C "$RLX_SOURCE" status --porcelain)"
test -x "$PYTHON_BIN"
test -f "$RUN_ARGS_FILE"

mapfile -t RUN_ARGS < "$RUN_ARGS_FILE"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$RLX_SOURCE"
export JAX_COMPILATION_CACHE_DIR="$JAX_CACHE_DIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_DIR="$RUN_DIR/wandb"
mkdir -p "$JAX_CACHE_DIR" "$WANDB_DIR"

if grep -q '^--runner.track_wandb=True$' "$RUN_ARGS_FILE"; then
    source <(grep -m1 'export WANDB_API_KEY' /home/bohlinger/.bashrc)
fi

echo "SPINE_BUNKER_LAUNCH commit=$SPINE_COMMIT job=$SLURM_JOB_ID label=$RUN_LABEL args=$RUN_ARGS_FILE"
"$PYTHON_BIN" -c 'import jax, flax, mujoco, rl_x; devices = jax.devices(); print("jax=", jax.__version__, "flax=", flax.__version__, "mujoco=", mujoco.__version__); print("devices=", devices); print("rl_x=", rl_x.__file__); assert any(device.platform == "gpu" for device in devices)'

cd "$RUN_DIR"
"$PYTHON_BIN" "$RLX_SOURCE/experiments/experiment.py" \
    "${RUN_ARGS[@]}" \
    --environment.seed=0 \
    --runner.mode=train \
    --runner.jax_cache_dir="$JAX_CACHE_DIR" \
    --runner.run_name="${RUN_LABEL}_seed0_${SLURM_JOB_ID}"

echo "SPINE_BUNKER_FINISHED_OK job=$SLURM_JOB_ID label=$RUN_LABEL"
