#!/bin/bash
#SBATCH --job-name=v22_randomized
#SBATCH --output=out_and_err_v22_randomized_%j.txt
#SBATCH --error=out_and_err_v22_randomized_%j.txt
#SBATCH --partition=proxima
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Continuation of job 9758915 (the model tried on HB40 on 2026-11-27) with domain randomization, 200,015,872 steps.
# Real drops rolled to the right and the model, trained on one exact robot, does the same in sim after small changes.
# Per episode: trunk mass +-2 kg, total centre of mass +-3 cm in x and y, drive limit 14-18 Nm per joint, kp and kd
# +-10 % per joint. Plus a reward for a level, left-right symmetric stance after touchdown. Everything else as in 9758915
# (pose +-14 cm, tilt and spin 30 deg(/s), speeds to 7 m/s, hip penalty 3); learning rate 3e-5 -> 1e-5 over the run.
# The parent (35ecacf2...) is gitignored: copy modelsv22_tilt_const/latest.model to the cluster, or set PARENT_SYM.
# Submit from the repository root: sbatch experiments/v22_randomized.sh --runner.track_wandb=False
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
export LANDING_V22_RANDOM_MASS_KG=2
export LANDING_V22_RANDOM_COM_M=0.03
export LANDING_V22_TORQUE_MIN_NM=14
export LANDING_V22_TORQUE_MAX_NM=18
export LANDING_V22_RANDOM_PD=0.1
export LANDING_V22_POSTURE_REWARD=0.5
export LANDING_V22_TOTAL_STEPS=200015872
export LANDING_V22_CONSTRAINTS_PER_ENV=256
export LANDING_V22_CONTACTS_PER_ENV=24
export LANDING_V22_CHECKPOINT="${PARENT_SYM:-$ROOT/modelsv22_tilt_const/latest.model}"
export LANDING_V22_PARENT_STEPS=60030976
grep -q LANDING_V22_POSTURE_REWARD "$ROOT/experiments/train_landing_real_drives_mjx_v22.py" || {
    echo "Old code: the train script does not know LANDING_V22_POSTURE_REWARD. Sync the repository first." >&2; exit 1; }
echo "35ecacf20ba76f0fec30a2776948b00c0cec9b0c36202fbf769d497f58619a4d  $LANDING_V22_CHECKPOINT" | sha256sum --check --quiet
export RUN_NAME=v22_randomized_${SLURM_JOB_ID:-local}
echo "Parent: $LANDING_V22_CHECKPOINT  randomized: mass +-2 kg, com +-3 cm, torque 14-18 Nm, PD +-10 %  posture reward: 0.5  steps: 200M"
exec bash "$ROOT/experiments/real_drives_mjx_v22.sh" "$@"
