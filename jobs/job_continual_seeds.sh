#!/bin/bash
#SBATCH --job-name=masc_continual_seeds
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=26:00:00
#SBATCH --array=0-4
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ww1a23@soton.ac.uk

# One continual learning task (TASK, default 0-01-1: Sto0 -> Sto01 -> Sto1, 3 rounds = 9 phases) with several random
# seeds: one SLURM array task per seed (array index = seed), all in <models dir>/<task>/<RUN_NAME>/seed=<seed>/.
# No figure / CSV during the training; the figure of all the seeds (mean +- std) is drawn by jobs/job_continual_plot.sh
# when every seed has finished:
#   mkdir -p runs && cd runs
#   RUN=$(date +%Y%m%d-%H%M%S)
#   J=$(RUN_NAME=$RUN sbatch --parsable ../jobs/job_continual_seeds.sh --lr 0.0001 --timesteps-per-phase 1000000 \
#       --num-env-runners 4 --no-stockout-termination)
#   sbatch --dependency=afterany:${J%%_*} ../jobs/job_continual_plot.sh PPOcon-sto_models/0-01-1/$RUN
# Extra arguments go to masc-continual.

TASK=${TASK:-0-01-1}
RUN_NAME=${RUN_NAME:-job${SLURM_ARRAY_JOB_ID:-local}}
SEED=${SLURM_ARRAY_TASK_ID:-0}

set -eo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load conda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-MASC}"   # conda environment with masc installed (pip install -e <repo>, see README)

export PYTHONUNBUFFERED=1             # print the training progress to the .out file immediately

command -v masc-continual > /dev/null || { echo "masc-continual not found: pip install -e <repo>"; exit 2; }
echo "job ${SLURM_JOB_ID:-local} on $(hostname), ${SLURM_CPUS_PER_TASK:-?} cpus, continual task $TASK, seed $SEED, run $RUN_NAME"
sleep $((RANDOM % 60))   # many jobs can start on the same node at the same time
masc-continual --task "$TASK" --seed "$SEED" --run-name "$RUN_NAME" --no-figure --no-monitor "$@"
