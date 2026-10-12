#!/bin/bash
#SBATCH --job-name=masc_continual
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=26:00:00
#SBATCH --array=0-5
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ww1a23@soton.ac.uk

# Continual learning (masc-continual), one SLURM array task per continual learning task:
#   task 0: 0-01-1     std 0 -> 0.1 -> 1          task 1: 1-01-0       std 1 -> 0.1 -> 0
#   task 2: 3-7-10     batch 3 -> 7 -> 10         task 3: 10-7-3       batch 10 -> 7 -> 3
#   task 4: bat10-sto1 batch 10 -> std 1          task 5: sto1-bat10   std 1 -> batch 10
# Each task repeats its order --rounds times (default 3, 5 for bat10-sto1 / sto1-bat10), writes the models and the
# TensorBoard curves to PPOcon-*_models/ and plots PPOcon_<task>_r<rounds>_lr<lr>_<date-time>.png, all in the
# directory the job is submitted from.
#
# Submit from an output directory, e.g.
#   mkdir -p runs && cd runs
#   sbatch ../jobs/job_continual.sh                               all six tasks
#   sbatch --array=5 ../jobs/job_continual.sh                     only sto1-bat10
#   sbatch --array=2,3 ../jobs/job_continual.sh --lr 0.001 0.0003 --rounds 1   extra arguments go to masc-continual

TASKS=(0-01-1 1-01-0 3-7-10 10-7-3 bat10-sto1 sto1-bat10)

set -eo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load conda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-MASC}"   # conda environment with masc installed (pip install -e <repo>, see README)

export PYTHONUNBUFFERED=1             # print the training progress to the .out file immediately

command -v masc-continual > /dev/null || { echo "masc-continual not found: pip install -e <repo>"; exit 2; }

TASK=${TASKS[${SLURM_ARRAY_TASK_ID:-0}]}
echo "job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname), ${SLURM_CPUS_PER_TASK:-?} cpus, continual task $TASK, dir $(pwd)"
sleep $((RANDOM % 60))   # many jobs can start on the same node at the same time
masc-continual --task "$TASK" "$@"
