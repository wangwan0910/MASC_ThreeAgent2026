#!/bin/bash
#SBATCH --job-name=job_bat_continual
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=26:00:00
#SBATCH --array=0-1
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ww1a23@soton.ac.uk

# Continual learning, one SLURM array task per continual learning task (see continual_bat.py):
#   task 0: 3-7-10   batch size 3 -> 7 -> 10
#   task 1: 10-7-3   batch size 10 -> 7 -> 3
# Each task repeats its order --rounds times (default 3), writes the training curve to TensorBoard and plots
# PPOcon_<task>_r<rounds>_lr<lr>_<date-time>.png in continual_learning/ at the end.
#
# Submit from this directory (continual_learning/):
#   sbatch job_bat_continual.sh                                                    both tasks
#   sbatch --array=0 job_bat_continual.sh                                          only task 0
#   sbatch job_bat_continual.sh --lr 0.03 0.003 0.0003 --num-env-runners 8 --rounds 1   extra arguments go to the script

TASKS=(3-7-10 10-7-3)

set -eo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

module load conda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-MASC}"   # conda environment, see requirements.txt; other name: CONDA_ENV=xxx sbatch ...

export PYTHONUNBUFFERED=1             # print the training progress to the .out file immediately

TASK=${TASKS[${SLURM_ARRAY_TASK_ID:-0}]}
echo "job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname), ${SLURM_CPUS_PER_TASK:-?} cpus, continual task $TASK, dir $(pwd)"
# works when submitted from continual_learning/ (where this script is) or from its parent directory
SCRIPT=continual_bat.py
[ -f "$SCRIPT" ] || SCRIPT=continual_learning/continual_bat.py
python "$SCRIPT" --task "$TASK" "$@"
