#!/bin/bash
#SBATCH --job-name=job_batsto_sto1-bat10
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=26:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ww1a23@soton.ac.uk

# Continual learning task sto1-bat10 (sto std 1 -> bat batch 10), repeated --rounds times (default 5), see continual_batsto.py.
# Writes the training curve to TensorBoard and plots PPOcon_sto1-bat10_r<rounds>_lr<lr>_<date-time>.png in this directory.
# Submit from this directory:
#   sbatch job_batsto_sto1-bat10.sh
#   sbatch job_batsto_sto1-bat10.sh --lr 0.03 0.003 0.0003 --num-env-runners 8 --rounds 1   extra arguments go to the script

set -eo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

module load conda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-MASC}"   # conda environment, see requirements.txt; other name: CONDA_ENV=xxx sbatch ...

export PYTHONUNBUFFERED=1             # print the training progress to the .out file immediately

echo "job ${SLURM_JOB_ID:-local} on $(hostname), ${SLURM_CPUS_PER_TASK:-?} cpus, continual task sto1-bat10, dir $(pwd)"
python continual_batsto.py --task sto1-bat10 "$@"
