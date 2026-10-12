#!/bin/bash
#SBATCH --job-name=job_bat_ppo
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=26:00:00
#SBATCH --array=0-2
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ww1a23@soton.ac.uk

# PPO training on the three environments of this directory, one SLURM array task per environment
# (each task is a separate job with its own 26 h limit, the three run in parallel):
#   task 0: BatSupplyEnv-v1  (batch 10)   -> PPO_Bat10.png
#   task 1: BatSupplyEnv-v2  (batch 7)    -> PPO_Bat7.png
#   task 2: BatSupplyEnv-v3  (batch 3)    -> PPO_Bat3.png
# The reward curve is checked during the training (see main_ppo.py): a run stops when it has converged healthily,
# or early when it is unhealthy (diverged / collapsed / stalled / no learning), or at TIME_BUDGET_HOURS.
# Only a converged run plots PPO_<env>.png in this directory (TensorBoard logs in tu_logs/<env>/); otherwise the
# reason is in tu_logs/<env>/<time>/training_status.json and the task exits with code 1 (SLURM state FAILED).
#
# Submit from this directory:
#   sbatch job_bat_ppo.sh                                  all three environments
#   sbatch --array=0 job_bat_ppo.sh                        only task 0
#   sbatch job_bat_ppo.sh --lr 0.03 0.003 0.0003           extra arguments go to main_ppo.py (here: lr grid search,
#                                                         the trials run in parallel inside each task)

ENVS=(BatSupplyEnv-v1 BatSupplyEnv-v2 BatSupplyEnv-v3)
NUM_ENV_RUNNERS=4        # sampling workers per trial (1 + 4 CPUs per trial; 16 CPUs -> up to 3 lr trials at once)
TIME_BUDGET_HOURS=25     # below --time=26:00:00: leaves time for the final checkpoint and the figure

set -eo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

module load conda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-MASC}"   # conda environment, see requirements.txt; other name: CONDA_ENV=xxx sbatch ...

export PYTHONUNBUFFERED=1             # print the training progress to the .out file immediately

ENV=${ENVS[${SLURM_ARRAY_TASK_ID:-0}]}
echo "job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname), ${SLURM_CPUS_PER_TASK:-?} cpus, env $ENV, dir $(pwd)"
python -c "import ray, torch, gymnasium, tensorboard; print('ray', ray.__version__, 'torch', torch.__version__)"
python main_ppo.py --env "$ENV" --num-env-runners "$NUM_ENV_RUNNERS" --time-budget-hours "$TIME_BUDGET_HOURS" "$@"
