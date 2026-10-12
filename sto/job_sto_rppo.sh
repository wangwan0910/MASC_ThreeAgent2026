#!/bin/bash
#SBATCH --job-name=job_sto_rppo
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

# RPPO training on the three environments of this directory, one SLURM array task per environment
# (each task is a separate job with its own 26 h limit, the three run in parallel):
#   task 0: StoSupplyEnv-v3  (std 1)      -> RPPO_Sto1.png
#   task 1: StoSupplyEnv-v2  (std 0.1)    -> RPPO_Sto01.png
#   task 2: StoSupplyEnv-v1  (std 0)      -> RPPO_Sto0.png
# The reward curve is checked during the training (see main_ppo.py): a run stops when it has converged healthily,
# or early when it is unhealthy (diverged / collapsed / stalled / no learning), or at TIME_BUDGET_HOURS.
# Only a converged run plots RPPO_<env>.png in this directory (TensorBoard logs in tu1_logs/<env>/); otherwise the
# reason is in tu1_logs/<env>/<time>/training_status.json and the task exits with code 1 (SLURM state FAILED).
#
# Submit from this directory:
#   sbatch job_sto_rppo.sh                                  all three environments
#   sbatch --array=0 job_sto_rppo.sh                        only task 0
#   sbatch job_sto_rppo.sh --lr 0.001 0.0003 0.0001          extra arguments go to main_rppo.py (here: lr grid search,
#                                                         the trials run in parallel inside each task)

ENVS=(StoSupplyEnv-v3 StoSupplyEnv-v2 StoSupplyEnv-v1)
NUM_ENV_RUNNERS=4        # sampling workers per trial (1 + 4 CPUs per trial; 16 CPUs -> up to 3 lr trials at once)
TIME_BUDGET_HOURS=25     # below --time=26:00:00: leaves time for the final checkpoint and the figure
REWARD_MODE=${REWARD_MODE:-team}   # team / individual / mixed, e.g. REWARD_MODE=individual sbatch job_sto_rppo.sh
# no stockout termination: every episode runs the T = 100 days. With the termination after 12 stockout days the agents
# learned to end the episodes early instead of serving the demand (smoke tests of 2026-10-12)
ENV_ARGS="--reward-mode $REWARD_MODE --no-stockout-termination --dir tu1_$REWARD_MODE"   # logs in tu1_<mode>_logs/

set -eo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

module load conda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-MASC}"   # conda environment, see requirements.txt; other name: CONDA_ENV=xxx sbatch ...

export PYTHONUNBUFFERED=1             # print the training progress to the .out file immediately

ENV=${ENVS[${SLURM_ARRAY_TASK_ID:-0}]}
echo "job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname), ${SLURM_CPUS_PER_TASK:-?} cpus, env $ENV, dir $(pwd)"
python -c "import ray, torch, gymnasium, tensorboard; print('ray', ray.__version__, 'torch', torch.__version__)"
python main_rppo.py --env "$ENV" --num-env-runners "$NUM_ENV_RUNNERS" --time-budget-hours "$TIME_BUDGET_HOURS" $ENV_ARGS "$@"
