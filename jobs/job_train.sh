#!/bin/bash
#SBATCH --job-name=masc_train
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

# Single-environment training (masc-train, or masc-train-rppo with ALGO=rppo), one SLURM array task per environment
# (each task is a separate job with its own 26 h limit, they run in parallel):
#   task 0: BatSupplyEnv-v1 (Bat10)   task 1: BatSupplyEnv-v2 (Bat7)    task 2: BatSupplyEnv-v3 (Bat3)
#   task 3: StoSupplyEnv-v3 (Sto1)    task 4: StoSupplyEnv-v2 (Sto01)   task 5: StoSupplyEnv-v1 (Sto0)
# Everything is written to the directory the job is submitted from (logs <dir>_logs/<env>/, figures PPO_<env>.png,
# csv_results/, SLURM output). A run that does not converge exits with code 1 (SLURM state FAILED), the reason is in
# <dir>_logs/<env>/<time>/training_status.json.
#
# Submit from an output directory, e.g.
#   mkdir -p runs && cd runs
#   sbatch ../jobs/job_train.sh                                   all six environments, PPO, team reward
#   sbatch --array=0-2 ../jobs/job_train.sh                       only the batch environments
#   ALGO=rppo sbatch ../jobs/job_train.sh                         recurrent PPO (LSTM policies)
#   REWARD_MODE=individual sbatch ../jobs/job_train.sh            individual rewards, logs in tu_individual_logs/
#   sbatch ../jobs/job_train.sh --seeds 0 1 2 3 4 --lr 0.0001 --no-early-stop --total-timesteps 2000000 \
#       --num-env-runners 2 --dir tu_seeds                        5 seeds, mean +- std figures
# Extra arguments go to masc-train after the defaults below, so they can override them.

ENVS=(BatSupplyEnv-v1 BatSupplyEnv-v2 BatSupplyEnv-v3 StoSupplyEnv-v3 StoSupplyEnv-v2 StoSupplyEnv-v1)
ALGO=${ALGO:-ppo}                    # ppo: masc-train, rppo: masc-train-rppo
NUM_ENV_RUNNERS=4                    # sampling workers per trial (1 + 4 CPUs per trial; 16 CPUs -> up to 3 trials)
TIME_BUDGET_HOURS=25                 # below --time=26:00:00: leaves time for the final checkpoint and the figure
REWARD_MODE=${REWARD_MODE:-team}     # team / individual / mixed
# no stockout termination: every episode runs the T = 100 days. With the termination after 12 stockout days the agents
# learned to end the episodes early instead of serving the demand (smoke tests of 2026-10-12)
LOG_PREFIX=tu; [ "$ALGO" = rppo ] && LOG_PREFIX=tu1
DEFAULT_ARGS="--reward-mode $REWARD_MODE --no-stockout-termination --dir ${LOG_PREFIX}_$REWARD_MODE"

set -eo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load conda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-MASC}"   # conda environment with masc installed (pip install -e <repo>, see README)

export PYTHONUNBUFFERED=1             # print the training progress to the .out file immediately

case "$ALGO" in
    ppo) CMD=masc-train ;;
    rppo) CMD=masc-train-rppo ;;
    *) echo "unknown ALGO=$ALGO (ppo or rppo)"; exit 2 ;;
esac
command -v "$CMD" > /dev/null || { echo "$CMD not found: pip install -e <repo> in the conda environment"; exit 2; }

ENV=${ENVS[${SLURM_ARRAY_TASK_ID:-0}]}
echo "job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname), ${SLURM_CPUS_PER_TASK:-?} cpus, $CMD --env $ENV, dir $(pwd)"
python -c "import masc, ray, torch; print('masc', masc.__version__, 'ray', ray.__version__, 'torch', torch.__version__)"

# many jobs can start on the same node at the same time: start at a random delay, and run again once if python fails
# within 5 minutes (the local Ray instance did not start; a real training run takes much longer)
sleep $((RANDOM % 60))
for attempt in 1 2; do
    start=$(date +%s)
    status=0
    "$CMD" --env "$ENV" --num-env-runners "$NUM_ENV_RUNNERS" --time-budget-hours "$TIME_BUDGET_HOURS" $DEFAULT_ARGS "$@" || status=$?
    if [ "$status" = 0 ] || [ $(( $(date +%s) - start )) -gt 300 ] || [ "$attempt" = 2 ]; then
        exit "$status"
    fi
    echo "$CMD failed after $(( $(date +%s) - start )) s (exit $status), retrying in 60 s"
    sleep 60
done
