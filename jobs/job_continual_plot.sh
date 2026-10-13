#!/bin/bash
#SBATCH --job-name=masc_continual_plot
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ww1a23@soton.ac.uk

# Figure of all the seeds of a continual learning experiment (mean +- std of every phase), e.g. after
# jobs/job_continual_seeds.sh (see there):  sbatch --dependency=afterany:<array job> job_continual_plot.sh <experiment dir>
# The figure (PPOcon_<envs>_<rounds>rounds_<steps>Msteps-per-phase_lr<lr>_<n>seeds_<run>.png) goes to the directory the
# job is submitted from.

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load conda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-MASC}"
masc-plot-continual --seeds "$1"
