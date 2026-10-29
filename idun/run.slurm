#!/bin/bash
# Runs main.py as a Slurm batch job on IDUN. Every argument goes on to main.py. Submit from the
# repo root, e.g. ten SAC runs with solver seeds 0 to 9:
#   mkdir -p logs
#   sbatch --array=0-9 idun/run.slurm --solver sac --n 1000 --level L0 --random-start each
# In a job array, each task gets --solver-seed <its index>, so the tasks differ only in their
# seed, and its run folder is results/runs/<level>_<solver>_seed<index>_job<array job id>.
# The account comes from SBATCH_ACCOUNT (see README), or add --account=<account> to sbatch.
# Resources below can be overridden on the sbatch command line, e.g. sbatch --time=0-04:00:00 ...
#SBATCH --job-name=ast
#SBATCH --partition=CPUQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export MPLBACKEND=Agg  # No display on the compute nodes: plots are only saved
export PYTHONUNBUFFERED=1  # Print to the log right away, not in large blocks
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  # PyTorch and numpy use at most the job's cores
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PATH="$HOME/.local/bin:$PATH"  # uv

seedArgs=()
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    seedArgs=(--solver-seed "$SLURM_ARRAY_TASK_ID")
fi

echo "Job ${SLURM_JOB_ID} on $(hostname), started $(date)"
echo "main.py --no-plots $* ${seedArgs[*]:-}"
uv run --frozen python main.py --no-plots "$@" ${seedArgs[@]+"${seedArgs[@]}"}
echo "Finished $(date)"
