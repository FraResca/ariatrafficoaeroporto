#!/bin/bash
#SBATCH --job-name=aira_paper_figs
#SBATCH --nodes=1
#SBATCH --error=aira_paper_figs.err
#SBATCH --output=aira_paper_figs.out
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

cd "${SLURM_SUBMIT_DIR:-$PWD}"
ENV_PREFIX="$(pwd)/.conda-env"

module load miniconda3/
eval "$(/hpc/apps/miniconda3/24.4.0/bin/conda shell.bash hook)"
conda activate "$ENV_PREFIX"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-cache"

mkdir -p "$MPLCONFIGDIR"

echo "Running from: $(pwd)"
"$CONDA_PREFIX/bin/python" prepare_paper_figures.py
