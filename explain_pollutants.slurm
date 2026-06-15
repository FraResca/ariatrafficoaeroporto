#!/bin/bash
#SBATCH --job-name=aira_explain_full
#SBATCH --nodes=1
#SBATCH --error=aira_explain_full.err
#SBATCH --output=aira_explain_full.out
#SBATCH --partition=compute
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH --time=24:00:00

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

echo "Running from: $(pwd)"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-not-set}"

"$CONDA_PREFIX/bin/python" explain_pollutants_by_feature_groups.py \
  --workers 0 \
  --output-dir Analysis/slurm_full_explain
