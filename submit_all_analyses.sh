#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

explain_job_id="$(sbatch --parsable explain_pollutants.slurm)"
echo "Submitted explain_pollutants.slurm as job ${explain_job_id}"

upwind_job_id="$(sbatch --parsable upwind_downwind.slurm)"
echo "Submitted upwind_downwind.slurm as job ${upwind_job_id}"

airport_response_job_id="$(sbatch --parsable airport_response_analysis.slurm)"
echo "Submitted airport_response_analysis.slurm as job ${airport_response_job_id}"

cross_job_id="$(sbatch --parsable --dependency=afterok:${explain_job_id}:${upwind_job_id} cross_pollutant_analysis.slurm)"
echo "Submitted cross_pollutant_analysis.slurm as job ${cross_job_id} (afterok:${explain_job_id}:${upwind_job_id})"

figures_job_id="$(sbatch --parsable --dependency=afterok:${cross_job_id} prepare_paper_figures.slurm)"
echo "Submitted prepare_paper_figures.slurm as job ${figures_job_id} (afterok:${cross_job_id})"
