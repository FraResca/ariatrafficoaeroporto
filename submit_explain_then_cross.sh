#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

explain_job_id="$(sbatch --parsable explain_pollutants.slurm)"
echo "Submitted explain_pollutants.slurm as job ${explain_job_id}"

cross_job_id="$(sbatch --parsable --dependency=afterok:${explain_job_id} cross_pollutant_analysis.slurm)"
echo "Submitted cross_pollutant_analysis.slurm as job ${cross_job_id} (afterok:${explain_job_id})"
