#!/bin/bash
#SBATCH --job-name=gea_gqap_rpd
#SBATCH --error=gea_gqap_rpd-%j.err
#SBATCH --output=gea_gqap_rpd-%j.log
#SBATCH --time=336:00:00
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --gpus=0

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Parallelism (processes). Keep BLAS single-threaded above.
export NUM_WORKERS=16
export RPD_NUM_WORKERS=16

# Tuning controls
export RPD_SHEET="GEA_2"
export RPD_BLOCK="adaptive"
export RPD_NUM_RUNS=5
export RPD_ITERATIONS=1000
export RPD_TIME_LIMIT=1000
# export RPD_MAX_TAGUCHI_ROWS=27

module load Python
source activate gea_gqap_env

poetry run python gea_gqap_adaptive_python/tests/RPD/run_rpd_tuning_debug.py

