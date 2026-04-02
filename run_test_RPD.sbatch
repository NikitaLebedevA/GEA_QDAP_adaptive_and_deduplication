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

export NUM_WORKERS=16
export RPD_NUM_WORKERS=16

export RPD_EVAL_MODELS=GA,GEA_1,GEA_2,GEA_3,GEA
# Все 4 типа → на строку Taguchi 5×4 прогона модель×тип (дольше, чем только adaptive)
export RPD_EVAL_TYPES=adaptive,non_adaptive,adaptive_wo_duplicates,non_adaptive_wo_duplicates
export RPD_BLOCKS=adaptive,base
export RPD_NUM_RUNS=5
export RPD_ITERATIONS=1000
export RPD_TIME_LIMIT=1000

module load Python
source activate gea_gqap_env

poetry run python gea_gqap_adaptive_python/tests/RPD/run_rpd_tune_all_sheets.py
