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

# Пять архитектур как в run_comparison_test: GA, GEA_1, GEA_2, GEA_3, GEA
export RPD_EVAL_MODELS=GA,GEA_1,GEA_2,GEA_3,GEA

# Типы прогона для усреднения отклика перед MEPFM (нагрузка × число токенов):
#   adaptive | non_adaptive | adaptive_wo_duplicates | non_adaptive_wo_duplicates
# По умолчанию только adaptive (как классический RPD по адаптивному GEA).
# Раскомментируйте строку ниже, чтобы учитывать все четыре типа (сильно дольше):
# export RPD_EVAL_TYPES=adaptive,non_adaptive,adaptive_wo_duplicates,non_adaptive_wo_duplicates
export RPD_EVAL_TYPES=adaptive

export RPD_BLOCK=adaptive
export RPD_NUM_RUNS=5
export RPD_ITERATIONS=1000
export RPD_TIME_LIMIT=1000

module load Python
source activate gea_gqap_env

# Все листы из taguchi_config_RPD.json (GEA_1, GEA_2, GEA_3) → отдельные JSON в tests/RPD/results/
poetry run python gea_gqap_adaptive_python/tests/RPD/run_rpd_tune_all_sheets.py
