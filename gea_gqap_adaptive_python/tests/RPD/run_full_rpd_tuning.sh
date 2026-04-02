#!/usr/bin/env bash
# Пример полного перезапуска RPD по всем листам из taguchi_config_RPD.json (GEA_1, GEA_2, GEA_3).
# Запускать из каталога gea_gqap_adaptive_python/tests/RPD или передать путь.
set -euo pipefail

cd "$(dirname "$0")"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# Полный прогон (переопределите под квоту кластера)
export RPD_ITERATIONS="${RPD_ITERATIONS:-1000}"
export RPD_TIME_LIMIT="${RPD_TIME_LIMIT:-1000}"
export RPD_NUM_RUNS="${RPD_NUM_RUNS:-30}"
export RPD_NUM_WORKERS="${RPD_NUM_WORKERS:-16}"
export RPD_BLOCK="${RPD_BLOCK:-adaptive}"
export RPD_EVAL_MODELS="${RPD_EVAL_MODELS:-GA,GEA_1,GEA_2,GEA_3,GEA}"
export RPD_EVAL_TYPES="${RPD_EVAL_TYPES:-adaptive}"

echo "RPD_ITERATIONS=$RPD_ITERATIONS RPD_TIME_LIMIT=$RPD_TIME_LIMIT RPD_NUM_RUNS=$RPD_NUM_RUNS RPD_BLOCK=$RPD_BLOCK"

python3 run_rpd_tune_all_sheets.py

echo "Готово. Результаты: $(pwd)/results/rpd_tuning_*_${RPD_BLOCK}.json"
echo "Далее: python3 export_recommended_to_test_config.py --results-dir results --update ../test_20260402/test_config.json"
