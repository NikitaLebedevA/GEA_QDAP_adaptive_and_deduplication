# RPD + Taguchi: полный перезапуск и перенос параметров в тест

## Что есть в репозитории

| Файл | Назначение |
|------|------------|
| `taguchi_config_RPD.json` | Уровни факторов и таблица Taguchi по листам (блоки **base** и **adaptive**). |
| `run_rpd_tune_all_sheets.py` | **Единственная точка входа:** все листы × `RPD_BLOCKS` или один лист через `RPD_SHEETS` / `RPD_BLOCKS`. Внутри — `run_single_rpd_tuning()` по `RPD_SHEET` + `RPD_BLOCK`. Лог: `results/rpd_tune_all_sheets.log` (`RPD_MAIN_LOG`, `RPD_MAIN_LOG_FILE`). |
| `results/` | JSON `rpd_tuning_<лист>_<блок>.json` (cost, RPD, MEPFM, **`recommended_by_model`**). |

Отдельных листов **GA** и **полного GEA** в текущем `taguchi_config_RPD.json` нет. Их нужно добавить из Excel (и перегенерировать JSON через `excel_taguchi_parser.py`, если пользуетесь им) или задать коэффициенты для `GA` / `GEA` вручную в `test_config.json` после экспорта.

## Переменные окружения (полный прогон на кластере)

Имеет смысл выставить **реалистичные** итерации и время, иначе оптимизация будет по «коротким» прогонам.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# Один лист Taguchi = много строк × датасеты × num_runs — заложите запас по walltime
export RPD_ITERATIONS=1000
export RPD_TIME_LIMIT=1000
export RPD_NUM_RUNS=30
export RPD_NUM_WORKERS=16
export RPD_BLOCK=adaptive
```

### Пять архитектур и типы алгоритма

По умолчанию для **каждой** строки Taguchi считается средний `best_cost` по датасетам и прогонам для моделей из `RPD_EVAL_MODELS` (как в сравнительном тесте):

- `GA`, `GEA_1`, `GEA_2`, `GEA_3`, `GEA` — значения по умолчанию: все пять.

Для каждой модели отклик **усредняется** по токенам из `RPD_EVAL_TYPES` (затем отдельный MEPFM и свой `recommended`):

- `adaptive`, `non_adaptive`, `adaptive_wo_duplicates`, `non_adaptive_wo_duplicates`  
- По умолчанию: только `adaptive`. Четыре токена увеличивают число задач в **4 раза**.

Итог в JSON: `recommended_by_model` (по одному набору коэффициентов на модель), плюс поля `recommended` / `response_mean_best_cost` для **листа** (`primary_model_for_legacy_fields` = имя листа, если оно есть в `RPD_EVAL_MODELS`).

Кластерный запуск всех листов: корневой **`run_test_RPD.sbatch`** (вызывает `run_rpd_tune_all_sheets.py`).

Опционально:

- `RPD_TAGUCHI_CONFIG` — путь к своему `taguchi_config_RPD.json`.
- `RPD_MAX_TAGUCHI_ROWS=0` — все строки таблицы (`0` = все, см. `run_rpd_tune_all_sheets.py`).
- `RPD_MAIN_LOG=0` — не писать объединённый лог при `run_rpd_tune_all_sheets.py`.
- `RPD_MAIN_LOG_FILE=/path/to.log` — путь лога (по умолчанию `results/rpd_tune_all_sheets.log`).

## Шаг 1: тюнинг по всем листам

Из каталога `gea_gqap_adaptive_python/tests/RPD`:

```bash
cd gea_gqap_adaptive_python/tests/RPD
python3 run_rpd_tune_all_sheets.py
```

Только выбранные листы:

```bash
export RPD_SHEETS=GEA_1,GEA_2,GEA_3
python3 run_rpd_tune_all_sheets.py
```

Один лист и один блок:

```bash
export RPD_SHEETS=GEA_2
export RPD_BLOCKS=adaptive
python3 run_rpd_tune_all_sheets.py
```

Артефакты: `results/rpd_tuning_<SHEET>_<BLOCK>.json` (например `rpd_tuning_GEA_2_adaptive.json`).

## Шаг 2: перенос в `test_config.json`

Вручную перенесите **`recommended_by_model[...].config_kwargs`** из `results/rpd_tuning_*_*.json` в **`algorithm_by_variant`** (ключи `MODEL|тип`) и при необходимости в **`algorithm`** общие поля.

## Шаг 3: сравнительный тест

```bash
cd ../test_20260402
python3 run_comparison_test.py
```

## Примечание про блок `base` vs `adaptive`

- **`adaptive`** — в Taguchi входят `alpha`, `lambda_min`, `lambda_max`; подходит для адаптивного GEA в `run_adaptive_ga`.
- **`base`** — без адаптивных факторов; используйте, если сравниваете только неадаптивные настройки.

Для четырёх типов алгоритма обычно нужны пары JSON `*_adaptive.json` / `*_base.json` по каждому листу и ручная сборка `algorithm_by_variant`.
