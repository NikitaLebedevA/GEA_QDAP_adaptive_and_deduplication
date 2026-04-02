# RPD + Taguchi: полный перезапуск и перенос параметров в тест

## Что есть в репозитории

| Файл | Назначение |
|------|------------|
| `taguchi_config_RPD.json` | Уровни факторов и таблица Taguchi по листам Excel (сейчас листы **GEA_1**, **GEA_2**, **GEA_3**; блоки **base** и **adaptive**). |
| `run_rpd_tuning_debug.py` | Один прогон RPD для **одного** листа (`RPD_SHEET`) и блока (`RPD_BLOCK`, по умолчанию `adaptive`). |
| `run_rpd_tune_all_sheets.py` | Последовательный запуск тюнинга **по всем листам** из `taguchi_config_RPD.json` (или список через `RPD_SHEETS`). |
| `export_recommended_to_test_config.py` | Сбор поля `recommended.config_kwargs` из `results/rpd_tuning_*_*.json` и запись в `algorithm_by_model` в `test_config.json`. |
| `results/` | Сюда пишутся JSON с сырыми cost, RPD, MEPFM и **`recommended`** (готовые коэффициенты для кода). |

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
- `RPD_MAX_TAGUCHI_ROWS=0` — все строки таблицы (`0` = все, см. `run_rpd_tuning_debug.py`).

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

Для **одного** листа (как раньше):

```bash
export RPD_SHEET=GEA_2
export RPD_BLOCK=adaptive
python3 run_rpd_tuning_debug.py
```

Артефакты: `results/rpd_tuning_<SHEET>_<BLOCK>.json` (например `rpd_tuning_GEA_2_adaptive.json`).

## Шаг 2: перенос лучших коэффициентов в `test_config.json`

Скрипт читает все подходящие файлы в `results/`, извлекает `recommended.config_kwargs` и обновляет секцию **`algorithm_by_model`** (имя ключа модели = имя листа: `GEA_1`, `GEA_2`, `GEA_3`).

```bash
python3 export_recommended_to_test_config.py \
  --results-dir results \
  --update ../test_20260402/test_config.json
```

Просмотр без записи:

```bash
python3 export_recommended_to_test_config.py --results-dir results --dry-run
```

Общие поля (`time_limit`, `adaptive_epsilon`, `mask_mutation_index`) задавайте в **`algorithm`** в том же JSON вручную или оставьте как есть.

## Шаг 3: сравнительный тест

```bash
cd ../test_20260402
python3 run_comparison_test.py
```

## Примечание про блок `base` vs `adaptive`

- **`adaptive`** — в Taguchi входят `alpha`, `lambda_min`, `lambda_max`; подходит для адаптивного GEA в `run_adaptive_ga`.
- **`base`** — без адаптивных факторов; используйте, если сравниваете только неадаптивные настройки.

Для финального `test_config` с четырьмя типами алгоритмов обычно нужен прогон **`adaptive`** по листам и перенос в `algorithm_by_model`; при необходимости можно дополнительно прогнать `base` и объединить вручную.
