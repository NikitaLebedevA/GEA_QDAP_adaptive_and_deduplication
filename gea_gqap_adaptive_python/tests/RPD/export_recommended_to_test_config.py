#!/usr/bin/env python3
"""
Собирает recommended.config_kwargs из JSON результатов RPD (rpd_tuning_<sheet>_<block>.json)
и записывает их в test_config.json в algorithm_by_model.

Имя ключа модели = имя листа Taguchi (GEA_1, GEA_2, GEA_3, …). Листов GA / GEA в
текущем taguchi_config может не быть — тогда их блоки в test_config не создаются
скриптом (оставьте вручную или дополните конфиг Excel + taguchi_config_RPD.json).

Пример:
  python3 export_recommended_to_test_config.py --results-dir results --dry-run
  python3 export_recommended_to_test_config.py --results-dir results --update ../test_20260402/test_config.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Tuple


def _parse_rpd_filename(name: str) -> Tuple[str, str] | None:
    m = re.match(r"rpd_tuning_(.+)_([^.]+)\.json$", name)
    if not m:
        return None
    return m.group(1), m.group(2)


def _config_for_sheet_from_rpd_json(path: Path, sheet: str) -> Dict[str, Any]:
    """
    Берёт коэффициенты для листа Taguchi: recommended_by_model[sheet], иначе legacy recommended.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    rbm = data.get("recommended_by_model") or {}
    if sheet in rbm and isinstance(rbm[sheet], dict):
        ck = rbm[sheet].get("config_kwargs")
        if isinstance(ck, dict) and ck:
            return dict(ck)
    rec = data.get("recommended") or {}
    if isinstance(rec, dict):
        ck = rec.get("config_kwargs")
        if isinstance(ck, dict) and ck:
            return dict(ck)
    raise ValueError(f"Нет recommended_by_model[{sheet!r}] и recommended.config_kwargs в {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Экспорт RPD recommended → algorithm_by_model")
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Каталог с rpd_tuning_*.json",
    )
    ap.add_argument(
        "--update",
        type=Path,
        default=None,
        help="Путь к test_config.json для обновления (иначе только stdout)",
    )
    ap.add_argument(
        "--block",
        default="adaptive",
        help="Учитывать только файлы с таким суффиксом блока (по умолчанию adaptive)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Не писать файл, только вывести algorithm_by_model",
    )
    args = ap.parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.is_dir():
        print(f"Нет каталога: {results_dir}", file=sys.stderr)
        sys.exit(1)

    algorithm_by_model: Dict[str, Dict[str, Any]] = {}
    for path in sorted(results_dir.glob("rpd_tuning_*.json")):
        if path.name == "summary_all_datasets.json":
            continue
        parsed = _parse_rpd_filename(path.name)
        if not parsed:
            continue
        sheet, block = parsed
        if block != args.block:
            continue
        try:
            algorithm_by_model[sheet] = _config_for_sheet_from_rpd_json(path, sheet)
        except ValueError as e:
            print(f"Пропуск {path.name}: {e}", file=sys.stderr)

    if not algorithm_by_model:
        print(
            f"Не найдено ни одного rpd_tuning_*_{args.block}.json с recommended.config_kwargs в {results_dir}",
            file=sys.stderr,
        )
        sys.exit(2)

    out_obj = {"algorithm_by_model": algorithm_by_model}
    text = json.dumps(out_obj, indent=2, ensure_ascii=False)

    if args.dry_run or args.update is None:
        print(text)
        if args.update is None and not args.dry_run:
            print(
                "\n(Укажите --update путь/к/test_config.json чтобы записать.)",
                file=sys.stderr,
            )
        return

    cfg_path = args.update.resolve()
    if not cfg_path.exists():
        print(f"Файл не найден: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    full = json.loads(cfg_path.read_text(encoding="utf-8"))
    existing = full.get("algorithm_by_model")
    if not isinstance(existing, dict):
        existing = {}
    # Перезаписываем только ключи, для которых есть свежий RPD
    merged = dict(existing)
    merged.update(algorithm_by_model)
    full["algorithm_by_model"] = merged

    cfg_path.write_text(json.dumps(full, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Обновлено: {cfg_path}", file=sys.stderr)
    print(f"Ключи algorithm_by_model из RPD: {sorted(algorithm_by_model.keys())}", file=sys.stderr)


if __name__ == "__main__":
    main()
