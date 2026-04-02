#!/usr/bin/env python3
"""
Сводная таблица по 4 типам алгоритмов на каждую модель:
  adaptive, non_adaptive, adaptive_wo_duplicates, non_adaptive_wo_duplicates.
Параметры теста: 1000 итераций, популяция 350, лимит времени 1000 с. Датасеты: c + T.
"""

import json
from pathlib import Path
from typing import Dict, List, Any

TEST_DIR = Path(__file__).parent


def load_config() -> Dict[str, Any]:
    config_path = TEST_DIR / "test_config.json"
    if not config_path.exists():
        return {"model_variants": ["GA", "GEA_1", "GEA_2", "GEA_3", "GEA"], "algorithm_types": []}
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("algorithm_types", [
        "adaptive", "non_adaptive", "adaptive_wo_duplicates", "non_adaptive_wo_duplicates"
    ])
    return cfg


def load_results(results_dir: Path) -> List[Dict[str, Any]]:
    results = []
    for json_file in sorted(results_dir.glob("*_results.json")):
        if json_file.name == "summary_all_datasets.json":
            continue
        try:
            with json_file.open("r", encoding="utf-8") as f:
                results.append(json.load(f))
        except Exception as e:
            print(f"Ошибка при загрузке {json_file.name}: {e}")
    return results


def create_excel_report(results_dir: Path, output_file: Path):
    import pandas as pd
    cfg = load_config()
    model_variants = list(cfg.get("model_variants", {}).keys()) or ["GA", "GEA_1", "GEA_2", "GEA_3", "GEA"]
    type_keys = cfg.get("algorithm_types") or [
        "adaptive", "non_adaptive", "adaptive_wo_duplicates", "non_adaptive_wo_duplicates"
    ]
    results = load_results(results_dir)
    if not results:
        print("Нет результатов для обработки!")
        return

    best_cost_rows = []
    elapsed_time_rows = []
    paper_rows = []

    for result in results:
        dataset = result["dataset"]
        models = result.get("models", {})
        row_bc = {"Dataset": dataset}
        row_et = {"Dataset": dataset}
        row_paper = {"Dataset": dataset}

        for mk in model_variants:
            m = models.get(mk, {})
            for tk in type_keys:
                t = m.get(tk, {})
                bc = t.get("best_cost", {})
                et = t.get("elapsed_time", {})
                prefix = f"{mk}_{tk}_"
                row_bc[f"{prefix}min"] = bc.get("min", 0)
                row_bc[f"{prefix}max"] = bc.get("max", 0)
                row_bc[f"{prefix}std"] = bc.get("std", 0)
                row_bc[f"{prefix}median"] = bc.get("median", 0)
                row_bc[f"{prefix}mean"] = bc.get("mean", 0)
                row_et[f"{prefix}min"] = et.get("min", 0)
                row_et[f"{prefix}max"] = et.get("max", 0)
                row_et[f"{prefix}std"] = et.get("std", 0)
                row_et[f"{prefix}median"] = et.get("median", 0)
                row_et[f"{prefix}mean"] = et.get("mean", 0)
                row_paper[f"{prefix}Best"] = bc.get("min", 0)
                row_paper[f"{prefix}Mean"] = bc.get("mean", 0)
                row_paper[f"{prefix}Worst"] = bc.get("max", 0)
                row_paper[f"{prefix}Std"] = bc.get("std", 0)
                row_paper[f"{prefix}TimeMean"] = et.get("mean", 0)

        best_cost_rows.append(row_bc)
        elapsed_time_rows.append(row_et)
        paper_rows.append(row_paper)

    df_bc = pd.DataFrame(best_cost_rows)
    df_et = pd.DataFrame(elapsed_time_rows)
    df_paper = pd.DataFrame(paper_rows)
    info_rows = [
        {"Параметр": "Типы алгоритмов", "Значение": ", ".join(type_keys)},
        {"Параметр": "Модели", "Значение": ", ".join(model_variants)},
        {"Параметр": "Датасетов", "Значение": str(len(results))},
    ]
    df_info = pd.DataFrame(info_rows)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_info.to_excel(writer, sheet_name="Сводка", index=False)
        df_bc.to_excel(writer, sheet_name="best_cost", index=False)
        df_et.to_excel(writer, sheet_name="elapsed_time", index=False)
        df_paper.to_excel(writer, sheet_name="paper_table", index=False)

    print(f"Excel создан: {output_file}")


if __name__ == "__main__":
    results_dir = TEST_DIR / "results"
    output_file = TEST_DIR / "comparison_results.xlsx"
    create_excel_report(results_dir, output_file)
