#!/usr/bin/env python3
"""
Графики эволюции cost по итерациям для каждого датасета и каждой модели.
На каждом подграфике — 4 типа: adaptive, non_adaptive, adaptive_wo_duplicates, non_adaptive_wo_duplicates.
Ось Y: снизу меньше, сверху больше.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import sys
import numpy as np

TEST_DIR = Path(__file__).parent
LOCAL_LIB = TEST_DIR / ".venv_lib"
if LOCAL_LIB.exists():
    sys.path.insert(0, str(LOCAL_LIB))

import matplotlib.pyplot as plt

# Цвета и метки для 4 типов
TYPE_STYLE = {
    "adaptive": {"label": "Adaptive", "color": "#2ca02c", "linestyle": "-"},
    "non_adaptive": {"label": "Non-Adaptive", "color": "#d62728", "linestyle": "-"},
    "adaptive_wo_duplicates": {"label": "Adaptive w/o dup", "color": "#1f77b4", "linestyle": "--"},
    "non_adaptive_wo_duplicates": {"label": "Non-Adaptive w/o dup", "color": "#ff7f0e", "linestyle": "--"},
}


def load_config() -> Dict[str, Any]:
    config_path = TEST_DIR / "test_config.json"
    if not config_path.exists():
        return {"model_variants": ["GA", "GEA_1", "GEA_2", "GEA_3", "GEA"], "algorithm_types": list(TYPE_STYLE.keys())}
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("algorithm_types", list(TYPE_STYLE.keys()))
    return cfg


def load_dataset_results(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    datasets = {}
    for json_file in sorted(results_dir.glob("*_results.json")):
        if json_file.name == "summary_all_datasets.json":
            continue
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                datasets[data["dataset"]] = data
        except Exception as e:
            print(f"Ошибка при загрузке {json_file.name}: {e}")
    return datasets


def averaged_costs_from_runs(runs: List[Dict]) -> List[float]:
    """Усредняет best_cost по запускам для каждой итерации."""
    if not runs:
        return []
    max_iter = max(len(r.get("iterations", [])) for r in runs)
    if max_iter == 0:
        return []
    sums = [0.0] * max_iter
    cnt = [0] * max_iter
    for run in runs:
        for i, it in enumerate(run.get("iterations", [])):
            if i >= max_iter:
                break
            c = it.get("best_cost")
            if c is not None:
                sums[i] += c
                cnt[i] += 1
    return [sums[i] / max(cnt[i], 1) for i in range(max_iter)]


def _set_y_axis_order(ax, all_costs: List[float]):
    if not all_costs:
        return
    cost_min = min(all_costs)
    cost_max = max(all_costs)
    margin = (cost_max - cost_min) * 0.05 if cost_max > cost_min else (abs(cost_max) * 0.05 or 1.0)
    ax.set_ylim(cost_min - margin, cost_max + margin)


def plot_dataset_all_models(
    dataset_name: str,
    dataset_data: Dict[str, Any],
    output_dir: Path,
    model_variants: List[str],
    type_keys: List[str],
):
    models = dataset_data.get("models", {})
    if not models:
        print(f"Нет данных по моделям для {dataset_name}")
        return

    n = len(model_variants)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, model_key in enumerate(model_variants):
        if idx >= len(axes):
            break
        ax = axes[idx]
        m = models.get(model_key, {})
        all_costs_flat = []
        for tk in type_keys:
            runs = m.get(tk, {}).get("runs", [])
            costs = averaged_costs_from_runs(runs)
            style = TYPE_STYLE.get(tk, {"label": tk, "color": "gray", "linestyle": "-"})
            if costs:
                iters = list(range(1, len(costs) + 1))
                ax.plot(iters, costs, label=style["label"], linewidth=1.5, color=style["color"], linestyle=style["linestyle"])
                all_costs_flat.extend(costs)

        ax.set_title(model_key, fontsize=11, fontweight="bold")
        ax.set_xlabel("Iteration", fontsize=9)
        ax.set_ylabel("Average Best Cost", fontsize=9)
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3)
        _set_y_axis_order(ax, all_costs_flat)

    for j in range(len(model_variants), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Cost Evolution: {dataset_name}\n(4 algorithm types, averaged over runs)", fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_file = output_dir / f"{dataset_name}_cost_evolution_all_models.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"График сохранен: {out_file}")


def main():
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"
    output_dir = script_dir / "cost_plots"
    output_dir.mkdir(exist_ok=True)

    cfg = load_config()
    model_variants = list(cfg.get("model_variants", {}).keys()) or ["GA", "GEA_1", "GEA_2", "GEA_3", "GEA"]
    type_keys = cfg.get("algorithm_types") or list(TYPE_STYLE.keys())
    datasets = load_dataset_results(results_dir)
    if not datasets:
        print("Нет результатов в results/")
        return

    print(f"Модели: {model_variants}, типы: {type_keys}")
    print(f"Датасетов: {len(datasets)}, сохранение в {output_dir}\n")
    for name, data in sorted(datasets.items()):
        print(f"Обработка {name}...")
        plot_dataset_all_models(name, data, output_dir, model_variants, type_keys)
    print("\nГрафики созданы.")


if __name__ == "__main__":
    main()
