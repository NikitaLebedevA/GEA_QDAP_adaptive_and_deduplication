#!/usr/bin/env python3
"""
Графики эволюции адаптивных коэффициентов (лямбд) по итерациям.
Для каждого датасета и каждой модели — два адаптивных типа: adaptive и adaptive_wo_duplicates.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

TEST_DIR = Path(__file__).parent
LOCAL_LIB = TEST_DIR / ".venv_lib"
if LOCAL_LIB.exists():
    sys.path.insert(0, str(LOCAL_LIB))

import matplotlib.pyplot as plt

LAMBDA_KEYS = ["crossover", "mutation", "scenario1", "scenario2", "scenario3"]
COLORS = {
    "crossover": "#1f77b4",
    "mutation": "#ff7f0e",
    "scenario1": "#2ca02c",
    "scenario2": "#d62728",
    "scenario3": "#9467bd",
}
ADAPTIVE_TYPE_KEYS = ["adaptive", "adaptive_wo_duplicates"]
ADAPTIVE_LABELS = {"adaptive": "Adaptive", "adaptive_wo_duplicates": "Adaptive w/o dup"}


def load_config() -> Dict[str, Any]:
    config_path = TEST_DIR / "test_config.json"
    if not config_path.exists():
        return {"model_variants": ["GA", "GEA_1", "GEA_2", "GEA_3", "GEA"]}
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def averaged_lambdas_from_runs(adaptive_runs: List[Dict]) -> Dict[str, List[float]]:
    if not adaptive_runs:
        return {}
    max_iter = max(len(r.get("iterations", [])) for r in adaptive_runs)
    if max_iter == 0:
        return {}
    sums = {k: [0.0] * max_iter for k in LAMBDA_KEYS}
    counts = [0] * max_iter
    for run in adaptive_runs:
        for i, it in enumerate(run.get("iterations", [])):
            if i >= max_iter:
                break
            lv = it.get("lambda_values", {})
            if lv:
                for k in LAMBDA_KEYS:
                    sums[k][i] += lv.get(k, 1.0)
                counts[i] += 1
    return {
        k: [sums[k][i] / max(counts[i], 1) for i in range(max_iter)]
        for k in LAMBDA_KEYS
    }


def plot_dataset_all_models(
    dataset_name: str,
    dataset_data: Dict[str, Any],
    output_dir: Path,
    model_variants: List[str],
):
    models = dataset_data.get("models", {})
    if not models:
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
        all_vals = []
        has_data = False
        for atk in ADAPTIVE_TYPE_KEYS:
            runs = m.get(atk, {}).get("runs", [])
            avg = averaged_lambdas_from_runs(runs)
            if not avg or not avg.get("crossover"):
                continue
            has_data = True
            iters = list(range(1, len(avg["crossover"]) + 1))
            linestyle = "-" if atk == "adaptive" else "--"
            label_prefix = ADAPTIVE_LABELS.get(atk, atk)
            for key in LAMBDA_KEYS:
                ax.plot(
                    iters, avg[key],
                    label=f"{key} ({label_prefix})",
                    linewidth=1.5, color=COLORS[key], linestyle=linestyle,
                )
                all_vals.extend(avg[key])

        if not has_data:
            ax.text(0.5, 0.5, f"No lambda data\n{model_key}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(model_key, fontsize=11)
            continue

        ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        y_min = max(0.3, min(all_vals) * 0.95)
        y_max = min(1.6, max(all_vals) * 1.05)
        ax.set_ylim(y_min, y_max)
        ax.set_title(model_key, fontsize=11, fontweight="bold")
        ax.set_xlabel("Iteration", fontsize=9)
        ax.set_ylabel("Lambda", fontsize=9)
        ax.legend(loc="best", fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

    for j in range(len(model_variants), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Evolution of Adaptive Lambdas: {dataset_name}\n(Adaptive vs Adaptive w/o duplicates)", fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_file = output_dir / f"{dataset_name}_lambda_evolution_all_models.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"График коэффициентов сохранен: {out_file}")


def main():
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"
    output_dir = script_dir / "lambda_plots"
    output_dir.mkdir(exist_ok=True)
    cfg = load_config()
    model_variants = list(cfg.get("model_variants", {}).keys()) or ["GA", "GEA_1", "GEA_2", "GEA_3", "GEA"]
    datasets = load_dataset_results(results_dir)
    if not datasets:
        print("Нет результатов в results/")
        return
    print(f"Модели: {model_variants}, сохранение в {output_dir}\n")
    for name, data in sorted(datasets.items()):
        print(f"Обработка {name}...")
        plot_dataset_all_models(name, data, output_dir, model_variants)
    print("Графики коэффициентов созданы.")


if __name__ == "__main__":
    main()
