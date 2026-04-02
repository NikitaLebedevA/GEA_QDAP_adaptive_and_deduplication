#!/usr/bin/env python3
"""
Графики по итоговому JSON run_rpd_tuning_debug.py (MEPFM + кривые отклика по строкам Taguchi).

Пример:
  python3 plot_rpd_from_json.py results/rpd_tuning_GEA_2_adaptive.json
  python3 plot_rpd_from_json.py results/rpd_tuning_GEA_2_adaptive.json --variant GEA_2|adaptive --out-dir plots/

Зависимость: matplotlib (см. pyproject.toml).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def _plot_mepfm_bars(
    mepfm: Dict[str, Any],
    param_names: List[str],
    title: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    levels: List[List[int]] = mepfm["levels"]
    means: List[List[float]] = mepfm["means_by_level"]
    n = len(param_names)
    ncols = min(4, max(1, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), squeeze=False)
    for j, pname in enumerate(param_names):
        r, c = divmod(j, ncols)
        ax = axes[r][c]
        xs = [str(x) for x in levels[j]]
        ys = means[j]
        ax.bar(xs, ys, color="steelblue", edgecolor="black", linewidth=0.5)
        ax.set_title(pname, fontsize=10)
        ax.set_xlabel("уровень")
        ax.set_ylabel("средн. RPD-метрика")
        ax.grid(axis="y", alpha=0.3)
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_response_curves(
    y_mean: List[List[float]],
    y_min: List[List[float]] | None,
    title: str,
    out_path: Path,
) -> None:
    """y_*: shape (n_rows, n_ds) — кривая по строкам Taguchi."""
    import matplotlib.pyplot as plt

    ym = np.asarray(y_mean, dtype=float)
    row_mean = ym.mean(axis=1)
    row_min = ym.min(axis=1)
    row_max = ym.max(axis=1)
    x = np.arange(ym.shape[0])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(x, row_min, row_max, alpha=0.25, label="min–max по датасетам")
    ax.plot(x, row_mean, "o-", ms=3, label="среднее по датасетам (mean реплик)")
    if y_min is not None:
        ymi = np.asarray(y_min, dtype=float)
        ax.plot(
            x,
            ymi.mean(axis=1),
            "s--",
            ms=3,
            alpha=0.85,
            label="среднее по датасетам (min реплик)",
        )
    ax.set_xlabel("индекс строки Taguchi (0…)")
    ax.set_ylabel("best_cost")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Графики из rpd_tuning_*.json")
    ap.add_argument("json_path", type=Path, help="Финальный JSON RPD")
    ap.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Ключ MODEL|тип для by_variant графиков (по умолчанию progress_eval_key из JSON)",
    )
    ap.add_argument("--out-dir", type=Path, default=None, help="Каталог для PNG (по умолчанию рядом с JSON)")
    args = ap.parse_args()

    path = args.json_path.resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    param_names: List[str] = list(data.get("param_names") or [])

    out_dir = args.out_dir.resolve() if args.out_dir else path.parent / f"{path.stem}_plots"
    stem = path.stem

    mepfm = data.get("mepfm")
    if isinstance(mepfm, dict) and param_names:
        _plot_mepfm_bars(mepfm, param_names, f"{stem} primary MEPFM", out_dir / f"{stem}_mepfm_primary.png")

    vk = args.variant
    if vk is None:
        pek = data.get("rpd_eval", {}).get("progress_eval_key")
        if isinstance(pek, list) and len(pek) == 3:
            mk, kind, ded = pek[0], pek[1], pek[2]
            if kind == "adaptive" and not ded:
                vk = f"{mk}|adaptive"
            elif kind == "adaptive" and ded:
                vk = f"{mk}|adaptive_wo_duplicates"
            elif kind == "non_adaptive" and not ded:
                vk = f"{mk}|non_adaptive"
            else:
                vk = f"{mk}|non_adaptive_wo_duplicates"

    by_m = data.get("mepfm_by_variant") or {}
    if vk and isinstance(by_m.get(vk), dict) and param_names:
        _plot_mepfm_bars(
            by_m[vk],
            param_names,
            f"{stem} MEPFM {vk}",
            out_dir / f"{stem}_mepfm_{vk.replace('|', '_')}.png",
        )

    by_min = data.get("mepfm_by_variant_min") or {}
    if vk and isinstance(by_min.get(vk), dict) and param_names:
        _plot_mepfm_bars(
            by_min[vk],
            param_names,
            f"{stem} MEPFM(min-ячейки) {vk}",
            out_dir / f"{stem}_mepfm_min_{vk.replace('|', '_')}.png",
        )

    rmean = (data.get("response_mean_best_cost_by_variant") or {})
    rmin = data.get("response_min_best_cost_by_variant") or {}
    if vk and vk in rmean:
        _plot_response_curves(
            rmean[vk],
            rmin.get(vk) if isinstance(rmin, dict) else None,
            f"{stem} отклик по строкам Taguchi — {vk}",
            out_dir / f"{stem}_response_rows_{vk.replace('|', '_')}.png",
        )

    print(f"Графики: {out_dir}")


if __name__ == "__main__":
    main()
