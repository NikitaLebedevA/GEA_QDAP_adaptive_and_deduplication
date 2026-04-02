#!/usr/bin/env python3
"""
Plot tuning results similarly to `Tuning_Result_Plotting_OR.ipynb`.

Input: JSON produced by `run_rpd_tuning_debug.py`.
Output: PDF with main effects plots for means (MEPFM).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import numpy as np


def main() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    here = Path(__file__).resolve().parent
    results_dir = here / "results"
    in_path = Path(os.environ.get("RPD_JSON", str(results_dir / "rpd_tuning_GEA_2_adaptive.json"))).resolve()
    payload = json.loads(in_path.read_text(encoding="utf-8"))

    table = np.array(payload["taguchi_table"], dtype=int)
    mean_rpd = np.array(payload["mean_rpd"], dtype=float).reshape(-1)
    param_names: List[str] = list(payload["param_names"])

    n_params = table.shape[1]
    ncols = 4
    nrows = int(np.ceil(n_params / ncols))

    sns.set(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    best_levels = []
    for i in range(n_params):
        levels = np.unique(table[:, i])
        means = []
        for lvl in levels:
            idx = np.where(table[:, i] == lvl)[0]
            means.append(float(np.mean(mean_rpd[idx])))
        means = np.array(means, dtype=float)
        best_lvl = int(levels[int(np.argmin(means))])
        best_levels.append(best_lvl)

        ax = axes[i]
        ax.plot(levels, means, marker="o", linestyle="-", color="b")
        ax.set_title(param_names[i], fontsize=12, weight="bold")
        ax.set_xlabel("Levels")
        ax.set_ylabel("Mean RPD")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_xticks(levels)
        ax.set_xticklabels([f"{chr(65+i)} ({lvl})" for lvl in levels])
        ax.scatter(best_lvl, float(np.min(means)), color="red", s=80, zorder=3, label="Best")
        ax.legend(loc="best", fontsize=8)

    for j in range(n_params, len(axes)):
        fig.delaxes(axes[j])

    meta = payload.get("taguchi_config") or payload.get("excel") or {}
    sheet = meta.get("sheet", "?")
    block = meta.get("block", "?")
    title = f"Tuning results for {sheet} ({block})"
    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    out_pdf = in_path.with_suffix(".pdf")
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    print("Saved:", out_pdf)
    plt.show()


if __name__ == "__main__":
    main()

