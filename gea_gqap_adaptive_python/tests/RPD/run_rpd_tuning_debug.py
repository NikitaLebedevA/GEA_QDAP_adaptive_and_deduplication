#!/usr/bin/env python3
"""
Debug runner for parameter tuning using RPD + Taguchi tables.

What it does:
- Picks 5 `c*` datasets uniformly (top-to-bottom) + 2 simplest `T*` datasets.
- For a chosen Excel sheet + block ("adaptive" by default), iterates over Taguchi runs.
- For each Taguchi row, runs the configured algorithm variants (see RPD_EVAL_MODELS / RPD_EVAL_TYPES),
  `num_runs` times per dataset, collects best_cost.
- Builds a response matrix per model (and per eval type); per model, averages types then runs RPD + MEPFM.
- Env `RPD_EVAL_MODELS` defaults to GA,GEA_1,GEA_2,GEA_3,GEA. `RPD_EVAL_TYPES` defaults to adaptive only.

Outputs:
- JSON with raw costs + RPD + chosen best levels in `gea_gqap_adaptive_python/tests/RPD/results/`
- Final payload includes `recommended.config_kwargs` for pasting into `test_config.json` (`algorithm_by_model`).

Full restart / cluster settings / batch by sheet: see `README_RPD.md` in this folder.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np

# GA / GEA_1 / GEA_2 / GEA_3 / GEA — как в run_comparison_test
DEFAULT_MODEL_VARIANTS: Dict[str, Tuple[bool, bool, bool]] = {
    "GA": (False, False, False),
    "GEA_1": (True, False, False),
    "GEA_2": (False, True, False),
    "GEA_3": (False, False, True),
    "GEA": (True, True, True),
}

ADAPTIVE_ONLY_PARAM_KEYS = frozenset(
    {"adaptive_alpha", "adaptive_lambda_min", "adaptive_lambda_max", "adaptive_epsilon"}
)

# keep single-threaded BLAS for repeatability / laptop safety
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent.parent.parent
sys_path_added = False


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _fmt_duration(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min {s % 60} s"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h} h {m} min {sec} s"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _ensure_sys_path() -> None:
    global sys_path_added
    if sys_path_added:
        return
    import sys

    sys.path.insert(0, str(REPO_ROOT / "gea_gqap_adaptive_python"))
    sys_path_added = True


def _parse_eval_models() -> List[str]:
    raw = os.environ.get("RPD_EVAL_MODELS", "GA,GEA_1,GEA_2,GEA_3,GEA").strip()
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    for p in parts:
        if p not in DEFAULT_MODEL_VARIANTS:
            raise ValueError(
                f"Unknown RPD_EVAL_MODELS entry {p!r}; allowed: {sorted(DEFAULT_MODEL_VARIANTS)}"
            )
    return parts


def _parse_eval_type_specs() -> List[Tuple[str, bool]]:
    """
    (kind, dedupe): kind is 'adaptive' | 'non_adaptive'.
    """
    raw = os.environ.get("RPD_EVAL_TYPES", "adaptive").strip().lower()
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    out: List[Tuple[str, bool]] = []
    for p in parts:
        if p == "adaptive":
            out.append(("adaptive", False))
        elif p == "non_adaptive":
            out.append(("non_adaptive", False))
        elif p == "adaptive_wo_duplicates":
            out.append(("adaptive", True))
        elif p == "non_adaptive_wo_duplicates":
            out.append(("non_adaptive", True))
        else:
            raise ValueError(
                f"Unknown RPD_EVAL_TYPES token {p!r}; use adaptive, non_adaptive, "
                "adaptive_wo_duplicates, non_adaptive_wo_duplicates"
            )
    return out


def _eval_key_str(model_key: str, algo_kind: str, dedupe: bool) -> str:
    if algo_kind == "adaptive" and not dedupe:
        return f"{model_key}|adaptive"
    if algo_kind == "non_adaptive" and not dedupe:
        return f"{model_key}|non_adaptive"
    if algo_kind == "adaptive" and dedupe:
        return f"{model_key}|adaptive_wo_duplicates"
    return f"{model_key}|non_adaptive_wo_duplicates"


def _worker_process(
    task: Tuple[str, int, Dict[str, Any], int, float, str, str, Tuple[bool, bool, bool], str, bool],
) -> Tuple[str, int, str, str, bool, float]:
    """Multiprocessing worker (must be top-level to be picklable)."""
    import dataclasses
    from pathlib import Path
    import sys

    (
        ds,
        run_number,
        cfg_kwargs,
        iterations,
        time_limit,
        repo_root_str,
        model_key,
        enable_scenario,
        algo_kind,
        dedupe,
    ) = task

    repo_root = Path(repo_root_str)
    sys.path.insert(0, str(repo_root / "gea_gqap_adaptive_python"))
    sys.path.insert(0, str(repo_root / "GEA_GQAP_Python"))

    it = int(iterations)
    tl = float(time_limit)

    if algo_kind == "adaptive":
        from gea_gqap_adaptive_python import AdaptiveAlgorithmConfig, load_model, run_adaptive_ga

        names = {f.name for f in dataclasses.fields(AdaptiveAlgorithmConfig)}
        kwargs = {k: v for k, v in cfg_kwargs.items() if k in names}
        model = load_model(ds)
        cfg = AdaptiveAlgorithmConfig(
            iterations=it,
            time_limit=tl,
            enable_scenario=enable_scenario,
            deduplicate=dedupe,
            **kwargs,
        )
        res = run_adaptive_ga(model, config=cfg)
    else:
        from gea_gqap_python import load_model as load_model_na
        from gea_gqap_python.algorithm import run_ga, AlgorithmConfig

        base = {k: v for k, v in cfg_kwargs.items() if k not in ADAPTIVE_ONLY_PARAM_KEYS}
        names = {f.name for f in dataclasses.fields(AlgorithmConfig)}
        kwargs = {k: v for k, v in base.items() if k in names}
        model = load_model_na(ds)
        cfg = AlgorithmConfig(
            iterations=it,
            time_limit=tl,
            enable_scenario=enable_scenario,
            deduplicate=dedupe,
            **kwargs,
        )
        res = run_ga(model, config=cfg)
    return ds, int(run_number), model_key, algo_kind, dedupe, float(res.best_cost)


def _build_recommended_payload(
    param_levels: Dict[str, Tuple[float, float, float]],
    param_names: List[str],
    mep,
) -> Dict[str, Any]:
    best_level_by_param = {p: int(lvl) for p, lvl in zip(param_names, mep.best_levels)}
    best_values_by_param: Dict[str, float | int] = {}
    best_config_kwargs: Dict[str, float | int] = {}
    for p in param_names:
        lvl = best_level_by_param[p]
        if lvl not in (1, 2, 3):
            continue
        val = float(param_levels[p][lvl - 1])
        field = _map_param(p)
        if field is None:
            best_values_by_param[p] = val
            continue
        if field in ("population_size", "mask_mutation_index"):
            best_config_kwargs[field] = int(round(val))
            best_values_by_param[p] = int(round(val))
        else:
            best_config_kwargs[field] = float(val)
            best_values_by_param[p] = float(val)
    return {
        "best_level_by_param": best_level_by_param,
        "best_value_by_param": best_values_by_param,
        "config_kwargs": best_config_kwargs,
    }


def _pick_datasets() -> List[str]:
    _ensure_sys_path()
    from gea_gqap_adaptive_python import list_available_models, load_model
    try:
        from .rpd_utils import pick_uniform  # type: ignore
    except Exception:  # pragma: no cover
        import sys

        sys.path.insert(0, str(TEST_DIR.parent))
        from rpd_utils import pick_uniform  # type: ignore

    names = list_available_models()
    c_names = sorted([n for n in names if n.startswith("c")])
    t_names = sorted([n for n in names if n.startswith("T")])

    # 5 c datasets uniformly from top to bottom
    c_sel = pick_uniform(c_names, 5)

    # 2 simplest T datasets: smallest I*J
    t_info: List[Tuple[str, int]] = []
    for n in t_names:
        m = load_model(n)
        t_info.append((n, int(m.I) * int(m.J)))
    t_info.sort(key=lambda x: (x[1], x[0]))
    t_sel = [t_info[0][0], t_info[1][0]]
    return c_sel + t_sel


def _map_param(name: str) -> str | None:
    """
    Map Excel parameter names to config fields.
    Unknown parameters are ignored (returned as None).
    """
    return {
        "Npop": "population_size",
        "PCrossover": "crossover_rate",
        "PMutation": "mutation_rate",
        "PScenario1": "p_scenario1",
        "PScenario2": "p_scenario2",
        "PScenario3": "p_scenario3",
        "PFixedX": "p_fixed_x",
        "PCrossoverScenario1": "scenario_crossover_rate",
        "PMutationScenario2": "scenario_mutation_rate",
        "PMutationScenario3": "scenario_mutation_rate",
        # adaptive params
        "alpha": "adaptive_alpha",
        "lambda_min": "adaptive_lambda_min",
        "lambda_max": "adaptive_lambda_max",
    }.get(name)


def _build_config_kwargs(param_levels: Dict[str, Tuple[float, float, float]], levels_row: List[int]) -> Dict:
    # Keep stable param order as appears in Excel (dict preserves insertion in py3.7+)
    names = list(param_levels.keys())
    if len(names) != len(levels_row):
        raise ValueError(f"Table columns != params: {len(levels_row)} vs {len(names)}")
    kwargs: Dict = {}
    for p_name, lvl in zip(names, levels_row):
        field = _map_param(p_name)
        if field is None:
            continue
        if lvl not in (1, 2, 3):
            raise ValueError(f"Unexpected level {lvl} for {p_name}")
        val = float(param_levels[p_name][lvl - 1])
        if field in ("population_size",):
            kwargs[field] = int(round(val))
        elif field in ("mask_mutation_index",):
            kwargs[field] = int(round(val))
        else:
            kwargs[field] = float(val)
    return kwargs


def _load_taguchi_from_repo_config(sheet_name: str, block_name: str) -> tuple[Dict[str, Tuple[float, float, float]], List[List[int]], Dict[str, Any]]:
    """
    Load Taguchi parameter levels + table from a repo-tracked JSON config.
    This removes the runtime dependency on the Excel file on the cluster.
    """
    cfg_path = Path(os.environ.get("RPD_TAGUCHI_CONFIG", str(TEST_DIR / "taguchi_config_RPD.json"))).resolve()
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    sheets = data.get("sheets", {})
    if sheet_name not in sheets:
        raise KeyError(f"Sheet '{sheet_name}' not found in {cfg_path}. Available: {sorted(sheets.keys())}")
    sheet = sheets[sheet_name]
    if block_name not in sheet:
        raise KeyError(f"Block '{block_name}' not found for sheet '{sheet_name}' in {cfg_path}.")
    block = sheet[block_name]

    param_levels_raw = block["param_levels"]
    param_levels: Dict[str, Tuple[float, float, float]] = {
        str(k): (float(v[0]), float(v[1]), float(v[2])) for k, v in param_levels_raw.items()
    }
    table = [[int(x) for x in row] for row in block["table"]]
    meta = {"config_path": str(cfg_path), "sheet": sheet_name, "block": block_name}
    return param_levels, table, meta


def main() -> None:
    _ensure_sys_path()
    try:
        from .rpd_utils import mepfm, rpd_matlab  # type: ignore
    except Exception:  # pragma: no cover
        import sys

        sys.path.insert(0, str(TEST_DIR.parent))
        from rpd_utils import mepfm, rpd_matlab  # type: ignore

    from concurrent.futures import ProcessPoolExecutor, as_completed

    sheet_name = os.environ.get("RPD_SHEET", "GEA_2")
    block_name = os.environ.get("RPD_BLOCK", "adaptive")  # "base" or "adaptive"

    iterations = int(os.environ.get("RPD_ITERATIONS", "60"))
    time_limit = float(os.environ.get("RPD_TIME_LIMIT", "3.0"))
    num_runs = int(os.environ.get("RPD_NUM_RUNS", "5"))
    max_rows = int(os.environ.get("RPD_MAX_TAGUCHI_ROWS", "0"))  # 0 => all
    num_workers = int(os.environ.get("RPD_NUM_WORKERS", os.environ.get("NUM_WORKERS", "16")))

    eval_models = _parse_eval_models()
    type_specs = _parse_eval_type_specs()
    variants = DEFAULT_MODEL_VARIANTS
    eval_keys: List[Tuple[str, str, bool]] = []
    for mk in eval_models:
        for kind, ded in type_specs:
            eval_keys.append((mk, kind, ded))

    datasets = _pick_datasets()
    param_levels, table_list, taguchi_meta = _load_taguchi_from_repo_config(sheet_name, block_name)

    param_names = list(param_levels.keys())
    table = np.array(table_list, dtype=int)
    if max_rows and max_rows < table.shape[0]:
        table = table[:max_rows, :]

    n_ds = len(datasets)
    total_rows = int(table.shape[0])
    response_by_key: Dict[Tuple[str, str, bool], np.ndarray] = {
        k: np.zeros((total_rows, n_ds), dtype=float) for k in eval_keys
    }
    response_by_model: Dict[str, np.ndarray] = {
        mk: np.zeros((total_rows, n_ds), dtype=float) for mk in eval_models
    }
    raw_costs_detail: List[Dict[str, List[List[float]]]] = []

    primary_model = sheet_name if sheet_name in eval_models else eval_models[0]
    if sheet_name not in eval_models:
        print(
            f"[{_ts()}] WARN: sheet {sheet_name!r} not in RPD_EVAL_MODELS; "
            f"legacy response/recommended use primary_model={primary_model!r}",
            flush=True,
        )

    out_dir = TEST_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rpd_tuning_{sheet_name}_{block_name}.json"

    started_at = time.monotonic()
    tasks_per_row = n_ds * num_runs * len(eval_keys)
    total_tasks_all = total_rows * tasks_per_row
    print(f"[{_ts()}] === RPD tuning start ===", flush=True)
    print(f"[{_ts()}] Sheet={sheet_name} block={block_name}", flush=True)
    print(f"[{_ts()}] Taguchi config: {taguchi_meta['config_path']}", flush=True)
    print(f"[{_ts()}] Taguchi rows={total_rows} (max_rows={max_rows or 'all'})", flush=True)
    print(f"[{_ts()}] Datasets={n_ds} runs_per_dataset={num_runs} workers={num_workers}", flush=True)
    print(f"[{_ts()}] RPD_EVAL_MODELS={','.join(eval_models)}", flush=True)
    print(f"[{_ts()}] RPD_EVAL_TYPES={os.environ.get('RPD_EVAL_TYPES', 'adaptive')}", flush=True)
    print(f"[{_ts()}] Iterations={iterations} time_limit={time_limit}", flush=True)
    print(f"[{_ts()}] Tasks per Taguchi row={tasks_per_row} total_tasks={total_tasks_all}", flush=True)
    print(f"[{_ts()}] Output JSON: {out_path}", flush=True)

    ds_to_idx = {ds: i for i, ds in enumerate(datasets)}

    for run_idx in range(total_rows):
        row_started_at = time.monotonic()
        row_levels = table[run_idx, :].tolist()
        cfg_kwargs = _build_config_kwargs(param_levels, row_levels)

        costs_acc: Dict[Tuple[str, str, bool], List[List[float]]] = {
            k: [[float("nan")] * num_runs for _ in range(n_ds)] for k in eval_keys
        }

        tasks: List[Any] = []
        for ds in datasets:
            for r in range(num_runs):
                for mk in eval_models:
                    es = variants[mk]
                    for kind, ded in type_specs:
                        tasks.append(
                            (
                                ds,
                                r,
                                cfg_kwargs,
                                iterations,
                                time_limit,
                                str(REPO_ROOT),
                                mk,
                                es,
                                kind,
                                ded,
                            )
                        )

        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(_worker_process, t) for t in tasks]
            done = 0
            errors = 0

            heartbeat_stop = threading.Event()

            def _heartbeat() -> None:
                while not heartbeat_stop.wait(300):
                    elapsed = time.monotonic() - row_started_at
                    print(
                        f"[{_ts()}] heartbeat row {run_idx + 1}/{total_rows}: {done}/{len(tasks)} done "
                        f"({done/len(tasks)*100:.0f}%) | {_fmt_duration(elapsed)}",
                        flush=True,
                    )

            hb = threading.Thread(target=_heartbeat, daemon=True)
            hb.start()

            for fut in as_completed(futures):
                try:
                    ds, r, mk, kind, ded, cost = fut.result()
                    di = ds_to_idx[ds]
                    costs_acc[(mk, kind, ded)][di][r] = cost
                except Exception as e:
                    errors += 1
                    print(f"[{_ts()}] ERR row {run_idx + 1}: {e}", flush=True)
                finally:
                    done += 1
                    if done % max(1, num_workers) == 0 or done == len(tasks):
                        elapsed = time.monotonic() - row_started_at
                        print(
                            f"[{_ts()}] row {run_idx + 1}/{total_rows}: {done}/{len(tasks)} "
                            f"({done/len(tasks)*100:.0f}%) | {_fmt_duration(elapsed)}",
                            flush=True,
                        )

            heartbeat_stop.set()

        row_raw: Dict[str, List[List[float]]] = {}
        for k in eval_keys:
            key_s = _eval_key_str(k[0], k[1], k[2])
            row_raw[key_s] = []
            for di in range(n_ds):
                cst = costs_acc[k][di]
                if any(not np.isfinite(x) for x in cst):
                    raise RuntimeError(
                        f"Missing/invalid costs model={k[0]} kind={k[1]} dedupe={k[2]} "
                        f"dataset={datasets[di]} taguchi_row={run_idx}"
                    )
                row_raw[key_s].append(list(cst))
                response_by_key[k][run_idx, di] = float(np.mean(cst))
        raw_costs_detail.append(row_raw)

        for mk in eval_models:
            rows_stack = np.vstack(
                [response_by_key[(mk, kind, ded)][run_idx, :] for kind, ded in type_specs]
            )
            response_by_model[mk][run_idx, :] = np.mean(rows_stack, axis=0)

        response_so_far = response_by_model[primary_model][: run_idx + 1, :].copy()
        rpd_so_far = rpd_matlab(response_so_far, flag=1)
        mean_rpd_so_far = rpd_so_far.mean(axis=1)
        _write_json(
            out_path,
            {
                "taguchi_config": taguchi_meta,
                "rpd_eval": {
                    "eval_models": eval_models,
                    "eval_type_specs": [{"kind": k, "dedupe": d} for k, d in type_specs],
                    "primary_model_for_legacy_fields": primary_model,
                },
                "datasets": datasets,
                "debug_overrides": {"iterations": iterations, "time_limit": time_limit, "num_runs": num_runs},
                "param_names": param_names,
                "param_levels": {k: list(v) for k, v in param_levels.items()},
                "taguchi_table": table_list,
                "progress": {
                    "completed_rows": int(run_idx + 1),
                    "total_rows": int(total_rows),
                    "elapsed_seconds": round(time.monotonic() - started_at, 1),
                },
                "response_mean_best_cost": response_so_far.tolist(),
                "raw_best_costs_by_eval": raw_costs_detail[: run_idx + 1],
                "rpd": rpd_so_far.tolist(),
                "mean_rpd": mean_rpd_so_far.tolist(),
            },
        )
        print(
            f"[{_ts()}] Saved progress ({run_idx + 1}/{total_rows}) -> {out_path}",
            flush=True,
        )

    response = response_by_model[primary_model]
    rpd = rpd_matlab(response, flag=1)
    mean_rpd = rpd.mean(axis=1)
    mep_primary = mepfm(mean_rpd, table, param_names=param_names)

    recommended_by_model: Dict[str, Dict[str, Any]] = {}
    mepfm_by_model: Dict[str, Any] = {}
    rpd_by_model: Dict[str, Any] = {}
    for mk in eval_models:
        rpd_m = rpd_matlab(response_by_model[mk], flag=1)
        mean_rpd_m = rpd_m.mean(axis=1)
        mep_m = mepfm(mean_rpd_m, table, param_names=param_names)
        recommended_by_model[mk] = _build_recommended_payload(param_levels, param_names, mep_m)
        mepfm_by_model[mk] = {
            "best_levels": list(mep_m.best_levels),
            "levels": [list(x) for x in mep_m.levels],
            "means_by_level": [list(map(float, x)) for x in mep_m.means_by_level],
        }
        rpd_by_model[mk] = rpd_m.tolist()

    payload = {
        "taguchi_config": taguchi_meta,
        "rpd_eval": {
            "eval_models": eval_models,
            "eval_type_specs": [{"kind": k, "dedupe": d} for k, d in type_specs],
            "primary_model_for_legacy_fields": primary_model,
        },
        "datasets": datasets,
        "debug_overrides": {"iterations": iterations, "time_limit": time_limit, "num_runs": num_runs},
        "param_names": param_names,
        "param_levels": {k: list(v) for k, v in param_levels.items()},
        "taguchi_table": table_list,
        "response_mean_best_cost": response.tolist(),
        "response_mean_best_cost_by_model": {mk: response_by_model[mk].tolist() for mk in eval_models},
        "raw_best_costs_by_eval": raw_costs_detail,
        "rpd": rpd.tolist(),
        "rpd_by_model": rpd_by_model,
        "mean_rpd": mean_rpd.tolist(),
        "mepfm": {
            "best_levels": list(mep_primary.best_levels),
            "levels": [list(x) for x in mep_primary.levels],
            "means_by_level": [list(map(float, x)) for x in mep_primary.means_by_level],
        },
        "mepfm_by_model": mepfm_by_model,
        "recommended": recommended_by_model[primary_model],
        "recommended_by_model": recommended_by_model,
    }

    _write_json(out_path, payload)
    elapsed_total = time.monotonic() - started_at
    print(f"[{_ts()}] Saved final: {out_path}", flush=True)
    print(f"[{_ts()}] Primary ({primary_model}) best levels: {mep_primary.best_levels}", flush=True)
    print(
        f"[{_ts()}] Primary recommended config kwargs: {recommended_by_model[primary_model]['config_kwargs']}",
        flush=True,
    )
    print(f"[{_ts()}] Total elapsed: {_fmt_duration(elapsed_total)}", flush=True)


if __name__ == "__main__":
    main()

