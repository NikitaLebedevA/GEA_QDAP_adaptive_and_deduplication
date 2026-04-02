#!/usr/bin/env python3
"""Все листы из taguchi JSON (или RPD_SHEETS) × RPD_BLOCKS (по умолчанию adaptive,base). Подробности: README_RPD.md."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent


def _sheet_names_from_config() -> list[str]:
    cfg_path = Path(os.environ.get("RPD_TAGUCHI_CONFIG", str(TEST_DIR / "taguchi_config_RPD.json"))).resolve()
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    return sorted(data.get("sheets", {}).keys())


def main() -> None:
    raw_sheets = os.environ.get("RPD_SHEETS", "").strip()
    sheets = [s.strip() for s in raw_sheets.split(",") if s.strip()] if raw_sheets else _sheet_names_from_config()
    blocks = [b.strip() for b in os.environ.get("RPD_BLOCKS", "adaptive,base").split(",") if b.strip()]

    print(f"Sheets: {sheets}", flush=True)
    print(f"Blocks: {blocks}", flush=True)

    sys.path.insert(0, str(TEST_DIR))
    import run_rpd_tuning_debug as rpd  # noqa: E402

    for sheet in sheets:
        os.environ["RPD_SHEET"] = sheet
        for block in blocks:
            os.environ["RPD_BLOCK"] = block
            print(f">>> RPD {sheet} / {block}", flush=True)
            rpd.main()


if __name__ == "__main__":
    main()
