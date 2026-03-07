from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List

import numpy as np

from .models import Model


DATA_DIR = Path(__file__).resolve().parent.parent / "datasets"


def list_available_models() -> List[str]:
    return sorted(p.stem for p in DATA_DIR.glob("*.m"))


def _parse_numeric_sequence(block: str) -> np.ndarray:
    cleaned = block.replace("\n", " ").replace("\r", " ").strip()
    parts = [p for p in re.split(r"[,\s]+", cleaned) if p]
    return np.array([float(p) for p in parts], dtype=float)


def _parse_matrix(block: str) -> np.ndarray:
    rows = [row.strip() for row in block.strip().split(";") if row.strip()]
    data = []
    for row in rows:
        entries = [entry for entry in re.split(r"[,\s]+", row) if entry]
        data.append([float(entry) for entry in entries])
    return np.array(data, dtype=float)


def _extract_block(content: str, var_name: str) -> str:
    pattern = rf"\b{var_name}\b\s*=\s*\[(.*?)\];"
    match = re.search(pattern, content, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not locate block for {var_name}")
    return match.group(1)


def _extract_scalar(content: str, var_name: str) -> int:
    pattern = rf"\b{var_name}\b\s*=\s*(\d+);"
    match = re.search(pattern, content)
    if not match:
        raise ValueError(f"Could not locate scalar for {var_name}")
    return int(match.group(1))


@lru_cache
def load_model(name: str) -> Model:
    path = DATA_DIR / f"{name}.m"
    if not path.exists():
        raise FileNotFoundError(f"No MATLAB dataset named '{name}' found in {DATA_DIR}")

    content = path.read_text(encoding="utf-8")
    I = _extract_scalar(content, "I")
    J = _extract_scalar(content, "J")
    cij = _parse_matrix(_extract_block(content, "cij"))
    aij = _parse_matrix(_extract_block(content, "aij"))
    bi = _parse_numeric_sequence(_extract_block(content, "bi"))
    X = _parse_numeric_sequence(_extract_block(content, "X"))
    Y = _parse_numeric_sequence(_extract_block(content, "Y"))
    XX = _parse_numeric_sequence(_extract_block(content, "XX"))
    YY = _parse_numeric_sequence(_extract_block(content, "YY"))

    cij = cij.reshape(I, J)
    aij = aij.reshape(I, J)
    bi = bi.reshape(I)
    X = X.reshape(I)
    Y = Y.reshape(I)
    XX = XX.reshape(J)
    YY = YY.reshape(J)

    DIS = np.sqrt((X[:, None] - X[None, :]) ** 2 + (Y[:, None] - Y[None, :]) ** 2)
    F = np.sqrt((XX[:, None] - XX[None, :]) ** 2 + (YY[:, None] - YY[None, :]) ** 2)

    return Model(I=I, J=J, cij=cij, aij=aij, bi=bi, DIS=DIS, F=F)

