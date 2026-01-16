from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import polars as pl
import yaml

_BOM = "\ufeff"


def load_config(config_path: Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg if isinstance(cfg, dict) else {}


def resolve_path(base_dir: Path, maybe_path: str | Path) -> Path:
    path = Path(maybe_path)
    if path.is_absolute():
        return path
    return (Path(base_dir) / path).resolve()


def bom_rename_map(columns: Sequence[str]) -> Dict[str, str]:
    rename: Dict[str, str] = {}
    for col in columns:
        if col.startswith(_BOM):
            rename[col] = col.lstrip(_BOM)
    return rename


def strip_bom_columns(df: pl.DataFrame) -> pl.DataFrame:
    rename = bom_rename_map(df.columns)
    return df.rename(rename) if rename else df

