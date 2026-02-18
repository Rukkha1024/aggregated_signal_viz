from __future__ import annotations

"""Shared helpers for config-driven aggregation mode parsing and filtering."""

from typing import Any, Dict, List

import polars as pl


def as_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if not text:
                continue
            out.append(text)
        return out
    text = str(value).strip()
    return [text] if text else []


def get_mode_cfg(config: Dict[str, Any], mode_name: str) -> Dict[str, Any]:
    modes = config.get("aggregation_modes", {})
    if not isinstance(modes, dict):
        raise TypeError("config['aggregation_modes'] must be a mapping.")
    if mode_name not in modes:
        available = ", ".join(sorted(modes.keys()))
        raise KeyError(f"Unknown mode '{mode_name}'. Available: {available}")
    mode_cfg = modes[mode_name]
    if not isinstance(mode_cfg, dict):
        raise TypeError(f"aggregation_modes.{mode_name} must be a mapping.")
    if not mode_cfg.get("enabled", True):
        print(f"Warning: aggregation_modes.{mode_name}.enabled is false (running anyway).")
    return mode_cfg


def coerce_value_for_dtype(value: Any, dtype: pl.DataType) -> Any:
    if dtype in [
        pl.Int64,
        pl.Int32,
        pl.Int16,
        pl.Int8,
        pl.UInt64,
        pl.UInt32,
        pl.UInt16,
        pl.UInt8,
    ]:
        try:
            return int(value)
        except (ValueError, TypeError):
            return value
    if dtype in [pl.Float64, pl.Float32]:
        try:
            return float(value)
        except (ValueError, TypeError):
            return value
    return value


def coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "y", "on"):
        return True
    if text in ("false", "0", "no", "n", "off"):
        return False
    return bool(default)


def apply_filters(df: pl.DataFrame, filters: Dict[str, Any]) -> pl.DataFrame:
    if not filters:
        return df

    filter_exprs = []
    for col_name, raw_value in filters.items():
        if col_name not in df.columns:
            print(f"Warning: Filter column '{col_name}' not found in data (skipping).")
            continue

        col_dtype = df[col_name].dtype
        col_value = coerce_value_for_dtype(raw_value, col_dtype)
        filter_exprs.append(pl.col(col_name) == col_value)
        print(f"Applying filter: {col_name} == {col_value!r}")

    if filter_exprs:
        df = df.filter(pl.all_horizontal(filter_exprs))
        print(f"Data shape after filtering: {df.shape}")
    return df


__all__ = [
    "apply_filters",
    "as_str_list",
    "coerce_bool",
    "coerce_value_for_dtype",
    "get_mode_cfg",
]
