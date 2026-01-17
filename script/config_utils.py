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


def get_output_base_dir(base_dir: Path, config: Dict[str, Any]) -> Path:
    output_base = (config.get("output") or {}).get("base_dir", "output")
    return resolve_path(base_dir, output_base)


def resolve_output_dir(base_dir: Path, config: Dict[str, Any], output_dir: str | Path) -> Path:
    """
    Resolve a mode's output directory with backward-compatible semantics.

    Preferred config form:
      - output.base_dir: "output"
      - aggregation_modes.<mode>.output_dir: "step_TF"
        -> <base_dir>/output/step_TF

    Legacy config form (still supported):
      - aggregation_modes.<mode>.output_dir: "output/step_TF"
        -> <base_dir>/output/step_TF (no double "output/output")
    """
    out_dir_path = Path(output_dir)
    if out_dir_path.is_absolute():
        return out_dir_path

    raw_base = (config.get("output") or {}).get("base_dir", "output")
    raw_base_path = Path(raw_base)
    output_base_dir = get_output_base_dir(base_dir, config)

    if not raw_base_path.is_absolute():
        base_parts = raw_base_path.parts
        if base_parts and out_dir_path.parts[: len(base_parts)] == base_parts:
            return resolve_path(base_dir, out_dir_path)

    return (output_base_dir / out_dir_path).resolve()


def get_frame_ratio(data_cfg: Dict[str, Any]) -> int:
    device_rate = float(data_cfg.get("device_sample_rate", 1000))
    mocap_rate = float(data_cfg.get("mocap_sample_rate", 100))
    configured = data_cfg.get("frame_ratio")
    if configured is not None:
        try:
            return int(configured)
        except (TypeError, ValueError):
            pass
    return int(device_rate / mocap_rate)


def bom_rename_map(columns: Sequence[str]) -> Dict[str, str]:
    rename: Dict[str, str] = {}
    for col in columns:
        if col.startswith(_BOM):
            rename[col] = col.lstrip(_BOM)
    return rename


def strip_bom_columns(df: pl.DataFrame) -> pl.DataFrame:
    rename = bom_rename_map(df.columns)
    return df.rename(rename) if rename else df
