"""Generate onset-timing error-bar (forest) plots.

Merges parquet data with a features CSV, computes per-muscle onset statistics,
and renders error-bar figures grouped by facet/hue from config.yaml aggregation_modes.
Classes: VizConfig. Key functions: load_and_merge_data(), process_stats(), main().
"""
from __future__ import annotations

import argparse
import os
import sys
import string
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator
import numpy as np
import polars as pl

_HERE = Path(__file__).resolve()
_SCRIPTS_DIR = next(p for p in (_HERE.parent, *_HERE.parents) if p.name == "scripts")
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _repo import ensure_repo_on_path

repo_root = ensure_repo_on_path()

from src.core.config import load_config, resolve_path, strip_bom_columns  # noqa: E402
from src.plotting.matplotlib.errorbar_onset import plot_onset_timing_errorbar  # noqa: E402

# =============================================================================
# 1. VISUALIZATION CONFIGURATION (CONSTANTS)
# =============================================================================

@dataclass
class VizConfig:
    """
    Onset summary-plot configuration (visualization + fixed columns).

    Important
    ---------
    - Aggregation behavior (filter/groupby/overlay/filename_pattern/output_dir) is controlled
      exclusively via `config.yaml: aggregation_modes`.
    - `VizConfig` intentionally does NOT contain aggregation-related knobs (no facet/hue/filters).
      It only defines:
        1) which numeric column to visualize (features CSV),
        2) the muscle-name column (features CSV),
        3) plot appearance (colors/markers/fonts/layout),
        4) the plot-type subfolder name under `output.base_dir`.
    """

    # -----------------------------------------------------------------------------
    # Data Columns (features CSV)
    # -----------------------------------------------------------------------------
    # Any numeric feature column in the features CSV can be used here.
    target_column: str = "TKEO_AGLR_emg_onset_timing"
    # Column name in CSV holding muscle labels like 'TA', 'SOL' etc.
    muscle_column_in_feature: str = "emg_channel"
    x_label: str = "Onset Timing (ms)"
    # Base filename used when `aggregation_modes.<mode>.filename_pattern` is not provided.
    file_prefix: str = "onset_viz"

    # -----------------------------------------------------------------------------
    # Plot Dimensions & Style (visualization-only; not in config.yaml)
    # -----------------------------------------------------------------------------
    # Plot Dimensions & Style
    # (가로, 세로) inches. facet(패널) 개수/라벨 길이에 따라 조절 권장.
    figure_size: Tuple[float, float] = (12, 10)
    dpi: int = 300  # 저장 이미지 해상도. 논문/보고서용이면 300 이상 권장
    font_family: str = "NanumGothic"  # 한글 깨짐 방지용 폰트 패밀리(시스템에 설치되어 있어야 함)
    
    # Colors (Hue Palette)
    # hue(= aggregation_modes로부터 유도된 overlay 그룹) 범주별 색상 팔레트(고대비 기본값).
    # hue 값들은 정렬(sorted)된 순서대로 색상/마커가 할당됩니다.
    # 범주 수가 많아지면 리스트를 확장하세요.
    colors: List[str] = field(default_factory=lambda: [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ])
    
    # Bar & Marker Style
    marker_size: int = 10  # 평균값 마커 크기(points)
    
    # Marker Palette (Hue별 자동 할당)
    # hue 값들을 정렬 후 순서대로 마커 할당 (색상 팔레트와 동일한 방식)
    # 마커 종류: o(원), ^(위삼각), s(사각형), D(다이아몬드), v(아래삼각), *(별), <(왼삼각), >(오른삼각), p(오각형), h(육각형)
    marker_palette: List[str] = field(default_factory=lambda: [
        "o",    # 원
        "^",    # 위쪽 삼각형
        "s",    # 사각형
        "D",    # 다이아몬드
        "v",    # 아래쪽 삼각형
        "*",    # 별
        "<",    # 왼쪽 삼각형
        ">",    # 오른쪽 삼각형
        "p",    # 오각형
        "h",    # 육각형
    ])
    
    bar_width: float = 0.6  # 한 muscle tick에 할당되는 전체 폭(여러 hue가 있으면 내부에서 분할)
    marker_alpha: float = 1.0  # 마커 투명도(0~1)
    
    # Error Bar Detail Style
    errorbar_alpha: float = 0.8  # 에러바 선/캡 투명도(0~1)
    cap_size: int = 4  # 에러바 끝 캡(cap) 크기
    errorbar_linewidth: float = 2  # 오차 막대 선 두께
    errorbar_capthick: float = 3  # 오차 막대 캡 선 두께
    
    # Text Style
    text_fontsize: int = 8  # bar 옆에 표시되는 값(예: mean±std, n) 텍스트 크기
    text_offset_x: float = 5.0  # 텍스트를 x축 방향으로 밀어내는 오프셋(데이터 단위; onset timing이면 ms로 해석)
    title_fontsize: int = 20  # 전체 제목(suptitle) 크기
    xlabel_fontsize: int = 15  # x축 라벨 크기
    legend_fontsize: int = 10  # 범례 크기
    tick_labelsize: int = 20  # y축 tick 라벨 크기(근육명)
    xtick_labelsize: int = 20  # x축 tick 라벨 크기(숫자)
    
    # Grid Style
    grid_alpha: float = 1  # 그리드 선 투명도(0~1)
    grid_linestyle: str = "--"  # 그리드 선 스타일 ('-', '--', '-.', ':', 'solid', 'dashed', 'dashdot', 'dotted')
    grid_linewidth: float = 0.8  # 그리드 선 두께
    grid_color: str = "gray"  # 그리드 선 색상
    
    # Layout
    layout_rect: Tuple[float, float, float, float] = (0, 0, 1, 0.95)  # tight_layout 적용 영역(left, bottom, right, top)
    # Title
    title: Optional[str] = None  # None이면 각 facet의 기본 제목 사용
    show_title: bool = False  # True, False
    show_legend: bool = False
    show_counts_text: bool = False  # e.g., "count/total_count" next to marker
    show_xlabel: bool = False
    show_xtick_labels: bool = False
    show_ytick_labels: bool = True

    # Sorting
    # Muscle ordering within each facet:
    #   - None: use order from `config.signal_groups.emg.columns` (or data-driven fallback)
    #   - "ascending"/"descending": sort by facet-wise mean (averaged across hues)
    sort_by_mean: Optional[Literal["ascending", "descending"]] = None
    
    # Output
    # NOTE: output base dir comes from config.yaml (output.base_dir).
    output_dir: str = "onset"  # plot-type subfolder under output.base_dir (e.g., output/onset)

VIZ_CFG = VizConfig()

# Faceted error-bar plot grid policy (hardcoded by design).
# 중앙제어(config.yaml)에서 관리하지 않습니다.
# 너무 많은 facet이 생겨도 figure가 가로로 무한히 늘어나지 않도록 상한을 둡니다.
ONSET_SUMMARY_MAX_COLS = 6

def _apply_onset_show_options_from_config(config: Dict[str, Any]) -> None:
    """
    vis_onset show_* policy:
      - By default, follow `plot_style.common.show_*` (single source of truth).
    """
    common = (config.get("plot_style") or {}).get("common") or {}
    if not isinstance(common, dict):
        common = {}

    def _as_bool(value: Any) -> Optional[bool]:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in ("true", "1", "yes", "y", "on"):
            return True
        if text in ("false", "0", "no", "n", "off"):
            return False
        return None

    mapping = {
        "show_title": ("show_subplot_titles", VIZ_CFG.show_title),
        "show_legend": ("show_legend", VIZ_CFG.show_legend),
        "show_xlabel": ("show_xlabel", VIZ_CFG.show_xlabel),
        "show_xtick_labels": ("show_xtick_labels", VIZ_CFG.show_xtick_labels),
        "show_ytick_labels": ("show_ytick_labels", VIZ_CFG.show_ytick_labels),
    }

    for attr, (key, fallback) in mapping.items():
        raw = common.get(key, fallback)
        b = _as_bool(raw)
        if b is None:
            continue
        setattr(VIZ_CFG, attr, bool(b))


def _as_str_list(value: Any) -> List[str]:
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


def _get_mode_cfg(config: Dict[str, Any], mode_name: str) -> Dict[str, Any]:
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


def _coerce_value_for_dtype(value: Any, dtype: pl.DataType) -> Any:
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


def _coerce_bool(value: Any, default: bool = False) -> bool:
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


def _apply_filters(df: pl.DataFrame, filters: Dict[str, Any]) -> pl.DataFrame:
    if not filters:
        return df

    filter_exprs = []
    for col_name, raw_value in filters.items():
        if col_name not in df.columns:
            print(f"Warning: Filter column '{col_name}' not found in data (skipping).")
            continue

        col_dtype = df[col_name].dtype
        col_value = _coerce_value_for_dtype(raw_value, col_dtype)
        filter_exprs.append(pl.col(col_name) == col_value)
        print(f"Applying filter: {col_name} == {col_value!r}")

    if filter_exprs:
        df = df.filter(pl.all_horizontal(filter_exprs))
        print(f"Data shape after filtering: {df.shape}")
    return df


def _parse_x_axis_zeroing_config(
    config: Dict[str, Any],
) -> Tuple[bool, str, str]:
    id_cfg = (config.get("data") or {}).get("id_columns") or {}
    onset_col = str(id_cfg.get("onset") or "platform_onset").strip() or "platform_onset"

    zero_cfg = config.get("x_axis_zeroing", {})
    if not isinstance(zero_cfg, dict):
        return False, onset_col, onset_col

    enabled = _coerce_bool(zero_cfg.get("enabled"), False)
    ref_event = str(zero_cfg.get("reference_event") or onset_col).strip() or onset_col
    return enabled, ref_event, onset_col


def _combo_key_expr(cols: List[str]) -> pl.Expr:
    parts: List[pl.Expr] = []
    for col in cols:
        parts.append(
            pl.concat_str(
                [
                    pl.lit(f"{col}="),
                    pl.col(col).cast(pl.Utf8),
                ],
                separator="",
            )
        )
    return pl.concat_str(parts, separator=", ")


def _infer_facet_and_hue_columns(
    mode_cfg: Dict[str, Any],
    df_columns: List[str],
) -> Tuple[Optional[str], Optional[str], List[str], List[str]]:
    """
    Returns (facet_col, hue_col, facet_fields, hue_fields).

    Policy (aggregation_modes-compatible, onset-friendly):
      - Prefer `overlay_within` for hue (if present), else `color_by`.
      - Use remaining `groupby` fields (minus `overlay_within`) as facet candidates.
      - If multiple candidates exist, build a combined key column.
    """
    overlay_within = _as_str_list(mode_cfg.get("overlay_within"))
    color_by = _as_str_list(mode_cfg.get("color_by"))
    groupby = _as_str_list(mode_cfg.get("groupby"))
    overlay = bool(mode_cfg.get("overlay", False))

    hue_fields: List[str] = []
    if overlay and overlay_within:
        hue_fields = overlay_within
    elif color_by:
        hue_fields = color_by

    hue_fields = [c for c in hue_fields if c in df_columns]
    if len(hue_fields) == 1:
        hue_col = hue_fields[0]
    elif len(hue_fields) > 1:
        hue_col = "__hue_combo"
    else:
        hue_col = None

    facet_candidates = groupby
    if overlay and overlay_within:
        facet_candidates = [c for c in facet_candidates if c not in overlay_within]
    if hue_col and hue_col in facet_candidates:
        facet_candidates = [c for c in facet_candidates if c != hue_col]
    facet_candidates = [c for c in facet_candidates if c in df_columns]

    if len(facet_candidates) == 1:
        facet_col = facet_candidates[0]
    elif len(facet_candidates) > 1:
        facet_col = "__facet_combo"
    else:
        facet_col = None

    return facet_col, hue_col, facet_candidates, hue_fields


def _format_mode_filename(
    pattern: str,
    df: pl.DataFrame,
    filters: Dict[str, Any],
) -> str:
    fmt = string.Formatter()
    fields = [field_name for _, field_name, _, _ in fmt.parse(pattern) if field_name]

    context: Dict[str, Any] = {"signal_group": "onset"}
    for key, value in (filters or {}).items():
        context[key] = value

    for field in fields:
        if field in context:
            continue
        if field not in df.columns:
            continue

        series = df.get_column(field)
        unique_vals = series.drop_nulls().unique().to_list()
        if len(unique_vals) == 1:
            context[field] = unique_vals[0]

    missing = [f for f in fields if f not in context]
    if missing:
        raise KeyError(f"filename_pattern missing keys: {missing}")

    filename = pattern.format(**context)
    filename = filename.replace(os.sep, "_")
    return filename


def load_and_merge_data(
    config: Dict[str, Any],
    base_dir: Path,
    extra_input_columns: Optional[List[str]] = None,
) -> pl.DataFrame:
    """Load input (parquet) + features (csv) and join by trial keys."""
    input_path_str = config["data"]["input_file"]
    input_path = resolve_path(base_dir, input_path_str)
    
    print(f"Loading input file: {input_path}")
    lf_input = pl.scan_parquet(str(input_path))
    
    key_cols = [
        config["data"]["id_columns"]["subject"],
        config["data"]["id_columns"]["trial"],
        config["data"]["id_columns"]["velocity"],
    ]
    extra_cols: List[str] = []
    for col in (extra_input_columns or []):
        name = str(col).strip()
        if not name or name in key_cols or name in extra_cols:
            continue
        extra_cols.append(name)

    input_schema = lf_input.collect_schema()
    available_input_cols = set(input_schema.keys())
    agg_exprs: List[pl.Expr] = []
    for col in extra_cols:
        if col not in available_input_cols:
            continue
        agg_exprs.append(pl.col(col).max().alias(col))
    df_trials = lf_input.group_by(key_cols, maintain_order=False).agg(agg_exprs).collect()

    feature_path_str = config["data"]["features_file"]
    feature_path = resolve_path(base_dir, feature_path_str)
    
    print(f"Loading features file: {feature_path}")
    df_features = pl.read_csv(str(feature_path))

    df_features = strip_bom_columns(df_features)

    merged = df_trials.join(df_features, on=key_cols, how="inner")
    
    print(f"Merged Data Shape: {merged.shape}")
    return merged

def process_stats(
    df: pl.DataFrame, 
    muscle_order: List[str], 
    facet_col: Optional[str], 
    hue_col: Optional[str],
    trial_key_cols: List[str],
    target_col: Optional[str] = None,
) -> pl.DataFrame:
    """Group by (facet, hue, muscle) and compute mean/std/count (+ total_count).

    Notes
    -----
    - `count`: number of unique trials with valid (non-null, non-NaN) values for the target column.
    - `total_count`: number of unique trials in the (facet, hue) group (same denominator across muscles).
    """
    target_col = str(target_col or VIZ_CFG.target_column)
    muscle_col = VIZ_CFG.muscle_column_in_feature
    
    print(f"\n[Process Stats] Target Column: {target_col}")
    print(f"[Process Stats] Initial Data Shape: {df.shape}")

    missing_trial_cols = [c for c in trial_key_cols if c not in df.columns]
    if missing_trial_cols:
        raise KeyError(
            "Missing trial key columns required for counting unique trials: "
            f"{missing_trial_cols}. Available columns: {df.columns}"
        )

    df_muscle_filtered = df.filter(pl.col(muscle_col).is_in(muscle_order))
    print(f"[Process Stats] Shape after filtering muscles: {df_muscle_filtered.shape}")

    # Drop rows where target is NULL/NaN (polars treats them differently).
    df_clean = df_muscle_filtered.filter(
        pl.col(target_col).is_not_null() &
        ~pl.col(target_col).is_nan()
    )
    print(f"[Process Stats] Shape after filtering nulls/NaNs('{target_col}'): {df_clean.shape}")

    if df_clean.is_empty():
        print("[Process Stats] WARNING: Data is empty after filtering!")
        return pl.DataFrame()

    group_cols = [muscle_col]
    if facet_col and facet_col in df.columns:
        group_cols.append(facet_col)
    if hue_col and hue_col in df.columns:
        group_cols.append(hue_col)

    denom_group_cols = [c for c in group_cols if c != muscle_col]
    if not denom_group_cols:
        denom_group_cols = ["_denom_dummy"]
        df_muscle_filtered = df_muscle_filtered.with_columns(pl.lit(1).alias("_denom_dummy"))
        df_clean = df_clean.with_columns(pl.lit(1).alias("_denom_dummy"))
        group_cols.append("_denom_dummy")

    trial_key_expr = pl.struct(trial_key_cols).alias("_trial_key")

    totals_df = df_muscle_filtered.group_by(denom_group_cols).agg([
        trial_key_expr.n_unique().alias("total_count")
    ])

    stats = df_clean.group_by(group_cols).agg([
        pl.col(target_col).mean().alias("mean"),
        pl.col(target_col).std().alias("std"),
        trial_key_expr.n_unique().alias("count"),
    ])

    stats = stats.join(totals_df, on=denom_group_cols, how="left")
    stats = stats.with_columns(pl.col("total_count").fill_null(0))

    print("\n[Process Stats] Calculated Stats Summary:")
    print(stats)

    return stats

# 3. PLOTTING

def resolve_summary_grid_layout(
    n_panels: int,
) -> Tuple[int, int]:
    """
    Summary-plot grid policy (hardcoded, onset-only):
      - cols is capped by ONSET_SUMMARY_MAX_COLS
      - rows is derived from n_panels and cols
    """
    if n_panels <= 0:
        return 1, 1

    max_cols = max(1, int(ONSET_SUMMARY_MAX_COLS))
    cols = min(max_cols, n_panels)
    rows = int(ceil(n_panels / cols))
    return rows, cols


def plot_onset_timing(
    stats_df: pl.DataFrame,
    muscle_order: List[str],
    facet_col: Optional[str],
    hue_col: Optional[str],
    output_dir: Path,
    output_filename: str,
    config: Dict[str, Any],
    sort_by_mean: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
):
    # Keep VizConfig / visualization constants in the script; delegate rendering to `src/`.
    plot_onset_timing_errorbar(
        stats_df=stats_df,
        muscle_order=muscle_order,
        facet_col=facet_col,
        hue_col=hue_col,
        output_dir=output_dir,
        output_filename=output_filename,
        viz_cfg=VIZ_CFG,
        max_cols=int(ONSET_SUMMARY_MAX_COLS),
        sort_by_mean=sort_by_mean,
        filters=filters,
    )

# 4. MAIN EXECUTION

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Onset Timing (Vertical Forest Plot)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--velocity", type=float, default=None, help="Filter by velocity (optional)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    config_path = Path(args.config)
    config = load_config(config_path)
    _apply_onset_show_options_from_config(config)
    base_dir = config_path.parent

    x_axis_zeroing_enabled, x_axis_zeroing_reference_event, onset_col = _parse_x_axis_zeroing_config(config)
    extra_input_cols: List[str] = []
    if x_axis_zeroing_enabled:
        extra_input_cols.extend([onset_col, x_axis_zeroing_reference_event])

    df = load_and_merge_data(config, base_dir, extra_input_columns=extra_input_cols)

    try:
        muscle_order = config["signal_groups"]["emg"]["columns"]
        print(f"Using muscle order from config: {muscle_order}")
    except KeyError:
        print("Warning: EMG columns not found in config. Using data-driven muscle list.")
        muscle_order = sorted(df[VIZ_CFG.muscle_column_in_feature].unique().to_list())
        print(f"Using data-driven muscle order: {muscle_order}")

    id_cols_cfg = config.get("data", {}).get("id_columns", {})
    trial_key_cols = [
        id_cols_cfg.get("subject", "subject"),
        id_cols_cfg.get("velocity", "velocity"),
        id_cols_cfg.get("trial", "trial_num"),
    ]
    input_path = resolve_path(base_dir, config["data"]["input_file"])
    input_schema = pl.scan_parquet(str(input_path)).collect_schema()
    input_columns = set(input_schema.keys())
    zero_ref_from_input = x_axis_zeroing_reference_event in input_columns

    modes_cfg = config.get("aggregation_modes", {})
    if not isinstance(modes_cfg, dict) or not modes_cfg:
        raise KeyError("config.yaml must define non-empty 'aggregation_modes' to run vis_onset.")

    output_base_dir = Path(config.get("output", {}).get("base_dir", "output"))
    if not output_base_dir.is_absolute():
        output_base_dir = (base_dir / output_base_dir).resolve()

    ran_any = False
    for mode_name, mode_cfg in modes_cfg.items():
        if not isinstance(mode_cfg, dict):
            print(f"Skipping mode '{mode_name}': config is not a mapping.")
            continue
        if not mode_cfg.get("enabled", True):
            continue

        print(f"\n=== [Mode] {mode_name} ===")
        ran_any = True

        active_filters = mode_cfg.get("filter", {})
        if active_filters is None:
            active_filters = {}
        if not isinstance(active_filters, dict):
            raise TypeError(f"aggregation_modes.{mode_name}.filter must be a mapping.")

        df_mode = _apply_filters(df, dict(active_filters))

        if args.velocity is not None:
            if "velocity" in df_mode.columns:
                print(f"Filtering velocity = {args.velocity}")
                df_mode = df_mode.filter(pl.col("velocity") == args.velocity)
                print(f"Data shape after velocity filtering: {df_mode.shape}")

        target_col_for_mode = VIZ_CFG.target_column
        if x_axis_zeroing_enabled:
            if VIZ_CFG.target_column not in df_mode.columns:
                raise KeyError(
                    f"[x_axis_zeroing] target column '{VIZ_CFG.target_column}' not found for mode '{mode_name}'."
                )
            df_mode = df_mode.with_columns(
                pl.col(VIZ_CFG.target_column).cast(pl.Float64, strict=False).fill_nan(None).alias("__target_base")
            )

            if x_axis_zeroing_reference_event == onset_col:
                df_mode = df_mode.with_columns(pl.col("__target_base").alias("__target_zeroed"))
            else:
                if x_axis_zeroing_reference_event not in df_mode.columns:
                    raise KeyError(
                        "[x_axis_zeroing] reference_event column not found in vis_onset data: "
                        f"'{x_axis_zeroing_reference_event}' (mode='{mode_name}')"
                    )
                if zero_ref_from_input:
                    if onset_col not in df_mode.columns:
                        raise KeyError(
                            f"[x_axis_zeroing] onset column '{onset_col}' is required but missing (mode='{mode_name}')."
                        )
                    ref_ms_expr = (
                        (
                            pl.col(x_axis_zeroing_reference_event).cast(pl.Float64, strict=False)
                            - pl.col(onset_col).cast(pl.Float64, strict=False)
                        )
                        * (1000.0 / float(config["data"].get("mocap_sample_rate", 100)))
                    )
                else:
                    ref_ms_expr = pl.col(x_axis_zeroing_reference_event).cast(pl.Float64, strict=False)

                df_mode = df_mode.with_columns(ref_ms_expr.fill_nan(None).alias("__xzero_ref_ms"))
                valid_ref = df_mode.filter(pl.col("__xzero_ref_ms").is_not_null() & ~pl.col("__xzero_ref_ms").is_nan())
                if valid_ref.is_empty():
                    raise ValueError(
                        "[x_axis_zeroing] reference_event has no valid values in vis_onset: "
                        f"'{x_axis_zeroing_reference_event}' (mode='{mode_name}')"
                    )
                df_mode = df_mode.with_columns((pl.col("__target_base") - pl.col("__xzero_ref_ms")).alias("__target_zeroed"))

            target_col_for_mode = "__target_zeroed"

        facet_col, hue_col, facet_fields, hue_fields = _infer_facet_and_hue_columns(
            mode_cfg=mode_cfg,
            df_columns=df_mode.columns,
        )
        if hue_col == "__hue_combo" and hue_fields:
            df_mode = df_mode.with_columns(_combo_key_expr(hue_fields).alias("__hue_combo"))
        if facet_col == "__facet_combo" and facet_fields:
            df_mode = df_mode.with_columns(_combo_key_expr(facet_fields).alias("__facet_combo"))

        stats = process_stats(
            df_mode,
            muscle_order=muscle_order,
            facet_col=facet_col,
            hue_col=hue_col,
            trial_key_cols=trial_key_cols,
            target_col=target_col_for_mode,
        )

        if stats.is_empty():
            print(f"[Mode] {mode_name}: no data after filtering (skipping).")
            continue

        print(f"[Mode] {mode_name}: final stats ready for plotting.")

        mode_out_subdir = mode_cfg.get("output_dir") or str(mode_name)
        output_dir = output_base_dir / VIZ_CFG.output_dir / str(mode_out_subdir)

        filename_pattern = mode_cfg.get("filename_pattern")
        if filename_pattern is not None and str(filename_pattern).strip():
            try:
                filename = _format_mode_filename(str(filename_pattern), df=df_mode, filters=dict(active_filters))
            except KeyError as e:
                print(f"Warning: {e}. Falling back to default filename policy.")
                filename = f"{VIZ_CFG.file_prefix}_{mode_name}.png"
        else:
            filename = f"{VIZ_CFG.file_prefix}_{mode_name}.png"

        plot_onset_timing(
            stats_df=stats,
            muscle_order=muscle_order,
            facet_col=facet_col,
            hue_col=hue_col,
            output_dir=output_dir,
            output_filename=filename,
            config=config,
            sort_by_mean=VIZ_CFG.sort_by_mean,
            filters=dict(active_filters),
        )

    if not ran_any:
        print("No enabled modes found in config.yaml aggregation_modes. Nothing to do.")

if __name__ == "__main__":
    main()
