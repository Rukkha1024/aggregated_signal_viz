from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import polars as pl
import yaml

# =============================================================================
# 1. VISUALIZATION CONFIGURATION (CONSTANTS)
# =============================================================================

@dataclass
class VizConfig:
    """Configuration for visualization style and parameters."""

    # File & Column Settings
    target_column: str = "TKEO_AGLR_emg_onset_timing"
    muscle_column_in_feature: str = "emg_channel"  # Column name in CSV holding 'TA', 'SOL' etc.

    # Facet & Hue Configuration (Default values)
    facet_column: Optional[str] = None  # No faceting by default
    hue_column: str = "step_TF"

    # Data Filtering (자유롭게 컬럼명: 값 추가/제거 가능)
    # 예: {"mixed": "1", "age_group": "young", "velocity": 10}
    # 필터 안 쓰려면 빈 딕셔너리로: {}
    filters: Dict[str, Any] = field(default_factory=lambda: {
        "mixed": "1",
        "age_group": "young"
    })

    # Plot Dimensions & Style
    figure_size: Tuple[int, int] = (12, 10)  # (가로, 세로) 인치. facet 개수/라벨 길이에 따라 조절 권장
    dpi: int = 300  # 저장 이미지 해상도. 논문/보고서용이면 300 이상 권장
    font_family: str = "NanumGothic"  # 한글 깨짐 방지용 폰트 패밀리(시스템에 설치되어 있어야 함)
    
    # Colors (Hue Palette)
    # hue_col 범주별 색상 팔레트(고대비 기본값). 범주 수가 많아지면 리스트를 확장하세요.
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
    errorbar_alpha: float = 0.6  # 에러바 선/캡 투명도(0~1)
    
    # Error Bar Detail Style
    cap_size: int = 4  # 에러바 끝 캡(cap) 크기
    errorbar_linewidth: float = 1.5  # 오차 막대 선 두께
    errorbar_capthick: float = 1.0  # 오차 막대 캡 선 두께
    
    # Text Style
    text_fontsize: int = 8  # bar 옆에 표시되는 값(예: mean±std, n) 텍스트 크기
    text_offset_x: float = 5.0  # 텍스트를 x축 방향으로 밀어내는 오프셋(데이터 단위; onset timing이면 ms로 해석)
    title_fontsize: int = 20  # 전체 제목(suptitle) 크기
    xlabel_fontsize: int = 15  # x축 라벨 크기
    legend_fontsize: int = 10  # 범례 크기
    tick_labelsize: int = 20  # y축 tick 라벨 크기(근육명)
    xtick_labelsize: int = 20  # x축 tick 라벨 크기(숫자)
    
    # Layout
    grid_alpha: float = 0.3  # 그리드 선 투명도(가독성 조절)
    layout_rect: Tuple[float, float, float, float] = (0, 0, 1, 0.95)  # tight_layout 적용 영역(left, bottom, right, top)

    # Output
    # NOTE: output base dir comes from config.yaml (output.base_dir).
    output_dir: str = "onset"  # plot-type subfolder under output.base_dir (e.g., output/onset)

VIZ_CFG = VizConfig()

# 2. DATA LOADER & PROCESSOR

def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_and_merge_data(config: Dict[str, Any], base_dir: Path) -> pl.DataFrame:
    """Load input (parquet) + features (csv) and join by trial keys."""
    input_path_str = config["data"]["input_file"]
    input_path = (base_dir / input_path_str).resolve() if not Path(input_path_str).is_absolute() else Path(input_path_str)
    
    print(f"Loading input file: {input_path}")
    lf_input = pl.scan_parquet(str(input_path))
    
    id_cols = list(config["data"]["id_columns"].values())
    key_cols = [config["data"]["id_columns"]["subject"], 
                config["data"]["id_columns"]["trial"], 
                config["data"]["id_columns"]["velocity"]]
    
    df_trials = lf_input.select(key_cols).unique().collect()

    feature_path_str = config["data"]["features_file"]
    feature_path = (base_dir / feature_path_str).resolve() if not Path(feature_path_str).is_absolute() else Path(feature_path_str)
    
    print(f"Loading features file: {feature_path}")
    df_features = pl.read_csv(str(feature_path))
    
    df_features = df_features.rename({c: c.lstrip("\ufeff") for c in df_features.columns})

    # Inner join keeps only trials present in both files.
    for col in key_cols:
        if col in df_features.columns and col in df_trials.columns:
            pass

    merged = df_trials.join(df_features, on=key_cols, how="inner")
    
    print(f"Merged Data Shape: {merged.shape}")
    return merged

def process_stats(
    df: pl.DataFrame, 
    muscle_order: List[str], 
    facet_col: Optional[str], 
    hue_col: Optional[str]
) -> pl.DataFrame:
    """Group by (facet, hue, muscle) and compute mean/std/count (+ total_count)."""
    target_col = VIZ_CFG.target_column
    muscle_col = VIZ_CFG.muscle_column_in_feature
    
    print(f"\n[Process Stats] Target Column: {target_col}")
    print(f"[Process Stats] Initial Data Shape: {df.shape}")

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

    totals_df = df_muscle_filtered.group_by(group_cols).agg([
        pl.col(target_col).count().alias("total_count")
    ])

    stats = df_clean.group_by(group_cols).agg([
        pl.col(target_col).mean().alias("mean"),
        pl.col(target_col).std().alias("std"),
        pl.col(target_col).count().alias("count")
    ])

    stats = stats.join(totals_df, on=group_cols, how="left")
    stats = stats.with_columns(pl.col("total_count").fill_null(0))

    print("\n[Process Stats] Calculated Stats Summary:")
    print(stats)

    return stats

# 3. PLOTTING

def resolve_summary_grid_layout(
    config: Dict[str, Any],
    plot_type: str,
    n_panels: int,
) -> Tuple[int, int]:
    """
    Summary-plot grid policy (config-driven):
      - figure_layout.summary_plots.<plot_type>.max_cols controls columns upper bound
      - rows is derived from n_panels and cols
    """
    if n_panels <= 0:
        return 1, 1

    max_cols_raw = (
        config.get("figure_layout", {})
        .get("summary_plots", {})
        .get(plot_type, {})
        .get("max_cols")
    )
    try:
        max_cols = int(max_cols_raw) if max_cols_raw is not None else n_panels
    except (TypeError, ValueError):
        max_cols = n_panels
    max_cols = max(1, max_cols)

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
):
    """Generate horizontal error-bar plot and save to output."""
    import matplotlib.pyplot as plt
    
    # Build title from active filters
    filter_title_parts = []
    for col_name, col_value in VIZ_CFG.filters.items():
        filter_title_parts.append(f"{col_name}={col_value}")
    filter_title = ", ".join(filter_title_parts) if filter_title_parts else "All Data"
    
    if facet_col and facet_col in stats_df.columns:
        facets = sorted(stats_df[facet_col].unique().to_list())
    else:
        facets = [filter_title]
        stats_df = stats_df.with_columns(pl.lit(filter_title).alias("_facet_dummy"))
        facet_col = "_facet_dummy"

    if hue_col and hue_col in stats_df.columns:
        hues = sorted(stats_df[hue_col].unique().to_list())
    else:
        hues = ["_single_hue"]
    
    n_facets = len(facets)
    n_hues = len(hues)

    grid_rows, grid_cols = resolve_summary_grid_layout(
        config=config,
        plot_type=VIZ_CFG.output_dir,
        n_panels=n_facets,
    )
    
    fig, axes = plt.subplots(
        grid_rows,
        grid_cols,
        figsize=(
            VIZ_CFG.figure_size[0] * (grid_cols / 2 + 0.5),
            VIZ_CFG.figure_size[1] * grid_rows,
        ),
        dpi=VIZ_CFG.dpi, 
        sharey=True,
        squeeze=False
    )
    axes_flat = axes.flatten()
    
    plt.rcParams["font.family"] = VIZ_CFG.font_family
    plt.rcParams["axes.unicode_minus"] = False

    group_height = VIZ_CFG.bar_width / n_hues
    
    existing_muscles = set(stats_df[VIZ_CFG.muscle_column_in_feature].unique().to_list())
    valid_muscles = [m for m in muscle_order if m in existing_muscles]
    valid_muscles_reversed = valid_muscles[::-1]
    
    y_indices = np.arange(len(valid_muscles_reversed))
    muscle_to_y = {m: i for i, m in enumerate(valid_muscles_reversed)}

    for ax, facet_val in zip(axes_flat, facets):
        facet_data = stats_df.filter(pl.col(facet_col) == facet_val)
        
        for hue_idx, hue_val in enumerate(hues):
            if hue_col:
                subset = facet_data.filter(pl.col(hue_col) == hue_val)
                label = f"{hue_col}: {hue_val}"
            else:
                subset = facet_data
                label = None
                
            if subset.is_empty():
                continue
                
            muscles = subset[VIZ_CFG.muscle_column_in_feature].to_list()
            means = subset["mean"].to_numpy()
            stds = subset["std"].to_numpy()
            counts = subset["count"].to_numpy()  # valid count
            total_counts = subset["total_count"].to_numpy()  # total count
            
            offset = -1 * (hue_idx - (n_hues - 1) / 2) * group_height

            ys = [muscle_to_y[m] + offset for m in muscles]
            color = VIZ_CFG.colors[hue_idx % len(VIZ_CFG.colors)]
            
            # Auto-assign marker from palette (same logic as colors)
            marker = VIZ_CFG.marker_palette[hue_idx % len(VIZ_CFG.marker_palette)]
            
            # Plot errorbar and get handles to set separate alphas
            line, caps, bars = ax.errorbar(
                means,
                ys,
                xerr=stds,
                fmt=marker,
                markersize=VIZ_CFG.marker_size,
                capsize=VIZ_CFG.cap_size,
                capthick=VIZ_CFG.errorbar_capthick,
                elinewidth=VIZ_CFG.errorbar_linewidth,
                color=color,
                label=""  # No label in errorbar (use proxy artist for legend)
            )
            
            # Set separate alphas for marker and errorbars
            line.set_alpha(VIZ_CFG.marker_alpha)
            for bar in bars:
                bar.set_alpha(VIZ_CFG.errorbar_alpha)
            for cap in caps:
                cap.set_alpha(VIZ_CFG.errorbar_alpha)
            
            for idx, (x, y, c, tc) in enumerate(zip(means, ys, counts, total_counts)):
                if np.isnan(x):
                    continue
                text_x = x + stds[idx] + VIZ_CFG.text_offset_x
                ax.text(
                    text_x,
                    y,
                    f"{int(c)}/{int(tc)}",
                    va="center",
                    ha="left",
                    fontsize=VIZ_CFG.text_fontsize,
                    color=color
                )

        title = f"{facet_val}" if facet_col != "_facet_dummy" else filter_title
        ax.set_title(title, fontsize=VIZ_CFG.title_fontsize, fontweight="bold")
        ax.set_yticks(y_indices)
        ax.set_yticklabels(valid_muscles_reversed, fontsize=VIZ_CFG.tick_labelsize)
        ax.set_xlabel("Onset Timing (ms)", fontsize=VIZ_CFG.xlabel_fontsize)
        ax.tick_params(axis="x", labelsize=VIZ_CFG.xtick_labelsize)
        ax.grid(True, axis="x", alpha=VIZ_CFG.grid_alpha, linestyle="--")
        
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes_flat[len(facets):]:
        ax.axis("off")

    if hue_col:
        # Create proxy artists for legend (marker only, no errorbar)
        legend_handles = []
        for hue_idx, hue_val in enumerate(hues):
            color = VIZ_CFG.colors[hue_idx % len(VIZ_CFG.colors)]
            marker = VIZ_CFG.marker_palette[hue_idx % len(VIZ_CFG.marker_palette)]
            label = f"{hue_col}: {hue_val}"

            # Create proxy artist with marker only
            proxy_artist = mlines.Line2D(
                [], [],
                marker=marker,
                color=color,
                markersize=VIZ_CFG.marker_size,
                linestyle='None',
                label=label
            )
            legend_handles.append(proxy_artist)

        # Add legend to first facet only
        first_ax = axes_flat[0]
        first_ax.legend(
            handles=legend_handles,
            loc="best",
            frameon=False,
            fontsize=VIZ_CFG.legend_fontsize,
        )

    plt.tight_layout(rect=VIZ_CFG.layout_rect)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / output_filename
    print(f"Saving plot to: {out_path}")
    plt.savefig(out_path, dpi=VIZ_CFG.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

# 4. MAIN EXECUTION

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Onset Timing (Vertical Forest Plot)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--facet",
        type=str,
        default=VIZ_CFG.facet_column,
        help=f"Column for Faceting (Subplots). Default: {VIZ_CFG.facet_column}"
    )
    parser.add_argument(
        "--hue",
        type=str,
        default=VIZ_CFG.hue_column,
        help=f"Column for Hue (Colors). Default: {VIZ_CFG.hue_column}"
    )
    parser.add_argument("--velocity", type=float, default=None, help="Filter by velocity (optional)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    config_path = Path(args.config)
    config = load_config(config_path)
    base_dir = config_path.parent
    
    df = load_and_merge_data(config, base_dir)
    
    # Apply filters from VizConfig.filters
    filter_exprs = []
    for col_name, col_value in VIZ_CFG.filters.items():
        if col_name in df.columns:
            # Auto-convert col_value type to match column dtype
            col_dtype = df[col_name].dtype
            if col_dtype in [pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8]:
                try:
                    col_value = int(col_value)
                except (ValueError, TypeError):
                    pass
            elif col_dtype in [pl.Float64, pl.Float32]:
                try:
                    col_value = float(col_value)
                except (ValueError, TypeError):
                    pass
            
            filter_exprs.append(pl.col(col_name) == col_value)
            print(f"Applying filter: {col_name} == {col_value!r}")
        else:
            print(f"Warning: Filter column '{col_name}' not found in data")
    
    if filter_exprs:
        df = df.filter(pl.all_horizontal(filter_exprs))
        print(f"Data shape after filtering: {df.shape}")
    
    if args.velocity is not None:
        if "velocity" in df.columns:
            print(f"Filtering velocity = {args.velocity}")
            df = df.filter(pl.col("velocity") == args.velocity)
    
    try:
        muscle_order = config["signal_groups"]["emg"]["columns"]
        print(f"Using muscle order from config: {muscle_order}")
    except KeyError:
        print("Warning: EMG columns not found in config. Using data-driven muscle list.")
        muscle_order = sorted(df[VIZ_CFG.muscle_column_in_feature].unique().to_list())
        print(f"Using data-driven muscle order: {muscle_order}")
        
    stats = process_stats(
        df, 
        muscle_order=muscle_order, 
        facet_col=args.facet, 
        hue_col=args.hue
    )
    
    if stats.is_empty():
        print("No data available after filtering. Exiting.")
        return

    print("Final Stats for Plotting:")
    print(stats)

    # Build filename from filters
    fname_parts = ["onset_viz"]
    for col_name, col_value in VIZ_CFG.filters.items():
        fname_parts.append(f"{col_name}-{col_value}")
    if args.facet: 
        fname_parts.append(f"facet-{args.facet}")
    if args.hue: 
        fname_parts.append(f"hue-{args.hue}")
    filename = "_".join(fname_parts) + ".png"

    output_base_dir = Path(config.get("output", {}).get("base_dir", "output"))
    if not output_base_dir.is_absolute():
        output_base_dir = (base_dir / output_base_dir).resolve()
    output_dir = output_base_dir / VIZ_CFG.output_dir
    
    plot_onset_timing(
        stats_df=stats,
        muscle_order=muscle_order,
        facet_col=args.facet,
        hue_col=args.hue,
        output_dir=output_dir,
        output_filename=filename,
        config=config,
    )

if __name__ == "__main__":
    main()
