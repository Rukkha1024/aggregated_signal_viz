from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
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
    facet_column: str = "age_group"
    hue_column: str = "step_TF"

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
    marker_fmt: str = "o"  # 평균값 마커 모양(Matplotlib marker fmt)
    marker_size: int = 6  # 평균값 마커 크기(points)
    cap_size: int = 4  # 에러바 끝 캡(cap) 크기
    bar_width: float = 0.6  # 한 muscle tick에 할당되는 전체 폭(여러 hue가 있으면 내부에서 분할)
    alpha: float = 0.8  # 마커/에러바 투명도(0~1)
    
    # Text Style
    text_fontsize: int = 8  # bar 옆에 표시되는 값(예: mean±std, n) 텍스트 크기
    text_offset_x: float = 5.0  # 텍스트를 x축 방향으로 밀어내는 오프셋(데이터 단위; onset timing이면 ms로 해석)
    title_fontsize: int = 14  # 전체 제목(suptitle) 크기
    label_fontsize: int = 10  # 축 라벨 크기
    tick_labelsize: int = 9  # tick 라벨 크기(근육명/범례 등)
    
    # Layout
    grid_alpha: float = 0.3  # 그리드 선 투명도(가독성 조절)
    layout_rect: Tuple[float, float, float, float] = (0, 0, 1, 0.95)  # tight_layout 적용 영역(left, bottom, right, top)

    # Output
    output_dir: str = "output/onset"  # 결과 이미지 저장 폴더(상대경로 기준: 프로젝트 루트)

VIZ_CFG = VizConfig()

# =============================================================================
# 2. DATA LOADER & PROCESSOR
# =============================================================================

def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_and_merge_data(config: Dict[str, Any], base_dir: Path) -> pl.DataFrame:
    """
    Loads 'input_file' (Parquet) and 'features_file' (CSV), then merges them.
    This ensures we have both the task filtering info and the calculated features.
    """
    # 1. Load Parquet (Input File) - Main source for Trial IDs
    input_path_str = config["data"]["input_file"]
    input_path = (base_dir / input_path_str).resolve() if not Path(input_path_str).is_absolute() else Path(input_path_str)
    
    print(f"Loading input file: {input_path}")
    lf_input = pl.scan_parquet(str(input_path))
    
    # Select only necessary ID columns from input to avoid memory bloat
    # We assume 'final_dataset.csv' has the metrics, but we use input for filtering context if needed.
    id_cols = list(config["data"]["id_columns"].values())
    # Ensure distinct trial identification
    key_cols = [config["data"]["id_columns"]["subject"], 
                config["data"]["id_columns"]["trial"], 
                config["data"]["id_columns"]["velocity"]]
    
    # Get unique trials from input file (aggregated by key)
    # This acts as a filter for valid trials present in the signal data
    df_trials = lf_input.select(key_cols).unique().collect()

    # 2. Load Features (CSV) - Source of 'TKEO_AGLR_emg_onset_timing'
    feature_path_str = config["data"]["features_file"]
    feature_path = (base_dir / feature_path_str).resolve() if not Path(feature_path_str).is_absolute() else Path(feature_path_str)
    
    print(f"Loading features file: {feature_path}")
    df_features = pl.read_csv(str(feature_path))
    
    # Remove BOM if present
    df_features = df_features.rename({c: c.lstrip("\ufeff") for c in df_features.columns})

    # 3. Merge
    # We prefer Inner Join to keep only trials that exist in both
    # Cast key columns to ensure types match (csv often reads as int, parquet as float or int)
    for col in key_cols:
        if col in df_features.columns and col in df_trials.columns:
            # simple cast to matching types if needed, usually polars handles this well
            pass

    merged = df_trials.join(df_features, on=key_cols, how="inner")
    
    print(f"Merged Data Shape: {merged.shape}")
    # print(f"Merged Columns: {merged.columns}")
    return merged

def process_stats(
    df: pl.DataFrame, 
    muscle_order: List[str], 
    facet_col: Optional[str], 
    hue_col: Optional[str]
) -> pl.DataFrame:
    """
    Filters data, groups by (Facet, Hue, Muscle), and calculates Mean, STD, Count.
    Excludes NaNs in the target column.
    """
    target_col = VIZ_CFG.target_column
    muscle_col = VIZ_CFG.muscle_column_in_feature
    
    print(f"\n[Process Stats] Target Column: {target_col}")
    print(f"[Process Stats] Initial Data Shape: {df.shape}")

    # 1. Filter by configured muscles FIRST
    df_muscle_filtered = df.filter(pl.col(muscle_col).is_in(muscle_order))
    print(f"[Process Stats] Shape after filtering muscles: {df_muscle_filtered.shape}")

    # 2. Filter Valid Data (for valid_count)
    # Drop rows where target is null/NaN
    # CRITICAL: polars distinguishes between NULL and NaN
    # - drop_nulls() only removes SQL NULL values
    # - We must also filter out float NaN values explicitly
    df_clean = df_muscle_filtered.filter(
        pl.col(target_col).is_not_null() &
        ~pl.col(target_col).is_nan()
    )
    print(f"[Process Stats] Shape after filtering nulls/NaNs('{target_col}'): {df_clean.shape}")

    if df_clean.is_empty():
        print("[Process Stats] WARNING: Data is empty after filtering!")
        return pl.DataFrame()

    # 2. Grouping
    group_cols = [muscle_col]
    if facet_col and facet_col in df.columns:
        group_cols.append(facet_col)
    if hue_col and hue_col in df.columns:
        group_cols.append(hue_col)

    # 3. Calculate total_count (before NaN filtering)
    totals_df = df_muscle_filtered.group_by(group_cols).agg([
        pl.col(target_col).count().alias("total_count")
    ])

    # 4. Calculate valid stats (after NaN filtering)
    stats = df_clean.group_by(group_cols).agg([
        pl.col(target_col).mean().alias("mean"),
        pl.col(target_col).std().alias("std"),
        pl.col(target_col).count().alias("count")
    ])

    # 5. Merge total_count into stats
    stats = stats.join(totals_df, on=group_cols, how="left")
    stats = stats.with_columns(pl.col("total_count").fill_null(0))

    print("\n[Process Stats] Calculated Stats Summary:")
    print(stats)

    return stats

# =============================================================================
# 3. PLOTTING
# =============================================================================

def plot_onset_timing(
    stats_df: pl.DataFrame,
    muscle_order: List[str],
    facet_col: Optional[str],
    hue_col: Optional[str],
    output_filename: str
):
    """
    Generates the Horizontal Error Bar Plot (Vertical Forest Plot style).
    """
    import matplotlib.pyplot as plt
    
    # Setup Data Structures
    # Determine unique facets
    if facet_col and facet_col in stats_df.columns:
        facets = sorted(stats_df[facet_col].unique().to_list())
    else:
        facets = ["All Data"]
        stats_df = stats_df.with_columns(pl.lit("All Data").alias("_facet_dummy"))
        facet_col = "_facet_dummy"

    # Determine unique hues
    if hue_col and hue_col in stats_df.columns:
        hues = sorted(stats_df[hue_col].unique().to_list())
    else:
        hues = ["_single_hue"]
    
    n_facets = len(facets)
    n_hues = len(hues)
    
    # Configure Figure
    fig, axes = plt.subplots(
        1, n_facets, 
        figsize=(VIZ_CFG.figure_size[0] * (n_facets / 2 + 0.5), VIZ_CFG.figure_size[1]), 
        dpi=VIZ_CFG.dpi, 
        sharey=True,
        squeeze=False
    )
    axes = axes.flatten()
    
    # Font setup
    plt.rcParams["font.family"] = VIZ_CFG.font_family
    plt.rcParams["axes.unicode_minus"] = False

    # Calculate dodging (offset) logic
    # Total width available per muscle tick = VIZ_CFG.bar_width
    # Width per hue group
    group_height = VIZ_CFG.bar_width / n_hues
    
    # Filter muscles that actually exist in the stats (Remove empty rows)
    existing_muscles = set(stats_df[VIZ_CFG.muscle_column_in_feature].unique().to_list())
    valid_muscles = [m for m in muscle_order if m in existing_muscles]
    # Reverse order for plotting (Top=First in list)
    valid_muscles_reversed = valid_muscles[::-1]
    
    y_indices = np.arange(len(valid_muscles_reversed))
    muscle_to_y = {m: i for i, m in enumerate(valid_muscles_reversed)}

    # --- Plotting Loop ---
    for ax, facet_val in zip(axes, facets):
        # Filter for current facet
        facet_data = stats_df.filter(pl.col(facet_col) == facet_val)
        
        for hue_idx, hue_val in enumerate(hues):
            # Filter for current hue
            if hue_col:
                subset = facet_data.filter(pl.col(hue_col) == hue_val)
                label = f"{hue_col}: {hue_val}"
            else:
                subset = facet_data
                label = None
                
            if subset.is_empty():
                continue
                
            # Extract data
            muscles = subset[VIZ_CFG.muscle_column_in_feature].to_list()
            means = subset["mean"].to_numpy()
            stds = subset["std"].to_numpy()
            counts = subset["count"].to_numpy()  # valid count
            total_counts = subset["total_count"].to_numpy()  # total count
            
            # Calculate Y positions
            # Center is y_indices. Shift up/down based on hue_idx
            # Example: 2 hues. idx 0 -> shift +0.15, idx 1 -> shift -0.15
            # Formula: offset = (hue_idx - (n_hues - 1) / 2) * group_height * -1 (inverted axis logic)
            # Actually, standard logic: 
            # Top of group = index + width/2
            # Bottom of group = index - width/2
            
            offset = (hue_idx - (n_hues - 1) / 2) * group_height
            # Invert offset direction because Y axis 0 is usually bottom, but we correspond list index
            # Let's stick to standard math: 
            # if hue_idx is large (last), we want it lower visually if we fill from top? 
            # Usually: Hue 0 (Top), Hue 1 (Bottom) within the band.
            # So offset should decrease as hue_idx increases.
            offset = -1 * offset 

            ys = [muscle_to_y[m] + offset for m in muscles]
            color = VIZ_CFG.colors[hue_idx % len(VIZ_CFG.colors)]
            
            # Error Bar Plot
            ax.errorbar(
                means, 
                ys, 
                xerr=stds, 
                fmt=VIZ_CFG.marker_fmt,
                markersize=VIZ_CFG.marker_size,
                capsize=VIZ_CFG.cap_size,
                color=color,
                label=label,
                alpha=VIZ_CFG.alpha
            )
            
            # Add 'valid/total' text
            for idx, (x, y, c, tc) in enumerate(zip(means, ys, counts, total_counts)):
                # Skip NaN values
                if np.isnan(x):
                    continue
                # Place text to the right of the error bar
                # x + std + offset
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

        # Style the Ax
        # Dynamic title generation
        title = f"{facet_col}: {facet_val}"
        ax.set_title(title, fontsize=VIZ_CFG.title_fontsize, fontweight="bold")
        ax.set_yticks(y_indices)
        ax.set_yticklabels(valid_muscles_reversed, fontsize=VIZ_CFG.tick_labelsize)
        ax.set_xlabel("Onset Timing (ms)", fontsize=VIZ_CFG.label_fontsize)
        ax.grid(True, axis="x", alpha=VIZ_CFG.grid_alpha, linestyle="--")
        
        # Remove top/right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Global Stuff
    # Add legend only to the first plot (or outside)
    if hue_col:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles, labels, 
                loc="lower center", 
                bbox_to_anchor=(0.5, 0.01), 
                ncol=n_hues, 
                frameon=False,
                fontsize=VIZ_CFG.label_fontsize
            )

    plt.tight_layout(rect=VIZ_CFG.layout_rect)
    
    # Save
    out_dir = Path(VIZ_CFG.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / output_filename
    print(f"Saving plot to: {out_path}")
    plt.savefig(out_path, dpi=VIZ_CFG.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

# =============================================================================
# 4. MAIN EXECUTION
# =============================================================================

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
    
    # 1. Load Config & Data
    config_path = Path(args.config)
    config = load_config(config_path)
    base_dir = config_path.parent
    
    df = load_and_merge_data(config, base_dir)
    
    # 2. Pre-filter (User constraints)
    # Filter velocity if specified in args or strictly use config logic if needed.
    # Here we support command line override for flexibility
    if args.velocity is not None:
        if "velocity" in df.columns:
            print(f"Filtering velocity = {args.velocity}")
            df = df.filter(pl.col("velocity") == args.velocity)
    
    # 3. Get Muscle Order from Config
    try:
        muscle_order = config["signal_groups"]["emg"]["columns"]
        print(f"Using muscle order from config: {muscle_order}")
    except KeyError:
        print("Warning: EMG columns not found in config. Using data-driven muscle list.")
        muscle_order = sorted(df[VIZ_CFG.muscle_column_in_feature].unique().to_list())
        print(f"Using data-driven muscle order: {muscle_order}")
        
    # 4. Calculate Stats
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

    # 5. Generate Plot
    # Construct filename
    fname_parts = ["onset_viz"]
    if args.facet: fname_parts.append(f"facet-{args.facet}")
    if args.hue: fname_parts.append(f"hue-{args.hue}")
    filename = "_".join(fname_parts) + ".png"
    
    plot_onset_timing(
        stats_df=stats,
        muscle_order=muscle_order,
        facet_col=args.facet,
        hue_col=args.hue,
        output_filename=filename
    )

if __name__ == "__main__":
    main()
