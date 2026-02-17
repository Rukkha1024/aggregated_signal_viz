from __future__ import annotations

"""Onset summary error-bar plot (Matplotlib).

This module contains the reusable rendering function for the legacy onset
summary plot (`scripts/plotting/errorbar/vis_onset.py`). Visualization constants
and stylistic choices remain in the script (VizConfig), while this module
focuses on generating the figure deterministically from precomputed summary
statistics.
"""

from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl


def resolve_summary_grid_layout(n_panels: int, *, max_cols: int) -> Tuple[int, int]:
    """
    Summary-plot grid policy (onset-only):
      - cols is capped by max_cols
      - rows is derived from n_panels and cols
    """
    if n_panels <= 0:
        return 1, 1

    cols = min(max(1, int(max_cols)), n_panels)
    rows = int(ceil(n_panels / cols))
    return rows, cols


def plot_onset_timing_errorbar(
    *,
    stats_df: pl.DataFrame,
    muscle_order: List[str],
    facet_col: Optional[str],
    hue_col: Optional[str],
    output_dir: Path,
    output_filename: str,
    viz_cfg: Any,
    max_cols: int = 6,
    sort_by_mean: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Path:
    """Render and save an onset timing horizontal error-bar summary plot."""

    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    plt.rcParams["font.family"] = str(getattr(viz_cfg, "font_family", "sans-serif"))
    plt.rcParams["axes.unicode_minus"] = False

    muscle_col = str(getattr(viz_cfg, "muscle_column_in_feature", "emg_channel"))

    # Build title from active filters.
    filter_title_parts: List[str] = []
    for col_name, col_value in (filters or {}).items():
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

    grid_rows, grid_cols = resolve_summary_grid_layout(n_panels=n_facets, max_cols=max_cols)

    fig_w, fig_h = getattr(viz_cfg, "figure_size", (12, 10))
    fig, axes = plt.subplots(
        grid_rows,
        grid_cols,
        figsize=(float(fig_w) * (grid_cols / 2 + 0.5), float(fig_h) * grid_rows),
        dpi=int(getattr(viz_cfg, "dpi", 300)),
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes.flatten()

    bar_width = float(getattr(viz_cfg, "bar_width", 0.6))
    group_height = bar_width / max(1, n_hues)

    existing_muscles = set(stats_df[muscle_col].unique().to_list())
    valid_muscles = [m for m in muscle_order if m in existing_muscles]

    colors = list(getattr(viz_cfg, "colors", [])) or ["#1f77b4"]
    marker_palette = list(getattr(viz_cfg, "marker_palette", [])) or ["o"]

    for ax, facet_val in zip(axes_flat, facets):
        facet_data = stats_df.filter(pl.col(facet_col) == facet_val)

        # Sort muscles by mean value if requested (per facet).
        if sort_by_mean in ["ascending", "descending"]:
            facet_means = (
                facet_data.group_by(muscle_col).agg(pl.col("mean").mean().alias("avg_mean"))
            )
            muscle_mean_dict = dict(zip(facet_means[muscle_col].to_list(), facet_means["avg_mean"].to_list()))
            sorted_muscles = sorted(
                valid_muscles,
                key=lambda m: muscle_mean_dict.get(m, float("inf")),
                reverse=(sort_by_mean == "descending"),
            )
            valid_muscles_reversed = sorted_muscles[::-1]
        else:
            valid_muscles_reversed = valid_muscles[::-1]

        y_indices = np.arange(len(valid_muscles_reversed))
        muscle_to_y = {m: i for i, m in enumerate(valid_muscles_reversed)}

        for hue_idx, hue_val in enumerate(hues):
            if hue_col:
                subset = facet_data.filter(pl.col(hue_col) == hue_val)
            else:
                subset = facet_data

            if subset.is_empty():
                continue

            muscles = subset[muscle_col].to_list()
            means = subset["mean"].to_numpy()
            stds = subset["std"].to_numpy()
            counts = subset["count"].to_numpy()
            total_counts = subset["total_count"].to_numpy()

            offset = -1.0 * (hue_idx - (n_hues - 1) / 2) * group_height
            ys = [muscle_to_y[m] + offset for m in muscles]

            color = colors[hue_idx % len(colors)]
            marker = marker_palette[hue_idx % len(marker_palette)]

            line, caps, bars = ax.errorbar(
                means,
                ys,
                xerr=stds,
                fmt=marker,
                markersize=int(getattr(viz_cfg, "marker_size", 10)),
                capsize=int(getattr(viz_cfg, "cap_size", 4)),
                capthick=float(getattr(viz_cfg, "errorbar_capthick", 3)),
                elinewidth=float(getattr(viz_cfg, "errorbar_linewidth", 2)),
                color=color,
                label="",
            )

            line.set_alpha(float(getattr(viz_cfg, "marker_alpha", 1.0)))
            for bar in bars:
                bar.set_alpha(float(getattr(viz_cfg, "errorbar_alpha", 0.8)))
            for cap in caps:
                cap.set_alpha(float(getattr(viz_cfg, "errorbar_alpha", 0.8)))

            if bool(getattr(viz_cfg, "show_counts_text", False)):
                text_offset_x = float(getattr(viz_cfg, "text_offset_x", 5.0))
                text_fontsize = int(getattr(viz_cfg, "text_fontsize", 8))
                for idx, (x, y, c, tc) in enumerate(zip(means, ys, counts, total_counts)):
                    if np.isnan(x):
                        continue
                    text_x = x + stds[idx] + text_offset_x
                    ax.text(
                        text_x,
                        y,
                        f"{int(c)}/{int(tc)}",
                        va="center",
                        ha="left",
                        fontsize=text_fontsize,
                        color=color,
                        fontfamily=str(getattr(viz_cfg, "font_family", "sans-serif")),
                    )

        default_title = f"{facet_val}" if facet_col != "_facet_dummy" else filter_title
        title = getattr(viz_cfg, "title", None) or default_title
        if bool(getattr(viz_cfg, "show_title", False)):
            ax.set_title(
                title,
                fontsize=int(getattr(viz_cfg, "title_fontsize", 20)),
                fontweight="bold",
                fontfamily=str(getattr(viz_cfg, "font_family", "sans-serif")),
            )

        ax.set_yticks(y_indices)
        if bool(getattr(viz_cfg, "show_ytick_labels", True)):
            ax.set_yticklabels(
                valid_muscles_reversed,
                fontsize=int(getattr(viz_cfg, "tick_labelsize", 20)),
                fontfamily=str(getattr(viz_cfg, "font_family", "sans-serif")),
            )
        else:
            ax.tick_params(axis="y", labelleft=False)

        if bool(getattr(viz_cfg, "show_xlabel", False)):
            ax.set_xlabel(
                str(getattr(viz_cfg, "x_label", "Onset Timing (ms)")),
                fontsize=int(getattr(viz_cfg, "xlabel_fontsize", 15)),
                fontfamily=str(getattr(viz_cfg, "font_family", "sans-serif")),
            )

        ax.tick_params(axis="x", labelsize=int(getattr(viz_cfg, "xtick_labelsize", 20)))
        if not bool(getattr(viz_cfg, "show_xtick_labels", False)):
            ax.tick_params(axis="x", labelbottom=False)

        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.grid(
            True,
            axis="x",
            alpha=float(getattr(viz_cfg, "grid_alpha", 1.0)),
            linestyle=str(getattr(viz_cfg, "grid_linestyle", "--")),
            linewidth=float(getattr(viz_cfg, "grid_linewidth", 0.8)),
            color=str(getattr(viz_cfg, "grid_color", "gray")),
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes_flat[len(facets) :]:
        ax.axis("off")

    if hue_col and bool(getattr(viz_cfg, "show_legend", False)):
        legend_handles = []
        for hue_idx, hue_val in enumerate(hues):
            color = colors[hue_idx % len(colors)]
            marker = marker_palette[hue_idx % len(marker_palette)]
            label = f"{hue_col}: {hue_val}"
            proxy_artist = mlines.Line2D(
                [],
                [],
                marker=marker,
                color=color,
                markersize=int(getattr(viz_cfg, "marker_size", 10)),
                linestyle="None",
                label=label,
            )
            legend_handles.append(proxy_artist)

        axes_flat[0].legend(
            handles=legend_handles,
            loc="best",
            frameon=False,
            prop={
                "family": str(getattr(viz_cfg, "font_family", "sans-serif")),
                "size": int(getattr(viz_cfg, "legend_fontsize", 10)),
            },
        )

    plt.tight_layout(rect=tuple(getattr(viz_cfg, "layout_rect", (0, 0, 1, 0.95))))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / str(output_filename)
    print(f"Saving plot to: {out_path}")
    plt.savefig(out_path, dpi=int(getattr(viz_cfg, "dpi", 300)), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path

