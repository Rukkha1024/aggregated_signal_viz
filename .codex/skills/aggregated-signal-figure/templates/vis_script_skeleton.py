from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import polars as pl
import yaml


@dataclass(frozen=True)
class VizConfig:
    """
    Keep visualization parameters here (do not use config.yaml plot_style).
    """

    plot_type: str = "onset"  # output subdir under output.base_dir + key under figure_layout.summary_plots
    dpi: int = 300
    font_family: str = "NanumGothic"

    grid_alpha: float = 0.3
    tick_labelsize: int = 9
    label_fontsize: int = 10
    title_fontsize: int = 14
    title_fontweight: str = "bold"
    title_pad: int = 5

    legend_fontsize: int = 9
    legend_loc: str = "best"
    legend_framealpha: float = 0.8

    savefig_bbox_inches: str = "tight"
    savefig_facecolor: str = "white"

    # Example palette for grouped/overlay plots
    colors: List[str] = field(
        default_factory=lambda: [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
    )


VIZ = VizConfig()


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_matplotlib() -> None:
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # noqa: F401

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = VIZ.font_family


def resolve_output_dir(config: Dict[str, Any], base_dir: Path) -> Path:
    output_base = Path(config.get("output", {}).get("base_dir", "output"))
    if not output_base.is_absolute():
        output_base = (base_dir / output_base).resolve()
    return output_base / VIZ.plot_type


def resolve_summary_grid_layout(config: Dict[str, Any], n_panels: int) -> Tuple[int, int]:
    if n_panels <= 0:
        return 1, 1
    max_cols_raw = (
        config.get("figure_layout", {}).get("summary_plots", {}).get(VIZ.plot_type, {}).get("max_cols")
    )
    try:
        max_cols = int(max_cols_raw) if max_cols_raw is not None else n_panels
    except (TypeError, ValueError):
        max_cols = n_panels
    max_cols = max(1, max_cols)
    cols = min(max_cols, n_panels)
    rows = int(ceil(n_panels / cols))
    return rows, cols


def add_subplot_legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(
        handles,
        labels,
        fontsize=VIZ.legend_fontsize,
        loc=VIZ.legend_loc,
        framealpha=VIZ.legend_framealpha,
    )


def finalize_axes(ax) -> None:
    ax.grid(True, alpha=VIZ.grid_alpha, linestyle="--")
    ax.tick_params(labelsize=VIZ.tick_labelsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_png(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=VIZ.dpi,
        bbox_inches=VIZ.savefig_bbox_inches,
        facecolor=VIZ.savefig_facecolor,
    )


def load_and_merge_data(config: Dict[str, Any], base_dir: Path) -> pl.DataFrame:
    """
    Example loader: parquet + features csv join by trial keys.
    Customize per plot.
    """
    input_path = Path(config["data"]["input_file"])
    if not input_path.is_absolute():
        input_path = (base_dir / input_path).resolve()

    features_path = Path(config["data"]["features_file"])
    if not features_path.is_absolute():
        features_path = (base_dir / features_path).resolve()

    id_cfg = config["data"]["id_columns"]
    key_cols = [id_cfg["subject"], id_cfg["trial"], id_cfg["velocity"]]

    df_trials = pl.scan_parquet(str(input_path)).select(key_cols).unique().collect()
    df_features = pl.read_csv(str(features_path)).rename(lambda c: c.lstrip("\ufeff"))
    return df_trials.join(df_features, on=key_cols, how="inner")


def plot_summary_grid_example(stats_df: pl.DataFrame, facet_col: str, hue_col: Optional[str], output_path: Path) -> None:
    """
    Example: facet panels on a grid (summary plot).
    Ensure each subplot has a legend (ax.legend).
    """
    import matplotlib.pyplot as plt

    facets = sorted(stats_df[facet_col].unique().to_list())
    rows, cols = resolve_summary_grid_layout(config={}, n_panels=len(facets))

    fig, axes = plt.subplots(rows, cols, figsize=(12 * (cols / 2 + 0.5), 10 * rows), dpi=VIZ.dpi, squeeze=False)
    axes_flat = axes.flatten()

    for ax, facet in zip(axes_flat, facets):
        sub = stats_df.filter(pl.col(facet_col) == facet)
        if hue_col and hue_col in sub.columns:
            hues = sorted(sub[hue_col].unique().to_list())
            for i, hue in enumerate(hues):
                sub_h = sub.filter(pl.col(hue_col) == hue)
                ax.plot(sub_h["x"].to_numpy(), sub_h["y"].to_numpy(), color=VIZ.colors[i], label=str(hue))
        else:
            ax.plot(sub["x"].to_numpy(), sub["y"].to_numpy(), label="value")

        ax.set_title(str(facet), fontsize=VIZ.title_fontsize, fontweight=VIZ.title_fontweight, pad=VIZ.title_pad)
        finalize_axes(ax)
        add_subplot_legend(ax)

    for ax in axes_flat[len(facets) :]:
        ax.axis("off")

    fig.tight_layout()
    save_png(fig, output_path)
    plt.close(fig)


def plot_channel_grid_example(
    channels: List[str],
    grid_layout: Tuple[int, int],
    x: np.ndarray,
    ys_by_channel: Dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """
    Example: channel grid plot using config-driven grid_layout.
    Ensure each subplot has a legend (ax.legend).
    """
    import matplotlib.pyplot as plt

    rows, cols = grid_layout
    fig, axes = plt.subplots(rows, cols, figsize=(18, 9), dpi=VIZ.dpi)
    axes_flat = axes.flatten()

    for ax, ch in zip(axes_flat, channels):
        y = ys_by_channel.get(ch)
        if y is None:
            ax.axis("off")
            continue
        ax.plot(x, y, color="gray", linewidth=1.2, alpha=0.8, label="mean")
        ax.set_title(ch, fontsize=VIZ.title_fontsize, fontweight=VIZ.title_fontweight, pad=VIZ.title_pad)
        finalize_axes(ax)
        add_subplot_legend(ax)

    for ax in axes_flat[len(channels) :]:
        ax.axis("off")

    fig.tight_layout()
    save_png(fig, output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot skeleton (copy and customize)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    base_dir = config_path.parent

    setup_matplotlib()
    _ = resolve_output_dir(config, base_dir)  # used by real scripts

    # Implement your data loading / processing / plotting here.
    # This skeleton intentionally does not run end-to-end.


if __name__ == "__main__":
    main()

