from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class MplStyle:
    dpi: int = 300
    font_family: Optional[str] = "NanumGothic"
    axes_unicode_minus: bool = False

    grid_alpha: float = 0.3
    title_fontsize: int = 14
    title_fontweight: str = "bold"
    label_fontsize: int = 10
    tick_labelsize: int = 9
    legend_fontsize: int = 9
    legend_loc: str = "best"
    legend_framealpha: float = 0.8

    savefig_bbox_inches: str = "tight"
    savefig_facecolor: str = "white"


def setup_matplotlib(style: MplStyle) -> None:
    """
    Configure matplotlib for headless figure saving with consistent defaults.

    Intended usage in each vis script:
      - call this before heavy plotting
      - keep per-plot style parameters in the script (do not read plot_style from config.yaml)
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # noqa: F401

    plt.rcParams["axes.unicode_minus"] = bool(style.axes_unicode_minus)
    if style.font_family:
        plt.rcParams["font.family"] = style.font_family


def finalize_axes(ax, style: MplStyle) -> None:
    ax.grid(True, alpha=style.grid_alpha, linestyle="--")
    ax.tick_params(labelsize=style.tick_labelsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_subplot_legend(ax, style: MplStyle) -> None:
    """
    Always prefer subplot-level legends (ax.legend) over figure-level legends.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(
        fontsize=style.legend_fontsize,
        loc=style.legend_loc,
        framealpha=style.legend_framealpha,
    )


def save_png(fig, output_path: Path, style: MplStyle) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=style.dpi,
        bbox_inches=style.savefig_bbox_inches,
        facecolor=style.savefig_facecolor,
    )
