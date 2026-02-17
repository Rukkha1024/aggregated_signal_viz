from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any, Dict, Tuple


def resolve_summary_grid_layout(config: Dict[str, Any], plot_type: str, n_panels: int) -> Tuple[int, int]:
    """
    Summary grid layout policy for aggregated_signal_viz.

    Reads:
      config.figure_layout.summary_plots.<plot_type>.max_cols
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


def resolve_output_dir(config: Dict[str, Any], base_dir: Path, plot_type: str) -> Path:
    output_base = Path(config.get("output", {}).get("base_dir", "output"))
    if not output_base.is_absolute():
        output_base = (base_dir / output_base).resolve()
    return output_base / plot_type

