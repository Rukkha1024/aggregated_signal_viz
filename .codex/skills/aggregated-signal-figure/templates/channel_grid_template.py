from __future__ import annotations

from typing import Any, Dict, Tuple


def resolve_channel_grid_layout(config: Dict[str, Any], signal_group: str) -> Tuple[int, int]:
    """
    Channel grid layout policy for aggregated_signal_viz.

    Reads:
      config.signal_groups.<signal_group>.grid_layout
    """
    grid_layout = config["signal_groups"][signal_group]["grid_layout"]
    rows, cols = int(grid_layout[0]), int(grid_layout[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"Invalid grid_layout for {signal_group}: {grid_layout!r}")
    return rows, cols

