from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_SCRIPTS_DIR = next(p for p in (_HERE.parent, *_HERE.parents) if p.name == "scripts")
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _repo import ensure_repo_on_path


def main() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    ensure_repo_on_path()
    from src.plotting.matplotlib.common import (
        _apply_frame_tick_labels,
        _apply_window_definition_xticks,
        _ensure_time_zero_xtick,
    )

    fig, ax = plt.subplots(figsize=(6, 2), dpi=100)
    x = np.linspace(0.0, 1.0, 100)
    ax.plot(x, np.sin(2 * np.pi * x))

    # Build a scenario where 0ms (frame=0) is inside the time window
    # but not included by window boundary ticks.
    time_start_frame = -200.0
    time_end_frame = 800.0
    window_spans = [{"start": 0.25, "end": 0.35}]

    ticks = _apply_window_definition_xticks(ax, window_spans)
    ticks = _ensure_time_zero_xtick(
        ax,
        tick_positions=ticks,
        time_start_frame=time_start_frame,
        time_end_frame=time_end_frame,
    )
    _apply_frame_tick_labels(ax, time_start_frame=time_start_frame, time_end_frame=time_end_frame)

    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_xticklabels()]
    plt.close(fig)

    if "0" not in labels:
        raise AssertionError(f"Expected tick label '0' not found. labels={labels}, ticks={ticks}")
    print("OK: tick label 0 is present")


if __name__ == "__main__":
    main()
