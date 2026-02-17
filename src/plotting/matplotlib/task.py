from __future__ import annotations

"""Process-safe Matplotlib plot task entrypoints.

This module provides top-level functions that can be pickled and executed in a
`concurrent.futures.ProcessPoolExecutor`. The heavy implementation lives in
`src.plotting.matplotlib.common` and is intentionally kept close to the legacy
behavior.
"""

from typing import Any, Dict, Optional

from . import common as _common


def plot_worker_init(font_family: Optional[str]) -> None:
    _common._plot_worker_init(font_family)


def plot_task(task: Dict[str, Any]) -> None:
    _common._plot_task(task)

