from __future__ import annotations

"""Repo-root discovery helpers for running scripts from subfolders.

Scripts in this repository are intended to be executed directly (e.g.
`python scripts/pipeline/run_visualizer.py`). Because the script file can live
at varying depths, we centralize repo-root detection and `sys.path` injection
here.
"""

import sys
from pathlib import Path
from typing import Optional


def repo_root(start: Optional[Path] = None) -> Path:
    here = Path(start or __file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "config.yaml").exists() or (candidate / ".git").exists():
            return candidate
    return here.parents[2]


def ensure_repo_on_path(start: Optional[Path] = None) -> Path:
    root = repo_root(start)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root

