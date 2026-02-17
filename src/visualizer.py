"""Compatibility layer for legacy imports.

Historically, the main visualizer lived at `src/visualizer.py`.
During refactoring it moved to `src/core/visualizer.py`, but some scripts
still import from `src.visualizer`.

This module re-exports the public API to keep old entrypoints working.
"""

from __future__ import annotations

from .core.visualizer import AggregatedSignalVisualizer, ensure_output_dirs

__all__ = ["AggregatedSignalVisualizer", "ensure_output_dirs"]

