from __future__ import annotations

"""Shared conversion helpers for matplotlib task/plot modules."""

from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np


def as_output_path(value: Any) -> Path:
    return Path(value)


def as_ndarray(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float)


def as_optional_ndarray(values: Any) -> Optional[np.ndarray]:
    if values is None:
        return None
    return np.asarray(values, dtype=float)


def as_tuple_key(key: Any) -> Tuple[Any, ...]:
    if isinstance(key, tuple):
        return key
    if isinstance(key, list):
        return tuple(key)
    if isinstance(key, np.ndarray):
        return tuple(key.tolist())
    return tuple(key)


def as_tuple_keys(keys: Sequence[Any]) -> List[Tuple[Any, ...]]:
    return [as_tuple_key(k) for k in keys]

