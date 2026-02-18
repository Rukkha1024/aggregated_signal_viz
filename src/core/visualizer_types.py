from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import polars as pl


@dataclass(frozen=True)
class ResampledGroup:
    meta_df: pl.DataFrame
    tensor: np.ndarray  # (n_trials, n_channels, target_len)
    channels: List[str]
