from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

from .config import resolve_path, strip_bom_columns
from ..plotting.matplotlib.common import _event_ms_col


class VisualizerMarkersMixin:
    def _collect_markers(
        self,
        signal_group: str,
        key: Tuple,
        group_fields: List[str],
        filter_cfg: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if self.features_df is None:
            return {}
        df = self.features_df
        if filter_cfg:
            # 다중 컬럼 필터링: {mixed: 1, age_group: "young"}
            # 모든 조건을 AND로 결합
            for col, val in filter_cfg.items():
                if col not in df.columns:
                    print(f"[markers] Warning: column '{col}' not found in features dataframe, skipping this filter")
                    continue
                if val is None:
                    print(f"[markers] skip filter: column '{col}' has None value; skipping marker collection")
                    return {}
                df = df.filter(pl.col(col) == val)
        for field, value in zip(group_fields, key):
            if field in df.columns:
                if value is None:
                    print(f"[markers] skip group: column '{field}' has None value; key={key}; skipping marker collection")
                    return {}
                df = df.filter(pl.col(field) == value)
        if df.is_empty():
            return {}
        if signal_group == "emg":
            return self._collect_emg_markers(df)
        if signal_group == "forceplate":
            return self._collect_forceplate_markers(df)
        return {}

    def _collect_emg_markers(self, df: pl.DataFrame) -> Dict[str, Dict[str, float]]:
        channels = self.config["signal_groups"]["emg"]["columns"]
        onset_cols = [
            "TKEO_AGLR_emg_onset_timing",
            "TKEO_TH_emg_onset_timing",
            "non_TKEO_TH_onset_timing",
        ]
        markers: Dict[str, Dict[str, float]] = {}
        for ch in channels:
            ch_df = df.filter(pl.col("emg_channel") == ch)
            if ch_df.is_empty():
                continue
            onset_val = None
            for col in onset_cols:
                if col in ch_df.columns:
                    onset_val = self._safe_mean(ch_df[col])
                    if onset_val is not None:
                        break
            max_val = self._safe_mean(ch_df["emg_max_amp_timing"]) if "emg_max_amp_timing" in ch_df.columns else None
            marker_info: Dict[str, float] = {}
            if onset_val is not None:
                marker_info["onset"] = onset_val
            if max_val is not None:
                marker_info["max"] = max_val
            if marker_info:
                markers[ch] = marker_info
        return markers

    def _collect_forceplate_markers(self, df: pl.DataFrame) -> Dict[str, Dict[str, float]]:
        mapping = {"Fx": "fx_onset_timing", "Fy": "fy_onset_timing", "Fz": "fz_onset_timing"}
        markers: Dict[str, Dict[str, float]] = {}
        for ch, col in mapping.items():
            if col in df.columns:
                onset_val = self._safe_mean(df[col])
                if onset_val is not None:
                    markers[ch] = {"onset": onset_val}
        return markers

    @staticmethod
    def _safe_mean(series: pl.Series) -> Optional[float]:
        arr = series.drop_nulls().to_numpy()
        if arr.size == 0:
            return None
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return None
        return float(arr.mean())

    def _load_features(self) -> Optional[pl.DataFrame]:
        features_path = self.config["data"].get("features_file")
        if not features_path:
            return None
        path = resolve_path(self.base_dir, features_path)
        if not path.exists():
            return None
        df = pl.read_csv(path)
        return strip_bom_columns(df)

    def _detect_emg_channel_specific_event_columns(self) -> set[str]:
        """
        Detect event columns that vary across `emg_channel` within the same subject-velocity-trial.

        These columns should be treated as channel-specific when rendering EMG event_vlines/windows.
        """
        if self.features_df is None:
            return set()
        df = self.features_df
        emg_channel_col = "emg_channel"
        if emg_channel_col not in df.columns:
            return set()

        subject_col = str(self.id_cfg.get("subject") or "").strip()
        velocity_col = str(self.id_cfg.get("velocity") or "").strip()
        trial_col = str(self.id_cfg.get("trial") or "").strip()
        key_cols = [subject_col, velocity_col, trial_col]
        if any(not c or c not in df.columns for c in key_cols):
            return set()

        candidates = [c for c in self.required_event_columns if c in df.columns]
        if not candidates:
            return set()

        base = df.select([*key_cols, emg_channel_col, *candidates])
        base = base.with_columns(
            [pl.col(c).cast(pl.Float64, strict=False).fill_nan(None).alias(c) for c in candidates]
        )
        agg_exprs = [pl.col(c).n_unique().alias(f"__nuniq_{c}") for c in candidates]
        grouped = base.group_by(key_cols, maintain_order=False).agg(agg_exprs)
        if grouped.is_empty():
            return set()

        max_cols = [pl.col(f"__nuniq_{c}").max().alias(f"__nuniq_{c}") for c in candidates]
        max_df = grouped.select(max_cols)
        if max_df.is_empty():
            return set()
        max_row = max_df.row(0)
        out: set[str] = set()
        for event_col, nuniq in zip(candidates, max_row):
            try:
                if nuniq is not None and int(nuniq) > 1:
                    out.add(event_col)
            except Exception:
                continue
        return out

    def _collect_feature_event_means_by_emg_channel(
        self,
        *,
        meta_df: pl.DataFrame,
        indices: np.ndarray,
        event_cols: Sequence[str],
    ) -> Dict[str, Dict[str, float]]:
        if self.features_df is None:
            return {}
        if indices.size == 0:
            return {}

        df = self.features_df
        emg_channel_col = "emg_channel"
        if emg_channel_col not in df.columns:
            return {}

        subject_col = str(self.id_cfg.get("subject") or "").strip()
        velocity_col = str(self.id_cfg.get("velocity") or "").strip()
        trial_col = str(self.id_cfg.get("trial") or "").strip()
        key_cols = [subject_col, velocity_col, trial_col]
        if any(not c or c not in df.columns or c not in meta_df.columns for c in key_cols):
            return {}

        requested = [str(c) for c in event_cols if str(c).strip() and str(c) in df.columns]
        if not requested:
            return {}

        keys_df = meta_df.select(key_cols)[indices].unique()

        base = df.select([*key_cols, emg_channel_col, *requested])
        casts: List[pl.Expr] = []
        for k in key_cols:
            dtype = meta_df.schema.get(k)
            if dtype is not None:
                casts.append(pl.col(k).cast(dtype, strict=False).alias(k))
        if casts:
            base = base.with_columns(casts)

        filtered = base.join(keys_df, on=key_cols, how="inner")
        if filtered.is_empty():
            return {}

        agg_exprs: List[pl.Expr] = []
        for col in requested:
            agg_exprs.append(pl.col(col).cast(pl.Float64, strict=False).fill_nan(None).mean().alias(col))

        grouped = filtered.group_by(emg_channel_col, maintain_order=False).agg(agg_exprs)
        if grouped.is_empty():
            return {}

        out: Dict[str, Dict[str, float]] = {}
        for row in grouped.iter_rows(named=True):
            ch = row.get(emg_channel_col)
            if ch is None:
                continue
            ch_name = str(ch)
            values: Dict[str, float] = {}
            for col in requested:
                val = row.get(col)
                if val is None:
                    continue
                try:
                    fval = float(val)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(fval):
                    continue
                values[col] = fval
            if values:
                out[ch_name] = values
        return out

    def _get_feature_event_ms_table(
        self,
        *,
        requested: Sequence[str],
        key_cols: Sequence[str],
        key_schema: Dict[str, Any],
    ) -> Optional[pl.DataFrame]:
        """
        subject-velocity-trial 단위로 feature 이벤트를 platform_onset 기준 ms로 해석한 테이블을 생성하고(캐시),
        재사용합니다.

        출력 컬럼명은 `_event_ms_col(<event_col>)` 규칙을 따릅니다.
        """
        if self.features_df is None:
            return None

        requested_cols = [str(c) for c in requested if str(c).strip()]
        if not requested_cols:
            return None
        requested_key = tuple(sorted(dict.fromkeys(requested_cols)))

        key_cols_list = [str(c) for c in key_cols]
        missing_keys = [k for k in key_cols_list if k not in self.features_df.columns or k not in key_schema]
        if missing_keys:
            return None

        key_sig = tuple((k, str(key_schema.get(k))) for k in key_cols_list)
        if (
            self._feature_event_cache is None
            or requested_key != self._feature_event_cache_cols
            or key_sig != self._feature_event_cache_key_sig
        ):
            base = self.features_df.select([*key_cols_list, *requested_key])

            casts: List[pl.Expr] = []
            for k in key_cols_list:
                dtype = key_schema.get(k)
                if dtype is not None:
                    casts.append(pl.col(k).cast(dtype, strict=False).alias(k))
            if casts:
                base = base.with_columns(casts)

            agg_exprs: List[pl.Expr] = []
            for col in requested_key:
                agg_exprs.append(
                    pl.col(col)
                    .cast(pl.Float64, strict=False)
                    .fill_nan(None)
                    .mean()
                    .alias(_event_ms_col(col))
                )

            self._feature_event_cache = base.group_by(key_cols_list, maintain_order=False).agg(agg_exprs)
            self._feature_event_cache_cols = requested_key
            self._feature_event_cache_key_sig = key_sig
        return self._feature_event_cache

    def _enrich_meta_with_feature_event_ms(self, meta_df: pl.DataFrame) -> pl.DataFrame:
        """
        기본 입력 parquet에 없는 이벤트 컬럼에 대해, `data.features_file`에서 `__event_<col>_ms` 값을 채웁니다.

        규칙:
        - 입력 parquet에 이벤트가 존재하면, 값은 `data.id_columns.onset`(mocap frame)과 동일 도메인으로 해석되며
          `_load_and_align_lazy()`에서 ms로 변환됩니다.
        - 입력 parquet에 없고 `data.features_file`에만 존재하면, 값은 platform_onset 기준 ms로 해석됩니다.
        """
        if self.features_df is None or not self.required_event_columns:
            return meta_df

        input_cols = self._input_columns or set()
        feature_event_cols = [c for c in self.required_event_columns if c not in input_cols and c in self.features_df.columns]
        if not feature_event_cols:
            return meta_df

        subject_col = self.id_cfg["subject"]
        velocity_col = self.id_cfg["velocity"]
        trial_col = self.id_cfg["trial"]
        key_cols = [subject_col, velocity_col, trial_col]

        missing_keys = [k for k in key_cols if k not in meta_df.columns or k not in self.features_df.columns]
        if missing_keys:
            if not self._feature_event_logged:
                print(f"[features_event_ms] Warning: cannot join features_file events; missing keys: {missing_keys}")
                self._feature_event_logged = True
            return meta_df

        requested = tuple(sorted(feature_event_cols))
        feature_table = self._get_feature_event_ms_table(
            requested=requested,
            key_cols=key_cols,
            key_schema=meta_df.schema,
        )
        if feature_table is None or feature_table.is_empty():
            return meta_df
        if not self._feature_event_logged:
            print(f"[features_event_ms] using features_file columns (ms): {list(requested)}")
            self._feature_event_logged = True

        joined = meta_df.join(feature_table, on=key_cols, how="left", suffix="__feat")

        exclude_cols: List[str] = []
        for event_col in requested:
            base_col = _event_ms_col(event_col)
            feat_col = f"{base_col}__feat"
            if base_col not in joined.columns or feat_col not in joined.columns:
                continue
            joined = joined.with_columns(pl.coalesce([pl.col(base_col), pl.col(feat_col)]).alias(base_col))
            exclude_cols.append(feat_col)

        if exclude_cols:
            joined = joined.drop(exclude_cols)
        return joined
