from __future__ import annotations

import concurrent.futures
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

from .config import bom_rename_map, resolve_path
from .visualizer_types import ResampledGroup
from ..plotting.matplotlib.common import _event_ms_col


_INTERP_TARGET_AXIS: Optional[np.ndarray] = None


def _interp_worker_init(target_axis: np.ndarray) -> None:
    global _INTERP_TARGET_AXIS
    _INTERP_TARGET_AXIS = target_axis


def _interp_trial(x_list: Sequence[float], ys_lists: Sequence[Sequence[float]]) -> np.ndarray:
    if _INTERP_TARGET_AXIS is None:
        raise RuntimeError("Interpolation worker not initialized.")

    x_all = np.asarray(x_list, dtype=float)
    n_channels = len(ys_lists)
    out = np.full((n_channels, _INTERP_TARGET_AXIS.size), np.nan, dtype=float)

    for i, y_list in enumerate(ys_lists):
        y_all = np.asarray(y_list, dtype=float)
        valid = ~(np.isnan(x_all) | np.isnan(y_all))
        if valid.sum() < 2:
            continue

        x = x_all[valid]
        y = y_all[valid]

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        # np.interp expects x to be increasing. If duplicates exist, average y per unique x.
        uniq_x, inv = np.unique(x, return_inverse=True)
        if uniq_x.size != x.size:
            sums = np.bincount(inv, weights=y)
            counts = np.bincount(inv)
            y = sums / counts
            x = uniq_x

        out[i] = np.interp(_INTERP_TARGET_AXIS, x, y, left=np.nan, right=np.nan).astype(float, copy=False)

    return out


class VisualizerDataMixin:
    def _max_workers(self) -> int:
        perf = self.config.get("performance", {}) if isinstance(self.config, dict) else {}
        cfg_val = perf.get("max_workers") if isinstance(perf, dict) else None
        if cfg_val is not None:
            try:
                return max(1, int(cfg_val))
            except (TypeError, ValueError):
                pass
        return max(1, min(8, os.cpu_count() or 1))

    def _signal_group_names(self, selected_groups: Optional[Iterable[str]]) -> List[str]:
        names = list(self.config["signal_groups"].keys())
        if selected_groups is None:
            return names
        return [n for n in names if n in selected_groups]
    def _collect_needed_meta_columns(self, enabled_modes: List[Tuple[str, Dict[str, Any]]]) -> List[str]:
        needed: List[str] = []
        for _, mode_cfg in enabled_modes:
            for col in mode_cfg.get("groupby", []) or []:
                needed.append(col)
            filter_cfg = mode_cfg.get("filter")
            if filter_cfg and isinstance(filter_cfg, dict):
                for col_name in filter_cfg.keys():
                    needed.append(col_name)
        out: List[str] = []
        for col in needed:
            if not col or col in out:
                continue
            out.append(str(col))
        return out

    def _load_and_align_lazy(self) -> pl.LazyFrame:
        input_path = resolve_path(self.base_dir, self.config["data"]["input_file"])

        lf = pl.scan_parquet(str(input_path))

        rename_map = bom_rename_map(self._lazy_columns(lf))
        if rename_map:
            lf = lf.rename(rename_map)

        self._input_columns = set(self._lazy_columns(lf))

        task_col = self.id_cfg.get("task")
        task_filter = self.config["data"].get("task_filter")
        if task_filter and task_col and task_col in self._lazy_columns(lf):
            lf = lf.filter(pl.col(task_col) == task_filter)

        subject_col = self.id_cfg["subject"]
        velocity_col = self.id_cfg["velocity"]
        trial_col = self.id_cfg["trial"]
        frame_col = self.id_cfg["frame"]
        mocap_col = self.id_cfg["mocap_frame"]
        onset_col = self.id_cfg["onset"]
        offset_col = self.id_cfg["offset"]

        group_cols = [subject_col, velocity_col, trial_col]
        onset_mocap = pl.col(onset_col).first().over(group_cols)
        mocap_start = pl.col(mocap_col).min().over(group_cols)
        onset_device = (onset_mocap - mocap_start) * self.frame_ratio
        onset_aligned = pl.col(frame_col) - onset_device
        aligned_mocap = (pl.col(mocap_col) - onset_mocap) * self.frame_ratio
        offset_rel = (
            (pl.col(offset_col).first().over(group_cols) - pl.col(onset_col).first().over(group_cols))
            * self.frame_ratio
        )

        extra_event_exprs: List[pl.Expr] = []
        if self.required_event_columns:
            available_cols = set(self._lazy_columns(lf))
            self._input_columns = available_cols
            ms_per_frame = 1000.0 / float(self.device_rate)
            for event_col in self.required_event_columns:
                if event_col not in available_cols:
                    continue
                # Interpret event in the same domain as `platform_onset` (mocap frame) and convert to ms.
                event_mocap = pl.col(event_col).max().over(group_cols)  # ignores nulls
                event_rel_frame = (event_mocap - onset_mocap) * self.frame_ratio
                extra_event_exprs.append((event_rel_frame * ms_per_frame).alias(_event_ms_col(event_col)))

        # Preserve explicit warnings for event_vlines configuration (only).
        if self.event_vline_columns:
            available_cols = self._input_columns or set(self._lazy_columns(lf))
            for event_col in self.event_vline_columns:
                if event_col in available_cols:
                    continue
                if self.features_df is not None and event_col in self.features_df.columns:
                    continue
                print(f"[event_vlines] Warning: column '{event_col}' not found in input/features; skipping")

        return lf.with_columns(
            [
                onset_device.alias("onset_device_frame"),
                onset_aligned.alias("aligned_frame"),
                aligned_mocap.alias("aligned_mocap_frame"),
                offset_rel.alias("offset_from_onset"),
            ]
            + extra_event_exprs
        )

    @staticmethod
    def _lazy_columns(lf: pl.LazyFrame) -> List[str]:
        try:
            schema = lf.collect_schema()
            return list(schema.keys())
        except Exception:
            return list(getattr(lf, "schema", {}).keys())

    def _filter_first_group(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        subject_col = self.id_cfg["subject"]
        velocity_col = self.id_cfg["velocity"]
        trial_col = self.id_cfg["trial"]

        first_row = lf.select([subject_col, velocity_col, trial_col]).limit(1).collect()
        if first_row.is_empty():
            raise ValueError("No data available after applying task or input filters.")
        first_subject, first_velocity, first_trial = first_row.row(0)

        return lf.filter(
            (pl.col(subject_col) == first_subject)
            & (pl.col(velocity_col) == first_velocity)
            & (pl.col(trial_col) == first_trial)
        )

    def _prepare_time_axis(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        stats = lf.select(
            [
                pl.col("aligned_frame").min().alias("min"),
                pl.col("aligned_frame").max().alias("max"),
            ]
        ).collect()
        if stats.is_empty():
            raise ValueError("No data available to build time axis.")
        frame_min = stats["min"].item()
        frame_max = stats["max"].item()
        if frame_min is None or frame_max is None:
            raise ValueError("No data available to build time axis.")

        data_start_ms = self._frame_to_ms(float(frame_min))
        data_end_ms = self._frame_to_ms(float(frame_max))

        interp_cfg = self.config.get("interpolation", {})
        start_ms_cfg = interp_cfg.get("start_ms")
        end_ms_cfg = interp_cfg.get("end_ms")
        time_start_ms = data_start_ms if start_ms_cfg is None else float(start_ms_cfg)
        time_end_ms = data_end_ms if end_ms_cfg is None else float(end_ms_cfg)

        if time_end_ms <= time_start_ms:
            raise ValueError("`interpolation.end_ms` must be greater than `interpolation.start_ms`.")

        time_start_frame = self._ms_to_frame(time_start_ms)
        time_end_frame = self._ms_to_frame(time_end_ms)

        cropped = lf.filter(
            ((pl.col("aligned_frame") >= time_start_frame) & (pl.col("aligned_frame") <= time_end_frame))
            | (
                (pl.col("aligned_mocap_frame") >= time_start_frame)
                & (pl.col("aligned_mocap_frame") <= time_end_frame)
            )
        )
        if cropped.select(pl.len()).collect().item() == 0:
            raise ValueError("No data remains after applying the interpolation time window.")

        self.time_start_ms = time_start_ms
        self.time_end_ms = time_end_ms
        self.time_start_frame = time_start_frame
        self.time_end_frame = time_end_frame
        self.target_axis = np.linspace(time_start_frame, time_end_frame, self.target_length)
        self.x_norm = np.linspace(0.0, 1.0, self.target_length)
        return cropped

    def _resample_signal_group(self, lf: pl.LazyFrame, signal_group: str, meta_cols: List[str]) -> ResampledGroup:
        if self.target_axis is None:
            raise RuntimeError("target_axis must be initialized before interpolation.")

        subject_col = self.id_cfg["subject"]
        velocity_col = self.id_cfg["velocity"]
        trial_col = self.id_cfg["trial"]
        group_cols = [subject_col, velocity_col, trial_col]

        group_cfg = self.config["signal_groups"][signal_group]
        channels = list(group_cfg["columns"])

        time_base = str(group_cfg.get("time_base", "device")).strip().lower()
        if time_base in ("mocap", "mocapframe"):
            x_col = "aligned_mocap_frame"
        else:
            x_col = "aligned_frame"

        available_cols = set(self._lazy_columns(lf))
        missing_channels = [ch for ch in channels if ch not in available_cols]
        if missing_channels:
            if bool(group_cfg.get("optional", False)):
                print(f"[{signal_group}] skip: missing channels: {missing_channels}")
                return ResampledGroup(meta_df=pl.DataFrame(), tensor=np.empty((0, 0, 0)), channels=[])
            raise ValueError(f"Missing required channels for '{signal_group}': {missing_channels}")
        if x_col not in available_cols:
            if bool(group_cfg.get("optional", False)):
                print(f"[{signal_group}] skip: missing time axis column: {x_col!r}")
                return ResampledGroup(meta_df=pl.DataFrame(), tensor=np.empty((0, 0, 0)), channels=[])
            raise ValueError(f"Missing required time axis column for '{signal_group}': {x_col!r}")

        cols = set(group_cols + [x_col] + channels + meta_cols)
        lf_sel = lf.select([pl.col(c) for c in cols if c in available_cols])
        if self.time_start_frame is not None and self.time_end_frame is not None:
            lf_sel = lf_sel.filter((pl.col(x_col) >= self.time_start_frame) & (pl.col(x_col) <= self.time_end_frame))

        present_meta_cols: List[str] = []
        missing_meta_cols: List[str] = []
        for col in meta_cols:
            if col in group_cols:
                continue
            if col in available_cols:
                present_meta_cols.append(col)
            else:
                missing_meta_cols.append(col)

        agg_exprs: List[pl.Expr] = [pl.col(x_col).sort().alias("__x")]
        for ch in channels:
            agg_exprs.append(pl.col(ch).sort_by(x_col).alias(ch))
        for col in present_meta_cols:
            agg_exprs.append(pl.col(col).first().alias(col))
            agg_exprs.append(pl.col(col).n_unique().alias(f"__nuniq_{col}"))

        # NOTE: Polars `group_by(..., maintain_order=False)` does not guarantee row order.
        # The resulting group order can vary across process runs (hash-map iteration order),
        # which can change downstream floating-point reduction order and make plots (notably COM)
        # non-deterministic at the PNG-byte level.
        # Sort by the trial key to ensure stable tensor row ordering and reproducible outputs.
        grouped = lf_sel.group_by(group_cols, maintain_order=False).agg(agg_exprs).sort(group_cols)
        df = grouped.collect()
        if df.is_empty():
            raise ValueError("No data available after applying task or input filters.")

        for col in present_meta_cols:
            nuniq_col = f"__nuniq_{col}"
            if nuniq_col not in df.columns:
                continue
            bad = df.filter(pl.col(nuniq_col) != 1)
            if not bad.is_empty():
                raise ValueError(f"Metadata column '{col}' is not constant within some trials.")

        x_lists = df["__x"].to_list()
        ys_by_ch = {ch: df[ch].to_list() for ch in channels}
        ys_per_trial = list(zip(*(ys_by_ch[ch] for ch in channels)))

        n_trials = len(x_lists)
        tensor = np.full((n_trials, len(channels), self.target_axis.size), np.nan, dtype=float)
        max_workers = self._max_workers()
        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_interp_worker_init,
                initargs=(self.target_axis,),
            ) as ex:
                chunk = max(1, n_trials // (max_workers * 4)) if n_trials else 1
                for i, arr in enumerate(ex.map(_interp_trial, x_lists, ys_per_trial, chunksize=chunk)):
                    tensor[i] = arr
        except PermissionError:
            _interp_worker_init(self.target_axis)
            for i, arr in enumerate(map(_interp_trial, x_lists, ys_per_trial)):
                tensor[i] = arr

        keep_cols: List[str] = []
        for c in group_cols + meta_cols:
            if c in df.columns and c not in keep_cols:
                keep_cols.append(c)
        meta_df = df.select(keep_cols)
        for col in missing_meta_cols:
            if col in meta_df.columns:
                continue
            meta_df = meta_df.with_columns(pl.lit(None).alias(col))

        meta_df = self._enrich_meta_with_feature_event_ms(meta_df)
        return ResampledGroup(meta_df=meta_df, tensor=tensor, channels=channels)

    def _ms_to_frame(self, ms: float) -> float:
        return ms * self.device_rate / 1000.0

    def _frame_to_ms(self, frame: float) -> float:
        return frame / self.device_rate * 1000.0
