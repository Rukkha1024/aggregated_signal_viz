from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

from .config import resolve_output_dir
from .visualizer_types import ResampledGroup
from ..plotting.matplotlib.common import (
    _calculate_filtered_group_fields,
    _nanmean_3d_over_first_axis,
    _overlay_vline_event_names,
    _sort_overlay_keys,
)


class VisualizerTasksMixin:
    def _validate_emg_trial_grid_by_channel_modes(
        self,
        enabled_modes: Sequence[Tuple[str, Dict[str, Any]]],
    ) -> None:
        """
        Strict policy (requested):
        - If `output.emg_trial_grid_by_channel` is enabled, the run must abort unless
          every selected mode is "trial-level" (groupby includes subject/velocity/trial).
        """
        subject_col = str(self.id_cfg.get("subject") or "subject")
        velocity_col = str(self.id_cfg.get("velocity") or "velocity")
        trial_col = str(self.id_cfg.get("trial") or "trial_num")
        required = [subject_col, velocity_col, trial_col]

        for mode_name, mode_cfg in enabled_modes:
            groupby = mode_cfg.get("groupby") or []
            if not isinstance(groupby, list):
                raise TypeError(f"[emg_trial_grid_by_channel] aggregation_modes.{mode_name}.groupby must be a list.")
            missing = [col for col in required if col not in groupby]
            if missing:
                missing_str = ", ".join(missing)
                required_str = ", ".join(required)
                raise ValueError(
                    "[emg_trial_grid_by_channel] enabled but mode is not trial-level: "
                    f"aggregation_modes.{mode_name}.groupby missing [{missing_str}]. "
                    f"Required groupby includes [{required_str}]."
                )

    def _emit_emg_trial_grid_by_channel(
        self,
        *,
        resampled: ResampledGroup,
        enabled_modes: Sequence[Tuple[str, Dict[str, Any]]],
        sample: bool,
    ) -> List[Path]:
        """
        Emit Plotly HTML trial-grid outputs for EMG (one file per subject x emg_channel).
        """
        if self.x_norm is None:
            raise RuntimeError("x_norm is not initialized.")

        from ..plotting.plotly.emg_trial_grid_by_channel import write_emg_trial_grid_html

        subject_col = str(self.id_cfg.get("subject") or "subject")
        velocity_col = str(self.id_cfg.get("velocity") or "velocity")
        trial_col = str(self.id_cfg.get("trial") or "trial_num")

        if subject_col not in resampled.meta_df.columns:
            raise ValueError(f"[emg_trial_grid_by_channel] missing metadata column: {subject_col!r}")
        if velocity_col not in resampled.meta_df.columns:
            raise ValueError(f"[emg_trial_grid_by_channel] missing metadata column: {velocity_col!r}")
        if trial_col not in resampled.meta_df.columns:
            raise ValueError(f"[emg_trial_grid_by_channel] missing metadata column: {trial_col!r}")

        emg_cfg = (self.config.get("signal_groups") or {}).get("emg") if isinstance(self.config, dict) else None
        grid_layout = (emg_cfg or {}).get("grid_layout") if isinstance(emg_cfg, dict) else None
        max_cols = 4
        if isinstance(grid_layout, (list, tuple)) and len(grid_layout) == 2:
            try:
                max_cols = max(1, int(grid_layout[1]))
            except (TypeError, ValueError):
                max_cols = 4

        # Stable channel index mapping (resampled tensor axis order).
        channel_to_idx = {str(ch): i for i, ch in enumerate(resampled.channels)}
        if not channel_to_idx:
            return []

        out_paths: List[Path] = []
        meta_df = resampled.meta_df
        tensor = resampled.tensor

        if self.time_start_frame is None or self.time_end_frame is None:
            raise RuntimeError("[emg_trial_grid_by_channel] time_start_frame/time_end_frame must be initialized.")
        start_frame = float(self.time_start_frame)
        end_frame = float(self.time_end_frame)
        span = end_frame - start_frame
        if span == 0:
            raise RuntimeError("[emg_trial_grid_by_channel] invalid time axis span (start==end).")

        x_norm = np.asarray(self.x_norm, dtype=float)
        x_tick_dtick: Optional[float] = None
        try:
            x_tick_dtick = float(self.device_rate) * 25.0 / 1000.0  # 25 ms ticks (frames at device_rate)
        except Exception:
            x_tick_dtick = None
        if x_tick_dtick is not None and (not np.isfinite(x_tick_dtick) or x_tick_dtick <= 0):
            x_tick_dtick = None

        def _safe(text: Any) -> str:
            out = str(text)
            for ch in ("/", "\\", ":", "\n", "\r", "\t"):
                out = out.replace(ch, "_")
            return out.strip() or "untitled"

        for mode_name, mode_cfg in enabled_modes:
            mode_filter = mode_cfg.get("filter") if isinstance(mode_cfg.get("filter"), dict) else None
            filtered_idx = self._apply_filter_indices(meta_df, mode_filter)
            if filtered_idx.size == 0:
                continue

            output_dir = resolve_output_dir(self.base_dir, self.config, mode_cfg["output_dir"])
            out_dir = output_dir / "trial_grid_by_channel"
            out_dir.mkdir(parents=True, exist_ok=True)

            # Unique subjects for this mode after filtering.
            subject_vals = meta_df[subject_col].to_numpy()
            unique_subjects: List[Any] = []
            for v in subject_vals[filtered_idx].tolist():
                if v not in unique_subjects:
                    unique_subjects.append(v)

            if sample and unique_subjects:
                unique_subjects = unique_subjects[:1]

            for subject in unique_subjects:
                subj_idx = filtered_idx[subject_vals[filtered_idx] == subject]
                if subj_idx.size == 0:
                    continue

                # Sort trials within this subject for deterministic grid layout.
                sub = (
                    meta_df.with_row_index("__row_idx")
                    .filter(pl.col(subject_col) == subject)
                    .filter(pl.col("__row_idx").is_in([int(i) for i in subj_idx.tolist()]))
                    .with_columns(
                        [
                            pl.col(velocity_col).cast(pl.Float64, strict=False).fill_null(float("inf")).alias("__v"),
                            pl.col(trial_col).cast(pl.Int64, strict=False).fill_null(10**9).alias("__t"),
                        ]
                    )
                )
                sort_cols: List[str] = ["__v", "__t"]
                if "step_TF" in sub.columns:
                    sort_cols.append("step_TF")
                sort_cols.append("__row_idx")
                sub = sub.sort(sort_cols)

                ordered_rows = sub.select(["__row_idx", velocity_col, trial_col] + (["step_TF"] if "step_TF" in sub.columns else [])).to_dicts()
                ordered_trial_indices = [int(r["__row_idx"]) for r in ordered_rows]
                if sample and ordered_trial_indices:
                    ordered_rows = ordered_rows[:1]
                    ordered_trial_indices = ordered_trial_indices[:1]

                # Use the subject-specific trial set to compute windows/vlines/zeroing.
                window_spans = self._compute_window_spans(meta_df, subj_idx)
                window_spans_by_channel = self._compute_window_spans_by_channel(meta_df, subj_idx)
                event_vlines = self._collect_event_vlines(meta_df, subj_idx)
                event_vlines_by_channel = self._collect_emg_event_vlines_by_channel(meta_df, subj_idx)
                time_zero_frame = self._resolve_time_zero_frame(
                    meta_df=meta_df,
                    indices=subj_idx,
                    mode_name=mode_name,
                    signal_group="emg",
                    key_label=f"subject={subject}",
                )
                time_zero_frame_by_channel = self._resolve_time_zero_frame_by_channel(
                    meta_df=meta_df,
                    indices=subj_idx,
                    mode_name=mode_name,
                    signal_group="emg",
                    key_label=f"subject={subject}",
                )

                for emg_channel in resampled.channels:
                    ch_name = str(emg_channel)
                    if ch_name not in channel_to_idx:
                        continue
                    ch_idx = int(channel_to_idx[ch_name])

                    ch_zero = float(time_zero_frame_by_channel.get(ch_name, time_zero_frame)) if time_zero_frame_by_channel else float(time_zero_frame)
                    x_frames = start_frame + x_norm * span - ch_zero

                    series_by_trial = [tensor[i, ch_idx, :] for i in ordered_trial_indices]
                    subplot_titles: List[str] = []
                    for row in ordered_rows:
                        vel = row.get(velocity_col)
                        tr = row.get(trial_col)
                        step_tf = row.get("step_TF") if "step_TF" in row else None
                        parts = [f"v={vel}", f"trial={tr}"]
                        if step_tf is not None and str(step_tf).strip():
                            parts.append(str(step_tf))
                        subplot_titles.append(" | ".join(parts))

                    spans = window_spans_by_channel.get(ch_name, window_spans) if window_spans_by_channel else window_spans
                    vlines = event_vlines_by_channel.get(ch_name, event_vlines) if event_vlines_by_channel else event_vlines

                    html_name = _safe(f"{mode_name}_{subject}_{ch_name}_trial_grid_emg.html")
                    html_path = out_dir / html_name
                    title = f"{mode_name} | emg:{ch_name} | trial-grid | subject={subject}"

                    out_paths.append(
                        write_emg_trial_grid_html(
                            html_path=html_path,
                            title=title,
                            x=x_frames,
                            series_by_trial=series_by_trial,
                            subplot_titles=subplot_titles,
                            max_cols=max_cols,
                            window_spans=[
                                {
                                    "start": float(start_frame + float(s["start"]) * span - ch_zero),
                                    "end": float(start_frame + float(s["end"]) * span - ch_zero),
                                    "color": s.get("color"),
                                    "label": s.get("label"),
                                }
                                for s in spans or []
                                if s.get("start") is not None and s.get("end") is not None
                            ],
                            event_vlines=[
                                {
                                    "x": float(start_frame + float(v["x"]) * span - ch_zero),
                                    "color": v.get("color"),
                                    "label": v.get("label"),
                                    "name": v.get("name"),
                                }
                                for v in vlines or []
                                if v.get("x") is not None
                            ],
                            window_span_alpha=float(self.emg_style.get("window_span_alpha", 0.15)),
                            event_vline_style=self.event_vline_style,
                            line_color=self.emg_style.get("line_color", "gray"),
                            line_width=float(self.emg_style.get("line_width", 1.2)),
                            line_alpha=float(self.emg_style.get("line_alpha", 0.85)),
                            show_grid=bool(self.common_style.get("show_grid", True)),
                            grid_alpha=self.common_style.get("grid_alpha", 0.5),
                            x_tick_dtick=x_tick_dtick,
                        )
                    )

                    if sample:
                        return out_paths[:1]

        return out_paths

    def _build_plot_tasks(
        self, resampled: ResampledGroup, signal_group: str, mode_name: str, mode_cfg: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        if self.x_norm is None or self.target_axis is None or self.time_start_ms is None or self.time_end_ms is None:
            raise RuntimeError("Time axis is not initialized.")

        meta_df = resampled.meta_df
        tensor = resampled.tensor
        channels = resampled.channels

        if meta_df.is_empty() or tensor.size == 0 or not channels:
            return []

        filtered_idx = self._apply_filter_indices(meta_df, mode_cfg.get("filter"))
        if filtered_idx.size == 0:
            return []

        group_fields = list(mode_cfg.get("groupby", []) or [])
        overlay = bool(mode_cfg.get("overlay", False))

        grouped = self._group_indices(meta_df, filtered_idx, group_fields)
        if not grouped:
            return []

        tasks: List[Dict[str, Any]] = []

        if overlay:
            overlay_within = mode_cfg.get("overlay_within")
            overlay_event_names = _overlay_vline_event_names(self.event_vline_overlay_cfg)

            def _filter_pooled_vlines(vlines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                if not overlay_event_names:
                    return vlines
                return [v for v in vlines if str(v.get("name") or "").strip() not in overlay_event_names]

            def _filter_pooled_vlines_by_channel(
                mapping: Dict[str, List[Dict[str, Any]]],
            ) -> Dict[str, List[Dict[str, Any]]]:
                if not overlay_event_names:
                    return mapping
                out: Dict[str, List[Dict[str, Any]]] = {}
                for ch, vlines in mapping.items():
                    kept = [v for v in vlines if str(v.get("name") or "").strip() not in overlay_event_names]
                    if kept:
                        out[ch] = kept
                return out

            # Aggregate all keys first (common for both old and new logic)
            aggregated_by_key: Dict[Tuple, Dict[str, np.ndarray]] = {}
            markers_by_key: Dict[Tuple, Dict[str, Any]] = {}
            for key, idx in grouped.items():
                aggregated_by_key[key] = self._aggregate_tensor(tensor, meta_df, idx, channels)
                markers_by_key[key] = self._collect_markers(signal_group, key, group_fields, mode_cfg.get("filter"))
            event_vlines_by_key = {key: self._collect_event_vlines(meta_df, idx) for key, idx in grouped.items()}
            event_vlines_by_key_by_channel: Optional[Dict[Tuple, Dict[str, List[Dict[str, Any]]]]] = None
            if signal_group == "emg":
                event_vlines_by_key_by_channel = {
                    key: self._collect_emg_event_vlines_by_channel(meta_df, idx) for key, idx in grouped.items()
                }

            output_dir = resolve_output_dir(self.base_dir, self.config, mode_cfg["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            filename_pattern = mode_cfg["filename_pattern"]

            # OLD BEHAVIOR: overlay_within not specified -> all keys in one file
            if not overlay_within:
                window_spans = self._compute_window_spans(meta_df, filtered_idx)
                window_spans_by_channel = (
                    self._compute_window_spans_by_channel(meta_df, filtered_idx) if signal_group == "emg" else None
                )
                pooled_event_vlines = _filter_pooled_vlines(self._collect_event_vlines(meta_df, filtered_idx))
                pooled_event_vlines_by_channel = (
                    _filter_pooled_vlines_by_channel(self._collect_emg_event_vlines_by_channel(meta_df, filtered_idx))
                    if signal_group == "emg"
                    else None
                )
                sorted_keys = _sort_overlay_keys(list(aggregated_by_key.keys()), group_fields)
                filtered_group_fields = _calculate_filtered_group_fields(
                    sorted_keys,
                    group_fields,
                    threshold=self.legend_label_threshold
                )

                try:
                    filename = self._render_filename(filename_pattern, ("all",), signal_group, group_fields=[])
                except KeyError as e:
                    print(
                        f"[overlay] filename_pattern missing key {e!s}; filling group fields with 'overlay' for mode '{mode_name}'"
                    )
                    mapping = {field: "overlay" for field in group_fields}
                    mapping["signal_group"] = signal_group
                    filename = filename_pattern.format(**mapping)
                output_path = output_dir / filename
                time_zero_frame = self._resolve_time_zero_frame(
                    meta_df=meta_df,
                    indices=filtered_idx,
                    mode_name=mode_name,
                    signal_group=signal_group,
                    key_label="all",
                )
                time_zero_frame_by_channel = (
                    self._resolve_time_zero_frame_by_channel(
                        meta_df=meta_df,
                        indices=filtered_idx,
                        mode_name=mode_name,
                        signal_group=signal_group,
                        key_label="all",
                    )
                    if signal_group == "emg"
                    else None
                )

                tasks.append(
                    self._task_overlay(
                        signal_group=signal_group,
                        aggregated_by_key=aggregated_by_key,
                        markers_by_key=markers_by_key,
                        event_vlines_by_key=event_vlines_by_key,
                        event_vlines_by_key_by_channel=event_vlines_by_key_by_channel,
                        pooled_event_vlines=pooled_event_vlines,
                        pooled_event_vlines_by_channel=pooled_event_vlines_by_channel,
                        output_path=output_path,
                        mode_name=mode_name,
                        group_fields=group_fields,
                        sorted_keys=sorted_keys,
                        window_spans=window_spans,
                        window_spans_by_channel=window_spans_by_channel,
                        time_zero_frame=time_zero_frame,
                        time_zero_frame_by_channel=time_zero_frame_by_channel,
                        filtered_group_fields=filtered_group_fields,
                        color_by_fields=mode_cfg.get("color_by") if signal_group in ("emg", "cop", "com") else None,
                    )
                )
                return tasks

            # NEW BEHAVIOR: overlay_within specified -> separate files by non-overlay fields
            from collections import defaultdict

            file_fields = [f for f in group_fields if f not in overlay_within]

            # Group keys by file_fields
            file_groups: Dict[Tuple, List[Tuple]] = defaultdict(list)
            for key in aggregated_by_key.keys():
                field_to_value = dict(zip(group_fields, key))
                file_key = tuple(field_to_value[f] for f in file_fields) if file_fields else ("all",)
                file_groups[file_key].append(key)

            # Create one task per file_key
            for file_key, keys_in_file in file_groups.items():
                file_aggregated = {k: aggregated_by_key[k] for k in keys_in_file}
                file_markers = {k: markers_by_key[k] for k in keys_in_file}
                file_event_vlines = {k: event_vlines_by_key.get(k, []) for k in keys_in_file}
                file_event_vlines_by_key_by_channel = (
                    {k: event_vlines_by_key_by_channel.get(k, {}) for k in keys_in_file}
                    if event_vlines_by_key_by_channel
                    else None
                )
                file_indices = np.concatenate([grouped[k] for k in keys_in_file if k in grouped]) if keys_in_file else filtered_idx
                window_spans = self._compute_window_spans(meta_df, file_indices)
                window_spans_by_channel = (
                    self._compute_window_spans_by_channel(meta_df, file_indices) if signal_group == "emg" else None
                )
                pooled_event_vlines = _filter_pooled_vlines(self._collect_event_vlines(meta_df, file_indices))
                pooled_event_vlines_by_channel = (
                    _filter_pooled_vlines_by_channel(self._collect_emg_event_vlines_by_channel(meta_df, file_indices))
                    if signal_group == "emg"
                    else None
                )

                sorted_keys = _sort_overlay_keys(keys_in_file, group_fields)
                filtered_group_fields = _calculate_filtered_group_fields(
                    sorted_keys,
                    group_fields,
                    threshold=self.legend_label_threshold
                )

                filename = self._render_filename(filename_pattern, file_key, signal_group, file_fields)
                output_path = output_dir / filename
                time_zero_frame = self._resolve_time_zero_frame(
                    meta_df=meta_df,
                    indices=file_indices,
                    mode_name=mode_name,
                    signal_group=signal_group,
                    key_label=str(file_key),
                )
                time_zero_frame_by_channel = (
                    self._resolve_time_zero_frame_by_channel(
                        meta_df=meta_df,
                        indices=file_indices,
                        mode_name=mode_name,
                        signal_group=signal_group,
                        key_label=str(file_key),
                    )
                    if signal_group == "emg"
                    else None
                )

                tasks.append(
                    self._task_overlay(
                        signal_group=signal_group,
                        aggregated_by_key=file_aggregated,
                        markers_by_key=file_markers,
                        event_vlines_by_key=file_event_vlines,
                        event_vlines_by_key_by_channel=file_event_vlines_by_key_by_channel,
                        pooled_event_vlines=pooled_event_vlines,
                        pooled_event_vlines_by_channel=pooled_event_vlines_by_channel,
                        output_path=output_path,
                        mode_name=mode_name,
                        group_fields=group_fields,
                        sorted_keys=sorted_keys,
                        window_spans=window_spans,
                        window_spans_by_channel=window_spans_by_channel,
                        time_zero_frame=time_zero_frame,
                        time_zero_frame_by_channel=time_zero_frame_by_channel,
                        filtered_group_fields=filtered_group_fields,
                        color_by_fields=mode_cfg.get("color_by") if signal_group in ("emg", "cop", "com") else None,
                    )
                )

            return tasks

        for key, idx in grouped.items():
            aggregated = self._aggregate_tensor(tensor, meta_df, idx, channels)
            filename = self._render_filename(mode_cfg["filename_pattern"], key, signal_group, group_fields)
            output_dir = resolve_output_dir(self.base_dir, self.config, mode_cfg["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / filename
            markers = self._collect_markers(signal_group, key, group_fields, mode_cfg.get("filter"))
            event_vlines = self._collect_event_vlines(meta_df, idx)
            window_spans = self._compute_window_spans(meta_df, idx)
            time_zero_frame = self._resolve_time_zero_frame(
                meta_df=meta_df,
                indices=idx,
                mode_name=mode_name,
                signal_group=signal_group,
                key_label=str(key),
            )

            if signal_group == "emg":
                event_vlines_by_channel = self._collect_emg_event_vlines_by_channel(meta_df, idx)
                window_spans_by_channel = self._compute_window_spans_by_channel(meta_df, idx)
                time_zero_frame_by_channel = self._resolve_time_zero_frame_by_channel(
                    meta_df=meta_df,
                    indices=idx,
                    mode_name=mode_name,
                    signal_group=signal_group,
                    key_label=str(key),
                )
                tasks.append(
                    self._task_emg(
                        aggregated=aggregated,
                        output_path=output_path,
                        key=key,
                        mode_name=mode_name,
                        group_fields=group_fields,
                        markers=markers,
                        event_vlines=event_vlines,
                        event_vlines_by_channel=event_vlines_by_channel,
                        window_spans=window_spans,
                        window_spans_by_channel=window_spans_by_channel,
                        time_zero_frame=time_zero_frame,
                        time_zero_frame_by_channel=time_zero_frame_by_channel,
                    )
                )
            elif signal_group == "forceplate":
                tasks.append(
                    self._task_forceplate(
                        aggregated=aggregated,
                        output_path=output_path,
                        key=key,
                        mode_name=mode_name,
                        group_fields=group_fields,
                        markers=markers,
                        event_vlines=event_vlines,
                        window_spans=window_spans,
                        time_zero_frame=time_zero_frame,
                    )
                )
            elif signal_group == "cop":
                tasks.append(
                    self._task_cop(
                        aggregated=aggregated,
                        output_path=output_path,
                        key=key,
                        mode_name=mode_name,
                        group_fields=group_fields,
                        markers=markers,
                        event_vlines=event_vlines,
                        window_spans=window_spans,
                        time_zero_frame=time_zero_frame,
                    )
                )
            elif signal_group == "com":
                tasks.append(
                    self._task_com(
                        aggregated=aggregated,
                        output_path=output_path,
                        key=key,
                        mode_name=mode_name,
                        group_fields=group_fields,
                        markers=markers,
                        event_vlines=event_vlines,
                        window_spans=window_spans,
                        time_zero_frame=time_zero_frame,
                    )
                )

        return tasks

    def _apply_filter_indices(self, meta_df: pl.DataFrame, filter_cfg: Optional[Dict[str, Any]]) -> np.ndarray:
        idx = np.arange(meta_df.height, dtype=int)
        if not filter_cfg:
            return idx

        # 다중 컬럼 필터링: {mixed: 1, age_group: "young"}
        # 모든 조건을 AND로 결합
        mask = np.ones(meta_df.height, dtype=bool)
        for col, value in filter_cfg.items():
            if col not in meta_df.columns:
                print(f"[filter] Warning: column '{col}' not found in metadata, skipping this filter")
                continue
            series = meta_df[col].to_numpy()
            mask &= (series == value)

        return idx[mask]

    def _group_indices(
        self, meta_df: pl.DataFrame, indices: np.ndarray, group_fields: List[str]
    ) -> Dict[Tuple, np.ndarray]:
        if not group_fields:
            return {("all",): indices}

        missing = [f for f in group_fields if f not in meta_df.columns]
        if missing:
            return {}

        cols = [meta_df[f].to_list() for f in group_fields]
        grouped: Dict[Tuple, List[int]] = {}
        for i in indices.tolist():
            key = tuple(col[i] for col in cols)
            grouped.setdefault(key, []).append(i)
        return {k: np.asarray(v, dtype=int) for k, v in grouped.items()}

    def _aggregate_tensor(
        self, tensor: np.ndarray, meta_df: pl.DataFrame, indices: np.ndarray, channels: List[str]
    ) -> Dict[str, np.ndarray]:
        subject_col = self.id_cfg["subject"]
        subject_vals = np.asarray(meta_df[subject_col].to_list(), dtype=object)
        sub_subjects = subject_vals[indices]
        unique_subjects = list(dict.fromkeys(sub_subjects.tolist()))

        if len(unique_subjects) > 1:
            subject_means: List[np.ndarray] = []
            for subj in unique_subjects:
                subj_idx = indices[sub_subjects == subj]
                mean_subj = _nanmean_3d_over_first_axis(tensor[subj_idx])
                subject_means.append(mean_subj)
            stacked = np.stack(subject_means, axis=0)
            mean_all = _nanmean_3d_over_first_axis(stacked)
        else:
            mean_all = _nanmean_3d_over_first_axis(tensor[indices])

        return {ch: mean_all[i] for i, ch in enumerate(channels)}

    def _render_filename(self, pattern: str, key: Tuple, signal_group: str, group_fields: List[str]) -> str:
        if key == ("all",):
            return pattern.format(signal_group=signal_group)
        mapping = {field: value for field, value in zip(group_fields, key)}
        mapping["signal_group"] = signal_group
        return pattern.format(**mapping)

    def _task_overlay(
        self,
        *,
        signal_group: str,
        aggregated_by_key: Dict[Tuple, Dict[str, np.ndarray]],
        markers_by_key: Dict[Tuple, Dict[str, Any]],
        event_vlines_by_key: Dict[Tuple, List[Dict[str, Any]]],
        event_vlines_by_key_by_channel: Optional[Dict[Tuple, Dict[str, List[Dict[str, Any]]]]] = None,
        pooled_event_vlines: Sequence[Dict[str, Any]] = (),
        pooled_event_vlines_by_channel: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        output_path: Path,
        mode_name: str,
        group_fields: List[str],
        sorted_keys: List[Tuple],
        window_spans: List[Dict[str, Any]],
        window_spans_by_channel: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        filtered_group_fields: List[str],
        time_zero_frame: float = 0.0,
        time_zero_frame_by_channel: Optional[Dict[str, float]] = None,
        color_by_fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        task: Dict[str, Any] = {
            "kind": "overlay",
            "signal_group": signal_group,
            "output_path": str(output_path),
            "plotly_html": bool(self.plotly_html_enabled),
            "mode_name": mode_name,
            "group_fields": group_fields,
            "sorted_keys": sorted_keys,
            "aggregated_by_key": aggregated_by_key,
            "markers_by_key": markers_by_key,
            "event_vlines_by_key": event_vlines_by_key,
            "event_vlines_by_key_by_channel": event_vlines_by_key_by_channel,
            "pooled_event_vlines": list(pooled_event_vlines) if pooled_event_vlines else [],
            "pooled_event_vlines_by_channel": pooled_event_vlines_by_channel,
            "event_vline_style": self.event_vline_style,
            "event_vline_overlay_cfg": self.event_vline_overlay_cfg,
            "event_vline_order": self.event_vline_columns,
            "x": self.x_norm,
            "window_spans": window_spans,
            "window_spans_by_channel": window_spans_by_channel,
            "time_zero_frame": float(time_zero_frame),
            "time_zero_frame_by_channel": dict(time_zero_frame_by_channel) if time_zero_frame_by_channel else None,
            "common_style": self.common_style,
            "filtered_group_fields": filtered_group_fields,
            "color_by_fields": color_by_fields,
            "time_start_frame": self.time_start_frame,
            "time_end_frame": self.time_end_frame,
        }

        if signal_group == "emg":
            task.update(
                {
                    "channels": self.config["signal_groups"]["emg"]["columns"],
                    "grid_layout": self.config["signal_groups"]["emg"]["grid_layout"],
                    "window_span_alpha": self.emg_style["window_span_alpha"],
                    "style": self.emg_style,
                    "time_start_ms": self.time_start_ms,
                    "time_end_ms": self.time_end_ms,
                }
            )
            return task

        if signal_group == "forceplate":
            task.update(
                {
                    "channels": self.config["signal_groups"]["forceplate"]["columns"],
                    "grid_layout": self.config["signal_groups"]["forceplate"]["grid_layout"],
                    "window_span_alpha": self.forceplate_style["window_span_alpha"],
                    "style": self.forceplate_style,
                    "time_start_ms": self.time_start_ms,
                    "time_end_ms": self.time_end_ms,
                }
            )
            return task

        if signal_group == "cop":
            task.update(
                {
                    "style": self.cop_style,
                    "cop_channels": self.config["signal_groups"]["cop"]["columns"],
                    "grid_layout": self.config["signal_groups"]["cop"]["grid_layout"],
                }
            )
            return task

        if signal_group == "com":
            task.update(
                {
                    "style": self.com_style,
                    "cop_channels": self.config["signal_groups"]["com"]["columns"],
                    "grid_layout": self.config["signal_groups"]["com"]["grid_layout"],
                }
            )
            return task

        raise ValueError(f"Unknown signal_group for overlay task: {signal_group!r}")

    def _task_emg(
        self,
        *,
        aggregated: Dict[str, np.ndarray],
        output_path: Path,
        key: Tuple,
        mode_name: str,
        group_fields: List[str],
        markers: Dict[str, Any],
        event_vlines: List[Dict[str, Any]],
        event_vlines_by_channel: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        window_spans: List[Dict[str, Any]],
        window_spans_by_channel: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        time_zero_frame: float = 0.0,
        time_zero_frame_by_channel: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        task: Dict[str, Any] = {
            "kind": "emg",
            "output_path": str(output_path),
            "plotly_html": bool(self.plotly_html_enabled),
            "key": key,
            "mode_name": mode_name,
            "group_fields": group_fields,
            "aggregated": aggregated,
            "markers": markers,
            "event_vlines": event_vlines,
            "event_vlines_by_channel": event_vlines_by_channel,
            "event_vline_style": self.event_vline_style,
            "event_vline_order": self.event_vline_columns,
            "x": self.x_norm,
            "channels": self.config["signal_groups"]["emg"]["columns"],
            "grid_layout": self.config["signal_groups"]["emg"]["grid_layout"],
            "window_spans": window_spans,
            "window_spans_by_channel": window_spans_by_channel,
            "time_zero_frame": float(time_zero_frame),
            "time_zero_frame_by_channel": dict(time_zero_frame_by_channel) if time_zero_frame_by_channel else None,
            "window_span_alpha": self.emg_style["window_span_alpha"],
            "emg_style": self.emg_style,
            "common_style": self.common_style,
            "time_start_ms": self.time_start_ms,
            "time_end_ms": self.time_end_ms,
            "time_start_frame": self.time_start_frame,
            "time_end_frame": self.time_end_frame,
        }
        return task

    def _task_forceplate(
        self,
        *,
        aggregated: Dict[str, np.ndarray],
        output_path: Path,
        key: Tuple,
        mode_name: str,
        group_fields: List[str],
        markers: Dict[str, Any],
        event_vlines: List[Dict[str, Any]],
        window_spans: List[Dict[str, Any]],
        time_zero_frame: float = 0.0,
    ) -> Dict[str, Any]:
        return {
            "kind": "forceplate",
            "output_path": str(output_path),
            "plotly_html": bool(self.plotly_html_enabled),
            "key": key,
            "mode_name": mode_name,
            "group_fields": group_fields,
            "aggregated": aggregated,
            "markers": markers,
            "event_vlines": event_vlines,
            "event_vline_style": self.event_vline_style,
            "event_vline_order": self.event_vline_columns,
            "x": self.x_norm,
            "channels": self.config["signal_groups"]["forceplate"]["columns"],
            "grid_layout": self.config["signal_groups"]["forceplate"]["grid_layout"],
            "window_spans": window_spans,
            "time_zero_frame": float(time_zero_frame),
            "window_span_alpha": self.forceplate_style["window_span_alpha"],
            "forceplate_style": self.forceplate_style,
            "common_style": self.common_style,
            "time_start_ms": self.time_start_ms,
            "time_end_ms": self.time_end_ms,
            "time_start_frame": self.time_start_frame,
            "time_end_frame": self.time_end_frame,
        }

    def _task_cop(
        self,
        *,
        aggregated: Dict[str, np.ndarray],
        output_path: Path,
        key: Tuple,
        mode_name: str,
        group_fields: List[str],
        markers: Dict[str, Any],
        event_vlines: List[Dict[str, Any]],
        window_spans: List[Dict[str, Any]],
        time_zero_frame: float = 0.0,
    ) -> Dict[str, Any]:
        return {
            "kind": "cop",
            "output_path": str(output_path),
            "plotly_html": bool(self.plotly_html_enabled),
            "key": key,
            "mode_name": mode_name,
            "group_fields": group_fields,
            "aggregated": aggregated,
            "markers": markers,
            "event_vlines": event_vlines,
            "event_vline_style": self.event_vline_style,
            "event_vline_order": self.event_vline_columns,
            "x_axis": self.x_norm,
            "target_axis": self.target_axis,
            "time_zero_frame": float(time_zero_frame),
            "time_start_ms": self.time_start_ms,
            "time_end_ms": self.time_end_ms,
            "time_start_frame": self.time_start_frame,
            "time_end_frame": self.time_end_frame,
            "device_rate": self.device_rate,
            "cop_channels": self.config["signal_groups"]["cop"]["columns"],
            "grid_layout": self.config["signal_groups"]["cop"]["grid_layout"],
            "cop_style": self.cop_style,
            "common_style": self.common_style,
            "window_spans": window_spans,
        }

    def _task_com(
        self,
        *,
        aggregated: Dict[str, np.ndarray],
        output_path: Path,
        key: Tuple,
        mode_name: str,
        group_fields: List[str],
        markers: Dict[str, Any],
        event_vlines: List[Dict[str, Any]],
        window_spans: List[Dict[str, Any]],
        time_zero_frame: float = 0.0,
    ) -> Dict[str, Any]:
        return {
            "kind": "com",
            "output_path": str(output_path),
            "plotly_html": bool(self.plotly_html_enabled),
            "key": key,
            "mode_name": mode_name,
            "group_fields": group_fields,
            "aggregated": aggregated,
            "markers": markers,
            "event_vlines": event_vlines,
            "event_vline_style": self.event_vline_style,
            "event_vline_order": self.event_vline_columns,
            "x_axis": self.x_norm,
            "time_zero_frame": float(time_zero_frame),
            "time_start_ms": self.time_start_ms,
            "time_end_ms": self.time_end_ms,
            "time_start_frame": self.time_start_frame,
            "time_end_frame": self.time_end_frame,
            "device_rate": self.device_rate,
            "com_channels": self.config["signal_groups"]["com"]["columns"],
            "grid_layout": self.config["signal_groups"]["com"]["grid_layout"],
            "com_style": self.com_style,
            "common_style": self.common_style,
            "window_spans": window_spans,
        }

