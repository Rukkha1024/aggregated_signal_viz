from __future__ import annotations

import concurrent.futures
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

from .config import (
    bom_rename_map,
    get_frame_ratio,
    load_config,
    resolve_output_dir,
    resolve_path,
    strip_bom_columns,
)


def ensure_output_dirs(base_path: Path, config: Dict[str, Any]) -> None:
    for mode_cfg in config.get("aggregation_modes", {}).values():
        out_dir = mode_cfg.get("output_dir")
        if not out_dir:
            continue
        resolve_output_dir(base_path, config, out_dir).mkdir(parents=True, exist_ok=True)


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



# Matplotlib plotting helpers were extracted into src.plotting.matplotlib.common
# to keep the core pipeline (data loading/resampling/task building) focused.
from ..plotting.matplotlib.common import (
    _build_event_vline_color_map,
    _calculate_filtered_group_fields,
    _coerce_bool,
    _coerce_float,
    _event_ms_col,
    _is_within_time_axis,
    _ms_to_norm,
    _nanmean_3d_over_first_axis,
    _nanmean_ignore_nan,
    _overlay_vline_event_names,
    _parse_event_labels,
    _parse_event_vlines_config,
    _parse_event_vlines_style,
    _parse_group_linestyles,
    _parse_window_boundary_spec,
    _parse_window_colors,
    _sort_overlay_keys,
)

from ..plotting.matplotlib.task import plot_task, plot_worker_init


@dataclass(frozen=True)
class _ResampledGroup:
    meta_df: pl.DataFrame
    tensor: np.ndarray  # (n_trials, n_channels, target_len)
    channels: List[str]


class AggregatedSignalVisualizer:
    def __init__(self, config_path: Path) -> None:
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)
        self.base_dir = self.config_path.parent

        self.id_cfg = self.config["data"]["id_columns"]
        self.device_rate = float(self.config["data"].get("device_sample_rate", 1000))
        self.frame_ratio = get_frame_ratio(self.config["data"])
        self._input_columns: Optional[set[str]] = None

        output_cfg = self.config.get("output", {})
        if not isinstance(output_cfg, dict):
            output_cfg = {}
        self.plotly_html_enabled = _coerce_bool(output_cfg.get("plotly_html"), False)
        self.emg_trial_grid_by_channel_enabled = _coerce_bool(output_cfg.get("emg_trial_grid_by_channel"), False)

        event_vlines_cfg = self.config.get("event_vlines")
        self.event_vline_columns = _parse_event_vlines_config(event_vlines_cfg)  # tick 라벨 우선순위는 이 순서를 따름
        self.event_vline_meta_cols = [_event_ms_col(col) for col in self.event_vline_columns]
        self.event_vline_style = _parse_event_vlines_style(event_vlines_cfg)
        self.event_vline_colors = _build_event_vline_color_map(self.event_vline_columns, event_vlines_cfg)
        self.event_vline_labels = _parse_event_labels(event_vlines_cfg)
        overlay_cfg = event_vlines_cfg.get("overlay_group") if isinstance(event_vlines_cfg, dict) else None
        self.event_vline_overlay_cfg: Optional[Dict[str, Any]] = overlay_cfg if isinstance(overlay_cfg, dict) else None

        windows_cfg = self.config.get("windows", {})
        self.window_reference_event: Optional[str] = None
        self.window_definition_specs: Dict[str, Dict[str, Tuple[str, Any]]] = {}
        window_event_cols: List[str] = []
        if isinstance(windows_cfg, dict):
            ref_event = windows_cfg.get("reference_event")
            if ref_event is not None:
                ref_num = _coerce_float(ref_event)
                if ref_num is not None and abs(float(ref_num)) <= 1e-9:
                    self.window_reference_event = None
                else:
                    ref_name = str(ref_event).strip()
                    if ref_name:
                        self.window_reference_event = ref_name
                        window_event_cols.append(ref_name)
            definitions = windows_cfg.get("definitions", {})
            if isinstance(definitions, dict):
                for key, cfg in definitions.items():
                    if not isinstance(cfg, dict):
                        continue
                    name = str(key).strip()
                    if not name:
                        continue
                    start_spec = _parse_window_boundary_spec(cfg.get("start_ms"))
                    end_spec = _parse_window_boundary_spec(cfg.get("end_ms"))
                    if start_spec is None or end_spec is None:
                        continue
                    self.window_definition_specs[name] = {"start": start_spec, "end": end_spec}
                    if start_spec[0] == "event":
                        window_event_cols.append(str(start_spec[1]))
                    if start_spec[0] == "event_offset":
                        window_event_cols.append(str(start_spec[1][0]))
                    if end_spec[0] == "event":
                        window_event_cols.append(str(end_spec[1]))
                    if end_spec[0] == "event_offset":
                        window_event_cols.append(str(end_spec[1][0]))

        self.window_event_columns = [c for c in dict.fromkeys(window_event_cols) if c]
        x_zero_cfg = self.config.get("x_axis_zeroing", {})
        onset_col_for_zero = str(self.id_cfg.get("onset") or "platform_onset").strip() or "platform_onset"
        self.x_axis_zeroing_enabled = False
        self.x_axis_zeroing_reference_event = onset_col_for_zero
        if isinstance(x_zero_cfg, dict):
            self.x_axis_zeroing_enabled = _coerce_bool(x_zero_cfg.get("enabled"), False)
            ref_raw = x_zero_cfg.get("reference_event")
            if ref_raw is not None and str(ref_raw).strip():
                self.x_axis_zeroing_reference_event = str(ref_raw).strip()

        x_zero_event_columns: List[str] = []
        if self.x_axis_zeroing_enabled:
            ref_col = str(self.x_axis_zeroing_reference_event or "").strip()
            if ref_col and ref_col != onset_col_for_zero:
                x_zero_event_columns.append(ref_col)

        self.required_event_columns = [
            c
            for c in dict.fromkeys([*self.event_vline_columns, *self.window_event_columns, *x_zero_event_columns])
            if c
        ]
        self.required_event_ms_meta_cols = [_event_ms_col(col) for col in self.required_event_columns]

        self.target_length = int(self.config["interpolation"]["target_length"])
        self.target_axis: Optional[np.ndarray] = None
        self.x_norm: Optional[np.ndarray] = None
        self.time_start_ms: Optional[float] = None
        self.time_end_ms: Optional[float] = None
        self.time_start_frame: Optional[float] = None
        self.time_end_frame: Optional[float] = None

        style_cfg = self.config["plot_style"]
        common_style_cfg = style_cfg["common"]
        self.common_style = self._build_common_style(common_style_cfg)
        self.emg_style = self._build_emg_style(style_cfg["emg"])
        self.forceplate_style = self._build_forceplate_style(style_cfg["forceplate"])
        self.cop_style = self._build_cop_style(style_cfg["cop"])
        self.com_style = self._build_com_style(style_cfg.get("com", {}), self.cop_style)
        self.window_colors = _parse_window_colors(common_style_cfg.get("window_colors"))
        if not self.window_colors:
            self.window_colors = _parse_window_colors(style_cfg.get("cop", {}).get("window_colors"))
        self.legend_label_threshold = self.common_style.get("legend_label_threshold", 6)

        self.features_df: Optional[pl.DataFrame] = self._load_features()
        self._emg_channel_specific_event_columns: set[str] = self._detect_emg_channel_specific_event_columns()
        self._feature_event_cache: Optional[pl.DataFrame] = None
        self._feature_event_cache_cols: Tuple[str, ...] = ()
        self._feature_event_cache_key_sig: Tuple[Tuple[str, str], ...] = ()
        self._feature_event_logged: bool = False
        self._x_axis_zeroing_missing_logged: bool = False
        self._windows_reference_event_logged: bool = False
        self._window_event_warning_logged: set[str] = set()

    def run(
        self,
        modes: Optional[Iterable[str]] = None,
        signal_groups: Optional[Iterable[str]] = None,
        sample: bool = False,
    ) -> None:
        selected_modes = set(modes) if modes else None
        selected_groups = set(signal_groups) if signal_groups else None

        enabled_modes: List[Tuple[str, Dict[str, Any]]] = []
        for mode_name, mode_cfg in self.config.get("aggregation_modes", {}).items():
            if not mode_cfg.get("enabled", True):
                continue
            if selected_modes and mode_name not in selected_modes:
                continue
            enabled_modes.append((mode_name, mode_cfg))

        if self.emg_trial_grid_by_channel_enabled:
            self._validate_emg_trial_grid_by_channel_modes(enabled_modes)

        lf = self._load_and_align_lazy()
        if sample:
            lf = self._filter_first_group(lf)
        lf = self._prepare_time_axis(lf)

        ensure_output_dirs(self.base_dir, self.config)

        meta_cols_needed = self._collect_needed_meta_columns(enabled_modes)
        for col in self.required_event_ms_meta_cols:
            if col not in meta_cols_needed:
                meta_cols_needed.append(col)

        generated_outputs: List[Path] = []
        group_names = self._signal_group_names(selected_groups)
        for group_name in group_names:
            resampled = self._resample_signal_group(lf, group_name, meta_cols_needed)
            if group_name == "emg" and self.emg_trial_grid_by_channel_enabled:
                generated_outputs.extend(
                    self._emit_emg_trial_grid_by_channel(
                        resampled=resampled,
                        enabled_modes=enabled_modes,
                        sample=sample,
                    )
                )
            tasks: List[Dict[str, Any]] = []
            for mode_name, mode_cfg in enabled_modes:
                tasks.extend(self._build_plot_tasks(resampled, group_name, mode_name, mode_cfg))
            self._run_plot_tasks(tasks)
            generated_outputs.extend(self._collect_existing_outputs(tasks))
            if self.plotly_html_enabled:
                generated_outputs.extend(self._collect_existing_plotly_html_outputs(tasks))
        self._log_generated_outputs(generated_outputs)

    def _run_plot_tasks(self, tasks: List[Dict[str, Any]]) -> None:
        if not tasks:
            return
        max_workers = self._max_workers()
        font_family = self.common_style.get("font_family")
        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=plot_worker_init,
                initargs=(font_family,),
            ) as ex:
                list(ex.map(plot_task, tasks, chunksize=max(1, len(tasks) // (max_workers * 4))))
        except PermissionError:
            plot_worker_init(font_family)
            for task in tasks:
                plot_task(task)

    @staticmethod
    def _collect_task_output_paths(tasks: List[Dict[str, Any]]) -> List[Path]:
        paths: List[Path] = []
        for task in tasks:
            output_path = task.get("output_path")
            if not output_path:
                continue
            paths.append(Path(output_path))
        return paths

    def _collect_existing_outputs(self, tasks: List[Dict[str, Any]]) -> List[Path]:
        return [path for path in self._collect_task_output_paths(tasks) if path.exists()]

    @staticmethod
    def _plotly_html_path_for_output_path(output_path: Path) -> Path:
        if output_path.suffix:
            return output_path.with_suffix(".html")
        return output_path.parent / f"{output_path.name}.html"

    def _collect_existing_plotly_html_outputs(self, tasks: List[Dict[str, Any]]) -> List[Path]:
        html_paths: List[Path] = []
        for task in tasks:
            if not bool(task.get("plotly_html", False)):
                continue
            output_path = task.get("output_path")
            if not output_path:
                continue
            html_path = self._plotly_html_path_for_output_path(Path(output_path))
            if html_path.exists():
                html_paths.append(html_path)
        return html_paths

    def _log_generated_outputs(self, output_paths: Iterable[Path]) -> None:
        seen: set[str] = set()
        unique_paths: List[Path] = []
        for path in output_paths:
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            unique_paths.append(path)

        if not unique_paths:
            print("[run] generated files: none")
            return

        base_dir = self.base_dir.resolve()
        counts: Dict[str, int] = {}
        for path in unique_paths:
            try:
                rel_path = path.resolve().relative_to(base_dir)
                parent = rel_path.parent
            except ValueError:
                parent = path.parent
            parent_str = str(parent)
            counts[parent_str] = counts.get(parent_str, 0) + 1

        print("[run] generated files (summary):")
        for parent_str in sorted(counts):
            print(f"  - {parent_str}: {counts[parent_str]} files")

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
        resampled: _ResampledGroup,
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

    def _resample_signal_group(self, lf: pl.LazyFrame, signal_group: str, meta_cols: List[str]) -> _ResampledGroup:
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
                return _ResampledGroup(meta_df=pl.DataFrame(), tensor=np.empty((0, 0, 0)), channels=[])
            raise ValueError(f"Missing required channels for '{signal_group}': {missing_channels}")
        if x_col not in available_cols:
            if bool(group_cfg.get("optional", False)):
                print(f"[{signal_group}] skip: missing time axis column: {x_col!r}")
                return _ResampledGroup(meta_df=pl.DataFrame(), tensor=np.empty((0, 0, 0)), channels=[])
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
        return _ResampledGroup(meta_df=meta_df, tensor=tensor, channels=channels)

    def _build_plot_tasks(
        self, resampled: _ResampledGroup, signal_group: str, mode_name: str, mode_cfg: Dict[str, Any]
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

    def _compute_window_spans(self, meta_df: pl.DataFrame, indices: np.ndarray) -> List[Dict[str, Any]]:
        if self.time_start_ms is None or self.time_end_ms is None:
            return []
        if indices.size == 0:
            return []
        if not self.window_definition_specs:
            return []

        shift_ms = self._compute_window_reference_shift_ms_from_meta(meta_df, indices)

        spans: List[Dict[str, Any]] = []
        for name, spec in self.window_definition_specs.items():
            start_ms = self._resolve_window_boundary_ms(spec["start"], meta_df, indices, shift_ms)
            end_ms = self._resolve_window_boundary_ms(spec["end"], meta_df, indices, shift_ms)
            if start_ms is None or end_ms is None:
                continue
            clamped_start = max(float(start_ms), float(self.time_start_ms))
            clamped_end = min(float(end_ms), float(self.time_end_ms))
            if clamped_start >= clamped_end:
                continue
            start_norm = _ms_to_norm(clamped_start, float(self.time_start_ms), float(self.time_end_ms))
            end_norm = _ms_to_norm(clamped_end, float(self.time_start_ms), float(self.time_end_ms))
            if start_norm is None or end_norm is None:
                continue
            spans.append(
                {
                    "name": name,
                    "start": float(start_norm),
                    "end": float(end_norm),
                    "label": f"{name} ({int(round(clamped_end - clamped_start))} ms)",
                    "color": self.window_colors.get(name, "#cccccc"),
                }
            )
        return spans

    def _resolve_time_zero_frame(
        self,
        *,
        meta_df: pl.DataFrame,
        indices: np.ndarray,
        mode_name: str,
        signal_group: str,
        key_label: str,
    ) -> float:
        if not self.x_axis_zeroing_enabled:
            return 0.0
        if indices.size == 0:
            raise ValueError(
                f"[x_axis_zeroing] empty trial set for mode='{mode_name}', signal_group='{signal_group}', key='{key_label}'"
            )

        onset_col = str(self.id_cfg.get("onset") or "").strip()
        ref_col = str(self.x_axis_zeroing_reference_event or "").strip()
        if not ref_col or ref_col == onset_col:
            return 0.0

        ms_col = _event_ms_col(ref_col)
        if ms_col not in meta_df.columns:
            raise ValueError(
                "[x_axis_zeroing] reference_event column is not available in this plot context: "
                f"reference_event='{ref_col}', mode='{mode_name}', signal_group='{signal_group}', key='{key_label}'"
            )

        vals = meta_df[ms_col].to_numpy()
        mean_ms = _nanmean_ignore_nan(vals[indices])
        if mean_ms is None:
            if not self._x_axis_zeroing_missing_logged:
                print(
                    "[x_axis_zeroing] Warning: reference_event has no valid values; falling back to 0. "
                    f"reference_event='{ref_col}', mode='{mode_name}', signal_group='{signal_group}', key='{key_label}'"
                )
                self._x_axis_zeroing_missing_logged = True
            return 0.0
        return self._ms_to_frame(float(mean_ms))

    def _resolve_time_zero_frame_by_channel(
        self,
        *,
        meta_df: pl.DataFrame,
        indices: np.ndarray,
        mode_name: str,
        signal_group: str,
        key_label: str,
    ) -> Dict[str, float]:
        if signal_group != "emg":
            return {}
        if not self.x_axis_zeroing_enabled:
            return {}

        onset_col = str(self.id_cfg.get("onset") or "").strip()
        ref_col = str(self.x_axis_zeroing_reference_event or "").strip()
        if not ref_col or ref_col == onset_col:
            return {}
        if ref_col not in self._emg_channel_specific_event_columns:
            return {}

        means_by_ch = self._collect_feature_event_means_by_emg_channel(
            meta_df=meta_df,
            indices=indices,
            event_cols=[ref_col],
        )
        if not means_by_ch:
            return {}

        out: Dict[str, float] = {}
        for ch, values in means_by_ch.items():
            ms = values.get(ref_col)
            if ms is None:
                continue
            try:
                out[str(ch)] = self._ms_to_frame(float(ms))
            except Exception:
                continue

        return out

    def _resolve_window_boundary_ms_from_means(
        self,
        spec: Tuple[str, Any],
        *,
        channel: str,
        event_means_by_channel: Dict[str, Dict[str, float]],
        global_event_means: Dict[str, float],
        shift_ms: float,
    ) -> Optional[float]:
        kind, value = spec
        if kind == "offset":
            return float(value) + float(shift_ms)

        offset_extra = 0.0
        if kind == "event":
            event_col = str(value).strip()
        elif kind == "event_offset":
            try:
                event_col = str(value[0]).strip()
                offset_extra = float(value[1])
            except (TypeError, ValueError, IndexError):
                return None
        else:
            return None

        if not event_col:
            return None

        mean_ms = event_means_by_channel.get(channel, {}).get(event_col)
        if mean_ms is None:
            mean_ms = global_event_means.get(event_col)
        if mean_ms is None:
            return None
        return float(mean_ms) + float(offset_extra)

    def _compute_window_spans_by_channel(
        self,
        meta_df: pl.DataFrame,
        indices: np.ndarray,
    ) -> Dict[str, List[Dict[str, Any]]]:
        if self.time_start_ms is None or self.time_end_ms is None:
            return {}
        if indices.size == 0:
            return {}
        if not self.window_definition_specs:
            return {}

        channels = self.config["signal_groups"]["emg"]["columns"]
        ref_col = str(self.window_reference_event or "").strip()
        onset_col = str(self.id_cfg.get("onset") or "").strip()

        needed_events: set[str] = set()
        if ref_col and ref_col != onset_col:
            needed_events.add(ref_col)
        for spec in self.window_definition_specs.values():
            for key in ("start", "end"):
                kind, value = spec[key]
                if kind == "event":
                    needed_events.add(str(value).strip())
                elif kind == "event_offset":
                    try:
                        needed_events.add(str(value[0]).strip())
                    except Exception:
                        continue

        needed_events = {c for c in needed_events if c}
        channel_events = [c for c in needed_events if c in self._emg_channel_specific_event_columns]

        event_means_by_channel = self._collect_feature_event_means_by_emg_channel(
            meta_df=meta_df,
            indices=indices,
            event_cols=channel_events,
        )

        global_event_means: Dict[str, float] = {}
        for event_col in needed_events:
            ms_col = _event_ms_col(event_col)
            if ms_col not in meta_df.columns:
                continue
            vals = meta_df[ms_col].to_numpy()
            mean_ms = _nanmean_ignore_nan(vals[indices])
            if mean_ms is None:
                continue
            global_event_means[event_col] = float(mean_ms)

        global_shift_ms = self._compute_window_reference_shift_ms_from_meta(meta_df, indices)

        out: Dict[str, List[Dict[str, Any]]] = {}
        for ch in channels:
            shift_ms = global_shift_ms
            if ref_col and ref_col != onset_col and ref_col in self._emg_channel_specific_event_columns:
                shift_ms = float(
                    event_means_by_channel.get(ch, {}).get(ref_col, global_event_means.get(ref_col, 0.0))
                )

            spans: List[Dict[str, Any]] = []
            for name, spec in self.window_definition_specs.items():
                start_ms = self._resolve_window_boundary_ms_from_means(
                    spec["start"],
                    channel=ch,
                    event_means_by_channel=event_means_by_channel,
                    global_event_means=global_event_means,
                    shift_ms=shift_ms,
                )
                end_ms = self._resolve_window_boundary_ms_from_means(
                    spec["end"],
                    channel=ch,
                    event_means_by_channel=event_means_by_channel,
                    global_event_means=global_event_means,
                    shift_ms=shift_ms,
                )
                if start_ms is None or end_ms is None:
                    continue
                clamped_start = max(float(start_ms), float(self.time_start_ms))
                clamped_end = min(float(end_ms), float(self.time_end_ms))
                if clamped_start >= clamped_end:
                    continue
                start_norm = _ms_to_norm(clamped_start, float(self.time_start_ms), float(self.time_end_ms))
                end_norm = _ms_to_norm(clamped_end, float(self.time_start_ms), float(self.time_end_ms))
                if start_norm is None or end_norm is None:
                    continue
                spans.append(
                    {
                        "name": name,
                        "start": float(start_norm),
                        "end": float(end_norm),
                        "label": f"{name} ({int(round(clamped_end - clamped_start))} ms)",
                        "color": self.window_colors.get(name, "#cccccc"),
                    }
                )
            if spans:
                out[ch] = spans
        return out

    def _compute_window_reference_shift_ms_from_meta(self, meta_df: pl.DataFrame, indices: np.ndarray) -> float:
        ref_col = str(self.window_reference_event or "").strip()
        if not ref_col:
            return 0.0

        onset_col = str(self.id_cfg.get("onset") or "").strip()
        if not onset_col or ref_col == onset_col:
            return 0.0

        ms_col = _event_ms_col(ref_col)
        if ms_col not in meta_df.columns:
            if not self._windows_reference_event_logged:
                print(
                    f"[windows] Warning: reference_event '{ref_col}' not available; numeric windows are not shifted"
                )
                self._windows_reference_event_logged = True
            return 0.0

        vals = meta_df[ms_col].to_numpy()
        mean_ms = _nanmean_ignore_nan(vals[indices])
        if mean_ms is None:
            if not self._windows_reference_event_logged:
                print(f"[windows] Warning: reference_event '{ref_col}' has no values; numeric windows are not shifted")
                self._windows_reference_event_logged = True
            return 0.0
        return float(mean_ms)

    def _resolve_window_boundary_ms(
        self,
        spec: Tuple[str, Any],
        meta_df: pl.DataFrame,
        indices: np.ndarray,
        shift_ms: float,
    ) -> Optional[float]:
        kind, value = spec
        if kind == "offset":
            return float(value) + float(shift_ms)
        offset_extra = 0.0
        if kind == "event":
            event_col = str(value).strip()
        elif kind == "event_offset":
            try:
                event_col = str(value[0]).strip()
                offset_extra = float(value[1])
            except (TypeError, ValueError, IndexError):
                return None
        else:
            return None
        if not event_col:
            return None
        ms_col = _event_ms_col(event_col)
        if ms_col not in meta_df.columns:
            if event_col not in self._window_event_warning_logged:
                print(f"[windows] Warning: event '{event_col}' not available for window boundaries; skipping")
                self._window_event_warning_logged.add(event_col)
            return None

        vals = meta_df[ms_col].to_numpy()
        mean_ms = _nanmean_ignore_nan(vals[indices])
        if mean_ms is None:
            if event_col not in self._window_event_warning_logged:
                print(f"[windows] Warning: event '{event_col}' has no values for window boundaries; skipping")
                self._window_event_warning_logged.add(event_col)
            return None
        return float(mean_ms) + float(offset_extra)

    def _collect_event_vlines(self, meta_df: pl.DataFrame, indices: np.ndarray) -> List[Dict[str, Any]]:
        if not self.event_vline_columns:
            return []
        if self.time_start_ms is None or self.time_end_ms is None:
            return []
        if indices.size == 0:
            return []

        out: List[Dict[str, Any]] = []
        for event_col in self.event_vline_columns:
            ms_col = _event_ms_col(event_col)
            if ms_col not in meta_df.columns:
                continue
            vals = meta_df[ms_col].to_numpy()
            mean_ms = _nanmean_ignore_nan(vals[indices])
            if mean_ms is None:
                continue
            if not _is_within_time_axis(mean_ms, self.time_start_ms, self.time_end_ms):
                continue
            x = _ms_to_norm(mean_ms, self.time_start_ms, self.time_end_ms)
            if x is None:
                continue
            out.append(
                {
                    "name": event_col,
                    "label": self.event_vline_labels.get(event_col, event_col),
                    "x": float(x),
                    "color": self.event_vline_colors.get(event_col),
                }
            )
        return out

    def _collect_emg_event_vlines_by_channel(
        self,
        meta_df: pl.DataFrame,
        indices: np.ndarray,
    ) -> Dict[str, List[Dict[str, Any]]]:
        if not self.event_vline_columns:
            return {}
        if self.time_start_ms is None or self.time_end_ms is None:
            return {}
        if indices.size == 0:
            return {}

        channels = self.config["signal_groups"]["emg"]["columns"]
        channel_event_cols = [c for c in self.event_vline_columns if c in self._emg_channel_specific_event_columns]
        event_means_by_channel = self._collect_feature_event_means_by_emg_channel(
            meta_df=meta_df,
            indices=indices,
            event_cols=channel_event_cols,
        )

        global_event_means: Dict[str, float] = {}
        for event_col in self.event_vline_columns:
            ms_col = _event_ms_col(event_col)
            if ms_col not in meta_df.columns:
                continue
            vals = meta_df[ms_col].to_numpy()
            mean_ms = _nanmean_ignore_nan(vals[indices])
            if mean_ms is None:
                continue
            global_event_means[event_col] = float(mean_ms)

        out: Dict[str, List[Dict[str, Any]]] = {}
        for ch in channels:
            vlines: List[Dict[str, Any]] = []
            for event_col in self.event_vline_columns:
                mean_ms = event_means_by_channel.get(ch, {}).get(event_col)
                if mean_ms is None:
                    mean_ms = global_event_means.get(event_col)
                if mean_ms is None:
                    continue
                if not _is_within_time_axis(mean_ms, self.time_start_ms, self.time_end_ms):
                    continue
                x = _ms_to_norm(mean_ms, self.time_start_ms, self.time_end_ms)
                if x is None:
                    continue
                vlines.append(
                    {
                        "name": event_col,
                        "label": self.event_vline_labels.get(event_col, event_col),
                        "x": float(x),
                        "color": self.event_vline_colors.get(event_col),
                    }
                )
            if vlines:
                out[ch] = vlines
        return out

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

    def _ms_to_frame(self, ms: float) -> float:
        return ms * self.device_rate / 1000.0

    def _frame_to_ms(self, frame: float) -> float:
        return frame / self.device_rate * 1000.0

    # All plot style parameters are configured via config.yaml under `plot_style`.
    # Module-level style constants were removed; do not reintroduce hard-coded styles.
    def _build_common_style(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "dpi": cfg["dpi"],
            "grid_alpha": cfg["grid_alpha"],
            "tick_labelsize": cfg["tick_labelsize"],
            "title_fontsize": cfg["title_fontsize"],
            "title_fontweight": cfg["title_fontweight"],
            "title_pad": cfg["title_pad"],
            "label_fontsize": cfg["label_fontsize"],
            "legend_loc": cfg["legend_loc"],
            "legend_framealpha": cfg["legend_framealpha"],
            "tight_layout_rect": cfg["tight_layout_rect"],
            "savefig_bbox_inches": cfg["savefig_bbox_inches"],
            "savefig_facecolor": cfg["savefig_facecolor"],
            "font_family": cfg["font_family"],
            "show_suptitle": bool(cfg.get("show_suptitle", True)),
            "show_subplot_titles": bool(cfg.get("show_subplot_titles", True)),
            "show_grid": bool(cfg.get("show_grid", True)),
            "show_legend": bool(cfg.get("show_legend", True)),
            "show_xlabel": bool(cfg.get("show_xlabel", True)),
            "show_ylabel": bool(cfg.get("show_ylabel", True)),
            "show_xtick_labels": bool(cfg.get("show_xtick_labels", True)),
            "show_ytick_labels": bool(cfg.get("show_ytick_labels", True)),
            "show_event_vlines": bool(cfg.get("show_event_vlines", True)),
            "show_windows": bool(cfg.get("show_windows", True)),
            "show_max_marker": cfg["show_max_marker"],
            "use_group_colors": bool(cfg.get("use_group_colors", False)),
            "group_linestyles": _parse_group_linestyles(cfg.get("group_linestyles")),
        }

    def _build_emg_style(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "subplot_size": tuple(cfg["subplot_size"]),
            "line_color": cfg["line_color"],
            "line_width": cfg["line_width"],
            "line_alpha": cfg["line_alpha"],
            "window_span_alpha": cfg["window_span_alpha"],
            "max_marker_color": cfg["max_marker_color"],
            "max_marker_linestyle": cfg["max_marker_linestyle"],
            "max_marker_linewidth": cfg["max_marker_linewidth"],
            "legend_fontsize": cfg["legend_fontsize"],
            "x_label": cfg["x_label"],
            "y_label": cfg["y_label"],
            "max_marker": {
                "color": cfg["max_marker_color"],
                "linestyle": cfg["max_marker_linestyle"],
                "linewidth": cfg["max_marker_linewidth"],
            },
        }

    def _build_forceplate_style(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "subplot_size": tuple(cfg["subplot_size"]),
            "line_colors": cfg["line_colors"],
            "axis_labels": dict(cfg.get("axis_labels", {})),
            "line_width": cfg["line_width"],
            "line_alpha": cfg["line_alpha"],
            "window_span_alpha": cfg["window_span_alpha"],
            "legend_fontsize": cfg["legend_fontsize"],
            "x_label": cfg["x_label"],
            "y_label": cfg["y_label"],
        }

    def _build_cop_style(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        out = {
            "subplot_size": tuple(cfg["subplot_size"]),
            "scatter_size": cfg["scatter_size"],
            "scatter_alpha": cfg["scatter_alpha"],
            "background_color": cfg["background_color"],
            "background_alpha": cfg["background_alpha"],
            "background_size": cfg["background_size"],
            "window_span_alpha": cfg.get("window_span_alpha", 0.15),
            "line_colors": cfg.get("line_colors", {"Cx": "gray", "Cy": "gray"}),
            "line_width": cfg.get("line_width", 0.8),
            "line_alpha": cfg.get("line_alpha", 0.8),
            "max_marker_color": cfg["max_marker_color"],
            "max_marker_size": cfg["max_marker_size"],
            "max_marker_symbol": cfg["max_marker_symbol"],
            "max_marker_edgecolor": cfg["max_marker_edgecolor"],
            "max_marker_linewidth": cfg["max_marker_linewidth"],
            "max_marker_zorder": cfg["max_marker_zorder"],
            "legend_fontsize": cfg["legend_fontsize"],
            "x_label_time": cfg.get("x_label_time", "Normalized time (0-1)"),
            "y_label_cx": cfg.get("y_label_cx", "Cx"),
            "y_label_cy": cfg.get("y_label_cy", "Cy"),
            "x_label": cfg["x_label"],
            "y_label": cfg["y_label"],
            "y_invert": bool(cfg["y_invert"]),
            "max_marker": {
                "size": cfg["max_marker_size"],
                "marker": cfg["max_marker_symbol"],
                "color": cfg["max_marker_color"],
                "edgecolor": cfg["max_marker_edgecolor"],
                "linewidth": cfg["max_marker_linewidth"],
                "zorder": cfg["max_marker_zorder"],
            },
            "overlay_scatter_edgewidth": cfg.get("overlay_scatter_edgewidth", 0.6),
        }
        return out

    @staticmethod
    def _build_com_style(com_cfg: Any, cop_style: Dict[str, Any]) -> Dict[str, Any]:
        base_line_colors = dict(cop_style.get("line_colors", {}) or {})
        comx_color = base_line_colors.get("Cx", "gray")
        comy_color = base_line_colors.get("Cy", "gray")
        comz_color = base_line_colors.get("Cz", "gray")

        if not isinstance(com_cfg, dict):
            com_cfg = {}

        def _cop_to_com(label: Any, fallback: str) -> str:
            if label is None:
                return fallback
            text = str(label)
            if not text:
                return fallback
            return text.replace("COP", "COM")

        default_y_label_comx = _cop_to_com(cop_style.get("y_label_cx", cop_style.get("x_label")), "COMx")
        default_y_label_comy = _cop_to_com(cop_style.get("y_label_cy", cop_style.get("y_label")), "COMy")
        default_x_label = _cop_to_com(cop_style.get("x_label"), "COMx")
        default_y_label = _cop_to_com(cop_style.get("y_label"), "COMy")

        def _cfg_or_default(key: str, default: str) -> str:
            val = com_cfg.get(key)
            if val is None:
                return default
            text = str(val)
            return text if text else default

        out = dict(cop_style)
        out["line_colors"] = {
            "COMx": comx_color,
            "COMx_zero": comx_color,
            "COMy": comy_color,
            "COMy_zero": comy_color,
            "COMz": comz_color,
            "COMz_zero": comz_color,
        }
        out["y_label_comx"] = _cfg_or_default("y_label_comx", default_y_label_comx)
        out["y_label_comy"] = _cfg_or_default("y_label_comy", default_y_label_comy)
        out["y_label_comz"] = _cfg_or_default("y_label_comz", "COMz")
        out["x_label"] = _cfg_or_default("x_label", default_x_label)
        out["y_label"] = _cfg_or_default("y_label", default_y_label)
        out.setdefault("x_label_time", "Normalized time (0-1)")
        out.setdefault("y_invert", False)
        return out

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
