from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .config import get_frame_ratio, load_config, resolve_output_dir
from .visualizer_data import VisualizerDataMixin
from .visualizer_events import VisualizerEventsMixin
from .visualizer_markers import VisualizerMarkersMixin
from .visualizer_style import VisualizerStyleMixin
from .visualizer_tasks import VisualizerTasksMixin
from ..plotting.matplotlib.common import (
    _build_event_vline_color_map,
    _coerce_bool,
    _coerce_float,
    _event_ms_col,
    _parse_event_labels,
    _parse_event_vlines_config,
    _parse_event_vlines_style,
    _parse_window_boundary_spec,
    _parse_window_colors,
)
from ..plotting.matplotlib.task import plot_task, plot_worker_init


def ensure_output_dirs(base_path: Path, config: Dict[str, Any]) -> None:
    for mode_cfg in config.get("aggregation_modes", {}).values():
        out_dir = mode_cfg.get("output_dir")
        if not out_dir:
            continue
        resolve_output_dir(base_path, config, out_dir).mkdir(parents=True, exist_ok=True)


class AggregatedSignalVisualizer(
    VisualizerTasksMixin,
    VisualizerEventsMixin,
    VisualizerMarkersMixin,
    VisualizerStyleMixin,
    VisualizerDataMixin,
):
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
        self.target_axis = None
        self.x_norm = None
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

        self.features_df = self._load_features()
        self._emg_channel_specific_event_columns: set[str] = self._detect_emg_channel_specific_event_columns()
        self._feature_event_cache = None
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
