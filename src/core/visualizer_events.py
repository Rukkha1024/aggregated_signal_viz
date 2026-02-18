from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from ..plotting.matplotlib.common import _event_ms_col, _is_within_time_axis, _ms_to_norm, _nanmean_ignore_nan


class VisualizerEventsMixin:
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
