from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from scipy.interpolate import interp1d

plt.rcParams["axes.unicode_minus"] = False  # Avoid font warnings for minus symbols.


@dataclass
class AggregatedRecord:
    subject: str
    velocity: float
    trial: int
    data: Dict[str, np.ndarray]

    def get(self, key: str) -> Optional[object]:
        return getattr(self, key, None)


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_output_dirs(base_path: Path, config: Dict) -> None:
    for mode_cfg in config.get("aggregation_modes", {}).values():
        out_dir = mode_cfg.get("output_dir")
        if not out_dir:
            continue
        Path(base_path, out_dir).mkdir(parents=True, exist_ok=True)


class AggregatedSignalVisualizer:
    def __init__(self, config_path: Path) -> None:
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)
        self.base_dir = self.config_path.parent
        self.id_cfg = self.config["data"]["id_columns"]
        self.device_rate = self.config["data"].get("device_sample_rate", 1000)
        mocap_rate = self.config["data"].get("mocap_sample_rate", 100)
        self.frame_ratio = self.config["data"].get("frame_ratio") or int(self.device_rate / mocap_rate)
        self.target_length = int(self.config["interpolation"]["target_length"])
        self.interp_method = self.config["interpolation"]["method"]
        self.target_axis: Optional[np.ndarray] = None
        self.resampled: Dict[str, List[AggregatedRecord]] = {}
        style_cfg = self.config.get("plot_style", {})
        self.common_style = self._build_common_style(style_cfg.get("common"))
        self.emg_style = self._build_emg_style(style_cfg.get("emg"))
        self.forceplate_style = self._build_forceplate_style(style_cfg.get("forceplate"))
        self.cop_style = self._build_cop_style(style_cfg.get("cop"))
        self.window_colors = self.cop_style.get("window_colors", {})
        font_family = self.common_style.get("font_family")
        if font_family:
            plt.rcParams["font.family"] = font_family
        self.window_frames = self._compute_window_frames()
        self.features_df: Optional[pl.DataFrame] = self._load_features()

    # All plot style parameters are configured via config.yaml under `plot_style`.
    # Module-level style constants were removed; do not reintroduce hard-coded styles.
    def _build_common_style(self, cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        defaults = {
            "dpi": 300,
            "grid_alpha": 0.3,
            "tick_labelsize": 7,
            "title_fontsize": 20,
            "title_fontweight": "bold",
            "title_pad": 5,
            "label_fontsize": 8,
            "legend_loc": "best",
            "legend_framealpha": 0.8,
            "tight_layout_rect": [0, 0, 1, 0.99],
            "savefig_bbox_inches": "tight",
            "savefig_facecolor": "white",
            "font_family": "Malgun Gothic",
        }
        return self._merge_style(defaults, cfg)

    def _build_emg_style(self, cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        defaults = {
            "subplot_size": (12, 6),
            "line_color": "blue",
            "line_width": 0.8,
            "line_alpha": 0.8,
            "window_span_alpha": 0.15,
            "onset_marker_color": "red",
            "onset_marker_linestyle": "--",
            "onset_marker_linewidth": 1.5,
            "max_marker_color": "orange",
            "max_marker_linestyle": "--",
            "max_marker_linewidth": 1.5,
            "legend_fontsize": 6,
            "x_label": "Frame (normalized)",
            "y_label": "{channel}",
        }
        style = self._merge_style(defaults, cfg)
        style["subplot_size"] = tuple(style.get("subplot_size", defaults["subplot_size"]))
        style["onset_marker"] = {
            "color": style["onset_marker_color"],
            "linestyle": style["onset_marker_linestyle"],
            "linewidth": style["onset_marker_linewidth"],
        }
        style["max_marker"] = {
            "color": style["max_marker_color"],
            "linestyle": style["max_marker_linestyle"],
            "linewidth": style["max_marker_linewidth"],
        }
        return style

    def _build_forceplate_style(self, cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        defaults = {
            "subplot_size": (12, 6),
            "line_colors": {"Fx": "purple", "Fy": "brown", "Fz": "green"},
            "line_width": 0.8,
            "line_alpha": 0.8,
            "window_span_alpha": 0.15,
            "onset_marker_color": "red",
            "onset_marker_linestyle": "--",
            "onset_marker_linewidth": 1.5,
            "legend_fontsize": 6,
            "x_label": "Frame (normalized)",
            "y_label": "{channel} Value",
        }
        style = self._merge_style(defaults, cfg)
        style["subplot_size"] = tuple(style.get("subplot_size", defaults["subplot_size"]))
        style["onset_marker"] = {
            "color": style["onset_marker_color"],
            "linestyle": style["onset_marker_linestyle"],
            "linewidth": style["onset_marker_linewidth"],
        }
        return style

    def _build_cop_style(self, cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        defaults = {
            "subplot_size": (8, 8),
            "scatter_size": 8,
            "scatter_alpha": 0.7,
            "background_color": "lightgray",
            "background_alpha": 0.3,
            "background_size": 6,
            "window_colors": {"p1": "#4E79A7", "p2": "#F28E2B", "p3": "#E15759", "p4": "#59A14F"},
            "max_marker_color": "#ED1C24",
            "max_marker_size": 80,
            "max_marker_symbol": "*",
            "max_marker_edgecolor": "white",
            "max_marker_linewidth": 1,
            "max_marker_zorder": 10,
            "legend_fontsize": 5,
            "x_label": "Cx (R+/L-)",
            "y_label": "Cy (A+)",
            "y_invert": True,
        }
        style = self._merge_style(defaults, cfg)
        style["subplot_size"] = tuple(style.get("subplot_size", defaults["subplot_size"]))
        style["max_marker"] = {
            "size": style["max_marker_size"],
            "marker": style["max_marker_symbol"],
            "color": style["max_marker_color"],
            "edgecolor": style["max_marker_edgecolor"],
            "linewidth": style["max_marker_linewidth"],
            "zorder": style["max_marker_zorder"],
        }
        style["y_invert"] = bool(style.get("y_invert"))
        return style

    @staticmethod
    def _merge_style(defaults: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        style = defaults.copy()
        if overrides:
            style.update({k: v for k, v in overrides.items() if v is not None})
        return style

    def run(
        self,
        modes: Optional[Iterable[str]] = None,
        signal_groups: Optional[Iterable[str]] = None,
        sample: bool = False,
    ) -> None:
        selected_modes = set(modes) if modes else None
        selected_groups = set(signal_groups) if signal_groups else None

        df = self._load_and_align()
        if sample:
            df = self._sample_single_group(df)
        self.target_axis = self._build_target_axis(df)
        self.resampled = self._resample_all(df, selected_groups)

        for mode_name, mode_cfg in self.config["aggregation_modes"].items():
            if not mode_cfg.get("enabled", True):
                continue
            if selected_modes and mode_name not in selected_modes:
                continue
            for signal_group in self._signal_group_names(selected_groups):
                self._run_mode(signal_group, mode_name, mode_cfg)

    def _signal_group_names(self, selected_groups: Optional[Iterable[str]]) -> List[str]:
        names = list(self.config["signal_groups"].keys())
        if selected_groups is None:
            return names
        return [n for n in names if n in selected_groups]

    def _load_and_align(self) -> pl.DataFrame:
        input_path = Path(self.config["data"]["input_file"])
        if not input_path.is_absolute():
            input_path = (self.base_dir / input_path).resolve()
        df = pl.read_csv(input_path)
        df = df.rename({c: c.lstrip("\ufeff") for c in df.columns})

        task_col = self.id_cfg.get("task")
        task_filter = self.config["data"].get("task_filter")
        if task_filter and task_col in df.columns:
            df = df.filter(pl.col(task_col) == task_filter)
        if df.is_empty():
            raise ValueError("No data available after applying task or input filters.")

        subject_col = self.id_cfg["subject"]
        velocity_col = self.id_cfg["velocity"]
        trial_col = self.id_cfg["trial"]
        frame_col = self.id_cfg["frame"]
        mocap_col = self.id_cfg["mocap_frame"]
        onset_col = self.id_cfg["onset"]
        offset_col = self.id_cfg["offset"]

        group_cols = [subject_col, velocity_col, trial_col]
        # Align DeviceFrame so platform_onset becomes 0 using mocap→device frame ratio.
        mocap_start = pl.col(mocap_col).min().over(group_cols)
        onset_device = (pl.col(onset_col).first().over(group_cols) - mocap_start) * self.frame_ratio
        onset_aligned = pl.col(frame_col) - onset_device
        offset_rel = (
            (pl.col(offset_col).first().over(group_cols) - pl.col(onset_col).first().over(group_cols))
            * self.frame_ratio
        )

        df = df.with_columns(
            [
                onset_device.alias("onset_device_frame"),
                onset_aligned.alias("aligned_frame"),
                offset_rel.alias("offset_from_onset"),
            ]
        ).sort(group_cols + ["aligned_frame"])
        return df

    def _sample_single_group(self, df: pl.DataFrame) -> pl.DataFrame:
        subject_col = self.id_cfg["subject"]
        velocity_col = self.id_cfg["velocity"]
        trial_col = self.id_cfg["trial"]
        if df.is_empty():
            return df

        first_subject, first_velocity, first_trial = df.select(
            pl.col(subject_col).first(),
            pl.col(velocity_col).first(),
            pl.col(trial_col).first(),
        ).row(0)

        return df.filter(
            (pl.col(subject_col) == first_subject)
            & (pl.col(velocity_col) == first_velocity)
            & (pl.col(trial_col) == first_trial)
        )

    def _build_target_axis(self, df: pl.DataFrame) -> np.ndarray:
        frame_min = df.select(pl.col("aligned_frame").min()).item()
        frame_max = df.select(pl.col("aligned_frame").max()).item()
        if frame_max == frame_min:
            frame_min -= 0.5
            frame_max += 0.5
        return np.linspace(frame_min, frame_max, self.target_length)

    def _resample_all(
        self, df: pl.DataFrame, selected_groups: Optional[Iterable[str]]
    ) -> Dict[str, List[AggregatedRecord]]:
        subject_col = self.id_cfg["subject"]
        velocity_col = self.id_cfg["velocity"]
        trial_col = self.id_cfg["trial"]
        group_cols = [subject_col, velocity_col, trial_col]

        records: Dict[str, List[AggregatedRecord]] = {k: [] for k in self.config["signal_groups"]}
        groups = df.group_by(group_cols, maintain_order=True)
        for key, subdf in groups:
            subdf_sorted = subdf.sort("aligned_frame")
            meta = {
                subject_col: key[0],
                velocity_col: float(key[1]),
                trial_col: int(key[2]),
            }
            for group_name, cfg in self.config["signal_groups"].items():
                if selected_groups and group_name not in selected_groups:
                    continue
                data = self._interpolate_group(subdf_sorted, cfg["columns"])
                records[group_name].append(
                    AggregatedRecord(
                        subject=meta[subject_col],
                        velocity=meta[velocity_col],
                        trial=meta[trial_col],
                        data=data,
                    )
                )
        return records

    def _interpolate_group(self, df: pl.DataFrame, columns: List[str]) -> Dict[str, np.ndarray]:
        assert self.target_axis is not None, "target_axis must be initialized before interpolation"
        x = df["aligned_frame"].to_numpy()
        data: Dict[str, np.ndarray] = {}
        for col in columns:
            y = df[col].to_numpy()
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() < 2:
                data[col] = np.full_like(self.target_axis, np.nan, dtype=float)
                continue
            f = interp1d(
                x[valid],
                y[valid],
                kind=self.interp_method,
                bounds_error=False,
                fill_value=np.nan,
                assume_sorted=True,
            )
            data[col] = f(self.target_axis)
        return data

    def _run_mode(self, signal_group: str, mode_name: str, mode_cfg: Dict) -> None:
        records = self.resampled.get(signal_group, [])
        if not records:
            return

        filtered_records = self._apply_filter(records, mode_cfg.get("filter"))
        if not filtered_records:
            return

        group_fields = mode_cfg.get("groupby", [])
        grouped = self._group_records(filtered_records, group_fields)

        for key, recs in grouped.items():
            aggregated = self._aggregate_group(recs)
            filename = self._render_filename(mode_cfg["filename_pattern"], key, signal_group, group_fields)
            output_dir = Path(self.base_dir, mode_cfg["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / filename
            markers = self._collect_markers(signal_group, key, group_fields, mode_cfg.get("filter"))
            self._plot(signal_group, aggregated, output_path, key, mode_name, group_fields, markers)

    def _apply_filter(
        self, records: List[AggregatedRecord], filter_cfg: Optional[Dict]
    ) -> List[AggregatedRecord]:
        if not filter_cfg:
            return records
        col = filter_cfg["column"]
        value = filter_cfg["value"]
        return [r for r in records if getattr(r, col, None) == value]

    def _group_records(
        self, records: List[AggregatedRecord], group_fields: List[str]
    ) -> Dict[Tuple, List[AggregatedRecord]]:
        if not group_fields:
            return {("all",): records}

        grouped: Dict[Tuple, List[AggregatedRecord]] = {}
        for rec in records:
            key = tuple(getattr(rec, f) for f in group_fields)
            grouped.setdefault(key, []).append(rec)
        return grouped

    def _aggregate_group(self, records: List[AggregatedRecord]) -> Dict[str, np.ndarray]:
        assert records, "No records to aggregate"
        channels = records[0].data.keys()
        aggregated: Dict[str, np.ndarray] = {}
        for ch in channels:
            stack = np.vstack([r.data[ch] for r in records])
            nan_template = np.full_like(self.target_axis, np.nan, dtype=float)  # type: ignore[arg-type]
            if np.all(np.isnan(stack)):
                aggregated[ch] = nan_template
            else:
                nan_cols = np.all(np.isnan(stack), axis=0)
                if (~nan_cols).any():
                    nan_template[~nan_cols] = np.nanmean(stack[:, ~nan_cols], axis=0)
                aggregated[ch] = nan_template
        return aggregated

    def _render_filename(
        self, pattern: str, key: Tuple, signal_group: str, group_fields: List[str]
    ) -> str:
        if key == ("all",):
            return pattern.format(signal_group=signal_group)
        mapping = {field: value for field, value in zip(group_fields, key)}
        mapping["signal_group"] = signal_group
        return pattern.format(**mapping)

    def _plot(
        self,
        signal_group: str,
        aggregated: Dict[str, np.ndarray],
        output_path: Path,
        key: Tuple,
        mode_name: str,
        group_fields: List[str],
        markers: Dict[str, Any],
    ) -> None:
        if signal_group == "emg":
            self._plot_emg(aggregated, output_path, key, mode_name, group_fields, markers)
        elif signal_group == "forceplate":
            self._plot_forceplate(aggregated, output_path, key, mode_name, group_fields, markers)
        elif signal_group == "cop":
            self._plot_cop(aggregated, output_path, key, mode_name, group_fields, markers)

    def _plot_emg(
        self,
        aggregated: Dict[str, np.ndarray],
        output_path: Path,
        key: Tuple,
        mode_name: str,
        group_fields: List[str],
        markers: Dict[str, Any],
    ) -> None:
        rows, cols = self.config["signal_groups"]["emg"]["grid_layout"]
        fig, axes = plt.subplots(rows, cols, figsize=self.emg_style["subplot_size"], dpi=self.common_style["dpi"])
        axes_flat = axes.flatten()
        x = self.target_axis
        channels = self.config["signal_groups"]["emg"]["columns"]
        for ax, ch in zip(axes_flat, channels):
            y = aggregated.get(ch)
            if y is None:
                ax.axis("off")
                continue
            ax.plot(
                x,
                y,
                self.emg_style["line_color"],
                linewidth=self.emg_style["line_width"],
                alpha=self.emg_style["line_alpha"],
                label=ch,
            )
            for name, (start, end) in self.window_frames.items():
                ax.axvspan(
                    start,
                    end,
                    color=self.window_colors.get(name, "#cccccc"),
                    alpha=self.emg_style["window_span_alpha"],
                )
            marker_info = markers.get(ch, {})
            onset_time = marker_info.get("onset")
            if onset_time is not None:
                ax.axvline(onset_time, **self.emg_style["onset_marker"], label="onset")
            max_time = marker_info.get("max")
            if max_time is not None:
                ax.axvline(max_time, **self.emg_style["max_marker"], label="max")
            ax.set_title(
                ch,
                fontsize=self.common_style["title_fontsize"],
                fontweight=self.common_style["title_fontweight"],
                pad=self.common_style["title_pad"],
            )
            ax.grid(True, alpha=self.common_style["grid_alpha"])
            ax.tick_params(labelsize=self.common_style["tick_labelsize"])
            ax.legend(
                fontsize=self.emg_style["legend_fontsize"],
                loc=self.common_style["legend_loc"],
                framealpha=self.common_style["legend_framealpha"],
            )
        for ax in axes_flat[len(channels) :]:
            ax.axis("off")
        fig.suptitle(
            self._format_title(signal_group="emg", mode_name=mode_name, group_fields=group_fields, key=key),
            fontsize=self.common_style["title_fontsize"],
            fontweight=self.common_style["title_fontweight"],
        )
        fig.supxlabel(self.emg_style["x_label"], fontsize=self.common_style["label_fontsize"])
        y_label = self._format_label(self.emg_style.get("y_label", "Amplitude"), channel="Amplitude")
        fig.supylabel(y_label, fontsize=self.common_style["label_fontsize"])
        fig.tight_layout(rect=self.common_style["tight_layout_rect"])
        fig.savefig(
            output_path,
            bbox_inches=self.common_style["savefig_bbox_inches"],
            facecolor=self.common_style["savefig_facecolor"],
        )
        plt.close(fig)

    def _plot_forceplate(
        self,
        aggregated: Dict[str, np.ndarray],
        output_path: Path,
        key: Tuple,
        mode_name: str,
        group_fields: List[str],
        markers: Dict[str, Any],
    ) -> None:
        rows, cols = self.config["signal_groups"]["forceplate"]["grid_layout"]
        fig, axes = plt.subplots(
            rows, cols, figsize=self.forceplate_style["subplot_size"], dpi=self.common_style["dpi"]
        )
        x = self.target_axis
        for ax, ch in zip(np.ravel(axes), self.config["signal_groups"]["forceplate"]["columns"]):
            y = aggregated[ch]
            color = self.forceplate_style["line_colors"].get(ch, "blue")
            ax.plot(
                x,
                y,
                color=color,
                linewidth=self.forceplate_style["line_width"],
                alpha=self.forceplate_style["line_alpha"],
                label=ch,
            )
            for name, (start, end) in self.window_frames.items():
                ax.axvspan(
                    start,
                    end,
                    color=self.window_colors.get(name, "#cccccc"),
                    alpha=self.forceplate_style["window_span_alpha"],
                )
            onset_time = markers.get(ch, {}).get("onset")
            if onset_time is not None:
                ax.axvline(onset_time, **self.forceplate_style["onset_marker"], label="onset")
            ax.set_title(
                ch,
                fontsize=self.common_style["title_fontsize"],
                fontweight=self.common_style["title_fontweight"],
                pad=self.common_style["title_pad"],
            )
            ax.grid(True, alpha=self.common_style["grid_alpha"])
            ax.tick_params(labelsize=self.common_style["tick_labelsize"])
            ax.legend(
                fontsize=self.forceplate_style["legend_fontsize"],
                loc=self.common_style["legend_loc"],
                framealpha=self.common_style["legend_framealpha"],
            )
            ax.set_xlabel(self.forceplate_style["x_label"], fontsize=self.common_style["label_fontsize"])
            y_label = self._format_label(self.forceplate_style.get("y_label", "{channel} Value"), channel=ch)
            ax.set_ylabel(y_label, fontsize=self.common_style["label_fontsize"])
        fig.suptitle(
            self._format_title(signal_group="forceplate", mode_name=mode_name, group_fields=group_fields, key=key),
            fontsize=self.common_style["title_fontsize"],
            fontweight=self.common_style["title_fontweight"],
        )
        fig.tight_layout(rect=self.common_style["tight_layout_rect"])
        fig.savefig(
            output_path,
            bbox_inches=self.common_style["savefig_bbox_inches"],
            facecolor=self.common_style["savefig_facecolor"],
        )
        plt.close(fig)

    def _plot_cop(
        self,
        aggregated: Dict[str, np.ndarray],
        output_path: Path,
        key: Tuple,
        mode_name: str,
        group_fields: List[str],
        markers: Dict[str, Dict[str, float]],
    ) -> None:
        cx = aggregated.get("Cx")
        cy = aggregated.get("Cy")
        if cx is None or cy is None:
            return
        x_vals = cx
        y_vals = -cy if self.cop_style["y_invert"] else cy
        fig, ax = plt.subplots(1, 1, figsize=self.cop_style["subplot_size"], dpi=self.common_style["dpi"])
        ax.scatter(
            x_vals,
            y_vals,
            color=self.cop_style["background_color"],
            alpha=self.cop_style["background_alpha"],
            s=self.cop_style["background_size"],
            label="trajectory",
        )
        for name, (start, end) in self.window_frames.items():
            mask = (self.target_axis >= start) & (self.target_axis <= end)
            if mask.any():
                ax.scatter(
                    x_vals[mask],
                    y_vals[mask],
                    s=self.cop_style["scatter_size"],
                    alpha=self.cop_style["scatter_alpha"],
                    color=self.window_colors.get(name, "#999999"),
                    label=name,
                )
        max_time = markers.get("max")
        if max_time is not None:
            idx = self._closest_index(self.target_axis, max_time)
            ax.scatter(
                x_vals[idx],
                y_vals[idx],
                s=self.cop_style["max_marker"]["size"],
                marker=self.cop_style["max_marker"]["marker"],
                color=self.cop_style["max_marker"]["color"],
                edgecolor=self.cop_style["max_marker"]["edgecolor"],
                linewidth=self.cop_style["max_marker"]["linewidth"],
                zorder=self.cop_style["max_marker"]["zorder"],
                label="max",
            )
        ax.grid(True, alpha=self.common_style["grid_alpha"])
        ax.tick_params(labelsize=self.common_style["tick_labelsize"])
        ax.legend(
            fontsize=self.cop_style["legend_fontsize"],
            loc=self.common_style["legend_loc"],
            framealpha=self.common_style["legend_framealpha"],
        )
        ax.set_xlabel(self.cop_style["x_label"], fontsize=self.common_style["label_fontsize"])
        ax.set_ylabel(self.cop_style["y_label"], fontsize=self.common_style["label_fontsize"])
        ax.set_aspect("equal", adjustable="box")
        fig.suptitle(
            self._format_title(signal_group="cop", mode_name=mode_name, group_fields=group_fields, key=key),
            fontsize=self.common_style["title_fontsize"],
            fontweight=self.common_style["title_fontweight"],
        )
        fig.tight_layout(rect=self.common_style["tight_layout_rect"])
        fig.savefig(
            output_path,
            bbox_inches=self.common_style["savefig_bbox_inches"],
            facecolor=self.common_style["savefig_facecolor"],
        )
        plt.close(fig)

    def _compute_window_frames(self) -> Dict[str, Tuple[float, float]]:
        frames = {}
        definitions = self.config.get("windows", {}).get("definitions", {})
        for name, cfg in definitions.items():
            # Convert ms offsets relative to onset into device-frame units for plotting.
            start = cfg["start_ms"] * self.device_rate / 1000
            end = cfg["end_ms"] * self.device_rate / 1000
            frames[name] = (start, end)
        return frames

    def _format_title(
        self, signal_group: str, mode_name: str, group_fields: List[str], key: Tuple
    ) -> str:
        if key == ("all",):
            return f"{mode_name} | {signal_group}"
        parts = [f"{field}={value}" for field, value in zip(group_fields, key)]
        return f"{mode_name} | {signal_group} | " + ", ".join(parts)

    def _collect_markers(
        self,
        signal_group: str,
        key: Tuple,
        group_fields: List[str],
        filter_cfg: Optional[Dict],
    ) -> Dict[str, Any]:
        if self.features_df is None:
            return {}
        df = self.features_df
        if filter_cfg:
            col = filter_cfg["column"]
            val = filter_cfg["value"]
            if col in df.columns:
                df = df.filter(pl.col(col) == val)
        for field, value in zip(group_fields, key):
            if field in df.columns:
                df = df.filter(pl.col(field) == value)
        if df.is_empty():
            return {}
        if signal_group == "emg":
            return self._collect_emg_markers(df)
        if signal_group == "forceplate":
            return self._collect_forceplate_markers(df)
        # COP plots no longer rely on feature-based markers.
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

    @staticmethod
    def _closest_index(arr: np.ndarray, value: float) -> int:
        return int(np.nanargmin(np.abs(arr - value)))

    @staticmethod
    def _format_label(template: Any, **kwargs: Any) -> str:
        if not isinstance(template, str):
            return str(template)
        try:
            return template.format(**kwargs)
        except (KeyError, ValueError):
            return template

    def _load_features(self) -> Optional[pl.DataFrame]:
        features_path = self.config["data"].get("features_file")
        if not features_path:
            return None
        path = Path(features_path)
        if not path.is_absolute():
            path = (self.base_dir / path).resolve()
        if not path.exists():
            return None
        df = pl.read_csv(path)
        df = df.rename({c: c.lstrip("\ufeff") for c in df.columns})
        return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregated signal visualization")
    default_config = Path(__file__).resolve().parent / "config.yaml"
    parser.add_argument("--config", type=str, default=str(default_config), help="Path to YAML config.")
    parser.add_argument("--sample", action="store_true", help="Run on a single sample (first subject-velocity-trial group).")
    parser.add_argument(
        "--modes",
        type=str,
        nargs="*",
        default=None,
        help="Aggregation modes to run (default: all enabled).",
    )
    parser.add_argument(
        "--groups",
        type=str,
        nargs="*",
        default=None,
        help="Signal groups to run (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visualizer = AggregatedSignalVisualizer(Path(args.config))
    ensure_output_dirs(visualizer.base_dir, visualizer.config)
    visualizer.run(modes=args.modes, signal_groups=args.groups)


if __name__ == "__main__":
    main()
