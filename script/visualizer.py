from __future__ import annotations

import argparse
import concurrent.futures
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl
import yaml


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_output_dirs(base_path: Path, config: Dict[str, Any]) -> None:
    for mode_cfg in config.get("aggregation_modes", {}).values():
        out_dir = mode_cfg.get("output_dir")
        if not out_dir:
            continue
        Path(base_path, out_dir).mkdir(parents=True, exist_ok=True)


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


_PLOT_FONT_FAMILY: Optional[str] = None


def _plot_worker_init(font_family: Optional[str]) -> None:
    global _PLOT_FONT_FAMILY
    _PLOT_FONT_FAMILY = font_family

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # noqa: F401

    plt.rcParams["axes.unicode_minus"] = False
    if font_family:
        plt.rcParams["font.family"] = font_family


def _ms_to_norm(value: float, time_start_ms: float, time_end_ms: float) -> Optional[float]:
    denom = time_end_ms - time_start_ms
    if denom == 0:
        return None
    return (value - time_start_ms) / denom


def _is_within_time_axis(value_ms: float, time_start_ms: float, time_end_ms: float) -> bool:
    return time_start_ms <= value_ms <= time_end_ms


def _ms_to_frame(ms: float, device_rate: float) -> float:
    return ms * device_rate / 1000.0


def _closest_index(arr: np.ndarray, value: float) -> int:
    return int(np.nanargmin(np.abs(arr - value)))


def _format_label(template: Any, **kwargs: Any) -> str:
    if not isinstance(template, str):
        return str(template)
    try:
        return template.format(**kwargs)
    except (KeyError, ValueError):
        return template


def _format_title(signal_group: str, mode_name: str, group_fields: List[str], key: Tuple) -> str:
    if key == ("all",):
        return f"{mode_name} | {signal_group}"
    parts = [f"{field}={value}" for field, value in zip(group_fields, key)]
    return f"{mode_name} | {signal_group} | " + ", ".join(parts)


def _resolve_cop_channel_names(channels: Sequence[str]) -> Tuple[str, str]:
    if len(channels) < 2:
        raise ValueError("COP requires at least 2 channels. Check signal_groups.cop.columns in config.yaml.")

    lower_names = [ch.lower() for ch in channels]
    cx_idx = next((i for i, name in enumerate(lower_names) if "cx" in name), None)
    cy_idx = next((i for i, name in enumerate(lower_names) if "cy" in name), None)

    cx_name = channels[cx_idx] if cx_idx is not None else None
    cy_name = channels[cy_idx] if cy_idx is not None else None

    if cx_name and cy_name:
        return cx_name, cy_name

    remaining = [ch for ch in channels if ch not in (cx_name, cy_name)]
    if cx_name is None:
        cx_name = remaining[0] if remaining else channels[0]
        remaining = [ch for ch in remaining if ch != cx_name]
    if cy_name is None:
        cy_name = remaining[0] if remaining else channels[1]

    return cx_name, cy_name


def _calculate_filtered_group_fields(
    sorted_keys: List[Tuple],
    group_fields: List[str],
    threshold: int = 6
) -> List[str]:
    """
    groupby column별로 unique 값 개수를 계산하여 threshold 미만인 field만 반환.

    Args:
        sorted_keys: groupby 결과 key 리스트
        group_fields: groupby에 사용된 field 이름 리스트
        threshold: unique 값 개수 임계값 (이상이면 제외)

    Returns:
        threshold 미만의 unique 값을 가진 field 리스트
    """
    if not sorted_keys or not group_fields:
        return []

    # ("all",) 특수 케이스 처리
    if any(key == ("all",) for key in sorted_keys):
        return []

    filtered = []
    for i, field in enumerate(group_fields):
        # 해당 field의 인덱스 i에서 모든 key의 값을 추출하여 unique 개수 계산
        unique_values = {key[i] for key in sorted_keys if i < len(key)}
        if len(unique_values) < threshold:
            filtered.append(field)

    return filtered


def _format_group_label(
    key: Tuple,
    group_fields: List[str],
    filtered_group_fields: Optional[List[str]] = None
) -> Optional[str]:
    """
    groupby key를 legend label 문자열로 변환.

    Args:
        key: groupby 결과 key tuple
        group_fields: 전체 groupby field 리스트
        filtered_group_fields: 필터링된 field 리스트 (None이면 전체 사용)

    Returns:
        legend label 문자열 또는 None (label 없이 plot)
    """
    if not group_fields or key == ("all",):
        return "all"

    # filtered_group_fields가 제공된 경우
    if filtered_group_fields is not None:
        # 빈 리스트 → 모든 column이 6개 이상 → label 없이 plot
        if not filtered_group_fields:
            return None

        # filtered_group_fields만 사용하여 label 생성
        field_to_value = dict(zip(group_fields, key))
        if len(filtered_group_fields) == 1:
            return str(field_to_value[filtered_group_fields[0]])
        return ", ".join(
            f"{field}={field_to_value[field]}"
            for field in filtered_group_fields
            if field in field_to_value
        )

    # 기존 로직 (backward compatibility)
    if len(group_fields) == 1:
        return str(key[0])
    return ", ".join(f"{field}={value}" for field, value in zip(group_fields, key))


def _sort_overlay_keys(keys: List[Tuple], group_fields: List[str]) -> List[Tuple]:
    if group_fields == ["step_TF"]:
        order = {"nonstep": 0, "step": 1}

        def sort_key(key: Tuple) -> Tuple:
            val = key[0] if key else None
            sval = "" if val is None else str(val)
            return (order.get(sval, 99), sval, tuple(str(v) for v in key))

        return sorted(keys, key=sort_key)
    return sorted(keys, key=lambda key: tuple(str(v) for v in key))


def _nanmean_3d_over_first_axis(arr: np.ndarray) -> np.ndarray:
    """
    arr: (N, C, T) -> mean over N, preserving NaN where all values are NaN.
    """
    if arr.size == 0:
        return np.full(arr.shape[1:], np.nan, dtype=float)
    all_nan = np.all(np.isnan(arr), axis=0)
    out = np.full(arr.shape[1:], np.nan, dtype=float)
    if (~all_nan).any():
        out[~all_nan] = np.nanmean(arr[:, ~all_nan], axis=0)
    return out


def _plot_task(task: Dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    kind = task["kind"]
    common_style = task["common_style"]
    output_path = Path(task["output_path"])

    if kind == "overlay":
        _plot_overlay_generic(
            signal_group=task["signal_group"],
            aggregated_by_key=task["aggregated_by_key"],
            markers_by_key=task.get("markers_by_key", {}),
            output_path=output_path,
            mode_name=task["mode_name"],
            group_fields=task["group_fields"],
            sorted_keys=[tuple(k) for k in task["sorted_keys"]],
            x=np.asarray(task["x"], dtype=float),
            channels=task.get("channels"),
            grid_layout=task.get("grid_layout"),
            cop_channels=task.get("cop_channels"),
            window_spans=task["window_spans"],
            window_span_alpha=task.get("window_span_alpha"),
            style=task["style"],
            common_style=common_style,
            time_start_ms=task.get("time_start_ms"),
            time_end_ms=task.get("time_end_ms"),
            filtered_group_fields=task["filtered_group_fields"],
            color_by_fields=task.get("color_by_fields"),
        )
        return

    if kind == "emg":
        _plot_emg(
            aggregated=task["aggregated"],
            output_path=output_path,
            key=tuple(task["key"]),
            mode_name=task["mode_name"],
            group_fields=task["group_fields"],
            markers=task["markers"],
            x=np.asarray(task["x"], dtype=float),
            channels=task["channels"],
            grid_layout=task["grid_layout"],
            window_spans=task["window_spans"],
            window_span_alpha=task["window_span_alpha"],
            emg_style=task["emg_style"],
            common_style=common_style,
            time_start_ms=task["time_start_ms"],
            time_end_ms=task["time_end_ms"],
        )
        return

    if kind == "forceplate":
        _plot_forceplate(
            aggregated=task["aggregated"],
            output_path=output_path,
            key=tuple(task["key"]),
            mode_name=task["mode_name"],
            group_fields=task["group_fields"],
            markers=task["markers"],
            x=np.asarray(task["x"], dtype=float),
            channels=task["channels"],
            grid_layout=task["grid_layout"],
            window_spans=task["window_spans"],
            window_span_alpha=task["window_span_alpha"],
            forceplate_style=task["forceplate_style"],
            common_style=common_style,
            time_start_ms=task["time_start_ms"],
            time_end_ms=task["time_end_ms"],
        )
        return

    if kind == "cop":
        _plot_cop(
            aggregated=task["aggregated"],
            output_path=output_path,
            key=tuple(task["key"]),
            mode_name=task["mode_name"],
            group_fields=task["group_fields"],
            markers=task["markers"],
            x_axis=np.asarray(task["x_axis"], dtype=float) if task["x_axis"] is not None else None,
            target_axis=np.asarray(task["target_axis"], dtype=float) if task["target_axis"] is not None else None,
            time_start_ms=task["time_start_ms"],
            time_end_ms=task["time_end_ms"],
            device_rate=float(task["device_rate"]),
            cop_channels=task["cop_channels"],
            cop_style=task["cop_style"],
            common_style=common_style,
            window_spans=task["window_spans"],
        )
        return

    plt.close("all")
    raise ValueError(f"Unknown plot task kind: {kind!r}")


def _plot_overlay_generic(
    *,
    signal_group: str,
    aggregated_by_key: Dict[Tuple, Dict[str, np.ndarray]],
    markers_by_key: Dict[Tuple, Dict[str, Any]],
    output_path: Path,
    mode_name: str,
    group_fields: List[str],
    sorted_keys: List[Tuple],
    x: np.ndarray,
    channels: Optional[List[str]],
    grid_layout: Optional[List[int]],
    cop_channels: Optional[Sequence[str]],
    window_spans: List[Dict[str, Any]],
    window_span_alpha: Optional[float],
    style: Dict[str, Any],
    common_style: Dict[str, Any],
    time_start_ms: Optional[float],
    time_end_ms: Optional[float],
    filtered_group_fields: List[str],
    color_by_fields: Optional[List[str]] = None,
) -> None:
    if signal_group == "cop":
        _plot_cop_overlay(
            aggregated_by_key=aggregated_by_key,
            output_path=output_path,
            mode_name=mode_name,
            group_fields=group_fields,
            sorted_keys=sorted_keys,
            x=x,
            window_spans=window_spans,
            cop_channels=cop_channels or (),
            cop_style=style,
            common_style=common_style,
            filtered_group_fields=filtered_group_fields,
            color_by_fields=color_by_fields,
        )
        return

    if channels is None or grid_layout is None or window_span_alpha is None or time_start_ms is None or time_end_ms is None:
        raise ValueError(f"Missing required overlay parameters for {signal_group=}")

    _plot_overlay_timeseries_grid(
        aggregated_by_key=aggregated_by_key,
        markers_by_key=markers_by_key,
        output_path=output_path,
        mode_name=mode_name,
        signal_group=signal_group,
        group_fields=group_fields,
        sorted_keys=sorted_keys,
        x=x,
        channels=channels,
        grid_layout=grid_layout,
        window_spans=window_spans,
        window_span_alpha=window_span_alpha,
        style=style,
        common_style=common_style,
        time_start_ms=time_start_ms,
        time_end_ms=time_end_ms,
        filtered_group_fields=filtered_group_fields,
        color_by_fields=color_by_fields,
    )


def _plot_emg(
    *,
    aggregated: Dict[str, np.ndarray],
    output_path: Path,
    key: Tuple,
    mode_name: str,
    group_fields: List[str],
    markers: Dict[str, Any],
    x: np.ndarray,
    channels: List[str],
    grid_layout: List[int],
    window_spans: List[Dict[str, Any]],
    window_span_alpha: float,
    emg_style: Dict[str, Any],
    common_style: Dict[str, Any],
    time_start_ms: float,
    time_end_ms: float,
) -> None:
    import matplotlib.pyplot as plt

    rows, cols = grid_layout
    fig, axes = plt.subplots(rows, cols, figsize=emg_style["subplot_size"], dpi=common_style["dpi"])
    axes_flat = axes.flatten()

    for ax, ch in zip(axes_flat, channels):
        y = aggregated.get(ch)
        if y is None:
            ax.axis("off")
            continue

        ax.plot(
            x,
            y,
            emg_style["line_color"],
            linewidth=emg_style["line_width"],
            alpha=emg_style["line_alpha"],
        )

        for span in window_spans:
            ax.axvspan(
                span["start"],
                span["end"],
                color=span["color"],
                alpha=window_span_alpha,
                label=span["label"],
            )

        marker_info = markers.get(ch, {})
        if common_style.get("show_onset_marker", True):
            onset_time = marker_info.get("onset")
            if onset_time is not None and _is_within_time_axis(onset_time, time_start_ms, time_end_ms):
                onset_norm = _ms_to_norm(onset_time, time_start_ms, time_end_ms)
                if onset_norm is not None:
                    ax.axvline(onset_norm, **emg_style["onset_marker"], label="onset")
        if common_style.get("show_max_marker", True):
            max_time = marker_info.get("max")
            if max_time is not None and _is_within_time_axis(max_time, time_start_ms, time_end_ms):
                max_norm = _ms_to_norm(max_time, time_start_ms, time_end_ms)
                if max_norm is not None:
                    ax.axvline(max_norm, **emg_style["max_marker"], label="max")

        ax.set_title(
            ch,
            fontsize=common_style["title_fontsize"],
            fontweight=common_style["title_fontweight"],
            pad=common_style["title_pad"],
        )
        ax.grid(True, alpha=common_style["grid_alpha"])
        ax.tick_params(labelsize=common_style["tick_labelsize"])
        ax.legend(
            fontsize=emg_style["legend_fontsize"],
            loc=common_style["legend_loc"],
            framealpha=common_style["legend_framealpha"],
        )

    for ax in axes_flat[len(channels) :]:
        ax.axis("off")

    fig.suptitle(
        _format_title(signal_group="emg", mode_name=mode_name, group_fields=group_fields, key=key),
        fontsize=common_style["title_fontsize"],
        fontweight=common_style["title_fontweight"],
    )
    fig.supxlabel(emg_style["x_label"], fontsize=common_style["label_fontsize"])
    y_label = _format_label(emg_style.get("y_label", "Amplitude"), channel="Amplitude")
    fig.supylabel(y_label, fontsize=common_style["label_fontsize"])
    fig.tight_layout(rect=common_style["tight_layout_rect"])
    fig.savefig(
        output_path,
        bbox_inches=common_style["savefig_bbox_inches"],
        facecolor=common_style["savefig_facecolor"],
    )
    plt.close(fig)


def _plot_overlay_timeseries_grid(
    *,
    aggregated_by_key: Dict[Tuple, Dict[str, np.ndarray]],
    markers_by_key: Dict[Tuple, Dict[str, Any]],
    output_path: Path,
    mode_name: str,
    signal_group: str,
    group_fields: List[str],
    sorted_keys: List[Tuple],
    x: np.ndarray,
    channels: List[str],
    grid_layout: List[int],
    window_spans: List[Dict[str, Any]],
    window_span_alpha: float,
    style: Dict[str, Any],
    common_style: Dict[str, Any],
    time_start_ms: float,
    time_end_ms: float,
    filtered_group_fields: List[str],
    color_by_fields: Optional[List[str]] = None,
) -> None:
    import matplotlib.pyplot as plt

    rows, cols = grid_layout
    fig, axes = plt.subplots(rows, cols, figsize=style["subplot_size"], dpi=common_style["dpi"])

    if signal_group == "emg":
        axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else np.asarray([axes])

        for ax, ch in zip(axes_flat, channels):
            seen_labels: set[str] = set()
            has_any_series = any(aggregated_by_key.get(key, {}).get(ch) is not None for key in sorted_keys)
            if not has_any_series:
                ax.axis("off")
                continue

            for span in window_spans:
                ax.axvspan(
                    span["start"],
                    span["end"],
                    color=span["color"],
                    alpha=window_span_alpha,
                    label=span["label"],
                )

            # If color_by_fields is specified, group keys by color_by field(s) combination
            if color_by_fields:
                # Group keys by color_by field(s) combination
                color_groups = {}
                for key in sorted_keys:
                    field_to_value = dict(zip(group_fields, key))
                    # Create tuple of values for color_by fields
                    color_key = tuple(field_to_value.get(field) for field in color_by_fields)
                    if color_key not in color_groups:
                        color_groups[color_key] = []
                    color_groups[color_key].append(key)

                # Assign colors by color groups
                import matplotlib.pyplot as plt

                colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
                seen_color_labels: set[str] = set()

                for color_idx, (color_key, keys_list) in enumerate(color_groups.items()):
                    color = colors[color_idx % len(colors)]
                    # Create color label for legend
                    if len(color_by_fields) == 1:
                        color_label = str(color_key[0])
                    else:
                        color_label = ", ".join(
                            f"{field}={value}" for field, value in zip(color_by_fields, color_key)
                        )

                    for key in keys_list:
                        y = aggregated_by_key.get(key, {}).get(ch)
                        if y is None:
                            continue

                        # Only add to legend for first series of this color group
                        if color_label not in seen_color_labels:
                            plot_label = color_label
                            seen_color_labels.add(color_label)
                        else:
                            plot_label = "_nolegend_"

                        ax.plot(
                            x,
                            y,
                            color=color,
                            linewidth=style["line_width"],
                            alpha=style["line_alpha"],
                            label=plot_label,
                        )
            else:
                # No color_by specified - use single color for all lines
                single_color = style.get("line_color", "blue")
                for key in sorted_keys:
                    y = aggregated_by_key.get(key, {}).get(ch)
                    if y is None:
                        continue
                    group_label = _format_group_label(key, group_fields, filtered_group_fields)
                    if group_label is None or group_label in seen_labels:
                        plot_label = "_nolegend_"
                    else:
                        plot_label = group_label
                        seen_labels.add(group_label)
                    ax.plot(
                        x,
                        y,
                        color=single_color,
                        linewidth=style["line_width"],
                        alpha=style["line_alpha"],
                        label=plot_label,
                    )

            for key in sorted_keys:
                marker_info = markers_by_key.get(key, {}).get(ch, {})
                marker_label = _format_group_label(key, group_fields)
                if common_style.get("show_onset_marker", True):
                    onset_time = marker_info.get("onset")
                    if onset_time is not None and _is_within_time_axis(onset_time, time_start_ms, time_end_ms):
                        onset_norm = _ms_to_norm(onset_time, time_start_ms, time_end_ms)
                        if onset_norm is not None:
                            ax.axvline(onset_norm, **style["onset_marker"], label=f"{marker_label} onset")
                if common_style.get("show_max_marker", True):
                    max_time = marker_info.get("max")
                    if max_time is not None and _is_within_time_axis(max_time, time_start_ms, time_end_ms):
                        max_norm = _ms_to_norm(max_time, time_start_ms, time_end_ms)
                        if max_norm is not None:
                            ax.axvline(max_norm, **style["max_marker"], label=f"{marker_label} max")

            ax.set_title(
                ch,
                fontsize=common_style["title_fontsize"],
                fontweight=common_style["title_fontweight"],
                pad=common_style["title_pad"],
            )
            ax.grid(True, alpha=common_style["grid_alpha"])
            ax.tick_params(labelsize=common_style["tick_labelsize"])
            ax.legend(
                fontsize=style["legend_fontsize"],
                loc=common_style["legend_loc"],
                framealpha=common_style["legend_framealpha"],
            )

        for ax in axes_flat[len(channels) :]:
            ax.axis("off")

        overlay_by = ", ".join(group_fields) if group_fields else "all"
        fig.suptitle(
            f"{mode_name} | emg | overlay by {overlay_by}",
            fontsize=common_style["title_fontsize"],
            fontweight=common_style["title_fontweight"],
        )
        fig.supxlabel(style["x_label"], fontsize=common_style["label_fontsize"])
        y_label = _format_label(style.get("y_label", "Amplitude"), channel="Amplitude")
        fig.supylabel(y_label, fontsize=common_style["label_fontsize"])
        fig.tight_layout(rect=common_style["tight_layout_rect"])
        fig.savefig(
            output_path,
            bbox_inches=common_style["savefig_bbox_inches"],
            facecolor=common_style["savefig_facecolor"],
        )
        plt.close(fig)
        return

    if signal_group == "forceplate":
        for ax, ch in zip(np.ravel(axes), channels):
            seen_labels: set[str] = set()
            for span in window_spans:
                ax.axvspan(
                    span["start"],
                    span["end"],
                    color=span["color"],
                    alpha=window_span_alpha,
                    label=span["label"],
                )

            for key in sorted_keys:
                y = aggregated_by_key.get(key, {}).get(ch)
                if y is None:
                    continue
                group_label = _format_group_label(key, group_fields, filtered_group_fields)
                if group_label is None or group_label in seen_labels:
                    plot_label = "_nolegend_"
                else:
                    plot_label = group_label
                    seen_labels.add(group_label)
                ax.plot(
                    x,
                    y,
                    linewidth=style["line_width"],
                    alpha=style["line_alpha"],
                    label=plot_label,
                )

            if common_style.get("show_onset_marker", True):
                for key in sorted_keys:
                    onset_time = markers_by_key.get(key, {}).get(ch, {}).get("onset")
                    if onset_time is None or not _is_within_time_axis(onset_time, time_start_ms, time_end_ms):
                        continue
                    onset_norm = _ms_to_norm(onset_time, time_start_ms, time_end_ms)
                    if onset_norm is None:
                        continue
                    marker_label = _format_group_label(key, group_fields)
                    ax.axvline(onset_norm, **style["onset_marker"], label=f"{marker_label} onset")

            ax.set_title(
                ch,
                fontsize=common_style["title_fontsize"],
                fontweight=common_style["title_fontweight"],
                pad=common_style["title_pad"],
            )
            ax.grid(True, alpha=common_style["grid_alpha"])
            ax.tick_params(labelsize=common_style["tick_labelsize"])
            ax.legend(
                fontsize=style["legend_fontsize"],
                loc=common_style["legend_loc"],
                framealpha=common_style["legend_framealpha"],
            )
            ax.set_xlabel(style["x_label"], fontsize=common_style["label_fontsize"])
            y_label = _format_label(style.get("y_label", "{channel} Value"), channel=ch)
            ax.set_ylabel(y_label, fontsize=common_style["label_fontsize"])

        overlay_by = ", ".join(group_fields) if group_fields else "all"
        fig.suptitle(
            f"{mode_name} | forceplate | overlay by {overlay_by}",
            fontsize=common_style["title_fontsize"],
            fontweight=common_style["title_fontweight"],
        )
        fig.tight_layout(rect=common_style["tight_layout_rect"])
        fig.savefig(
            output_path,
            bbox_inches=common_style["savefig_bbox_inches"],
            facecolor=common_style["savefig_facecolor"],
        )
        plt.close(fig)
        return

    plt.close(fig)
    raise ValueError(f"Unknown signal_group for overlay: {signal_group!r}")


def _plot_forceplate(
    *,
    aggregated: Dict[str, np.ndarray],
    output_path: Path,
    key: Tuple,
    mode_name: str,
    group_fields: List[str],
    markers: Dict[str, Any],
    x: np.ndarray,
    channels: List[str],
    grid_layout: List[int],
    window_spans: List[Dict[str, Any]],
    window_span_alpha: float,
    forceplate_style: Dict[str, Any],
    common_style: Dict[str, Any],
    time_start_ms: float,
    time_end_ms: float,
) -> None:
    import matplotlib.pyplot as plt

    rows, cols = grid_layout
    fig, axes = plt.subplots(rows, cols, figsize=forceplate_style["subplot_size"], dpi=common_style["dpi"])

    for ax, ch in zip(np.ravel(axes), channels):
        y = aggregated.get(ch)
        if y is None:
            ax.axis("off")
            continue

        color = forceplate_style["line_colors"].get(ch, "blue")
        ax.plot(
            x,
            y,
            color=color,
            linewidth=forceplate_style["line_width"],
            alpha=forceplate_style["line_alpha"],
        )

        for span in window_spans:
            ax.axvspan(
                span["start"],
                span["end"],
                color=span["color"],
                alpha=window_span_alpha,
                label=span["label"],
            )

        if common_style.get("show_onset_marker", True):
            onset_time = markers.get(ch, {}).get("onset")
            if onset_time is not None and _is_within_time_axis(onset_time, time_start_ms, time_end_ms):
                onset_norm = _ms_to_norm(onset_time, time_start_ms, time_end_ms)
                if onset_norm is not None:
                    ax.axvline(onset_norm, **forceplate_style["onset_marker"], label="onset")

        ax.set_title(
            ch,
            fontsize=common_style["title_fontsize"],
            fontweight=common_style["title_fontweight"],
            pad=common_style["title_pad"],
        )
        ax.grid(True, alpha=common_style["grid_alpha"])
        ax.tick_params(labelsize=common_style["tick_labelsize"])
        ax.legend(
            fontsize=forceplate_style["legend_fontsize"],
            loc=common_style["legend_loc"],
            framealpha=common_style["legend_framealpha"],
        )
        ax.set_xlabel(forceplate_style["x_label"], fontsize=common_style["label_fontsize"])
        y_label = _format_label(forceplate_style.get("y_label", "{channel} Value"), channel=ch)
        ax.set_ylabel(y_label, fontsize=common_style["label_fontsize"])

    fig.suptitle(
        _format_title(signal_group="forceplate", mode_name=mode_name, group_fields=group_fields, key=key),
        fontsize=common_style["title_fontsize"],
        fontweight=common_style["title_fontweight"],
    )
    fig.tight_layout(rect=common_style["tight_layout_rect"])
    fig.savefig(
        output_path,
        bbox_inches=common_style["savefig_bbox_inches"],
        facecolor=common_style["savefig_facecolor"],
    )
    plt.close(fig)


def _plot_cop(
    *,
    aggregated: Dict[str, np.ndarray],
    output_path: Path,
    key: Tuple,
    mode_name: str,
    group_fields: List[str],
    markers: Dict[str, Dict[str, float]],
    x_axis: Optional[np.ndarray],
    target_axis: Optional[np.ndarray],
    time_start_ms: float,
    time_end_ms: float,
    device_rate: float,
    cop_channels: Sequence[str],
    cop_style: Dict[str, Any],
    common_style: Dict[str, Any],
    window_spans: List[Dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    cx_name, cy_name = _resolve_cop_channel_names(cop_channels)
    cx = aggregated.get(cx_name)
    cy = aggregated.get(cy_name)
    if cx is None or cy is None:
        available = ", ".join(sorted(aggregated.keys()))
        print(f"[cop] missing channels: Cx={cx_name!r}, Cy={cy_name!r}. Available: {available}")
        return

    x = x_axis
    if x is None:
        x = np.linspace(0.0, 1.0, num=cx.size, dtype=float)

    x_vals = cx
    y_vals = -cy if cop_style["y_invert"] else cy

    fig, axes = plt.subplots(1, 3, figsize=cop_style["subplot_size"], dpi=common_style["dpi"])
    ax_cx, ax_cy, ax_scatter = axes

    window_span_alpha = float(cop_style.get("window_span_alpha", 0.15))

    for ax in (ax_cx, ax_cy):
        for span in window_spans:
            ax.axvspan(
                span["start"],
                span["end"],
                color=span["color"],
                alpha=window_span_alpha,
                label="_nolegend_",
            )

    cx_color = cop_style.get("line_colors", {}).get("Cx", "blue")
    cy_color = cop_style.get("line_colors", {}).get("Cy", "red")

    ax_cx.plot(
        x,
        cx,
        color=cx_color,
        linewidth=cop_style.get("line_width", 0.8),
        alpha=cop_style.get("line_alpha", 0.8),
        label="Cx",
    )
    ax_cy.plot(
        x,
        y_vals,
        color=cy_color,
        linewidth=cop_style.get("line_width", 0.8),
        alpha=cop_style.get("line_alpha", 0.8),
        label="Cy",
    )

    ax_scatter.scatter(
        x_vals,
        y_vals,
        color=cop_style["background_color"],
        alpha=cop_style["background_alpha"],
        s=cop_style["background_size"],
    )

    if x_axis is not None:
        for span in window_spans:
            mask = (x_axis >= span["start"]) & (x_axis <= span["end"])
            if mask.any():
                ax_scatter.scatter(
                    x_vals[mask],
                    y_vals[mask],
                    s=cop_style["scatter_size"],
                    alpha=cop_style["scatter_alpha"],
                    color=span["color"],
                    label=span["label"],
                )

    if common_style.get("show_max_marker", True):
        max_time = markers.get("max")
        if (
            max_time is not None
            and _is_within_time_axis(max_time, time_start_ms, time_end_ms)
            and target_axis is not None
        ):
            target_frame = _ms_to_frame(max_time, device_rate)
            idx = _closest_index(target_axis, target_frame)
            ax_scatter.scatter(
                x_vals[idx],
                y_vals[idx],
                s=cop_style["max_marker"]["size"],
                marker=cop_style["max_marker"]["marker"],
                color=cop_style["max_marker"]["color"],
                edgecolor=cop_style["max_marker"]["edgecolor"],
                linewidth=cop_style["max_marker"]["linewidth"],
                zorder=cop_style["max_marker"]["zorder"],
                label="max",
            )

    ax_cx.set_title(
        cx_name,
        fontsize=common_style["title_fontsize"],
        fontweight=common_style["title_fontweight"],
        pad=common_style["title_pad"],
    )
    ax_cy.set_title(
        cy_name,
        fontsize=common_style["title_fontsize"],
        fontweight=common_style["title_fontweight"],
        pad=common_style["title_pad"],
    )
    ax_scatter.set_title(
        "Cxy",
        fontsize=common_style["title_fontsize"],
        fontweight=common_style["title_fontweight"],
        pad=common_style["title_pad"],
    )

    for ax in (ax_cx, ax_cy, ax_scatter):
        ax.grid(True, alpha=common_style["grid_alpha"])
        ax.tick_params(labelsize=common_style["tick_labelsize"])
        ax.legend(
            fontsize=cop_style["legend_fontsize"],
            loc=common_style["legend_loc"],
            framealpha=common_style["legend_framealpha"],
        )

    ax_cx.set_xlabel(cop_style.get("x_label_time", "Normalized time (0-1)"), fontsize=common_style["label_fontsize"])
    ax_cx.set_ylabel(cop_style.get("y_label_cx", "Cx"), fontsize=common_style["label_fontsize"])
    ax_cy.set_xlabel(cop_style.get("x_label_time", "Normalized time (0-1)"), fontsize=common_style["label_fontsize"])
    ax_cy.set_ylabel(cop_style.get("y_label_cy", "Cy"), fontsize=common_style["label_fontsize"])

    ax_scatter.set_xlabel(cop_style["x_label"], fontsize=common_style["label_fontsize"])
    ax_scatter.set_ylabel(cop_style["y_label"], fontsize=common_style["label_fontsize"])
    ax_scatter.set_aspect("equal", adjustable="datalim")

    # Scatter 축에 window 색상 legend를 별도로 제공 (배경/라인 legend와 분리).
    try:
        import matplotlib.patches as mpatches

        window_handles = [
            mpatches.Patch(facecolor=span["color"], edgecolor="none", label=span["label"]) for span in window_spans
        ]
        if window_handles:
            window_legend = ax_scatter.legend(
                handles=window_handles,
                fontsize=cop_style["legend_fontsize"],
                loc="upper right",
                framealpha=common_style["legend_framealpha"],
                title="window",
            )
            ax_scatter.add_artist(window_legend)
    except Exception:
        pass

    fig.suptitle(
        _format_title(signal_group="cop", mode_name=mode_name, group_fields=group_fields, key=key),
        fontsize=common_style["title_fontsize"],
        fontweight=common_style["title_fontweight"],
    )
    fig.tight_layout(rect=common_style["tight_layout_rect"])
    fig.savefig(
        output_path,
        facecolor=common_style["savefig_facecolor"],
    )
    plt.close(fig)


def _plot_cop_overlay(
    *,
    aggregated_by_key: Dict[Tuple, Dict[str, np.ndarray]],
    output_path: Path,
    mode_name: str,
    group_fields: List[str],
    sorted_keys: List[Tuple],
    x: np.ndarray,
    window_spans: List[Dict[str, Any]],
    cop_channels: Sequence[str],
    cop_style: Dict[str, Any],
    common_style: Dict[str, Any],
    filtered_group_fields: List[str],
    color_by_fields: Optional[List[str]] = None,
) -> None:
    import matplotlib.pyplot as plt

    cx_name, cy_name = _resolve_cop_channel_names(cop_channels)

    fig, axes = plt.subplots(1, 3, figsize=cop_style["subplot_size"], dpi=common_style["dpi"])
    ax_cx, ax_cy, ax_scatter = axes

    window_span_alpha = float(cop_style.get("window_span_alpha", 0.15))
    for ax in (ax_cx, ax_cy):
        for span in window_spans:
            ax.axvspan(
                span["start"],
                span["end"],
                color=span["color"],
                alpha=window_span_alpha,
                label="_nolegend_",
            )

    import matplotlib as mpl

    base_colors = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "H"]
    key_to_marker = {k: marker_cycle[i % len(marker_cycle)] for i, k in enumerate(sorted_keys)}

    key_to_color: Dict[Tuple, str] = {}
    if color_by_fields:
        color_groups: Dict[Tuple, List[Tuple]] = {}
        for key in sorted_keys:
            field_to_value = dict(zip(group_fields, key))
            color_key = tuple(field_to_value.get(field) for field in color_by_fields)
            color_groups.setdefault(color_key, []).append(key)
        for color_idx, (color_key, keys_list) in enumerate(color_groups.items()):
            color = base_colors[color_idx % len(base_colors)]
            for key in keys_list:
                key_to_color[key] = color
    else:
        for i, key in enumerate(sorted_keys):
            key_to_color[key] = base_colors[i % len(base_colors)]

    # Time series: Cx / Cy
    for ax, ch, y_label in (
        (ax_cx, cx_name, cop_style.get("y_label_cx", "Cx")),
        (ax_cy, cy_name, cop_style.get("y_label_cy", "Cy")),
    ):
        seen_labels: set[str] = set()
        for key in sorted_keys:
            series = aggregated_by_key.get(key, {}).get(ch)
            if series is None:
                continue
            y = (-series) if (ch == cy_name and cop_style["y_invert"]) else series
            color = key_to_color.get(key, "C0")
            if color_by_fields:
                field_to_value = dict(zip(group_fields, key))
                color_key = tuple(field_to_value.get(field) for field in color_by_fields)
                if len(color_by_fields) == 1:
                    label = str(color_key[0])
                else:
                    label = ", ".join(f"{field}={value}" for field, value in zip(color_by_fields, color_key))
            else:
                label = _format_group_label(key, group_fields, filtered_group_fields) or "_nolegend_"

            plot_label = label if label not in seen_labels else "_nolegend_"
            if plot_label != "_nolegend_":
                seen_labels.add(label)
            ax.plot(
                x,
                y,
                color=color,
                linewidth=cop_style.get("line_width", 0.8),
                alpha=cop_style.get("line_alpha", 0.8),
                label=plot_label,
            )

        ax.grid(True, alpha=common_style["grid_alpha"])
        ax.tick_params(labelsize=common_style["tick_labelsize"])
        ax.legend(
            fontsize=cop_style["legend_fontsize"],
            loc=common_style["legend_loc"],
            framealpha=common_style["legend_framealpha"],
        )
        ax.set_xlabel(cop_style.get("x_label_time", "Normalized time (0-1)"), fontsize=common_style["label_fontsize"])
        ax.set_ylabel(y_label, fontsize=common_style["label_fontsize"])

    # Add titles to Cx and Cy subplots
    ax_cx.set_title(
        cx_name,
        fontsize=common_style["title_fontsize"],
        fontweight=common_style["title_fontweight"],
        pad=common_style["title_pad"],
    )
    ax_cy.set_title(
        cy_name,
        fontsize=common_style["title_fontsize"],
        fontweight=common_style["title_fontweight"],
        pad=common_style["title_pad"],
    )

    # Scatter: Cx vs Cy (window color as facecolor, group as marker/edgecolor)
    scatter_edgewidth = float(cop_style.get("overlay_scatter_edgewidth", 0.6))
    for span in window_spans:
        mask = (x >= span["start"]) & (x <= span["end"])
        if not mask.any():
            continue
        for key in sorted_keys:
            cx = aggregated_by_key.get(key, {}).get(cx_name)
            cy = aggregated_by_key.get(key, {}).get(cy_name)
            if cx is None or cy is None:
                continue
            y_vals = (-cy) if cop_style["y_invert"] else cy
            marker = key_to_marker.get(key, "o")
            edgecolor = key_to_color.get(key, "C0")
            ax_scatter.scatter(
                cx[mask],
                y_vals[mask],
                s=cop_style["scatter_size"],
                alpha=cop_style["scatter_alpha"],
                marker=marker,
                facecolors=span["color"],
                edgecolors=edgecolor,
                linewidths=scatter_edgewidth,
                label="_nolegend_",
            )

    ax_scatter.grid(True, alpha=common_style["grid_alpha"])
    ax_scatter.tick_params(labelsize=common_style["tick_labelsize"])
    # Scatter legend를 "window(색)" / "group(마커)"로 분리.
    try:
        import matplotlib.patches as mpatches
        import matplotlib.lines as mlines

        window_handles = [
            mpatches.Patch(facecolor=span["color"], edgecolor="none", label=span["label"]) for span in window_spans
        ]
        if window_handles:
            window_legend = ax_scatter.legend(
                handles=window_handles,
                fontsize=cop_style["legend_fontsize"],
                loc="upper right",
                framealpha=common_style["legend_framealpha"],
                title="window",
            )
            ax_scatter.add_artist(window_legend)

        group_handles: List[Any] = []
        seen_labels: set[str] = set()
        for key in sorted_keys:
            label = _format_group_label(key, group_fields, filtered_group_fields)
            if label is None or label in seen_labels:
                continue
            seen_labels.add(label)
            marker = key_to_marker.get(key, "o")
            edgecolor = key_to_color.get(key, "C0")
            group_handles.append(
                mlines.Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker=marker,
                    markersize=6,
                    markerfacecolor="none",
                    markeredgecolor=edgecolor,
                    markeredgewidth=scatter_edgewidth,
                    label=label,
                )
            )
        if group_handles:
            ax_scatter.legend(
                handles=group_handles,
                fontsize=cop_style["legend_fontsize"],
                loc="lower left",
                framealpha=common_style["legend_framealpha"],
                title="group",
            )
    except Exception:
        ax_scatter.legend(
            fontsize=cop_style["legend_fontsize"],
            loc=common_style["legend_loc"],
            framealpha=common_style["legend_framealpha"],
        )
    ax_scatter.set_title(
        "Cxy",
        fontsize=common_style["title_fontsize"],
        fontweight=common_style["title_fontweight"],
        pad=common_style["title_pad"],
    )
    ax_scatter.set_xlabel(cop_style["x_label"], fontsize=common_style["label_fontsize"])
    ax_scatter.set_ylabel(cop_style["y_label"], fontsize=common_style["label_fontsize"])
    ax_scatter.set_aspect("equal", adjustable="datalim")

    overlay_by = ", ".join(group_fields) if group_fields else "all"
    fig.suptitle(
        f"{mode_name} | cop | overlay by {overlay_by}",
        fontsize=common_style["title_fontsize"],
        fontweight=common_style["title_fontweight"],
    )
    fig.tight_layout(rect=common_style["tight_layout_rect"])
    fig.savefig(
        output_path,
        facecolor=common_style["savefig_facecolor"],
    )
    plt.close(fig)


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
        mocap_rate = float(self.config["data"].get("mocap_sample_rate", 100))
        self.frame_ratio = int(self.config["data"].get("frame_ratio") or int(self.device_rate / mocap_rate))

        self.target_length = int(self.config["interpolation"]["target_length"])
        self.target_axis: Optional[np.ndarray] = None
        self.x_norm: Optional[np.ndarray] = None
        self.time_start_ms: Optional[float] = None
        self.time_end_ms: Optional[float] = None
        self.time_start_frame: Optional[float] = None
        self.time_end_frame: Optional[float] = None
        self.window_norm_ranges: Dict[str, Tuple[float, float]] = {}
        self.window_ms_ranges: Dict[str, Tuple[float, float]] = {}

        style_cfg = self.config["plot_style"]
        self.common_style = self._build_common_style(style_cfg["common"])
        self.emg_style = self._build_emg_style(style_cfg["emg"])
        self.forceplate_style = self._build_forceplate_style(style_cfg["forceplate"])
        self.cop_style = self._build_cop_style(style_cfg["cop"])
        self.window_colors = self.cop_style.get("window_colors", {})
        self.legend_label_threshold = self.common_style.get("legend_label_threshold", 6)

        self.features_df: Optional[pl.DataFrame] = self._load_features()

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

        lf = self._load_and_align_lazy()
        if sample:
            lf = self._filter_first_group(lf)
        lf = self._prepare_time_axis(lf)

        ensure_output_dirs(self.base_dir, self.config)

        meta_cols_needed = self._collect_needed_meta_columns(enabled_modes)

        group_names = self._signal_group_names(selected_groups)
        for group_name in group_names:
            resampled = self._resample_signal_group(lf, group_name, meta_cols_needed)
            tasks: List[Dict[str, Any]] = []
            for mode_name, mode_cfg in enabled_modes:
                tasks.extend(self._build_plot_tasks(resampled, group_name, mode_name, mode_cfg))
            self._run_plot_tasks(tasks)

    def _run_plot_tasks(self, tasks: List[Dict[str, Any]]) -> None:
        if not tasks:
            return
        max_workers = self._max_workers()
        font_family = self.common_style.get("font_family")
        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_plot_worker_init,
                initargs=(font_family,),
            ) as ex:
                list(ex.map(_plot_task, tasks, chunksize=max(1, len(tasks) // (max_workers * 4))))
        except PermissionError:
            _plot_worker_init(font_family)
            for task in tasks:
                _plot_task(task)

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
                needed.append(filter_cfg.get("column"))
        out: List[str] = []
        for col in needed:
            if not col or col in out:
                continue
            out.append(str(col))
        return out

    def _load_and_align_lazy(self) -> pl.LazyFrame:
        input_path = Path(self.config["data"]["input_file"])
        if not input_path.is_absolute():
            input_path = (self.base_dir / input_path).resolve()

        lf = pl.scan_parquet(str(input_path))

        cols = self._lazy_columns(lf)
        rename_map = {c: c.lstrip("\ufeff") for c in cols if c.startswith("\ufeff")}
        if rename_map:
            lf = lf.rename(rename_map)

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
        mocap_start = pl.col(mocap_col).min().over(group_cols)
        onset_device = (pl.col(onset_col).first().over(group_cols) - mocap_start) * self.frame_ratio
        onset_aligned = pl.col(frame_col) - onset_device
        offset_rel = (
            (pl.col(offset_col).first().over(group_cols) - pl.col(onset_col).first().over(group_cols))
            * self.frame_ratio
        )

        return lf.with_columns(
            [
                onset_device.alias("onset_device_frame"),
                onset_aligned.alias("aligned_frame"),
                offset_rel.alias("offset_from_onset"),
            ]
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

        cropped = lf.filter((pl.col("aligned_frame") >= time_start_frame) & (pl.col("aligned_frame") <= time_end_frame))
        if cropped.select(pl.len()).collect().item() == 0:
            raise ValueError("No data remains after applying the interpolation time window.")

        self.time_start_ms = time_start_ms
        self.time_end_ms = time_end_ms
        self.time_start_frame = time_start_frame
        self.time_end_frame = time_end_frame
        self.target_axis = np.linspace(time_start_frame, time_end_frame, self.target_length)
        self.x_norm = np.linspace(0.0, 1.0, self.target_length)
        self.window_norm_ranges, self.window_ms_ranges = self._compute_window_norm_ranges()
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

        cols = set(group_cols + ["aligned_frame"] + channels + meta_cols)
        available_cols = set(self._lazy_columns(lf))
        lf_sel = lf.select([pl.col(c) for c in cols if c in available_cols])

        present_meta_cols: List[str] = []
        missing_meta_cols: List[str] = []
        for col in meta_cols:
            if col in group_cols:
                continue
            if col in available_cols:
                present_meta_cols.append(col)
            else:
                missing_meta_cols.append(col)

        agg_exprs: List[pl.Expr] = [pl.col("aligned_frame").sort().alias("__x")]
        for ch in channels:
            agg_exprs.append(pl.col(ch).sort_by("aligned_frame").alias(ch))
        for col in present_meta_cols:
            agg_exprs.append(pl.col(col).first().alias(col))
            agg_exprs.append(pl.col(col).n_unique().alias(f"__nuniq_{col}"))

        grouped = lf_sel.group_by(group_cols, maintain_order=False).agg(agg_exprs)
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
        return _ResampledGroup(meta_df=meta_df, tensor=tensor, channels=channels)

    def _build_plot_tasks(
        self, resampled: _ResampledGroup, signal_group: str, mode_name: str, mode_cfg: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        if self.x_norm is None or self.target_axis is None or self.time_start_ms is None or self.time_end_ms is None:
            raise RuntimeError("Time axis is not initialized.")

        meta_df = resampled.meta_df
        tensor = resampled.tensor
        channels = resampled.channels

        filtered_idx = self._apply_filter_indices(meta_df, mode_cfg.get("filter"))
        if filtered_idx.size == 0:
            return []

        group_fields = list(mode_cfg.get("groupby", []) or [])
        overlay = bool(mode_cfg.get("overlay", False))

        grouped = self._group_indices(meta_df, filtered_idx, group_fields)
        if not grouped:
            return []

        window_spans = self._window_spans()

        tasks: List[Dict[str, Any]] = []

        if overlay:
            overlay_within = mode_cfg.get("overlay_within")

            # Aggregate all keys first (common for both old and new logic)
            aggregated_by_key: Dict[Tuple, Dict[str, np.ndarray]] = {}
            markers_by_key: Dict[Tuple, Dict[str, Any]] = {}
            for key, idx in grouped.items():
                aggregated_by_key[key] = self._aggregate_tensor(tensor, meta_df, idx, channels)
                markers_by_key[key] = self._collect_markers(signal_group, key, group_fields, mode_cfg.get("filter"))

            output_dir = Path(self.base_dir, mode_cfg["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            filename_pattern = mode_cfg["filename_pattern"]

            # OLD BEHAVIOR: overlay_within not specified -> all keys in one file
            if not overlay_within:
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

                tasks.append(
                    self._task_overlay(
                        signal_group=signal_group,
                        aggregated_by_key=aggregated_by_key,
                        markers_by_key=markers_by_key,
                        output_path=output_path,
                        mode_name=mode_name,
                        group_fields=group_fields,
                        sorted_keys=sorted_keys,
                        window_spans=window_spans,
                        filtered_group_fields=filtered_group_fields,
                        color_by_fields=mode_cfg.get("color_by") if signal_group in ("emg", "cop") else None,
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

                sorted_keys = _sort_overlay_keys(keys_in_file, group_fields)
                filtered_group_fields = _calculate_filtered_group_fields(
                    sorted_keys,
                    group_fields,
                    threshold=self.legend_label_threshold
                )

                filename = self._render_filename(filename_pattern, file_key, signal_group, file_fields)
                output_path = output_dir / filename

                tasks.append(
                    self._task_overlay(
                        signal_group=signal_group,
                        aggregated_by_key=file_aggregated,
                        markers_by_key=file_markers,
                        output_path=output_path,
                        mode_name=mode_name,
                        group_fields=group_fields,
                        sorted_keys=sorted_keys,
                        window_spans=window_spans,
                        filtered_group_fields=filtered_group_fields,
                        color_by_fields=mode_cfg.get("color_by") if signal_group in ("emg", "cop") else None,
                    )
                )

            return tasks

        for key, idx in grouped.items():
            aggregated = self._aggregate_tensor(tensor, meta_df, idx, channels)
            filename = self._render_filename(mode_cfg["filename_pattern"], key, signal_group, group_fields)
            output_dir = Path(self.base_dir, mode_cfg["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / filename
            markers = self._collect_markers(signal_group, key, group_fields, mode_cfg.get("filter"))

            if signal_group == "emg":
                tasks.append(
                    self._task_emg(
                        aggregated=aggregated,
                        output_path=output_path,
                        key=key,
                        mode_name=mode_name,
                        group_fields=group_fields,
                        markers=markers,
                        window_spans=window_spans,
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
                        window_spans=window_spans,
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
                        window_spans=window_spans,
                    )
                )

        return tasks

    def _apply_filter_indices(self, meta_df: pl.DataFrame, filter_cfg: Optional[Dict[str, Any]]) -> np.ndarray:
        idx = np.arange(meta_df.height, dtype=int)
        if not filter_cfg:
            return idx

        # 기존 방식: {column: "mixed", value: 1} (하위 호환성 유지)
        if "column" in filter_cfg and "value" in filter_cfg:
            col = filter_cfg.get("column")
            value = filter_cfg.get("value")
            if col is None or col not in meta_df.columns:
                return idx if value is None else np.array([], dtype=int)
            series = meta_df[col].to_numpy()
            mask = series == value
            return idx[mask]

        # 새로운 방식: {mixed: 1, age_group: "young"} - 다중 컬럼 필터링
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

    def _window_spans(self) -> List[Dict[str, Any]]:
        spans: List[Dict[str, Any]] = []
        for name, (start, end) in self.window_norm_ranges.items():
            label = self._format_window_label(name)
            spans.append(
                {
                    "name": name,
                    "start": float(start),
                    "end": float(end),
                    "label": label,
                    "color": self.window_colors.get(name, "#cccccc"),
                }
            )
        return spans

    def _task_overlay(
        self,
        *,
        signal_group: str,
        aggregated_by_key: Dict[Tuple, Dict[str, np.ndarray]],
        markers_by_key: Dict[Tuple, Dict[str, Any]],
        output_path: Path,
        mode_name: str,
        group_fields: List[str],
        sorted_keys: List[Tuple],
        window_spans: List[Dict[str, Any]],
        filtered_group_fields: List[str],
        color_by_fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        task: Dict[str, Any] = {
            "kind": "overlay",
            "signal_group": signal_group,
            "output_path": str(output_path),
            "mode_name": mode_name,
            "group_fields": group_fields,
            "sorted_keys": sorted_keys,
            "aggregated_by_key": aggregated_by_key,
            "markers_by_key": markers_by_key,
            "x": self.x_norm,
            "window_spans": window_spans,
            "common_style": self.common_style,
            "filtered_group_fields": filtered_group_fields,
            "color_by_fields": color_by_fields,
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
        window_spans: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "kind": "emg",
            "output_path": str(output_path),
            "key": key,
            "mode_name": mode_name,
            "group_fields": group_fields,
            "aggregated": aggregated,
            "markers": markers,
            "x": self.x_norm,
            "channels": self.config["signal_groups"]["emg"]["columns"],
            "grid_layout": self.config["signal_groups"]["emg"]["grid_layout"],
            "window_spans": window_spans,
            "window_span_alpha": self.emg_style["window_span_alpha"],
            "emg_style": self.emg_style,
            "common_style": self.common_style,
            "time_start_ms": self.time_start_ms,
            "time_end_ms": self.time_end_ms,
        }

    def _task_forceplate(
        self,
        *,
        aggregated: Dict[str, np.ndarray],
        output_path: Path,
        key: Tuple,
        mode_name: str,
        group_fields: List[str],
        markers: Dict[str, Any],
        window_spans: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "kind": "forceplate",
            "output_path": str(output_path),
            "key": key,
            "mode_name": mode_name,
            "group_fields": group_fields,
            "aggregated": aggregated,
            "markers": markers,
            "x": self.x_norm,
            "channels": self.config["signal_groups"]["forceplate"]["columns"],
            "grid_layout": self.config["signal_groups"]["forceplate"]["grid_layout"],
            "window_spans": window_spans,
            "window_span_alpha": self.forceplate_style["window_span_alpha"],
            "forceplate_style": self.forceplate_style,
            "common_style": self.common_style,
            "time_start_ms": self.time_start_ms,
            "time_end_ms": self.time_end_ms,
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
        window_spans: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "kind": "cop",
            "output_path": str(output_path),
            "key": key,
            "mode_name": mode_name,
            "group_fields": group_fields,
            "aggregated": aggregated,
            "markers": markers,
            "x_axis": self.x_norm,
            "target_axis": self.target_axis,
            "time_start_ms": self.time_start_ms,
            "time_end_ms": self.time_end_ms,
            "device_rate": self.device_rate,
            "cop_channels": self.config["signal_groups"]["cop"]["columns"],
            "cop_style": self.cop_style,
            "common_style": self.common_style,
            "window_spans": window_spans,
        }

    def _compute_window_norm_ranges(self) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
        if self.time_start_ms is None or self.time_end_ms is None:
            return {}, {}
        norm_ranges: Dict[str, Tuple[float, float]] = {}
        ms_ranges: Dict[str, Tuple[float, float]] = {}
        definitions = self.config.get("windows", {}).get("definitions", {})
        for name, cfg in definitions.items():
            raw_start = float(cfg["start_ms"])
            raw_end = float(cfg["end_ms"])
            clamped_start = max(raw_start, self.time_start_ms)
            clamped_end = min(raw_end, self.time_end_ms)
            if clamped_start >= clamped_end:
                continue
            start_norm = _ms_to_norm(clamped_start, self.time_start_ms, self.time_end_ms)
            end_norm = _ms_to_norm(clamped_end, self.time_start_ms, self.time_end_ms)
            if start_norm is None or end_norm is None:
                continue
            norm_ranges[name] = (start_norm, end_norm)
            ms_ranges[name] = (clamped_start, clamped_end)
        return norm_ranges, ms_ranges

    def _format_window_label(self, name: str) -> str:
        ms_range = self.window_ms_ranges.get(name)
        if not ms_range:
            return name
        start_ms, end_ms = ms_range
        return f"{name} ({int(start_ms)}-{int(end_ms)} ms)"

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
            "show_onset_marker": cfg["show_onset_marker"],
            "show_max_marker": cfg["show_max_marker"],
        }

    def _build_emg_style(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "subplot_size": tuple(cfg["subplot_size"]),
            "line_color": cfg["line_color"],
            "line_width": cfg["line_width"],
            "line_alpha": cfg["line_alpha"],
            "window_span_alpha": cfg["window_span_alpha"],
            "onset_marker_color": cfg["onset_marker_color"],
            "onset_marker_linestyle": cfg["onset_marker_linestyle"],
            "onset_marker_linewidth": cfg["onset_marker_linewidth"],
            "max_marker_color": cfg["max_marker_color"],
            "max_marker_linestyle": cfg["max_marker_linestyle"],
            "max_marker_linewidth": cfg["max_marker_linewidth"],
            "legend_fontsize": cfg["legend_fontsize"],
            "x_label": cfg["x_label"],
            "y_label": cfg["y_label"],
            "onset_marker": {
                "color": cfg["onset_marker_color"],
                "linestyle": cfg["onset_marker_linestyle"],
                "linewidth": cfg["onset_marker_linewidth"],
            },
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
            "line_width": cfg["line_width"],
            "line_alpha": cfg["line_alpha"],
            "window_span_alpha": cfg["window_span_alpha"],
            "onset_marker_color": cfg["onset_marker_color"],
            "onset_marker_linestyle": cfg["onset_marker_linestyle"],
            "onset_marker_linewidth": cfg["onset_marker_linewidth"],
            "legend_fontsize": cfg["legend_fontsize"],
            "x_label": cfg["x_label"],
            "y_label": cfg["y_label"],
            "onset_marker": {
                "color": cfg["onset_marker_color"],
                "linestyle": cfg["onset_marker_linestyle"],
                "linewidth": cfg["onset_marker_linewidth"],
            },
        }

    def _build_cop_style(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "subplot_size": tuple(cfg["subplot_size"]),
            "scatter_size": cfg["scatter_size"],
            "scatter_alpha": cfg["scatter_alpha"],
            "background_color": cfg["background_color"],
            "background_alpha": cfg["background_alpha"],
            "background_size": cfg["background_size"],
            "window_span_alpha": cfg.get("window_span_alpha", 0.15),
            "line_colors": cfg.get("line_colors", {"Cx": "blue", "Cy": "red"}),
            "line_width": cfg.get("line_width", 0.8),
            "line_alpha": cfg.get("line_alpha", 0.8),
            "window_colors": cfg["window_colors"],
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
            col = filter_cfg["column"]
            val = filter_cfg["value"]
            if col in df.columns:
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
    default_config = Path(__file__).resolve().parent.parent / "config.yaml"
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
    visualizer.run(modes=args.modes, signal_groups=args.groups, sample=args.sample)


if __name__ == "__main__":
    main()
