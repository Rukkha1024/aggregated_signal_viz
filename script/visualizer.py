from __future__ import annotations

import argparse
import concurrent.futures
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

try:
    from script.config_utils import bom_rename_map, load_config, resolve_path, strip_bom_columns
except ModuleNotFoundError:  # Allows running as `python script/visualizer.py`
    from config_utils import bom_rename_map, load_config, resolve_path, strip_bom_columns


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


def _parse_event_vlines_config(cfg: Any) -> List[str]:
    """
    Returns a de-duplicated list of event column names to render as vertical lines.

    Supported YAML forms:
      - event_vlines: { columns: [step_onset, ...] }
      - event_vlines: [step_onset, ...]
    """
    if cfg is None:
        return []
    if isinstance(cfg, dict):
        cfg = cfg.get("columns")
    if not isinstance(cfg, list):
        return []
    out: List[str] = []
    for item in cfg:
        if item is None:
            continue
        name = str(item).strip()
        if not name or name in out:
            continue
        out.append(name)
    return out


def _parse_event_vlines_style(cfg: Any) -> Dict[str, Any]:
    """
    Returns a matplotlib `axvline` kwargs dict for event vertical lines.

    Supported YAML form:
      event_vlines:
        columns: [step_onset, ...]
        style: { color: "red", linestyle: "--", linewidth: 1.5, alpha: 0.9 }
    """
    defaults: Dict[str, Any] = {"color": "red", "linestyle": "--", "linewidth": 1.5, "alpha": 0.9}
    if cfg is None:
        return dict(defaults)
    if isinstance(cfg, dict):
        style = cfg.get("style")
    else:
        style = None
    if not isinstance(style, dict):
        return dict(defaults)

    out = dict(defaults)

    color = style.get("color")
    if color is not None and str(color).strip():
        out["color"] = str(color).strip()

    linestyle = style.get("linestyle")
    if linestyle is not None and str(linestyle).strip():
        out["linestyle"] = str(linestyle).strip()

    linewidth = style.get("linewidth")
    if linewidth is not None:
        try:
            out["linewidth"] = float(linewidth)
        except (TypeError, ValueError):
            pass

    alpha = style.get("alpha")
    if alpha is not None:
        try:
            out["alpha"] = float(alpha)
        except (TypeError, ValueError):
            pass

    return out


_DEFAULT_EVENT_VLINE_PALETTE: Tuple[str, ...] = (
    "C0",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "C9",
)


def _parse_event_vlines_palette(cfg: Any) -> Tuple[str, ...]:
    if not isinstance(cfg, dict):
        return _DEFAULT_EVENT_VLINE_PALETTE
    raw = cfg.get("palette")
    if not isinstance(raw, list):
        return _DEFAULT_EVENT_VLINE_PALETTE
    palette: List[str] = []
    for item in raw:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        palette.append(text)
    return tuple(palette) if palette else _DEFAULT_EVENT_VLINE_PALETTE


def _parse_event_vlines_color_overrides(cfg: Any) -> Dict[str, str]:
    if not isinstance(cfg, dict):
        return {}
    raw = cfg.get("colors")
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, str] = {}
    for key, value in raw.items():
        if key is None or value is None:
            continue
        name = str(key).strip()
        color = str(value).strip()
        if not name or not color:
            continue
        out[name] = color
    return out


def _build_event_vline_color_map(event_columns: Sequence[str], cfg: Any) -> Dict[str, str]:
    palette = _parse_event_vlines_palette(cfg)
    if not palette:
        palette = _DEFAULT_EVENT_VLINE_PALETTE
    overrides = _parse_event_vlines_color_overrides(cfg)
    out: Dict[str, str] = {}
    for idx, event in enumerate(event_columns):
        if event in overrides:
            out[event] = overrides[event]
        else:
            out[event] = palette[idx % len(palette)]
    return out


def _event_ms_col(event_col: str) -> str:
    return f"__event_{event_col}_ms"


def _nanmean_ignore_nan(values: np.ndarray) -> Optional[float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return None
    return float(arr.mean())


def _draw_event_vlines(ax: Any, vlines: Sequence[Dict[str, Any]], *, style: Dict[str, Any]) -> None:
    if not vlines:
        return
    base_kwargs = dict(style) if style else {"color": "red", "linestyle": "--", "linewidth": 1.5}
    base_kwargs.setdefault("alpha", 0.9)
    for v in vlines:
        x = v.get("x")
        if x is None:
            continue
        kwargs = dict(base_kwargs)
        color = v.get("color")
        if color is not None and str(color).strip():
            kwargs["color"] = str(color).strip()
        ax.axvline(float(x), label="_nolegend_", **kwargs)


def _build_window_legend_handles(window_spans: Sequence[Dict[str, Any]]) -> List[Any]:
    try:
        import matplotlib.patches as mpatches
    except Exception:
        return []

    handles: List[Any] = []
    seen: set[str] = set()
    for span in window_spans:
        label = str(span.get("label", "")).strip()
        if not label or label in seen:
            continue
        color = span.get("color")
        if color is None or not str(color).strip():
            continue
        seen.add(label)
        handles.append(mpatches.Patch(facecolor=str(color), edgecolor="none", label=label))
    return handles


def _build_group_legend_handles(
    sorted_keys: Sequence[Tuple],
    group_fields: Sequence[str],
    filtered_group_fields: Optional[Sequence[str]],
    key_to_linestyle: Dict[Tuple, Any],
    *,
    linewidth: float,
    color: str = "0.2",
) -> List[Any]:
    try:
        import matplotlib.lines as mlines
    except Exception:
        return []

    handles: List[Any] = []
    seen_labels: set[str] = set()
    for key in sorted_keys:
        label = _format_group_label(tuple(key), list(group_fields), list(filtered_group_fields) if filtered_group_fields is not None else None)
        if label is None or label in seen_labels:
            continue
        seen_labels.add(label)
        linestyle = key_to_linestyle.get(tuple(key), "-")
        handles.append(
            mlines.Line2D(
                [],
                [],
                linestyle=linestyle,
                color=color,
                linewidth=linewidth,
                label=label,
            )
        )
    return handles


def _apply_window_group_legends(
    ax: Any,
    *,
    window_spans: Sequence[Dict[str, Any]],
    group_handles: Sequence[Any],
    legend_fontsize: float,
    framealpha: float,
    loc: str = "best",
) -> None:
    handles: List[Any] = []
    seen_labels: set[str] = set()

    for handle in _build_window_legend_handles(window_spans):
        label = str(getattr(handle, "get_label", lambda: "")()).strip()
        if not label or label == "_nolegend_" or label in seen_labels:
            continue
        seen_labels.add(label)
        handles.append(handle)

    for handle in group_handles:
        label = str(getattr(handle, "get_label", lambda: "")()).strip()
        if not label or label == "_nolegend_" or label in seen_labels:
            continue
        seen_labels.add(label)
        handles.append(handle)

    existing_handles, existing_labels = ax.get_legend_handles_labels()
    for handle, label in zip(existing_handles, existing_labels):
        label = str(label).strip()
        if not label or label == "_nolegend_" or label in seen_labels:
            continue
        seen_labels.add(label)
        handles.append(handle)

    if handles:
        ax.legend(
            handles=handles,
            fontsize=legend_fontsize,
            loc=loc,
            framealpha=framealpha,
        )


def _format_label(template: Any, **kwargs: Any) -> str:
    if not isinstance(template, str):
        return str(template)
    try:
        return template.format(**kwargs)
    except (KeyError, ValueError):
        return template


def _resolve_forceplate_axis_label(channel: str, axis_labels: Dict[str, str]) -> str:
    if not axis_labels:
        return channel
    if channel in axis_labels:
        return axis_labels[channel]
    base = channel[:-5] if channel.endswith("_zero") else channel
    return axis_labels.get(base, channel)


def _resolve_forceplate_line_color(channel: str, line_colors: Dict[str, Any]) -> str:
    if not line_colors:
        return "blue"

    direct = line_colors.get(channel)
    if direct is not None and str(direct).strip():
        return str(direct).strip()

    base = channel[:-5] if channel.endswith("_zero") else channel
    base_color = line_colors.get(base)
    if base_color is not None and str(base_color).strip():
        return str(base_color).strip()

    return "blue"


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


def _resolve_com_channel_names(channels: Sequence[str]) -> Tuple[str, str, Optional[str]]:
    if len(channels) < 2:
        raise ValueError("COM requires at least 2 channels. Check signal_groups.com.columns in config.yaml.")

    lower_names = [ch.lower() for ch in channels]
    x_idx = next((i for i, name in enumerate(lower_names) if "comx" in name), None)
    y_idx = next((i for i, name in enumerate(lower_names) if "comy" in name), None)
    z_idx = next((i for i, name in enumerate(lower_names) if "comz" in name), None)

    x_name = channels[x_idx] if x_idx is not None else None
    y_name = channels[y_idx] if y_idx is not None else None
    z_name = channels[z_idx] if z_idx is not None else None

    if x_name and y_name and z_name:
        return x_name, y_name, z_name

    remaining = [ch for ch in channels if ch not in (x_name, y_name, z_name)]
    if x_name is None:
        x_name = remaining[0] if remaining else channels[0]
        remaining = [ch for ch in remaining if ch != x_name]
    if y_name is None:
        y_name = remaining[0] if remaining else channels[1]
        remaining = [ch for ch in remaining if ch != y_name]
    if z_name is None:
        if remaining:
            z_name = remaining[0]
        elif len(channels) > 2:
            z_name = channels[2]
        else:
            z_name = None

    return x_name, y_name, z_name


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


def _parse_group_linestyles(raw_values: Any) -> Tuple[Any, ...]:
    if raw_values is None:
        return ("-", "--", ":", "-.", (0, (1, 1)), (0, (3, 1, 1, 1)))
    styles: List[Any] = []
    for item in raw_values:
        if isinstance(item, (list, tuple)):
            if len(item) == 2 and isinstance(item[1], (list, tuple)):
                offset = float(item[0])
                pattern = tuple(float(v) for v in item[1])
                styles.append((offset, pattern))
            else:
                styles.append(tuple(item))
        else:
            styles.append(str(item))
    return tuple(styles) if styles else ("-", "--", ":", "-.")


def _build_group_linestyles(sorted_keys: List[Tuple], linestyles: Sequence[Any]) -> Dict[Tuple, Any]:
    if not sorted_keys:
        return {}
    return {key: linestyles[i % len(linestyles)] for i, key in enumerate(sorted_keys)}


def _build_group_color_map(
    sorted_keys: List[Tuple],
    group_fields: List[str],
    color_by_fields: Optional[List[str]],
    base_colors: Sequence[str],
) -> Dict[Tuple, str]:
    if not sorted_keys:
        return {}
    key_to_color: Dict[Tuple, str] = {}
    if color_by_fields:
        color_groups: Dict[Tuple, List[Tuple]] = {}
        for key in sorted_keys:
            field_to_value = dict(zip(group_fields, key))
            color_key = tuple(field_to_value.get(field) for field in color_by_fields)
            color_groups.setdefault(color_key, []).append(key)
        for color_idx, (_, keys_list) in enumerate(color_groups.items()):
            color = base_colors[color_idx % len(base_colors)]
            for key in keys_list:
                key_to_color[key] = color
    else:
        for i, key in enumerate(sorted_keys):
            key_to_color[key] = base_colors[i % len(base_colors)]
    return key_to_color


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
    event_vline_style = task.get("event_vline_style", {})

    if kind == "overlay":
        _plot_overlay_generic(
            signal_group=task["signal_group"],
            aggregated_by_key=task["aggregated_by_key"],
            markers_by_key=task.get("markers_by_key", {}),
            event_vlines_by_key=task.get("event_vlines_by_key", {}),
            event_vline_style=event_vline_style,
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
            event_vlines=task.get("event_vlines", []),
            event_vline_style=event_vline_style,
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
            event_vlines=task.get("event_vlines", []),
            event_vline_style=event_vline_style,
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
            event_vlines=task.get("event_vlines", []),
            event_vline_style=event_vline_style,
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

    if kind == "com":
        _plot_com(
            aggregated=task["aggregated"],
            output_path=output_path,
            key=tuple(task["key"]),
            mode_name=task["mode_name"],
            group_fields=task["group_fields"],
            markers=task["markers"],
            event_vlines=task.get("event_vlines", []),
            event_vline_style=event_vline_style,
            x_axis=np.asarray(task["x_axis"], dtype=float) if task["x_axis"] is not None else None,
            time_start_ms=task["time_start_ms"],
            time_end_ms=task["time_end_ms"],
            device_rate=float(task["device_rate"]),
            com_channels=task["com_channels"],
            com_style=task["com_style"],
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
    event_vlines_by_key: Dict[Tuple, List[Dict[str, Any]]],
    event_vline_style: Dict[str, Any],
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
            event_vlines_by_key=event_vlines_by_key,
            event_vline_style=event_vline_style,
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

    if signal_group == "com":
        _plot_com_overlay(
            aggregated_by_key=aggregated_by_key,
            event_vlines_by_key=event_vlines_by_key,
            event_vline_style=event_vline_style,
            output_path=output_path,
            mode_name=mode_name,
            group_fields=group_fields,
            sorted_keys=sorted_keys,
            x=x,
            window_spans=window_spans,
            com_channels=cop_channels or (),
            com_style=style,
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
        event_vlines_by_key=event_vlines_by_key,
        event_vline_style=event_vline_style,
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
    event_vlines: List[Dict[str, Any]],
    event_vline_style: Dict[str, Any],
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

        _draw_event_vlines(ax, event_vlines, style=event_vline_style)

        marker_info = markers.get(ch, {})
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
    event_vlines_by_key: Dict[Tuple, List[Dict[str, Any]]],
    event_vline_style: Dict[str, Any],
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
        import matplotlib as mpl

        base_colors = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
        use_group_colors = common_style.get("use_group_colors", False)
        key_to_color = _build_group_color_map(sorted_keys, group_fields, color_by_fields, base_colors) if use_group_colors else {}
        key_to_linestyle = _build_group_linestyles(sorted_keys, common_style.get("group_linestyles", ("-", "--", ":", "-.")))

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
                color = key_to_color.get(key, single_color) if use_group_colors else single_color
                linestyle = key_to_linestyle.get(key, "-")
                ax.plot(
                    x,
                    y,
                    color=color,
                    linestyle=linestyle,
                    linewidth=style["line_width"],
                    alpha=style["line_alpha"],
                    label=plot_label,
                )

            for key in sorted_keys:
                _draw_event_vlines(ax, event_vlines_by_key.get(key, []), style=event_vline_style)

            for key in sorted_keys:
                marker_info = markers_by_key.get(key, {}).get(ch, {})
                marker_label = _format_group_label(key, group_fields)
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
        import matplotlib as mpl

        base_colors = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
        use_group_colors = common_style.get("use_group_colors", False)
        key_to_color = _build_group_color_map(sorted_keys, group_fields, color_by_fields, base_colors) if use_group_colors else {}
        key_to_linestyle = _build_group_linestyles(sorted_keys, common_style.get("group_linestyles", ("-", "--", ":", "-.")))

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
                channel_color = _resolve_forceplate_line_color(ch, style.get("line_colors", {}) or {})
                color = key_to_color.get(key, channel_color) if use_group_colors else channel_color
                linestyle = key_to_linestyle.get(key, "-")
                ax.plot(
                    x,
                    y,
                    color=color,
                    linestyle=linestyle,
                    linewidth=style["line_width"],
                    alpha=style["line_alpha"],
                    label=plot_label,
                )

            for key in sorted_keys:
                _draw_event_vlines(ax, event_vlines_by_key.get(key, []), style=event_vline_style)

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
            axis_label = _resolve_forceplate_axis_label(ch, style.get("axis_labels", {}))
            y_label = _format_label(style.get("y_label", "{channel} Value"), channel=ch, axis_label=axis_label)
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
    event_vlines: List[Dict[str, Any]],
    event_vline_style: Dict[str, Any],
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

        color = _resolve_forceplate_line_color(ch, forceplate_style.get("line_colors", {}) or {})
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

        _draw_event_vlines(ax, event_vlines, style=event_vline_style)

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
        axis_label = _resolve_forceplate_axis_label(ch, forceplate_style.get("axis_labels", {}))
        y_label = _format_label(
            forceplate_style.get("y_label", "{channel} Value"),
            channel=ch,
            axis_label=axis_label,
        )
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
    event_vlines: List[Dict[str, Any]],
    event_vline_style: Dict[str, Any],
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

    ap_vals = cx
    ml_vals = -cy if cop_style["y_invert"] else cy

    n_panels = 3
    fig_size = cop_style["subplot_size"]
    try:
        fig_w, fig_h = fig_size
        fig_size = (float(fig_w) * (n_panels / 3.0), float(fig_h))
    except (TypeError, ValueError):
        pass

    fig, axes = plt.subplots(1, n_panels, figsize=fig_size, dpi=common_style["dpi"])
    axes = np.asarray(axes).ravel()
    ax_cx = axes[0]
    ax_cy = axes[1]
    ax_scatter = axes[2]

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
        ap_vals,
        color=cx_color,
        linewidth=cop_style.get("line_width", 0.8),
        alpha=cop_style.get("line_alpha", 0.8),
        label="Cx",
    )
    ax_cy.plot(
        x,
        ml_vals,
        color=cy_color,
        linewidth=cop_style.get("line_width", 0.8),
        alpha=cop_style.get("line_alpha", 0.8),
        label="Cy",
    )

    _draw_event_vlines(ax_cx, event_vlines, style=event_vline_style)
    _draw_event_vlines(ax_cy, event_vlines, style=event_vline_style)

    ax_scatter.scatter(
        ml_vals,
        ap_vals,
        color=cop_style["background_color"],
        alpha=cop_style["background_alpha"],
        s=cop_style["background_size"],
    )

    if x_axis is not None:
        for span in window_spans:
            mask = (x_axis >= span["start"]) & (x_axis <= span["end"])
            if mask.any():
                ax_scatter.scatter(
                    ml_vals[mask],
                    ap_vals[mask],
                    s=cop_style["scatter_size"],
                    alpha=cop_style["scatter_alpha"],
                    color=span["color"],
                    label="_nolegend_",
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
                ml_vals[idx],
                ap_vals[idx],
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

    for ax in (ax_cx, ax_cy):
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

    ax_scatter.grid(True, alpha=common_style["grid_alpha"])
    ax_scatter.tick_params(labelsize=common_style["tick_labelsize"])
    _apply_window_group_legends(
        ax_scatter,
        window_spans=window_spans,
        group_handles=[],
        legend_fontsize=cop_style["legend_fontsize"],
        framealpha=common_style["legend_framealpha"],
        loc=common_style["legend_loc"],
    )

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


def _plot_com(
    *,
    aggregated: Dict[str, np.ndarray],
    output_path: Path,
    key: Tuple,
    mode_name: str,
    group_fields: List[str],
    markers: Dict[str, Dict[str, float]],
    event_vlines: List[Dict[str, Any]],
    event_vline_style: Dict[str, Any],
    x_axis: Optional[np.ndarray],
    time_start_ms: float,
    time_end_ms: float,
    device_rate: float,
    com_channels: Sequence[str],
    com_style: Dict[str, Any],
    common_style: Dict[str, Any],
    window_spans: List[Dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    comx_name, comy_name, comz_name = _resolve_com_channel_names(com_channels)
    comx = aggregated.get(comx_name)
    comy = aggregated.get(comy_name)
    if comx is None or comy is None:
        available = ", ".join(sorted(aggregated.keys()))
        print(f"[com] missing channels: COMx={comx_name!r}, COMy={comy_name!r}. Available: {available}")
        return
    comz = aggregated.get(comz_name) if comz_name else None
    if comz_name is not None and comz is None:
        available = ", ".join(sorted(aggregated.keys()))
        print(f"[com] missing channel: COMz={comz_name!r}. Available: {available}")
        comz_name = None

    x = x_axis
    if x is None:
        x = np.linspace(0.0, 1.0, num=comx.size, dtype=float)

    ap_vals = comx
    ml_vals = -comy if com_style.get("y_invert", False) else comy
    mag_vals = np.sqrt((ap_vals**2) + (ml_vals**2))

    n_panels = 5 if comz_name is not None else 4
    fig_size = com_style["subplot_size"]
    try:
        fig_w, fig_h = fig_size
        fig_size = (float(fig_w) * (n_panels / 3.0), float(fig_h))
    except (TypeError, ValueError):
        pass

    fig, axes = plt.subplots(1, n_panels, figsize=fig_size, dpi=common_style["dpi"])
    axes = np.asarray(axes).ravel()
    ax_x = axes[0]
    ax_y = axes[1]
    if comz_name is not None:
        ax_z = axes[2]
        ax_mag = axes[3]
        ax_scatter = axes[4]
    else:
        ax_z = None
        ax_mag = axes[2]
        ax_scatter = axes[3]

    window_span_alpha = float(com_style.get("window_span_alpha", 0.15))
    time_axes = [ax_x, ax_y] + ([ax_z] if ax_z is not None else []) + [ax_mag]
    for ax in time_axes:
        for span in window_spans:
            ax.axvspan(
                span["start"],
                span["end"],
                color=span["color"],
                alpha=window_span_alpha,
                label="_nolegend_",
            )

    x_color = com_style.get("line_colors", {}).get(comx_name, "blue")
    y_color = com_style.get("line_colors", {}).get(comy_name, "red")
    z_color = com_style.get("line_colors", {}).get(comz_name, "green") if comz_name is not None else None

    ax_x.plot(
        x,
        ap_vals,
        color=x_color,
        linewidth=com_style.get("line_width", 0.8),
        alpha=com_style.get("line_alpha", 0.8),
        label=comx_name,
    )
    ax_y.plot(
        x,
        ml_vals,
        color=y_color,
        linewidth=com_style.get("line_width", 0.8),
        alpha=com_style.get("line_alpha", 0.8),
        label=comy_name,
    )
    if ax_z is not None and comz is not None:
        ax_z.plot(
            x,
            comz,
            color=z_color,
            linewidth=com_style.get("line_width", 0.8),
            alpha=com_style.get("line_alpha", 0.8),
            label=comz_name,
        )
    ax_mag.plot(
        x,
        mag_vals,
        color="0.7",
        linewidth=com_style.get("line_width", 0.8),
        alpha=0.6,
        label="_nolegend_",
    )
    if x_axis is not None:
        seen_window_labels: set[str] = set()
        for span in window_spans:
            mask = (x_axis >= span["start"]) & (x_axis <= span["end"])
            if not mask.any():
                continue
            label = span["label"]
            plot_label = label if label not in seen_window_labels else "_nolegend_"
            if plot_label != "_nolegend_":
                seen_window_labels.add(label)
            ax_mag.plot(
                x_axis[mask],
                mag_vals[mask],
                color=span["color"],
                linewidth=com_style.get("line_width", 0.8),
                alpha=com_style.get("line_alpha", 0.8),
                label="_nolegend_",
            )

    for ax in time_axes:
        _draw_event_vlines(ax, event_vlines, style=event_vline_style)

    ax_scatter.scatter(
        ml_vals,
        ap_vals,
        color=com_style["background_color"],
        alpha=com_style["background_alpha"],
        s=com_style["background_size"],
    )

    if x_axis is not None:
        for span in window_spans:
            mask = (x_axis >= span["start"]) & (x_axis <= span["end"])
            if mask.any():
                ax_scatter.scatter(
                    ml_vals[mask],
                    ap_vals[mask],
                    s=com_style["scatter_size"],
                    alpha=com_style["scatter_alpha"],
                    color=span["color"],
                    label="_nolegend_",
                )

    if common_style.get("show_max_marker", True):
        max_time = markers.get("max")
        if max_time is not None and _is_within_time_axis(max_time, time_start_ms, time_end_ms):
            pass

    ax_x.set_title(
        comx_name,
        fontsize=common_style["title_fontsize"],
        fontweight=common_style["title_fontweight"],
        pad=common_style["title_pad"],
    )
    ax_y.set_title(
        comy_name,
        fontsize=common_style["title_fontsize"],
        fontweight=common_style["title_fontweight"],
        pad=common_style["title_pad"],
    )
    if ax_z is not None:
        ax_z.set_title(
            comz_name,
            fontsize=common_style["title_fontsize"],
            fontweight=common_style["title_fontweight"],
            pad=common_style["title_pad"],
        )
    ax_mag.set_title(
        "Magnitude",
        fontsize=common_style["title_fontsize"],
        fontweight=common_style["title_fontweight"],
        pad=common_style["title_pad"],
    )
    ax_scatter.set_title(
        "COMxy",
        fontsize=common_style["title_fontsize"],
        fontweight=common_style["title_fontweight"],
        pad=common_style["title_pad"],
    )

    axes_to_style = [ax_x, ax_y] + ([ax_z] if ax_z is not None else [])
    for ax in axes_to_style:
        ax.grid(True, alpha=common_style["grid_alpha"])
        ax.tick_params(labelsize=common_style["tick_labelsize"])
        ax.legend(
            fontsize=com_style["legend_fontsize"],
            loc=common_style["legend_loc"],
            framealpha=common_style["legend_framealpha"],
        )

    ax_mag.grid(True, alpha=common_style["grid_alpha"])
    ax_mag.tick_params(labelsize=common_style["tick_labelsize"])
    _apply_window_group_legends(
        ax_mag,
        window_spans=window_spans,
        group_handles=[],
        legend_fontsize=com_style["legend_fontsize"],
        framealpha=common_style["legend_framealpha"],
        loc=common_style["legend_loc"],
    )

    ax_scatter.grid(True, alpha=common_style["grid_alpha"])
    ax_scatter.tick_params(labelsize=common_style["tick_labelsize"])
    _apply_window_group_legends(
        ax_scatter,
        window_spans=window_spans,
        group_handles=[],
        legend_fontsize=com_style["legend_fontsize"],
        framealpha=common_style["legend_framealpha"],
        loc=common_style["legend_loc"],
    )

    ax_x.set_xlabel(com_style.get("x_label_time", "Normalized time (0-1)"), fontsize=common_style["label_fontsize"])
    ax_x.set_ylabel(com_style.get("y_label_comx", comx_name), fontsize=common_style["label_fontsize"])
    ax_y.set_xlabel(com_style.get("x_label_time", "Normalized time (0-1)"), fontsize=common_style["label_fontsize"])
    ax_y.set_ylabel(com_style.get("y_label_comy", comy_name), fontsize=common_style["label_fontsize"])
    if ax_z is not None:
        ax_z.set_xlabel(com_style.get("x_label_time", "Normalized time (0-1)"), fontsize=common_style["label_fontsize"])
        ax_z.set_ylabel(com_style.get("y_label_comz", comz_name), fontsize=common_style["label_fontsize"])
    ax_mag.set_xlabel(com_style.get("x_label_time", "Normalized time (0-1)"), fontsize=common_style["label_fontsize"])
    ax_mag.set_ylabel("COM magnitude", fontsize=common_style["label_fontsize"])

    ax_scatter.set_xlabel(com_style.get("x_label", comx_name), fontsize=common_style["label_fontsize"])
    ax_scatter.set_ylabel(com_style.get("y_label", comy_name), fontsize=common_style["label_fontsize"])
    ax_scatter.set_aspect("equal", adjustable="datalim")

    # window legend handled via _apply_window_group_legends(ax_scatter, ...)

    fig.suptitle(
        _format_title(signal_group="com", mode_name=mode_name, group_fields=group_fields, key=key),
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
    event_vlines_by_key: Dict[Tuple, List[Dict[str, Any]]],
    event_vline_style: Dict[str, Any],
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

    n_panels = 3
    fig_size = cop_style["subplot_size"]
    try:
        fig_w, fig_h = fig_size
        fig_size = (float(fig_w) * (n_panels / 3.0), float(fig_h))
    except (TypeError, ValueError):
        pass

    fig, axes = plt.subplots(1, n_panels, figsize=fig_size, dpi=common_style["dpi"])
    axes = np.asarray(axes).ravel()
    ax_cx = axes[0]
    ax_cy = axes[1]
    ax_scatter = axes[2]

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
    use_group_colors = common_style.get("use_group_colors", False)
    key_to_linestyle = _build_group_linestyles(sorted_keys, common_style.get("group_linestyles", ("-", "--", ":", "-.")))
    key_to_color = _build_group_color_map(sorted_keys, group_fields, color_by_fields, base_colors) if use_group_colors else {}

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
            channel_color = cop_style.get("line_colors", {}).get("Cx" if ch == cx_name else "Cy", "C0")
            color = key_to_color.get(key, channel_color) if use_group_colors else channel_color
            linestyle = key_to_linestyle.get(key, "-")
            label = _format_group_label(key, group_fields, filtered_group_fields) or "_nolegend_"

            plot_label = label if label not in seen_labels else "_nolegend_"
            if plot_label != "_nolegend_":
                seen_labels.add(label)
            ax.plot(
                x,
                y,
                color=color,
                linestyle=linestyle,
                linewidth=cop_style.get("line_width", 0.8),
                alpha=cop_style.get("line_alpha", 0.8),
                label=plot_label,
            )

        for key in sorted_keys:
            vlines = event_vlines_by_key.get(key, [])
            if not vlines:
                continue
            _draw_event_vlines(ax, vlines, style=event_vline_style)

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

    # Overlay line segments: window color, group line style
    overlay_linewidth = float(cop_style.get("line_width", 0.8))
    overlay_alpha = float(cop_style.get("scatter_alpha", 0.7))
    for span in window_spans:
        mask = (x >= span["start"]) & (x <= span["end"])
        if not mask.any():
            continue
        for key in sorted_keys:
            cx = aggregated_by_key.get(key, {}).get(cx_name)
            cy = aggregated_by_key.get(key, {}).get(cy_name)
            if cx is None or cy is None:
                continue
            ml_vals = (-cy) if cop_style["y_invert"] else cy
            linestyle = key_to_linestyle.get(key, "-")
    ax_scatter.plot(
                ml_vals[mask],
                cx[mask],
                color=span["color"],
                linestyle=linestyle,
                linewidth=overlay_linewidth,
                alpha=overlay_alpha,
                label="_nolegend_",
            )

    ax_scatter.grid(True, alpha=common_style["grid_alpha"])
    ax_scatter.tick_params(labelsize=common_style["tick_labelsize"])
    group_handles = _build_group_legend_handles(
        sorted_keys,
        group_fields,
        filtered_group_fields,
        key_to_linestyle,
        linewidth=overlay_linewidth,
    )
    _apply_window_group_legends(
        ax_scatter,
        window_spans=window_spans,
        group_handles=group_handles,
        legend_fontsize=cop_style["legend_fontsize"],
        framealpha=common_style["legend_framealpha"],
        loc=common_style["legend_loc"],
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


def _plot_com_overlay(
    *,
    aggregated_by_key: Dict[Tuple, Dict[str, np.ndarray]],
    event_vlines_by_key: Dict[Tuple, List[Dict[str, Any]]],
    event_vline_style: Dict[str, Any],
    output_path: Path,
    mode_name: str,
    group_fields: List[str],
    sorted_keys: List[Tuple],
    x: np.ndarray,
    window_spans: List[Dict[str, Any]],
    com_channels: Sequence[str],
    com_style: Dict[str, Any],
    common_style: Dict[str, Any],
    filtered_group_fields: List[str],
    color_by_fields: Optional[List[str]] = None,
) -> None:
    import matplotlib.pyplot as plt

    comx_name, comy_name, comz_name = _resolve_com_channel_names(com_channels)
    if comz_name is not None and not any(
        aggregated_by_key.get(key, {}).get(comz_name) is not None for key in sorted_keys
    ):
        comz_name = None

    n_panels = 5 if comz_name is not None else 4
    fig_size = com_style["subplot_size"]
    try:
        fig_w, fig_h = fig_size
        fig_size = (float(fig_w) * (n_panels / 3.0), float(fig_h))
    except (TypeError, ValueError):
        pass

    fig, axes = plt.subplots(1, n_panels, figsize=fig_size, dpi=common_style["dpi"])
    axes = np.asarray(axes).ravel()
    ax_x = axes[0]
    ax_y = axes[1]
    if comz_name is not None:
        ax_z = axes[2]
        ax_mag = axes[3]
        ax_scatter = axes[4]
    else:
        ax_z = None
        ax_mag = axes[2]
        ax_scatter = axes[3]

    window_span_alpha = float(com_style.get("window_span_alpha", 0.15))
    time_axes = [ax_x, ax_y] + ([ax_z] if ax_z is not None else []) + [ax_mag]
    for ax in time_axes:
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
    use_group_colors = common_style.get("use_group_colors", False)
    key_to_linestyle = _build_group_linestyles(sorted_keys, common_style.get("group_linestyles", ("-", "--", ":", "-.")))
    key_to_color = _build_group_color_map(sorted_keys, group_fields, color_by_fields, base_colors) if use_group_colors else {}

    channel_axes = [
        (ax_x, comx_name, com_style.get("y_label_comx", comx_name)),
        (ax_y, comy_name, com_style.get("y_label_comy", comy_name)),
    ]
    if ax_z is not None and comz_name is not None:
        channel_axes.append((ax_z, comz_name, com_style.get("y_label_comz", comz_name)))

    for ax, ch, y_label in channel_axes:
        seen_labels: set[str] = set()
        for key in sorted_keys:
            series = aggregated_by_key.get(key, {}).get(ch)
            if series is None:
                continue
            y = (-series) if (ch == comy_name and com_style.get("y_invert", False)) else series
            channel_color = com_style.get("line_colors", {}).get(ch, "C0")
            color = key_to_color.get(key, channel_color) if use_group_colors else channel_color
            linestyle = key_to_linestyle.get(key, "-")
            label = _format_group_label(key, group_fields, filtered_group_fields) or "_nolegend_"

            plot_label = label if label not in seen_labels else "_nolegend_"
            if plot_label != "_nolegend_":
                seen_labels.add(label)
            ax.plot(
                x,
                y,
                color=color,
                linestyle=linestyle,
                linewidth=com_style.get("line_width", 0.8),
                alpha=com_style.get("line_alpha", 0.8),
                label=plot_label,
            )

        for key in sorted_keys:
            vlines = event_vlines_by_key.get(key, [])
            if not vlines:
                continue
            _draw_event_vlines(ax, vlines, style=event_vline_style)

        ax.grid(True, alpha=common_style["grid_alpha"])
        ax.tick_params(labelsize=common_style["tick_labelsize"])
        ax.legend(
            fontsize=com_style["legend_fontsize"],
            loc=common_style["legend_loc"],
            framealpha=common_style["legend_framealpha"],
        )
        ax.set_xlabel(com_style.get("x_label_time", "Normalized time (0-1)"), fontsize=common_style["label_fontsize"])
        ax.set_ylabel(y_label, fontsize=common_style["label_fontsize"])

    # Magnitude time series (window color, group line style)
    for key in sorted_keys:
        comx = aggregated_by_key.get(key, {}).get(comx_name)
        comy = aggregated_by_key.get(key, {}).get(comy_name)
        if comx is None or comy is None:
            continue
        ml_vals = (-comy) if com_style.get("y_invert", False) else comy
        mag_vals = np.sqrt((comx**2) + (ml_vals**2))
        linestyle = key_to_linestyle.get(key, "-")
        for span in window_spans:
            mask = (x >= span["start"]) & (x <= span["end"])
            if not mask.any():
                continue
            ax_mag.plot(
                x[mask],
                mag_vals[mask],
                color=span["color"],
                linestyle=linestyle,
                linewidth=com_style.get("line_width", 0.8),
                alpha=com_style.get("line_alpha", 0.8),
                label="_nolegend_",
            )

    for key in sorted_keys:
        vlines = event_vlines_by_key.get(key, [])
        if not vlines:
            continue
        _draw_event_vlines(ax_mag, vlines, style=event_vline_style)

    ax_mag.grid(True, alpha=common_style["grid_alpha"])
    ax_mag.tick_params(labelsize=common_style["tick_labelsize"])
    ax_mag.set_xlabel(com_style.get("x_label_time", "Normalized time (0-1)"), fontsize=common_style["label_fontsize"])
    ax_mag.set_ylabel("COM magnitude", fontsize=common_style["label_fontsize"])
    group_handles = _build_group_legend_handles(
        sorted_keys,
        group_fields,
        filtered_group_fields,
        key_to_linestyle,
        linewidth=float(com_style.get("line_width", 0.8)),
    )
    _apply_window_group_legends(
        ax_mag,
        window_spans=window_spans,
        group_handles=group_handles,
        legend_fontsize=com_style["legend_fontsize"],
        framealpha=common_style["legend_framealpha"],
        loc=common_style["legend_loc"],
    )

    ax_x.set_title(
        comx_name,
        fontsize=common_style["title_fontsize"],
        fontweight=common_style["title_fontweight"],
        pad=common_style["title_pad"],
    )
    ax_y.set_title(
        comy_name,
        fontsize=common_style["title_fontsize"],
        fontweight=common_style["title_fontweight"],
        pad=common_style["title_pad"],
    )
    if ax_z is not None and comz_name is not None:
        ax_z.set_title(
            comz_name,
            fontsize=common_style["title_fontsize"],
            fontweight=common_style["title_fontweight"],
            pad=common_style["title_pad"],
        )
    ax_mag.set_title(
        "Magnitude",
        fontsize=common_style["title_fontsize"],
        fontweight=common_style["title_fontweight"],
        pad=common_style["title_pad"],
    )

    overlay_linewidth = float(com_style.get("line_width", 0.8))
    overlay_alpha = float(com_style.get("scatter_alpha", 0.7))
    for span in window_spans:
        mask = (x >= span["start"]) & (x <= span["end"])
        if not mask.any():
            continue
        for key in sorted_keys:
            comx = aggregated_by_key.get(key, {}).get(comx_name)
            comy = aggregated_by_key.get(key, {}).get(comy_name)
            if comx is None or comy is None:
                continue
            ml_vals = (-comy) if com_style.get("y_invert", False) else comy
            linestyle = key_to_linestyle.get(key, "-")
            ax_scatter.plot(
                ml_vals[mask],
                comx[mask],
                color=span["color"],
                linestyle=linestyle,
                linewidth=overlay_linewidth,
                alpha=overlay_alpha,
                label="_nolegend_",
            )

    ax_scatter.grid(True, alpha=common_style["grid_alpha"])
    ax_scatter.tick_params(labelsize=common_style["tick_labelsize"])
    group_handles = _build_group_legend_handles(
        sorted_keys,
        group_fields,
        filtered_group_fields,
        key_to_linestyle,
        linewidth=overlay_linewidth,
    )
    _apply_window_group_legends(
        ax_scatter,
        window_spans=window_spans,
        group_handles=group_handles,
        legend_fontsize=com_style["legend_fontsize"],
        framealpha=common_style["legend_framealpha"],
        loc=common_style["legend_loc"],
    )

    ax_scatter.set_title(
        "COMxy",
        fontsize=common_style["title_fontsize"],
        fontweight=common_style["title_fontweight"],
        pad=common_style["title_pad"],
    )
    ax_scatter.set_xlabel(com_style.get("x_label", comx_name), fontsize=common_style["label_fontsize"])
    ax_scatter.set_ylabel(com_style.get("y_label", comy_name), fontsize=common_style["label_fontsize"])
    ax_scatter.set_aspect("equal", adjustable="datalim")

    overlay_by = ", ".join(group_fields) if group_fields else "all"
    fig.suptitle(
        f"{mode_name} | com | overlay by {overlay_by}",
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

        event_vlines_cfg = self.config.get("event_vlines")
        self.event_vline_columns = _parse_event_vlines_config(event_vlines_cfg)
        self.event_vline_meta_cols = [_event_ms_col(col) for col in self.event_vline_columns]
        self.event_vline_style = _parse_event_vlines_style(event_vlines_cfg)
        self.event_vline_colors = _build_event_vline_color_map(self.event_vline_columns, event_vlines_cfg)

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
        self.com_style = self._build_com_style(style_cfg.get("com", {}), self.cop_style)
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
        for col in self.event_vline_meta_cols:
            if col not in meta_cols_needed:
                meta_cols_needed.append(col)

        generated_outputs: List[Path] = []
        group_names = self._signal_group_names(selected_groups)
        for group_name in group_names:
            resampled = self._resample_signal_group(lf, group_name, meta_cols_needed)
            tasks: List[Dict[str, Any]] = []
            for mode_name, mode_cfg in enabled_modes:
                tasks.extend(self._build_plot_tasks(resampled, group_name, mode_name, mode_cfg))
            self._run_plot_tasks(tasks)
            generated_outputs.extend(self._collect_existing_outputs(tasks))
        self._log_generated_outputs(generated_outputs)

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
        if self.event_vline_columns:
            available_cols = set(self._lazy_columns(lf))
            ms_per_frame = 1000.0 / float(self.device_rate)
            for event_col in self.event_vline_columns:
                if event_col not in available_cols:
                    print(f"[event_vlines] Warning: column '{event_col}' not found in input; skipping")
                    continue
                # Interpret event in the same domain as `platform_onset` (mocap frame) and convert to ms.
                event_mocap = pl.col(event_col).max().over(group_cols)  # ignores nulls
                event_rel_frame = (event_mocap - onset_mocap) * self.frame_ratio
                extra_event_exprs.append((event_rel_frame * ms_per_frame).alias(_event_ms_col(event_col)))

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
            event_vlines_by_key = {key: self._collect_event_vlines(meta_df, idx) for key, idx in grouped.items()}

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
                        event_vlines_by_key=event_vlines_by_key,
                        output_path=output_path,
                        mode_name=mode_name,
                        group_fields=group_fields,
                        sorted_keys=sorted_keys,
                        window_spans=window_spans,
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
                        event_vlines_by_key=file_event_vlines,
                        output_path=output_path,
                        mode_name=mode_name,
                        group_fields=group_fields,
                        sorted_keys=sorted_keys,
                        window_spans=window_spans,
                        filtered_group_fields=filtered_group_fields,
                        color_by_fields=mode_cfg.get("color_by") if signal_group in ("emg", "cop", "com") else None,
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
            event_vlines = self._collect_event_vlines(meta_df, idx)

            if signal_group == "emg":
                tasks.append(
                    self._task_emg(
                        aggregated=aggregated,
                        output_path=output_path,
                        key=key,
                        mode_name=mode_name,
                        group_fields=group_fields,
                        markers=markers,
                        event_vlines=event_vlines,
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
                        event_vlines=event_vlines,
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
                        event_vlines=event_vlines,
                        window_spans=window_spans,
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
            out.append({"name": event_col, "x": float(x), "color": self.event_vline_colors.get(event_col)})
        return out

    def _task_overlay(
        self,
        *,
        signal_group: str,
        aggregated_by_key: Dict[Tuple, Dict[str, np.ndarray]],
        markers_by_key: Dict[Tuple, Dict[str, Any]],
        event_vlines_by_key: Dict[Tuple, List[Dict[str, Any]]],
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
            "event_vlines_by_key": event_vlines_by_key,
            "event_vline_style": self.event_vline_style,
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

        if signal_group == "com":
            task.update(
                {
                    "style": self.com_style,
                    "cop_channels": self.config["signal_groups"]["com"]["columns"],
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
            "event_vlines": event_vlines,
            "event_vline_style": self.event_vline_style,
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
        event_vlines: List[Dict[str, Any]],
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
            "event_vlines": event_vlines,
            "event_vline_style": self.event_vline_style,
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
        event_vlines: List[Dict[str, Any]],
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
            "event_vlines": event_vlines,
            "event_vline_style": self.event_vline_style,
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
    ) -> Dict[str, Any]:
        return {
            "kind": "com",
            "output_path": str(output_path),
            "key": key,
            "mode_name": mode_name,
            "group_fields": group_fields,
            "aggregated": aggregated,
            "markers": markers,
            "event_vlines": event_vlines,
            "event_vline_style": self.event_vline_style,
            "x_axis": self.x_norm,
            "time_start_ms": self.time_start_ms,
            "time_end_ms": self.time_end_ms,
            "device_rate": self.device_rate,
            "com_channels": self.config["signal_groups"]["com"]["columns"],
            "com_style": self.com_style,
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
        return out

    @staticmethod
    def _build_com_style(com_cfg: Any, cop_style: Dict[str, Any]) -> Dict[str, Any]:
        base_line_colors = dict(cop_style.get("line_colors", {}) or {})
        comx_color = base_line_colors.get("Cx", "blue")
        comy_color = base_line_colors.get("Cy", "red")
        comz_color = base_line_colors.get("Cz", "green")

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
