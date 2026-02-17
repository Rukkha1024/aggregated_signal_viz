from __future__ import annotations

"""Matplotlib plotting implementation (extracted from the legacy visualizer module).

This module intentionally keeps the original helper/plot function names to
minimize behavior changes during refactors. Higher-level wrappers live in:
- src/plotting/matplotlib/task.py
- src/plotting/matplotlib/line.py
- src/plotting/matplotlib/scatter.py
"""

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

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


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    return bool(default)


def _parse_window_boundary_spec(value: Any) -> Optional[Tuple[str, Any]]:
    """
    윈도우 경계(start_ms/end_ms) 정의를 파싱합니다.

    지원 형식:
    - 숫자(int/float 또는 숫자 문자열) -> ("offset", <float>)
    - 이벤트 컬럼명(문자열) -> ("event", <str>)
    - 이벤트 +/- 오프셋(문자열) -> ("event_offset", (<str>, <float>))
    """
    num = _coerce_float(value)
    if num is not None:
        return ("offset", float(num))
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    m = re.match(r"^\s*(?P<event>[^+-]+?)\s*(?P<op>[+-])\s*(?P<offset>\d+(?:\.\d+)?)\s*$", text)
    if m:
        event_name = (m.group("event") or "").strip()
        if event_name.startswith("(") and event_name.endswith(")"):
            event_name = event_name[1:-1].strip()
        op = (m.group("op") or "").strip()
        offset_text = (m.group("offset") or "").strip()
        try:
            offset_val = float(offset_text)
        except ValueError:
            offset_val = 0.0
        signed = offset_val if op == "+" else -offset_val
        if event_name:
            return ("event_offset", (event_name, float(signed)))

    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    return ("event", text)


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
        linestyle = v.get("linestyle")
        if linestyle is not None and str(linestyle).strip():
            kwargs["linestyle"] = str(linestyle).strip()
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


def _build_event_vline_legend_handles(
    vlines: Sequence[Dict[str, Any]],
    *,
    style: Dict[str, Any],
) -> List[Any]:
    if not vlines:
        return []
    try:
        import matplotlib.lines as mlines
    except Exception:
        return []

    base_kwargs = dict(style) if style else {"color": "red", "linestyle": "--", "linewidth": 1.5}
    base_kwargs.setdefault("alpha", 0.9)

    def _linestyle_for_legend(value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            if text == "--":
                return (0, (3, 2))
            if text == ":":
                return (0, (1, 2))
            if text == "-.":
                return (0, (3, 2, 1, 2))
            return text if text else "--"
        return value

    linestyle = _linestyle_for_legend(base_kwargs.get("linestyle", "--"))
    linewidth = float(base_kwargs.get("linewidth", 1.5))
    legend_linewidth = min(linewidth, 1.0)
    alpha = float(base_kwargs.get("alpha", 0.9))
    base_color = base_kwargs.get("color", "red")

    handles: List[Any] = []
    seen: set[str] = set()
    for v in vlines:
        raw_label = v.get("label")
        if raw_label is None or not str(raw_label).strip():
            raw_label = v.get("name")
        if raw_label is None:
            continue
        label = str(raw_label).strip()
        if not label or label in seen:
            continue
        seen.add(label)
        color = v.get("color")
        resolved_color = str(color).strip() if color is not None and str(color).strip() else str(base_color)
        v_linestyle = v.get("linestyle", base_kwargs.get("linestyle", "--"))
        linestyle = _linestyle_for_legend(v_linestyle)
        handles.append(
            mlines.Line2D(
                [],
                [],
                color=resolved_color,
                linestyle=linestyle,
                linewidth=legend_linewidth,
                alpha=alpha,
                label=label,
            )
        )
    return handles


def _apply_window_group_legends(
    ax: Any,
    *,
    window_spans: Sequence[Dict[str, Any]],
    group_handles: Sequence[Any],
    event_vlines: Sequence[Dict[str, Any]] = (),
    event_vline_style: Optional[Dict[str, Any]] = None,
    legend_fontsize: float,
    framealpha: float,
    loc: str = "best",
) -> None:
    # NOTE: Legend styling is intentionally kept in code (not config.yaml) per project rules.
    # We normalize Line2D legend widths to avoid mixed linewidths when combining:
    # - plotted series handles (often thicker)
    # - custom event vline handles (often thinner)
    legend_linewidth = 1.0

    def _clone_handle(handle: Any, *, label_override: Optional[str] = None) -> Any:
        try:
            import matplotlib.lines as mlines
            import matplotlib.patches as mpatches
        except Exception:
            return handle

        try:
            label = str(label_override if label_override is not None else getattr(handle, "get_label", lambda: "")()).strip()
        except Exception:
            label = ""

        if isinstance(handle, mlines.Line2D):
            try:
                return mlines.Line2D(
                    [],
                    [],
                    color=handle.get_color(),
                    linestyle=handle.get_linestyle(),
                    linewidth=legend_linewidth,
                    alpha=handle.get_alpha(),
                    label=label,
                )
            except Exception:
                return handle

        if isinstance(handle, mpatches.Patch):
            try:
                return mpatches.Patch(
                    facecolor=handle.get_facecolor(),
                    edgecolor=handle.get_edgecolor(),
                    alpha=handle.get_alpha(),
                    label=label,
                )
            except Exception:
                return handle

        return handle

    handles: List[Any] = []
    seen_labels: set[str] = set()

    for handle in _build_window_legend_handles(window_spans):
        label = str(getattr(handle, "get_label", lambda: "")()).strip()
        if not label or label == "_nolegend_" or label in seen_labels:
            continue
        seen_labels.add(label)
        handles.append(_clone_handle(handle, label_override=label))

    for handle in group_handles:
        label = str(getattr(handle, "get_label", lambda: "")()).strip()
        if not label or label == "_nolegend_" or label in seen_labels:
            continue
        seen_labels.add(label)
        handles.append(_clone_handle(handle, label_override=label))

    if event_vlines:
        for handle in _build_event_vline_legend_handles(event_vlines, style=event_vline_style or {}):
            label = str(getattr(handle, "get_label", lambda: "")()).strip()
            if not label or label == "_nolegend_" or label in seen_labels:
                continue
            seen_labels.add(label)
            handles.append(_clone_handle(handle, label_override=label))

    existing_handles, existing_labels = ax.get_legend_handles_labels()
    for handle, label in zip(existing_handles, existing_labels):
        label = str(label).strip()
        if not label or label == "_nolegend_" or label in seen_labels:
            continue
        seen_labels.add(label)
        handles.append(_clone_handle(handle, label_override=label))

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
        return "gray"

    direct = line_colors.get(channel)
    if direct is not None and str(direct).strip():
        return str(direct).strip()

    base = channel[:-5] if channel.endswith("_zero") else channel
    base_color = line_colors.get(base)
    if base_color is not None and str(base_color).strip():
        return str(base_color).strip()

    return "gray"


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


def _overlay_vline_event_names(overlay_cfg: Optional[Dict[str, Any]]) -> set[str]:
    if not isinstance(overlay_cfg, dict):
        return set()
    if not bool(overlay_cfg.get("enabled", False)):
        return set()
    raw_cols = overlay_cfg.get("columns")
    if not isinstance(raw_cols, (list, tuple)):
        return set()
    return {str(c).strip() for c in raw_cols if c is not None and str(c).strip()}


def _parse_event_labels(cfg: Any) -> Dict[str, str]:
    if not isinstance(cfg, dict):
        return {}
    raw = cfg.get("event_labels")
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, str] = {}
    for key, value in raw.items():
        if key is None or value is None:
            continue
        name = str(key).strip()
        label = str(value).strip()
        if not name or not label:
            continue
        out[name] = label
    return out


def _overlay_group_label(
    *,
    key: Tuple,
    group_fields: Sequence[str],
    filtered_group_fields: Sequence[str],
) -> str:
    label = _format_group_label(key, list(group_fields), list(filtered_group_fields))
    if label is not None and str(label).strip():
        return str(label).strip()
    # Fallback: stable string for arbitrary group keys
    parts = [str(v).strip() for v in key if v is not None and str(v).strip()]
    return ", ".join(parts) if parts else "group"


def _build_overlay_event_vline_legend_vlines(
    *,
    overlay_events: Sequence[str],
    sorted_keys: Sequence[Tuple],
    group_fields: Sequence[str],
    filtered_group_fields: Sequence[str],
    key_to_linestyle: Dict[Tuple, Any],
    event_labels: Dict[str, str],
    event_colors: Dict[str, str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for event in overlay_events:
        event_name = str(event).strip()
        if not event_name:
            continue
        event_label = event_labels.get(event_name, event_name)
        color = event_colors.get(event_name)
        for key in sorted_keys:
            group_label = _overlay_group_label(
                key=tuple(key),
                group_fields=group_fields,
                filtered_group_fields=filtered_group_fields,
            )
            linestyle = key_to_linestyle.get(tuple(key), "-")
            item: Dict[str, Any] = {
                "name": event_name,
                "label": f"{event_label} ({group_label})",
                "linestyle": linestyle,
            }
            if color is not None and str(color).strip():
                item["color"] = str(color).strip()
            out.append(item)
    return out


def _infer_event_labels_and_colors(
    *,
    overlay_events: Sequence[str],
    pooled_event_vlines: Sequence[Dict[str, Any]],
    event_vlines_by_key: Dict[Tuple, List[Dict[str, Any]]],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    overlay_set = {str(e).strip() for e in overlay_events if e is not None and str(e).strip()}
    if not overlay_set:
        return {}, {}

    labels: Dict[str, str] = {}
    colors: Dict[str, str] = {}

    def consider(v: Dict[str, Any]) -> None:
        name = v.get("name")
        if name is None:
            return
        event = str(name).strip()
        if not event or event not in overlay_set:
            return
        label = v.get("label")
        if event not in labels and label is not None and str(label).strip():
            labels[event] = str(label).strip()
        color = v.get("color")
        if event not in colors and color is not None and str(color).strip():
            colors[event] = str(color).strip()

    for v in pooled_event_vlines:
        consider(v)
    for vlines in event_vlines_by_key.values():
        for v in vlines:
            consider(v)

    return labels, colors


def _parse_window_colors(cfg: Any) -> Dict[str, str]:
    if not isinstance(cfg, dict):
        return {}
    out: Dict[str, str] = {}
    for key, value in cfg.items():
        if key is None or value is None:
            continue
        name = str(key).strip()
        color = str(value).strip()
        if not name or not color:
            continue
        out[name] = color
    return out


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


def _flatten_axes(axes: Any) -> np.ndarray:
    return axes.flatten() if isinstance(axes, np.ndarray) else np.asarray([axes])


def _draw_window_spans(
    ax: Any,
    window_spans: Sequence[Dict[str, Any]],
    *,
    alpha: float,
    with_labels: bool = True,
) -> None:
    label_default = "_nolegend_" if not with_labels else None
    for span in window_spans:
        label = span.get("label") if with_labels else label_default
        ax.axvspan(
            span["start"],
            span["end"],
            color=span["color"],
            alpha=alpha,
            label=label,
        )


def _apply_frame_tick_labels(
    ax: Any,
    *,
    time_start_frame: float,
    time_end_frame: float,
    time_zero_frame: float = 0.0,
    blank_positions: Sequence[float] = (),
    blank_tol: float = 1e-6,
) -> None:
    try:
        from matplotlib.ticker import FuncFormatter
    except Exception:
        return

    start = float(time_start_frame)
    end = float(time_end_frame)
    span = end - start
    zero_frame = float(time_zero_frame)
    if span == 0:
        return

    blanks: List[float] = []
    if blank_positions:
        for pos in blank_positions:
            try:
                x = float(pos)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(x):
                continue
            blanks.append(x)

    def _fmt(x: float, _pos: int) -> str:
        if blanks:
            try:
                xx = float(x)
            except Exception:
                xx = x
            for pos in blanks:
                if abs(float(xx) - float(pos)) <= float(blank_tol):
                    return ""
        try:
            frame = start + float(x) * span
        except Exception:
            return ""
        if not np.isfinite(frame):
            return ""
        frame -= zero_frame
        if not np.isfinite(frame):
            return ""
        return f"{frame:.0f}"

    ax.xaxis.set_major_formatter(FuncFormatter(_fmt))


def _apply_window_definition_xticks(
    ax: Any,
    window_spans: Sequence[Dict[str, Any]],
    *,
    include_edges: bool = True,
) -> List[float]:
    ax.set_xlim(0.0, 1.0)
    if not window_spans:
        return []

    ticks: List[float] = []
    if include_edges:
        ticks.extend([0.0, 1.0])

    for span in window_spans:
        for key in ("start", "end"):
            value = span.get(key)
            if value is None:
                continue
            try:
                ticks.append(float(value))
            except (TypeError, ValueError):
                continue

    ticks = [t for t in ticks if np.isfinite(t)]
    if not ticks:
        return []

    ticks_sorted = sorted(ticks)
    uniq: List[float] = []
    tol = 1e-6
    for t in ticks_sorted:
        if not uniq or abs(t - uniq[-1]) > tol:
            uniq.append(t)
    ax.set_xticks(uniq)
    return uniq


def _ensure_time_zero_xtick(
    ax: Any,
    *,
    tick_positions: Sequence[float],
    time_start_frame: float,
    time_end_frame: float,
    time_zero_frame: float = 0.0,
    tol: float = 1e-6,
) -> List[float]:
    start = float(time_start_frame)
    end = float(time_end_frame)
    zero_frame = float(time_zero_frame)
    span = end - start
    if span == 0 or not np.isfinite(start) or not np.isfinite(end):
        return list(tick_positions)

    x0 = (zero_frame - start) / span
    if not np.isfinite(x0):
        return list(tick_positions)
    if x0 < -tol or x0 > 1.0 + tol:
        return list(tick_positions)

    existing = list(tick_positions) if tick_positions else list(ax.get_xticks())
    existing = [float(t) for t in existing if np.isfinite(t)]
    if any(abs(float(t) - float(x0)) <= tol for t in existing):
        return existing

    existing.append(float(x0))
    existing_sorted = sorted(existing)
    uniq: List[float] = []
    for t in existing_sorted:
        if not uniq or abs(t - uniq[-1]) > tol:
            uniq.append(float(t))
    ax.set_xticks(uniq)
    return uniq


def _collect_overlay_event_vlines_for_ticks(
    *,
    sorted_keys: Sequence[Tuple],
    event_vlines_by_key: Dict[Tuple, List[Dict[str, Any]]],
    overlay_cfg: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not isinstance(overlay_cfg, dict):
        return []
    if not bool(overlay_cfg.get("enabled", False)):
        return []
    mode = str(overlay_cfg.get("mode") or "").strip().lower()
    if mode != "linestyle":
        return []
    overlay_events = _overlay_vline_event_names(overlay_cfg)
    if not overlay_events:
        return []
    if not event_vlines_by_key:
        return []

    out: List[Dict[str, Any]] = []
    for key in sorted_keys:
        vlines_all = event_vlines_by_key.get(tuple(key), [])
        if not vlines_all:
            continue
        for v in vlines_all:
            name = v.get("name")
            event = str(name).strip() if name is not None else ""
            if event and event in overlay_events:
                out.append(v)
    return out


def _event_tick_positions_by_name(
    event_vlines: Sequence[Dict[str, Any]],
    *,
    tol: float = 1e-6,
) -> Dict[str, float]:
    by_name: Dict[str, List[float]] = {}
    for v in event_vlines:
        name_raw = v.get("name")
        name = str(name_raw).strip() if name_raw is not None else ""
        if not name:
            continue
        x_raw = v.get("x")
        if x_raw is None:
            continue
        try:
            x = float(x_raw)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(x):
            continue
        if x < -tol or x > 1.0 + tol:
            continue
        by_name.setdefault(name, []).append(min(1.0, max(0.0, x)))

    out: Dict[str, float] = {}
    for name, xs in by_name.items():
        if not xs:
            continue
        out[name] = float(np.mean(np.asarray(xs, dtype=float)))
    return out


def _select_event_tick_labels_to_blank(
    ax: Any,
    *,
    event_positions: Dict[str, float],
    event_order: Sequence[str],
    tick_labelsize: float,
    tol: float = 1e-6,
) -> set[str]:
    if len(event_positions) < 2:
        return set()

    order_map = {str(name).strip(): i for i, name in enumerate(event_order) if name is not None and str(name).strip()}

    def priority(name: str) -> int:
        return order_map.get(name, len(order_map) + 1)

    width_px = float(getattr(getattr(ax, "bbox", None), "width", 0.0) or 0.0)
    min_sep_px = max(6.0, float(tick_labelsize) * 4.0)
    min_sep_norm = min_sep_px / width_px if width_px > 1e-6 else 0.03

    items = [(name, float(x)) for name, x in event_positions.items() if np.isfinite(float(x))]
    items.sort(key=lambda kv: kv[1])

    hidden: set[str] = set()
    while True:
        visible = [(name, x) for name, x in items if name not in hidden]
        if len(visible) < 2:
            break
        changed = False
        for (name_a, x_a), (name_b, x_b) in zip(visible, visible[1:]):
            if abs(float(x_b) - float(x_a)) + tol >= float(min_sep_norm):
                continue
            pr_a = priority(name_a)
            pr_b = priority(name_b)
            if pr_a == pr_b:
                hidden.add(name_b)
            elif pr_a < pr_b:
                hidden.add(name_b)
            else:
                hidden.add(name_a)
            changed = True
            break
        if not changed:
            break
    return hidden


def _apply_event_vline_xticks(
    ax: Any,
    *,
    tick_positions: Sequence[float],
    event_vlines: Sequence[Dict[str, Any]],
    event_order: Sequence[str],
    tick_labelsize: float,
    tol: float = 1e-6,
) -> Tuple[List[float], List[float]]:
    base = list(tick_positions) if tick_positions else list(ax.get_xticks())

    event_positions = _event_tick_positions_by_name(event_vlines, tol=tol)
    hidden_names = _select_event_tick_labels_to_blank(
        ax,
        event_positions=event_positions,
        event_order=event_order,
        tick_labelsize=tick_labelsize,
        tol=tol,
    )
    visible_positions = [event_positions[name] for name in event_positions.keys() if name not in hidden_names]
    blank_positions: List[float] = []
    for name in sorted(hidden_names):
        if name not in event_positions:
            continue
        x_hidden = float(event_positions[name])
        if any(abs(float(x_visible) - x_hidden) <= tol for x_visible in visible_positions):
            continue
        blank_positions.append(x_hidden)

    merged: List[float] = []
    for value in base:
        try:
            x = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(x):
            continue
        if x < -tol or x > 1.0 + tol:
            continue
        merged.append(min(1.0, max(0.0, x)))

    for x in event_positions.values():
        if not np.isfinite(float(x)):
            continue
        merged.append(min(1.0, max(0.0, float(x))))

    if not merged:
        return base, blank_positions

    merged_sorted = sorted(merged)
    uniq: List[float] = []
    for t in merged_sorted:
        if not uniq or abs(t - uniq[-1]) > tol:
            uniq.append(t)
    if uniq:
        ax.set_xticks(uniq)
    return uniq, blank_positions


def _auto_rotate_dense_xticklabels(
    ax: Any,
    *,
    tick_positions: Sequence[float],
    time_start_frame: float,
    time_end_frame: float,
    min_delta_frame: float = 40.0,
) -> None:
    if not tick_positions:
        return
    start = float(time_start_frame)
    end = float(time_end_frame)
    span = end - start
    if span == 0:
        return

    frames = [start + float(x) * span for x in tick_positions]
    frames = [f for f in frames if np.isfinite(f)]
    if len(frames) < 3:
        return

    frames_sorted = sorted(frames)
    deltas = [b - a for a, b in zip(frames_sorted, frames_sorted[1:])]
    if not deltas:
        return

    if min(deltas) < float(min_delta_frame):
        ax.tick_params(axis="x", labelrotation=90)
        for label in ax.get_xticklabels():
            label.set_ha("center")
            label.set_va("top")
    else:
        ax.tick_params(axis="x", labelrotation=0)


def _prepare_overlay_group_styles(
    *,
    sorted_keys: List[Tuple],
    group_fields: List[str],
    color_by_fields: Optional[List[str]],
    common_style: Dict[str, Any],
) -> Tuple[bool, Dict[Tuple, str], Dict[Tuple, Any]]:
    import matplotlib as mpl

    base_colors = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    use_group_colors = common_style.get("use_group_colors", False)
    key_to_color = (
        _build_group_color_map(sorted_keys, group_fields, color_by_fields, base_colors) if use_group_colors else {}
    )
    key_to_linestyle = _build_group_linestyles(
        sorted_keys,
        common_style.get("group_linestyles", ("-", "--", ":", "-.")),
    )
    return use_group_colors, key_to_color, key_to_linestyle


def _plot_overlay_channel_series(
    ax: Any,
    *,
    x: np.ndarray,
    channel: str,
    sorted_keys: List[Tuple],
    aggregated_by_key: Dict[Tuple, Dict[str, np.ndarray]],
    group_fields: List[str],
    filtered_group_fields: List[str],
    line_width: float,
    line_alpha: float,
    channel_color: str,
    use_group_colors: bool,
    key_to_color: Dict[Tuple, str],
    key_to_linestyle: Dict[Tuple, Any],
) -> None:
    seen_labels: set[str] = set()
    for key in sorted_keys:
        y = aggregated_by_key.get(key, {}).get(channel)
        if y is None:
            continue
        group_label = _format_group_label(key, group_fields, filtered_group_fields)
        if group_label is None or group_label in seen_labels:
            plot_label = "_nolegend_"
        else:
            plot_label = group_label
            seen_labels.add(group_label)
        color = key_to_color.get(key, channel_color) if use_group_colors else channel_color
        linestyle = key_to_linestyle.get(key, "-")
        ax.plot(
            x,
            y,
            color=color,
            linestyle=linestyle,
            linewidth=line_width,
            alpha=line_alpha,
            label=plot_label,
        )


def _draw_event_vlines_for_keys(
    ax: Any,
    *,
    sorted_keys: List[Tuple],
    event_vlines_by_key: Dict[Tuple, List[Dict[str, Any]]],
    style: Dict[str, Any],
    overlay_cfg: Optional[Dict[str, Any]] = None,
    key_to_linestyle: Optional[Dict[Tuple, Any]] = None,
) -> None:
    mode = ""
    enabled = False
    linestyles: Tuple[Any, ...] = ()
    overlay_events: set[str] = set()
    if isinstance(overlay_cfg, dict):
        enabled = bool(overlay_cfg.get("enabled", False))
        mode = str(overlay_cfg.get("mode") or "").strip().lower()
        if mode != "linestyle":
            mode = ""
        overlay_events = _overlay_vline_event_names(overlay_cfg)
        raw_ls = overlay_cfg.get("linestyles")
        if isinstance(raw_ls, (list, tuple)) and raw_ls:
            linestyles = _parse_group_linestyles(raw_ls)

    for key_idx, key in enumerate(sorted_keys):
        vlines_all = event_vlines_by_key.get(key, [])
        if not vlines_all:
            continue
        if not (enabled and mode and overlay_events):
            continue
        vlines = [v for v in vlines_all if str(v.get("name") or "").strip() in overlay_events]
        if not vlines:
            continue

        override_ls: Optional[Any] = None
        if key_to_linestyle is not None and key in key_to_linestyle:
            override_ls = key_to_linestyle.get(key)
        elif linestyles:
            override_ls = linestyles[key_idx % len(linestyles)]

        if override_ls is None:
            _draw_event_vlines(ax, vlines, style=style)
            continue

        styled: List[Dict[str, Any]] = []
        for v in vlines:
            vv = dict(v)
            vv["linestyle"] = override_ls
            styled.append(vv)
        _draw_event_vlines(ax, styled, style=style)


def _style_timeseries_axis(
    ax: Any,
    *,
    title: str,
    common_style: Dict[str, Any],
    legend_fontsize: float,
    window_spans: Sequence[Dict[str, Any]],
    group_handles: Sequence[Any] = (),
    event_vlines: Sequence[Dict[str, Any]],
    event_vline_style: Dict[str, Any],
) -> None:
    if common_style.get("show_subplot_titles", True):
        ax.set_title(
            title,
            fontsize=common_style["title_fontsize"],
            fontweight=common_style["title_fontweight"],
            pad=common_style["title_pad"],
        )

    if common_style.get("show_grid", True):
        ax.grid(True, alpha=common_style["grid_alpha"])
    else:
        ax.grid(False)

    show_xtick_labels = bool(common_style.get("show_xtick_labels", True))
    show_ytick_labels = bool(common_style.get("show_ytick_labels", True))
    ax.tick_params(axis="x", labelsize=common_style["tick_labelsize"], labelbottom=show_xtick_labels)
    ax.tick_params(axis="y", labelsize=common_style["tick_labelsize"], labelleft=show_ytick_labels)

    if common_style.get("show_legend", True):
        show_windows = bool(common_style.get("show_windows", True))
        show_event_vlines = bool(common_style.get("show_event_vlines", True))
        _apply_window_group_legends(
            ax,
            window_spans=window_spans if show_windows else (),
            group_handles=group_handles,
            event_vlines=event_vlines if show_event_vlines else (),
            event_vline_style=event_vline_style,
            legend_fontsize=legend_fontsize,
            framealpha=common_style["legend_framealpha"],
            loc=common_style["legend_loc"],
        )


def _apply_time_axis_ticks(
    ax: Any,
    *,
    common_style: Dict[str, Any],
    window_spans: Sequence[Dict[str, Any]],
    event_vlines: Sequence[Dict[str, Any]],
    event_order: Sequence[str],
    time_start_frame: float,
    time_end_frame: float,
    tick_labelsize: float,
    time_zero_frame: float = 0.0,
) -> None:
    ax.set_xlim(0.0, 1.0)

    show_windows = bool(common_style.get("show_windows", True))
    show_event_vlines = bool(common_style.get("show_event_vlines", True))

    if show_windows:
        ticks = _apply_window_definition_xticks(ax, window_spans)
    else:
        ticks = list(ax.get_xticks())
        if not ticks:
            ticks = [0.0, 1.0]

    ticks = _ensure_time_zero_xtick(
        ax,
        tick_positions=ticks,
        time_start_frame=time_start_frame,
        time_end_frame=time_end_frame,
        time_zero_frame=time_zero_frame,
    )

    blank_positions: List[float] = []
    if show_event_vlines:
        ticks, blank_positions = _apply_event_vline_xticks(
            ax,
            tick_positions=ticks,
            event_vlines=event_vlines,
            event_order=event_order,
            tick_labelsize=tick_labelsize,
        )

    _apply_frame_tick_labels(
        ax,
        time_start_frame=time_start_frame,
        time_end_frame=time_end_frame,
        time_zero_frame=time_zero_frame,
        blank_positions=blank_positions,
    )

    if common_style.get("show_xtick_labels", True):
        _auto_rotate_dense_xticklabels(
            ax,
            tick_positions=ticks,
            time_start_frame=time_start_frame,
            time_end_frame=time_end_frame,
        )


def _savefig_and_close(fig: Any, output_path: Path, common_style: Dict[str, Any], *, bbox: bool = True) -> None:
    import matplotlib.pyplot as plt

    fig.tight_layout(rect=common_style["tight_layout_rect"])
    save_kwargs: Dict[str, Any] = {"facecolor": common_style["savefig_facecolor"]}
    if bbox:
        save_kwargs["bbox_inches"] = common_style["savefig_bbox_inches"]
    fig.savefig(output_path, **save_kwargs)
    plt.close(fig)


def _maybe_export_plotly_html(task: Dict[str, Any], output_path: Path) -> None:
    if not bool(task.get("plotly_html", False)):
        return

    from ..plotly.html_export import export_task_html

    html_path = export_task_html(task, output_path=output_path)
    if html_path is not None:
        print(f"Wrote: {html_path}")


def _plot_task(task: Dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    kind = task["kind"]
    common_style = task["common_style"]
    output_path = Path(task["output_path"])
    event_vline_style = task.get("event_vline_style", {})
    event_vline_order = task.get("event_vline_order", [])

    if kind == "overlay":
        _plot_overlay_generic(
            signal_group=task["signal_group"],
            aggregated_by_key=task["aggregated_by_key"],
            markers_by_key=task.get("markers_by_key", {}),
            event_vlines_by_key=task.get("event_vlines_by_key", {}),
            event_vlines_by_key_by_channel=task.get("event_vlines_by_key_by_channel"),
            pooled_event_vlines=task.get("pooled_event_vlines", []),
            pooled_event_vlines_by_channel=task.get("pooled_event_vlines_by_channel"),
            event_vline_style=event_vline_style,
            event_vline_overlay_cfg=task.get("event_vline_overlay_cfg"),
            event_vline_order=event_vline_order,
            output_path=output_path,
            mode_name=task["mode_name"],
            group_fields=task["group_fields"],
            sorted_keys=[tuple(k) for k in task["sorted_keys"]],
            x=np.asarray(task["x"], dtype=float),
            channels=task.get("channels"),
            grid_layout=task.get("grid_layout"),
            cop_channels=task.get("cop_channels"),
            window_spans=task["window_spans"],
            window_spans_by_channel=task.get("window_spans_by_channel"),
            window_span_alpha=task.get("window_span_alpha"),
            style=task["style"],
            common_style=common_style,
            time_start_ms=task.get("time_start_ms"),
            time_end_ms=task.get("time_end_ms"),
            time_start_frame=task.get("time_start_frame"),
            time_end_frame=task.get("time_end_frame"),
            time_zero_frame=float(task.get("time_zero_frame", 0.0)),
            time_zero_frame_by_channel=task.get("time_zero_frame_by_channel"),
            filtered_group_fields=task["filtered_group_fields"],
            color_by_fields=task.get("color_by_fields"),
        )
        _maybe_export_plotly_html(task, output_path)
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
            event_vlines_by_channel=task.get("event_vlines_by_channel"),
            event_vline_style=event_vline_style,
            event_vline_order=event_vline_order,
            x=np.asarray(task["x"], dtype=float),
            channels=task["channels"],
            grid_layout=task["grid_layout"],
            window_spans=task["window_spans"],
            window_spans_by_channel=task.get("window_spans_by_channel"),
            window_span_alpha=task["window_span_alpha"],
            emg_style=task["emg_style"],
            common_style=common_style,
            time_start_ms=task["time_start_ms"],
            time_end_ms=task["time_end_ms"],
            time_start_frame=task.get("time_start_frame"),
            time_end_frame=task.get("time_end_frame"),
            time_zero_frame=float(task.get("time_zero_frame", 0.0)),
            time_zero_frame_by_channel=task.get("time_zero_frame_by_channel"),
        )
        _maybe_export_plotly_html(task, output_path)
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
            event_vline_order=event_vline_order,
            x=np.asarray(task["x"], dtype=float),
            channels=task["channels"],
            grid_layout=task["grid_layout"],
            window_spans=task["window_spans"],
            window_span_alpha=task["window_span_alpha"],
            forceplate_style=task["forceplate_style"],
            common_style=common_style,
            time_start_ms=task["time_start_ms"],
            time_end_ms=task["time_end_ms"],
            time_start_frame=task.get("time_start_frame"),
            time_end_frame=task.get("time_end_frame"),
            time_zero_frame=float(task.get("time_zero_frame", 0.0)),
        )
        _maybe_export_plotly_html(task, output_path)
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
            event_vline_order=event_vline_order,
            x_axis=np.asarray(task["x_axis"], dtype=float) if task["x_axis"] is not None else None,
            target_axis=np.asarray(task["target_axis"], dtype=float) if task["target_axis"] is not None else None,
            time_start_ms=task["time_start_ms"],
            time_end_ms=task["time_end_ms"],
            time_start_frame=task.get("time_start_frame"),
            time_end_frame=task.get("time_end_frame"),
            time_zero_frame=float(task.get("time_zero_frame", 0.0)),
            device_rate=float(task["device_rate"]),
            cop_channels=task["cop_channels"],
            grid_layout=task.get("grid_layout"),
            cop_style=task["cop_style"],
            common_style=common_style,
            window_spans=task["window_spans"],
        )
        _maybe_export_plotly_html(task, output_path)
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
            event_vline_order=event_vline_order,
            x_axis=np.asarray(task["x_axis"], dtype=float) if task["x_axis"] is not None else None,
            time_start_ms=task["time_start_ms"],
            time_end_ms=task["time_end_ms"],
            time_start_frame=task.get("time_start_frame"),
            time_end_frame=task.get("time_end_frame"),
            time_zero_frame=float(task.get("time_zero_frame", 0.0)),
            device_rate=float(task["device_rate"]),
            com_channels=task["com_channels"],
            grid_layout=task.get("grid_layout"),
            com_style=task["com_style"],
            common_style=common_style,
            window_spans=task["window_spans"],
        )
        _maybe_export_plotly_html(task, output_path)
        return

    plt.close("all")
    raise ValueError(f"Unknown plot task kind: {kind!r}")


def _plot_overlay_generic(
    *,
    signal_group: str,
    aggregated_by_key: Dict[Tuple, Dict[str, np.ndarray]],
    markers_by_key: Dict[Tuple, Dict[str, Any]],
    event_vlines_by_key: Dict[Tuple, List[Dict[str, Any]]],
    event_vlines_by_key_by_channel: Optional[Dict[Tuple, Dict[str, List[Dict[str, Any]]]]],
    pooled_event_vlines: Sequence[Dict[str, Any]],
    pooled_event_vlines_by_channel: Optional[Dict[str, List[Dict[str, Any]]]],
    event_vline_style: Dict[str, Any],
    event_vline_overlay_cfg: Optional[Dict[str, Any]],
    event_vline_order: Sequence[str],
    output_path: Path,
    mode_name: str,
    group_fields: List[str],
    sorted_keys: List[Tuple],
    x: np.ndarray,
    channels: Optional[List[str]],
    grid_layout: Optional[List[int]],
    cop_channels: Optional[Sequence[str]],
    window_spans: List[Dict[str, Any]],
    window_spans_by_channel: Optional[Dict[str, List[Dict[str, Any]]]],
    window_span_alpha: Optional[float],
    style: Dict[str, Any],
    common_style: Dict[str, Any],
    time_start_ms: Optional[float],
    time_end_ms: Optional[float],
    time_start_frame: Optional[float],
    time_end_frame: Optional[float],
    filtered_group_fields: List[str],
    time_zero_frame: float = 0.0,
    time_zero_frame_by_channel: Optional[Dict[str, float]] = None,
    color_by_fields: Optional[List[str]] = None,
) -> None:
    if signal_group == "cop":
        _plot_cop_overlay(
            aggregated_by_key=aggregated_by_key,
            event_vlines_by_key=event_vlines_by_key,
            pooled_event_vlines=pooled_event_vlines,
            event_vline_style=event_vline_style,
            event_vline_overlay_cfg=event_vline_overlay_cfg,
            event_vline_order=event_vline_order,
            output_path=output_path,
            mode_name=mode_name,
            group_fields=group_fields,
            sorted_keys=sorted_keys,
            x=x,
            window_spans=window_spans,
            cop_channels=cop_channels or (),
            grid_layout=grid_layout,
            cop_style=style,
            common_style=common_style,
            filtered_group_fields=filtered_group_fields,
            color_by_fields=color_by_fields,
            time_start_frame=time_start_frame,
            time_end_frame=time_end_frame,
            time_zero_frame=time_zero_frame,
        )
        return

    if signal_group == "com":
        _plot_com_overlay(
            aggregated_by_key=aggregated_by_key,
            event_vlines_by_key=event_vlines_by_key,
            pooled_event_vlines=pooled_event_vlines,
            event_vline_style=event_vline_style,
            event_vline_overlay_cfg=event_vline_overlay_cfg,
            event_vline_order=event_vline_order,
            output_path=output_path,
            mode_name=mode_name,
            group_fields=group_fields,
            sorted_keys=sorted_keys,
            x=x,
            window_spans=window_spans,
            com_channels=cop_channels or (),
            grid_layout=grid_layout,
            com_style=style,
            common_style=common_style,
            filtered_group_fields=filtered_group_fields,
            color_by_fields=color_by_fields,
            time_start_frame=time_start_frame,
            time_end_frame=time_end_frame,
            time_zero_frame=time_zero_frame,
        )
        return

    if channels is None or grid_layout is None or window_span_alpha is None or time_start_ms is None or time_end_ms is None:
        raise ValueError(f"Missing required overlay parameters for {signal_group=}")

    _plot_overlay_timeseries_grid(
        aggregated_by_key=aggregated_by_key,
        markers_by_key=markers_by_key,
        event_vlines_by_key=event_vlines_by_key,
        event_vlines_by_key_by_channel=event_vlines_by_key_by_channel,
        pooled_event_vlines=pooled_event_vlines,
        pooled_event_vlines_by_channel=pooled_event_vlines_by_channel,
        event_vline_style=event_vline_style,
        event_vline_overlay_cfg=event_vline_overlay_cfg,
        event_vline_order=event_vline_order,
        output_path=output_path,
        mode_name=mode_name,
        signal_group=signal_group,
        group_fields=group_fields,
        sorted_keys=sorted_keys,
        x=x,
        channels=channels,
        grid_layout=grid_layout,
        window_spans=window_spans,
        window_spans_by_channel=window_spans_by_channel,
        window_span_alpha=window_span_alpha,
        style=style,
        common_style=common_style,
        time_start_ms=time_start_ms,
        time_end_ms=time_end_ms,
        time_start_frame=time_start_frame,
        time_end_frame=time_end_frame,
        time_zero_frame=time_zero_frame,
        time_zero_frame_by_channel=time_zero_frame_by_channel,
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
    event_vlines_by_channel: Optional[Dict[str, List[Dict[str, Any]]]],
    event_vline_style: Dict[str, Any],
    event_vline_order: Sequence[str],
    x: np.ndarray,
    channels: List[str],
    grid_layout: List[int],
    window_spans: List[Dict[str, Any]],
    window_spans_by_channel: Optional[Dict[str, List[Dict[str, Any]]]],
    window_span_alpha: float,
    emg_style: Dict[str, Any],
    common_style: Dict[str, Any],
    time_start_ms: float,
    time_end_ms: float,
    time_start_frame: Optional[float],
    time_end_frame: Optional[float],
    time_zero_frame: float = 0.0,
    time_zero_frame_by_channel: Optional[Dict[str, float]] = None,
) -> None:
    import matplotlib.pyplot as plt

    rows, cols = grid_layout
    fig, axes = plt.subplots(rows, cols, figsize=emg_style["subplot_size"], dpi=common_style["dpi"])
    axes_flat = _flatten_axes(axes)

    for ax, ch in zip(axes_flat, channels):
        y = aggregated.get(ch)
        if y is None:
            ax.axis("off")
            continue

        ch_event_vlines = event_vlines_by_channel.get(ch) if event_vlines_by_channel else None
        ch_window_spans = window_spans_by_channel.get(ch) if window_spans_by_channel else None

        ax.plot(
            x,
            y,
            emg_style["line_color"],
            linewidth=emg_style["line_width"],
            alpha=emg_style["line_alpha"],
        )

        if common_style.get("show_windows", True):
            _draw_window_spans(ax, ch_window_spans or window_spans, alpha=window_span_alpha, with_labels=True)

        if common_style.get("show_event_vlines", True):
            _draw_event_vlines(ax, ch_event_vlines or event_vlines, style=event_vline_style)

        marker_info = markers.get(ch, {})
        if common_style.get("show_max_marker", True):
            max_time = marker_info.get("max")
            if max_time is not None and _is_within_time_axis(max_time, time_start_ms, time_end_ms):
                max_norm = _ms_to_norm(max_time, time_start_ms, time_end_ms)
                if max_norm is not None:
                    ax.axvline(max_norm, **emg_style["max_marker"], label="max")

        _style_timeseries_axis(
            ax,
            title=ch,
            common_style=common_style,
            legend_fontsize=emg_style["legend_fontsize"],
            window_spans=ch_window_spans or window_spans,
            event_vlines=ch_event_vlines or event_vlines,
            event_vline_style=event_vline_style,
        )
        if time_start_frame is not None and time_end_frame is not None:
            ch_time_zero_frame = (
                float(time_zero_frame_by_channel.get(ch, time_zero_frame))
                if time_zero_frame_by_channel is not None
                else float(time_zero_frame)
            )
            _apply_time_axis_ticks(
                ax,
                common_style=common_style,
                window_spans=ch_window_spans or window_spans,
                event_vlines=ch_event_vlines or event_vlines,
                event_order=event_vline_order,
                time_start_frame=time_start_frame,
                time_end_frame=time_end_frame,
                time_zero_frame=ch_time_zero_frame,
                tick_labelsize=float(common_style["tick_labelsize"]),
            )

    for ax in axes_flat[len(channels) :]:
        ax.axis("off")

    if common_style.get("show_suptitle", True):
        fig.suptitle(
            _format_title(signal_group="emg", mode_name=mode_name, group_fields=group_fields, key=key),
            fontsize=common_style["title_fontsize"],
            fontweight=common_style["title_fontweight"],
        )
    if common_style.get("show_xlabel", True):
        fig.supxlabel(emg_style["x_label"], fontsize=common_style["label_fontsize"])
    if common_style.get("show_ylabel", True):
        y_label = _format_label(emg_style.get("y_label", "Amplitude"), channel="Amplitude")
        fig.supylabel(y_label, fontsize=common_style["label_fontsize"])
    _savefig_and_close(fig, output_path, common_style, bbox=True)


def _plot_overlay_timeseries_grid(
    *,
    aggregated_by_key: Dict[Tuple, Dict[str, np.ndarray]],
    markers_by_key: Dict[Tuple, Dict[str, Any]],
    event_vlines_by_key: Dict[Tuple, List[Dict[str, Any]]],
    event_vlines_by_key_by_channel: Optional[Dict[Tuple, Dict[str, List[Dict[str, Any]]]]],
    pooled_event_vlines: Sequence[Dict[str, Any]],
    pooled_event_vlines_by_channel: Optional[Dict[str, List[Dict[str, Any]]]],
    event_vline_style: Dict[str, Any],
    event_vline_overlay_cfg: Optional[Dict[str, Any]],
    event_vline_order: Sequence[str],
    output_path: Path,
    mode_name: str,
    signal_group: str,
    group_fields: List[str],
    sorted_keys: List[Tuple],
    x: np.ndarray,
    channels: List[str],
    grid_layout: List[int],
    window_spans: List[Dict[str, Any]],
    window_spans_by_channel: Optional[Dict[str, List[Dict[str, Any]]]],
    window_span_alpha: float,
    style: Dict[str, Any],
    common_style: Dict[str, Any],
    time_start_ms: float,
    time_end_ms: float,
    time_start_frame: Optional[float] = None,
    time_end_frame: Optional[float] = None,
    filtered_group_fields: List[str],
    time_zero_frame: float = 0.0,
    time_zero_frame_by_channel: Optional[Dict[str, float]] = None,
    color_by_fields: Optional[List[str]] = None,
) -> None:
    import matplotlib.pyplot as plt

    rows, cols = grid_layout
    fig, axes = plt.subplots(rows, cols, figsize=style["subplot_size"], dpi=common_style["dpi"])
    overlay_event_names = sorted(_overlay_vline_event_names(event_vline_overlay_cfg))
    overlay_event_labels, overlay_event_colors = _infer_event_labels_and_colors(
        overlay_events=overlay_event_names,
        pooled_event_vlines=pooled_event_vlines,
        event_vlines_by_key=event_vlines_by_key,
    )

    if signal_group == "emg":
        axes_flat = _flatten_axes(axes)
        use_group_colors, key_to_color, key_to_linestyle = _prepare_overlay_group_styles(
            sorted_keys=sorted_keys,
            group_fields=group_fields,
            color_by_fields=color_by_fields,
            common_style=common_style,
        )
        overlay_legend_vlines = _build_overlay_event_vline_legend_vlines(
            overlay_events=overlay_event_names,
            sorted_keys=sorted_keys,
            group_fields=group_fields,
            filtered_group_fields=filtered_group_fields,
            key_to_linestyle=key_to_linestyle,
            event_labels=overlay_event_labels,
            event_colors=overlay_event_colors,
        )
        event_vlines_all = list(pooled_event_vlines) + overlay_legend_vlines
        legend_group_linewidth = min(float(style.get("line_width", 0.6)), 0.8)
        legend_group_handles = _build_group_legend_handles(
            sorted_keys,
            group_fields,
            filtered_group_fields,
            key_to_linestyle,
            linewidth=legend_group_linewidth,
        )

        for ax, ch in zip(axes_flat, channels):
            has_any_series = any(aggregated_by_key.get(key, {}).get(ch) is not None for key in sorted_keys)
            if not has_any_series:
                ax.axis("off")
                continue

            single_color = style.get("line_color", "blue")
            ch_window_spans = window_spans_by_channel.get(ch) if window_spans_by_channel else None
            if common_style.get("show_windows", True):
                _draw_window_spans(ax, ch_window_spans or window_spans, alpha=window_span_alpha, with_labels=True)
            _plot_overlay_channel_series(
                ax,
                x=x,
                channel=ch,
                sorted_keys=sorted_keys,
                aggregated_by_key=aggregated_by_key,
                group_fields=group_fields,
                filtered_group_fields=filtered_group_fields,
                line_width=style["line_width"],
                line_alpha=style["line_alpha"],
                channel_color=single_color,
                use_group_colors=use_group_colors,
                key_to_color=key_to_color,
                key_to_linestyle=key_to_linestyle,
            )

            pooled_vlines = pooled_event_vlines_by_channel.get(ch, []) if pooled_event_vlines_by_channel else pooled_event_vlines
            if pooled_vlines and common_style.get("show_event_vlines", True):
                _draw_event_vlines(ax, pooled_vlines, style=event_vline_style)

            event_vlines_by_key_for_tick = event_vlines_by_key
            if event_vlines_by_key_by_channel:
                event_vlines_by_key_for_tick = {
                    key: event_vlines_by_key_by_channel.get(key, {}).get(ch, []) for key in sorted_keys
                }

            if common_style.get("show_event_vlines", True):
                _draw_event_vlines_for_keys(
                    ax,
                    sorted_keys=sorted_keys,
                    event_vlines_by_key=event_vlines_by_key_for_tick,
                    style=event_vline_style,
                    overlay_cfg=event_vline_overlay_cfg,
                    key_to_linestyle=key_to_linestyle,
                )

            for key in sorted_keys:
                marker_info = markers_by_key.get(key, {}).get(ch, {})
                marker_label = _format_group_label(key, group_fields)
                if common_style.get("show_max_marker", True):
                    max_time = marker_info.get("max")
                    if max_time is not None and _is_within_time_axis(max_time, time_start_ms, time_end_ms):
                        max_norm = _ms_to_norm(max_time, time_start_ms, time_end_ms)
                        if max_norm is not None:
                            ax.axvline(max_norm, **style["max_marker"], label=f"{marker_label} max")

            _style_timeseries_axis(
                ax,
                title=ch,
                common_style=common_style,
                legend_fontsize=style["legend_fontsize"],
                window_spans=ch_window_spans or window_spans,
                group_handles=legend_group_handles,
                event_vlines=event_vlines_all,
                event_vline_style=event_vline_style,
            )
            if time_start_frame is not None and time_end_frame is not None:
                ch_time_zero_frame = (
                    float(time_zero_frame_by_channel.get(ch, time_zero_frame))
                    if time_zero_frame_by_channel is not None
                    else float(time_zero_frame)
                )
                tick_vlines = list(pooled_vlines) + _collect_overlay_event_vlines_for_ticks(
                    sorted_keys=sorted_keys,
                    event_vlines_by_key=event_vlines_by_key_for_tick,
                    overlay_cfg=event_vline_overlay_cfg,
                )
                _apply_time_axis_ticks(
                    ax,
                    common_style=common_style,
                    window_spans=ch_window_spans or window_spans,
                    event_vlines=tick_vlines,
                    event_order=event_vline_order,
                    time_start_frame=time_start_frame,
                    time_end_frame=time_end_frame,
                    time_zero_frame=ch_time_zero_frame,
                    tick_labelsize=float(common_style["tick_labelsize"]),
                )

        for ax in axes_flat[len(channels) :]:
            ax.axis("off")

        overlay_by = ", ".join(group_fields) if group_fields else "all"
        if common_style.get("show_suptitle", True):
            fig.suptitle(
                f"{mode_name} | emg | overlay by {overlay_by}",
                fontsize=common_style["title_fontsize"],
                fontweight=common_style["title_fontweight"],
            )
        if common_style.get("show_xlabel", True):
            fig.supxlabel(style["x_label"], fontsize=common_style["label_fontsize"])
        if common_style.get("show_ylabel", True):
            y_label = _format_label(style.get("y_label", "Amplitude"), channel="Amplitude")
            fig.supylabel(y_label, fontsize=common_style["label_fontsize"])
        _savefig_and_close(fig, output_path, common_style, bbox=True)
        return

    if signal_group == "forceplate":
        use_group_colors, key_to_color, key_to_linestyle = _prepare_overlay_group_styles(
            sorted_keys=sorted_keys,
            group_fields=group_fields,
            color_by_fields=color_by_fields,
            common_style=common_style,
        )
        overlay_legend_vlines = _build_overlay_event_vline_legend_vlines(
            overlay_events=overlay_event_names,
            sorted_keys=sorted_keys,
            group_fields=group_fields,
            filtered_group_fields=filtered_group_fields,
            key_to_linestyle=key_to_linestyle,
            event_labels=overlay_event_labels,
            event_colors=overlay_event_colors,
        )
        event_vlines_all = list(pooled_event_vlines) + overlay_legend_vlines

        for ax, ch in zip(np.ravel(axes), channels):
            if common_style.get("show_windows", True):
                _draw_window_spans(ax, window_spans, alpha=window_span_alpha, with_labels=True)

            channel_color = _resolve_forceplate_line_color(ch, style.get("line_colors", {}) or {})
            _plot_overlay_channel_series(
                ax,
                x=x,
                channel=ch,
                sorted_keys=sorted_keys,
                aggregated_by_key=aggregated_by_key,
                group_fields=group_fields,
                filtered_group_fields=filtered_group_fields,
                line_width=style["line_width"],
                line_alpha=style["line_alpha"],
                channel_color=channel_color,
                use_group_colors=use_group_colors,
                key_to_color=key_to_color,
                key_to_linestyle=key_to_linestyle,
            )

            if pooled_event_vlines and common_style.get("show_event_vlines", True):
                _draw_event_vlines(ax, pooled_event_vlines, style=event_vline_style)

            if common_style.get("show_event_vlines", True):
                _draw_event_vlines_for_keys(
                    ax,
                    sorted_keys=sorted_keys,
                    event_vlines_by_key=event_vlines_by_key,
                    style=event_vline_style,
                    overlay_cfg=event_vline_overlay_cfg,
                    key_to_linestyle=key_to_linestyle,
                )

            _style_timeseries_axis(
                ax,
                title=ch,
                common_style=common_style,
                legend_fontsize=style["legend_fontsize"],
                window_spans=window_spans,
                event_vlines=event_vlines_all,
                event_vline_style=event_vline_style,
            )
            if time_start_frame is not None and time_end_frame is not None:
                tick_vlines = list(pooled_event_vlines) + _collect_overlay_event_vlines_for_ticks(
                    sorted_keys=sorted_keys,
                    event_vlines_by_key=event_vlines_by_key,
                    overlay_cfg=event_vline_overlay_cfg,
                )
                _apply_time_axis_ticks(
                    ax,
                    common_style=common_style,
                    window_spans=window_spans,
                    event_vlines=tick_vlines,
                    event_order=event_vline_order,
                    time_start_frame=time_start_frame,
                    time_end_frame=time_end_frame,
                    time_zero_frame=time_zero_frame,
                    tick_labelsize=float(common_style["tick_labelsize"]),
                )
            if common_style.get("show_xlabel", True):
                ax.set_xlabel(style["x_label"], fontsize=common_style["label_fontsize"])
            axis_label = _resolve_forceplate_axis_label(ch, style.get("axis_labels", {}))
            y_label = _format_label(style.get("y_label", "{channel} Value"), channel=ch, axis_label=axis_label)
            if common_style.get("show_ylabel", True):
                ax.set_ylabel(y_label, fontsize=common_style["label_fontsize"])

        overlay_by = ", ".join(group_fields) if group_fields else "all"
        if common_style.get("show_suptitle", True):
            fig.suptitle(
                f"{mode_name} | forceplate | overlay by {overlay_by}",
                fontsize=common_style["title_fontsize"],
                fontweight=common_style["title_fontweight"],
            )
        _savefig_and_close(fig, output_path, common_style, bbox=True)
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
    event_vline_order: Sequence[str],
    x: np.ndarray,
    channels: List[str],
    grid_layout: List[int],
    window_spans: List[Dict[str, Any]],
    window_span_alpha: float,
    forceplate_style: Dict[str, Any],
    common_style: Dict[str, Any],
    time_start_ms: float,
    time_end_ms: float,
    time_start_frame: Optional[float],
    time_end_frame: Optional[float],
    time_zero_frame: float = 0.0,
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

        if common_style.get("show_windows", True):
            _draw_window_spans(ax, window_spans, alpha=window_span_alpha, with_labels=True)

        if common_style.get("show_event_vlines", True):
            _draw_event_vlines(ax, event_vlines, style=event_vline_style)

        _style_timeseries_axis(
            ax,
            title=ch,
            common_style=common_style,
            legend_fontsize=forceplate_style["legend_fontsize"],
            window_spans=window_spans,
            event_vlines=event_vlines,
            event_vline_style=event_vline_style,
        )
        if time_start_frame is not None and time_end_frame is not None:
            _apply_time_axis_ticks(
                ax,
                common_style=common_style,
                window_spans=window_spans,
                event_vlines=event_vlines,
                event_order=event_vline_order,
                time_start_frame=time_start_frame,
                time_end_frame=time_end_frame,
                time_zero_frame=time_zero_frame,
                tick_labelsize=float(common_style["tick_labelsize"]),
            )
        if common_style.get("show_xlabel", True):
            ax.set_xlabel(forceplate_style["x_label"], fontsize=common_style["label_fontsize"])
        axis_label = _resolve_forceplate_axis_label(ch, forceplate_style.get("axis_labels", {}))
        y_label = _format_label(
            forceplate_style.get("y_label", "{channel} Value"),
            channel=ch,
            axis_label=axis_label,
        )
        if common_style.get("show_ylabel", True):
            ax.set_ylabel(y_label, fontsize=common_style["label_fontsize"])

    if common_style.get("show_suptitle", True):
        fig.suptitle(
            _format_title(signal_group="forceplate", mode_name=mode_name, group_fields=group_fields, key=key),
            fontsize=common_style["title_fontsize"],
            fontweight=common_style["title_fontweight"],
        )
    _savefig_and_close(fig, output_path, common_style, bbox=True)


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
    event_vline_order: Sequence[str],
    x_axis: Optional[np.ndarray],
    target_axis: Optional[np.ndarray],
    time_start_ms: float,
    time_end_ms: float,
    time_start_frame: Optional[float],
    time_end_frame: Optional[float],
    time_zero_frame: float = 0.0,
    device_rate: float,
    cop_channels: Sequence[str],
    grid_layout: Optional[Sequence[int]],
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

    rows, cols = 1, 3
    if grid_layout and len(grid_layout) == 2:
        try:
            rows = max(1, int(grid_layout[0]))
            cols = max(1, int(grid_layout[1]))
        except (TypeError, ValueError):
            rows, cols = 1, 3
    n_panels = rows * cols
    if n_panels < 3:
        raise ValueError("COP grid_layout must have at least 3 panels (Cx, Cy, scatter).")
    fig_size = cop_style["subplot_size"]
    try:
        fig_w, fig_h = fig_size
        fig_size = (float(fig_w) * (n_panels / 3.0), float(fig_h))
    except (TypeError, ValueError):
        pass

    fig, axes = plt.subplots(rows, cols, figsize=fig_size, dpi=common_style["dpi"])
    axes = np.asarray(axes).ravel()
    ax_cx = axes[0]
    ax_cy = axes[1]
    ax_scatter = axes[2]
    for ax in axes[3:]:
        ax.axis("off")

    window_span_alpha = float(cop_style.get("window_span_alpha", 0.15))

    for ax in (ax_cx, ax_cy):
        if common_style.get("show_windows", True):
            _draw_window_spans(ax, window_spans, alpha=window_span_alpha, with_labels=False)

    cx_color = cop_style.get("line_colors", {}).get("Cx", "gray")
    cy_color = cop_style.get("line_colors", {}).get("Cy", "gray")

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

    if common_style.get("show_event_vlines", True):
        _draw_event_vlines(ax_cx, event_vlines, style=event_vline_style)
        _draw_event_vlines(ax_cy, event_vlines, style=event_vline_style)

    ax_scatter.scatter(
        ml_vals,
        ap_vals,
        color=cop_style["background_color"],
        alpha=cop_style["background_alpha"],
        s=cop_style["background_size"],
    )

    if x_axis is not None and common_style.get("show_windows", True):
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

    _style_timeseries_axis(
        ax_cx,
        title=cx_name,
        common_style=common_style,
        legend_fontsize=cop_style["legend_fontsize"],
        window_spans=window_spans,
        event_vlines=event_vlines,
        event_vline_style=event_vline_style,
    )
    _style_timeseries_axis(
        ax_cy,
        title=cy_name,
        common_style=common_style,
        legend_fontsize=cop_style["legend_fontsize"],
        window_spans=window_spans,
        event_vlines=event_vlines,
        event_vline_style=event_vline_style,
    )

    if common_style.get("show_xlabel", True):
        ax_cx.set_xlabel(cop_style.get("x_label_time", "Normalized time (0-1)"), fontsize=common_style["label_fontsize"])
        ax_cy.set_xlabel(cop_style.get("x_label_time", "Normalized time (0-1)"), fontsize=common_style["label_fontsize"])
    if common_style.get("show_ylabel", True):
        ax_cx.set_ylabel(cop_style.get("y_label_cx", "Cx"), fontsize=common_style["label_fontsize"])
        ax_cy.set_ylabel(cop_style.get("y_label_cy", "Cy"), fontsize=common_style["label_fontsize"])
    if time_start_frame is not None and time_end_frame is not None:
        for ax in (ax_cx, ax_cy):
            _apply_time_axis_ticks(
                ax,
                common_style=common_style,
                window_spans=window_spans,
                event_vlines=event_vlines,
                event_order=event_vline_order,
                time_start_frame=time_start_frame,
                time_end_frame=time_end_frame,
                time_zero_frame=time_zero_frame,
                tick_labelsize=float(common_style["tick_labelsize"]),
            )

    if common_style.get("show_xlabel", True):
        ax_scatter.set_xlabel(cop_style["x_label"], fontsize=common_style["label_fontsize"])
    if common_style.get("show_ylabel", True):
        ax_scatter.set_ylabel(cop_style["y_label"], fontsize=common_style["label_fontsize"])
    ax_scatter.set_aspect("equal", adjustable="datalim")
    _style_timeseries_axis(
        ax_scatter,
        title="Cxy",
        common_style=common_style,
        legend_fontsize=cop_style["legend_fontsize"],
        window_spans=window_spans,
        event_vlines=[],
        event_vline_style=event_vline_style,
    )

    if common_style.get("show_suptitle", True):
        fig.suptitle(
            _format_title(signal_group="cop", mode_name=mode_name, group_fields=group_fields, key=key),
            fontsize=common_style["title_fontsize"],
            fontweight=common_style["title_fontweight"],
        )
    _savefig_and_close(fig, output_path, common_style, bbox=False)


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
    event_vline_order: Sequence[str],
    x_axis: Optional[np.ndarray],
    time_start_ms: float,
    time_end_ms: float,
    time_start_frame: Optional[float],
    time_end_frame: Optional[float],
    time_zero_frame: float = 0.0,
    device_rate: float,
    com_channels: Sequence[str],
    grid_layout: Optional[Sequence[int]],
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

    rows, cols = 1, 4
    if grid_layout and len(grid_layout) == 2:
        try:
            rows = max(1, int(grid_layout[0]))
            cols = max(1, int(grid_layout[1]))
        except (TypeError, ValueError):
            rows, cols = 1, 4

    slots = rows * cols
    if comz_name is not None and slots < 3:
        comz_name = None
        comz = None

    time_panels = 3 if comz_name is not None else 2
    total_panels = time_panels + 1  # + scatter
    if slots < time_panels:
        raise ValueError("COM grid_layout must have enough panels for COMx/COMy/(COMz).")

    fig_size = com_style["subplot_size"]
    try:
        fig_w, fig_h = fig_size
        fig_size = (float(fig_w) * (total_panels / 3.0), float(fig_h))
    except (TypeError, ValueError):
        pass

    include_scatter_in_grid = slots >= total_panels
    if include_scatter_in_grid:
        fig, axes = plt.subplots(rows, cols, figsize=fig_size, dpi=common_style["dpi"])
        axes = np.asarray(axes).ravel()
    elif rows == 1:
        fig, axes = plt.subplots(1, total_panels, figsize=fig_size, dpi=common_style["dpi"])
        axes = np.asarray(axes).ravel()
    else:
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=fig_size, dpi=common_style["dpi"])
        gs = GridSpec(rows, cols + 1, figure=fig)
        axes_time = [fig.add_subplot(gs[r, c]) for r in range(rows) for c in range(cols)]
        ax_scatter = fig.add_subplot(gs[:, cols])
        axes = np.asarray(axes_time + [ax_scatter], dtype=object)

    ax_x = axes[0]
    ax_y = axes[1]
    if comz_name is not None:
        ax_z = axes[2]
    else:
        ax_z = None
    ax_scatter = axes[time_panels]

    for ax in axes[total_panels:]:
        ax.axis("off")

    window_span_alpha = float(com_style.get("window_span_alpha", 0.15))
    time_axes = [ax_x, ax_y] + ([ax_z] if ax_z is not None else [])
    if common_style.get("show_windows", True):
        for ax in time_axes:
            _draw_window_spans(ax, window_spans, alpha=window_span_alpha, with_labels=False)

    x_color = com_style.get("line_colors", {}).get(comx_name, "gray")
    y_color = com_style.get("line_colors", {}).get(comy_name, "gray")
    z_color = com_style.get("line_colors", {}).get(comz_name, "gray") if comz_name is not None else None

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

    if common_style.get("show_event_vlines", True):
        for ax in time_axes:
            _draw_event_vlines(ax, event_vlines, style=event_vline_style)

    ax_scatter.scatter(
        ml_vals,
        ap_vals,
        color=com_style["background_color"],
        alpha=com_style["background_alpha"],
        s=com_style["background_size"],
    )

    if x_axis is not None and common_style.get("show_windows", True):
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

    if common_style.get("show_subplot_titles", True):
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
        ax_scatter.set_title(
            "COMxy",
            fontsize=common_style["title_fontsize"],
            fontweight=common_style["title_fontweight"],
            pad=common_style["title_pad"],
        )

    axes_to_style = [ax_x, ax_y] + ([ax_z] if ax_z is not None else [])
    for ax in axes_to_style:
        if common_style.get("show_grid", True):
            ax.grid(True, alpha=common_style["grid_alpha"])
        else:
            ax.grid(False)
        ax.tick_params(
            axis="x",
            labelsize=common_style["tick_labelsize"],
            labelbottom=bool(common_style.get("show_xtick_labels", True)),
        )
        ax.tick_params(
            axis="y",
            labelsize=common_style["tick_labelsize"],
            labelleft=bool(common_style.get("show_ytick_labels", True)),
        )
        if common_style.get("show_legend", True):
            _apply_window_group_legends(
                ax,
                window_spans=window_spans if common_style.get("show_windows", True) else (),
                group_handles=[],
                event_vlines=event_vlines if common_style.get("show_event_vlines", True) else (),
                event_vline_style=event_vline_style,
                legend_fontsize=com_style["legend_fontsize"],
                framealpha=common_style["legend_framealpha"],
                loc=common_style["legend_loc"],
            )

    if common_style.get("show_grid", True):
        ax_scatter.grid(True, alpha=common_style["grid_alpha"])
    else:
        ax_scatter.grid(False)
    ax_scatter.tick_params(
        axis="x",
        labelsize=common_style["tick_labelsize"],
        labelbottom=bool(common_style.get("show_xtick_labels", True)),
    )
    ax_scatter.tick_params(
        axis="y",
        labelsize=common_style["tick_labelsize"],
        labelleft=bool(common_style.get("show_ytick_labels", True)),
    )
    if common_style.get("show_legend", True):
        _apply_window_group_legends(
            ax_scatter,
            window_spans=window_spans if common_style.get("show_windows", True) else (),
            group_handles=[],
            legend_fontsize=com_style["legend_fontsize"],
            framealpha=common_style["legend_framealpha"],
            loc=common_style["legend_loc"],
        )

    if common_style.get("show_xlabel", True):
        ax_x.set_xlabel(com_style.get("x_label_time", "Normalized time (0-1)"), fontsize=common_style["label_fontsize"])
        ax_y.set_xlabel(com_style.get("x_label_time", "Normalized time (0-1)"), fontsize=common_style["label_fontsize"])
        if ax_z is not None:
            ax_z.set_xlabel(
                com_style.get("x_label_time", "Normalized time (0-1)"),
                fontsize=common_style["label_fontsize"],
            )
    if common_style.get("show_ylabel", True):
        ax_x.set_ylabel(com_style.get("y_label_comx", comx_name), fontsize=common_style["label_fontsize"])
        ax_y.set_ylabel(com_style.get("y_label_comy", comy_name), fontsize=common_style["label_fontsize"])
        if ax_z is not None:
            ax_z.set_ylabel(com_style.get("y_label_comz", comz_name), fontsize=common_style["label_fontsize"])
    if time_start_frame is not None and time_end_frame is not None:
        for ax in axes_to_style:
            _apply_time_axis_ticks(
                ax,
                common_style=common_style,
                window_spans=window_spans,
                event_vlines=event_vlines,
                event_order=event_vline_order,
                time_start_frame=time_start_frame,
                time_end_frame=time_end_frame,
                time_zero_frame=time_zero_frame,
                tick_labelsize=float(common_style["tick_labelsize"]),
            )

    if common_style.get("show_xlabel", True):
        ax_scatter.set_xlabel(com_style.get("x_label", comx_name), fontsize=common_style["label_fontsize"])
    if common_style.get("show_ylabel", True):
        ax_scatter.set_ylabel(com_style.get("y_label", comy_name), fontsize=common_style["label_fontsize"])
    ax_scatter.set_aspect("equal", adjustable="datalim")

    # window legend handled via _apply_window_group_legends(ax_scatter, ...)

    if common_style.get("show_suptitle", True):
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
    pooled_event_vlines: Sequence[Dict[str, Any]],
    event_vline_style: Dict[str, Any],
    event_vline_overlay_cfg: Optional[Dict[str, Any]],
    event_vline_order: Sequence[str],
    output_path: Path,
    mode_name: str,
    group_fields: List[str],
    sorted_keys: List[Tuple],
    x: np.ndarray,
    window_spans: List[Dict[str, Any]],
    cop_channels: Sequence[str],
    grid_layout: Optional[Sequence[int]],
    cop_style: Dict[str, Any],
    common_style: Dict[str, Any],
    filtered_group_fields: List[str],
    color_by_fields: Optional[List[str]] = None,
    time_start_frame: Optional[float] = None,
    time_end_frame: Optional[float] = None,
    time_zero_frame: float = 0.0,
) -> None:
    import matplotlib.pyplot as plt

    cx_name, cy_name = _resolve_cop_channel_names(cop_channels)
    overlay_event_names = sorted(_overlay_vline_event_names(event_vline_overlay_cfg))

    rows, cols = 1, 3
    if grid_layout and len(grid_layout) == 2:
        try:
            rows = max(1, int(grid_layout[0]))
            cols = max(1, int(grid_layout[1]))
        except (TypeError, ValueError):
            rows, cols = 1, 3
    n_panels = rows * cols
    if n_panels < 3:
        raise ValueError("COP grid_layout must have at least 3 panels (Cx, Cy, scatter).")
    fig_size = cop_style["subplot_size"]
    try:
        fig_w, fig_h = fig_size
        fig_size = (float(fig_w) * (n_panels / 3.0), float(fig_h))
    except (TypeError, ValueError):
        pass

    fig, axes = plt.subplots(rows, cols, figsize=fig_size, dpi=common_style["dpi"])
    axes = np.asarray(axes).ravel()
    ax_cx = axes[0]
    ax_cy = axes[1]
    ax_scatter = axes[2]
    for ax in axes[3:]:
        ax.axis("off")

    window_span_alpha = float(cop_style.get("window_span_alpha", 0.15))
    for ax in (ax_cx, ax_cy):
        if common_style.get("show_windows", True):
            _draw_window_spans(ax, window_spans, alpha=window_span_alpha, with_labels=False)

    use_group_colors, key_to_color, key_to_linestyle = _prepare_overlay_group_styles(
        sorted_keys=sorted_keys,
        group_fields=group_fields,
        color_by_fields=color_by_fields,
        common_style=common_style,
    )
    overlay_event_labels, overlay_event_colors = _infer_event_labels_and_colors(
        overlay_events=overlay_event_names,
        pooled_event_vlines=pooled_event_vlines,
        event_vlines_by_key=event_vlines_by_key,
    )
    overlay_legend_vlines = _build_overlay_event_vline_legend_vlines(
        overlay_events=overlay_event_names,
        sorted_keys=sorted_keys,
        group_fields=group_fields,
        filtered_group_fields=filtered_group_fields,
        key_to_linestyle=key_to_linestyle,
        event_labels=overlay_event_labels,
        event_colors=overlay_event_colors,
    )
    event_vlines_all = list(pooled_event_vlines) + overlay_legend_vlines
    legend_group_linewidth = min(float(cop_style.get("line_width", 0.8)), 0.8)
    legend_group_handles = _build_group_legend_handles(
        sorted_keys,
        group_fields,
        filtered_group_fields,
        key_to_linestyle,
        linewidth=legend_group_linewidth,
    )

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
            channel_color = cop_style.get("line_colors", {}).get("Cx" if ch == cx_name else "Cy", "gray")
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

        if pooled_event_vlines and common_style.get("show_event_vlines", True):
            _draw_event_vlines(ax, pooled_event_vlines, style=event_vline_style)

        if common_style.get("show_event_vlines", True):
            _draw_event_vlines_for_keys(
                ax,
                sorted_keys=sorted_keys,
                event_vlines_by_key=event_vlines_by_key,
                style=event_vline_style,
                overlay_cfg=event_vline_overlay_cfg,
                key_to_linestyle=key_to_linestyle,
            )
        _style_timeseries_axis(
            ax,
            title=ch,
            common_style=common_style,
            legend_fontsize=cop_style["legend_fontsize"],
            window_spans=window_spans,
            group_handles=legend_group_handles,
            event_vlines=event_vlines_all,
            event_vline_style=event_vline_style,
        )
        if time_start_frame is not None and time_end_frame is not None:
            tick_vlines = list(pooled_event_vlines) + _collect_overlay_event_vlines_for_ticks(
                sorted_keys=sorted_keys,
                event_vlines_by_key=event_vlines_by_key,
                overlay_cfg=event_vline_overlay_cfg,
            )
            _apply_time_axis_ticks(
                ax,
                common_style=common_style,
                window_spans=window_spans,
                event_vlines=tick_vlines,
                event_order=event_vline_order,
                time_start_frame=time_start_frame,
                time_end_frame=time_end_frame,
                time_zero_frame=time_zero_frame,
                tick_labelsize=float(common_style["tick_labelsize"]),
            )
        if common_style.get("show_xlabel", True):
            ax.set_xlabel(cop_style.get("x_label_time", "Normalized time (0-1)"), fontsize=common_style["label_fontsize"])
        if common_style.get("show_ylabel", True):
            ax.set_ylabel(y_label, fontsize=common_style["label_fontsize"])

    # Overlay line segments: window color, group line style
    overlay_linewidth = float(cop_style.get("line_width", 0.8))
    overlay_alpha = float(cop_style.get("scatter_alpha", 0.7))
    if common_style.get("show_windows", True):
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

    if common_style.get("show_grid", True):
        ax_scatter.grid(True, alpha=common_style["grid_alpha"])
    else:
        ax_scatter.grid(False)
    ax_scatter.tick_params(
        axis="x",
        labelsize=common_style["tick_labelsize"],
        labelbottom=bool(common_style.get("show_xtick_labels", True)),
    )
    ax_scatter.tick_params(
        axis="y",
        labelsize=common_style["tick_labelsize"],
        labelleft=bool(common_style.get("show_ytick_labels", True)),
    )
    if common_style.get("show_legend", True):
        _apply_window_group_legends(
            ax_scatter,
            window_spans=window_spans if common_style.get("show_windows", True) else (),
            group_handles=legend_group_handles,
            legend_fontsize=cop_style["legend_fontsize"],
            framealpha=common_style["legend_framealpha"],
            loc=common_style["legend_loc"],
        )
    if common_style.get("show_subplot_titles", True):
        ax_scatter.set_title(
            "Cxy",
            fontsize=common_style["title_fontsize"],
            fontweight=common_style["title_fontweight"],
            pad=common_style["title_pad"],
        )
    if common_style.get("show_xlabel", True):
        ax_scatter.set_xlabel(cop_style["x_label"], fontsize=common_style["label_fontsize"])
    if common_style.get("show_ylabel", True):
        ax_scatter.set_ylabel(cop_style["y_label"], fontsize=common_style["label_fontsize"])
    ax_scatter.set_aspect("equal", adjustable="datalim")

    overlay_by = ", ".join(group_fields) if group_fields else "all"
    if common_style.get("show_suptitle", True):
        fig.suptitle(
            f"{mode_name} | cop | overlay by {overlay_by}",
            fontsize=common_style["title_fontsize"],
            fontweight=common_style["title_fontweight"],
        )
    _savefig_and_close(fig, output_path, common_style, bbox=False)


def _plot_com_overlay(
    *,
    aggregated_by_key: Dict[Tuple, Dict[str, np.ndarray]],
    event_vlines_by_key: Dict[Tuple, List[Dict[str, Any]]],
    pooled_event_vlines: Sequence[Dict[str, Any]],
    event_vline_style: Dict[str, Any],
    event_vline_overlay_cfg: Optional[Dict[str, Any]],
    event_vline_order: Sequence[str],
    output_path: Path,
    mode_name: str,
    group_fields: List[str],
    sorted_keys: List[Tuple],
    x: np.ndarray,
    window_spans: List[Dict[str, Any]],
    com_channels: Sequence[str],
    grid_layout: Optional[Sequence[int]],
    com_style: Dict[str, Any],
    common_style: Dict[str, Any],
    filtered_group_fields: List[str],
    color_by_fields: Optional[List[str]] = None,
    time_start_frame: Optional[float] = None,
    time_end_frame: Optional[float] = None,
    time_zero_frame: float = 0.0,
) -> None:
    import matplotlib.pyplot as plt

    comx_name, comy_name, comz_name = _resolve_com_channel_names(com_channels)
    overlay_event_names = sorted(_overlay_vline_event_names(event_vline_overlay_cfg))
    if comz_name is not None and not any(
        aggregated_by_key.get(key, {}).get(comz_name) is not None for key in sorted_keys
    ):
        comz_name = None

    rows, cols = 1, 4
    if grid_layout and len(grid_layout) == 2:
        try:
            rows = max(1, int(grid_layout[0]))
            cols = max(1, int(grid_layout[1]))
        except (TypeError, ValueError):
            rows, cols = 1, 4

    slots = rows * cols
    if comz_name is not None and slots < 3:
        comz_name = None

    time_panels = 3 if comz_name is not None else 2
    total_panels = time_panels + 1  # + scatter
    if slots < time_panels:
        raise ValueError("COM grid_layout must have enough panels for COMx/COMy/(COMz).")

    fig_size = com_style["subplot_size"]
    try:
        fig_w, fig_h = fig_size
        fig_size = (float(fig_w) * (total_panels / 3.0), float(fig_h))
    except (TypeError, ValueError):
        pass

    include_scatter_in_grid = slots >= total_panels
    if include_scatter_in_grid:
        fig, axes = plt.subplots(rows, cols, figsize=fig_size, dpi=common_style["dpi"])
        axes = np.asarray(axes).ravel()
    elif rows == 1:
        fig, axes = plt.subplots(1, total_panels, figsize=fig_size, dpi=common_style["dpi"])
        axes = np.asarray(axes).ravel()
    else:
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=fig_size, dpi=common_style["dpi"])
        gs = GridSpec(rows, cols + 1, figure=fig)
        axes_time = [fig.add_subplot(gs[r, c]) for r in range(rows) for c in range(cols)]
        ax_scatter = fig.add_subplot(gs[:, cols])
        axes = np.asarray(axes_time + [ax_scatter], dtype=object)

    ax_x = axes[0]
    ax_y = axes[1]
    if comz_name is not None:
        ax_z = axes[2]
    else:
        ax_z = None
    ax_scatter = axes[time_panels]

    for ax in axes[total_panels:]:
        ax.axis("off")

    window_span_alpha = float(com_style.get("window_span_alpha", 0.15))
    time_axes = [ax_x, ax_y] + ([ax_z] if ax_z is not None else [])
    if common_style.get("show_windows", True):
        for ax in time_axes:
            _draw_window_spans(ax, window_spans, alpha=window_span_alpha, with_labels=False)

    import matplotlib as mpl

    base_colors = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    use_group_colors = common_style.get("use_group_colors", False)
    key_to_linestyle = _build_group_linestyles(sorted_keys, common_style.get("group_linestyles", ("-", "--", ":", "-.")))
    key_to_color = _build_group_color_map(sorted_keys, group_fields, color_by_fields, base_colors) if use_group_colors else {}
    overlay_event_labels, overlay_event_colors = _infer_event_labels_and_colors(
        overlay_events=overlay_event_names,
        pooled_event_vlines=pooled_event_vlines,
        event_vlines_by_key=event_vlines_by_key,
    )
    overlay_legend_vlines = _build_overlay_event_vline_legend_vlines(
        overlay_events=overlay_event_names,
        sorted_keys=sorted_keys,
        group_fields=group_fields,
        filtered_group_fields=filtered_group_fields,
        key_to_linestyle=key_to_linestyle,
        event_labels=overlay_event_labels,
        event_colors=overlay_event_colors,
    )
    event_vlines_all = list(pooled_event_vlines) + overlay_legend_vlines
    legend_group_linewidth = min(float(com_style.get("line_width", 0.8)), 0.8)
    legend_group_handles = _build_group_legend_handles(
        sorted_keys,
        group_fields,
        filtered_group_fields,
        key_to_linestyle,
        linewidth=legend_group_linewidth,
    )

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
            channel_color = com_style.get("line_colors", {}).get(ch, "gray")
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

        if pooled_event_vlines and common_style.get("show_event_vlines", True):
            _draw_event_vlines(ax, pooled_event_vlines, style=event_vline_style)
        if common_style.get("show_event_vlines", True):
            _draw_event_vlines_for_keys(
                ax,
                sorted_keys=sorted_keys,
                event_vlines_by_key=event_vlines_by_key,
                style=event_vline_style,
                overlay_cfg=event_vline_overlay_cfg,
                key_to_linestyle=key_to_linestyle,
            )

        if common_style.get("show_grid", True):
            ax.grid(True, alpha=common_style["grid_alpha"])
        else:
            ax.grid(False)
        ax.tick_params(
            axis="x",
            labelsize=common_style["tick_labelsize"],
            labelbottom=bool(common_style.get("show_xtick_labels", True)),
        )
        ax.tick_params(
            axis="y",
            labelsize=common_style["tick_labelsize"],
            labelleft=bool(common_style.get("show_ytick_labels", True)),
        )
        if common_style.get("show_legend", True):
            _apply_window_group_legends(
                ax,
                window_spans=window_spans if common_style.get("show_windows", True) else (),
                group_handles=legend_group_handles,
                event_vlines=event_vlines_all if common_style.get("show_event_vlines", True) else (),
                event_vline_style=event_vline_style,
                legend_fontsize=com_style["legend_fontsize"],
                framealpha=common_style["legend_framealpha"],
                loc=common_style["legend_loc"],
            )
        if time_start_frame is not None and time_end_frame is not None:
            tick_vlines = list(pooled_event_vlines) + _collect_overlay_event_vlines_for_ticks(
                sorted_keys=sorted_keys,
                event_vlines_by_key=event_vlines_by_key,
                overlay_cfg=event_vline_overlay_cfg,
            )
            _apply_time_axis_ticks(
                ax,
                common_style=common_style,
                window_spans=window_spans,
                event_vlines=tick_vlines,
                event_order=event_vline_order,
                time_start_frame=time_start_frame,
                time_end_frame=time_end_frame,
                time_zero_frame=time_zero_frame,
                tick_labelsize=float(common_style["tick_labelsize"]),
            )
        if common_style.get("show_xlabel", True):
            ax.set_xlabel(com_style.get("x_label_time", "Normalized time (0-1)"), fontsize=common_style["label_fontsize"])
        if common_style.get("show_ylabel", True):
            ax.set_ylabel(y_label, fontsize=common_style["label_fontsize"])

    if common_style.get("show_subplot_titles", True):
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

    overlay_linewidth = float(com_style.get("line_width", 0.8))
    overlay_alpha = float(com_style.get("scatter_alpha", 0.7))
    if common_style.get("show_windows", True):
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

    if common_style.get("show_grid", True):
        ax_scatter.grid(True, alpha=common_style["grid_alpha"])
    else:
        ax_scatter.grid(False)
    ax_scatter.tick_params(
        axis="x",
        labelsize=common_style["tick_labelsize"],
        labelbottom=bool(common_style.get("show_xtick_labels", True)),
    )
    ax_scatter.tick_params(
        axis="y",
        labelsize=common_style["tick_labelsize"],
        labelleft=bool(common_style.get("show_ytick_labels", True)),
    )
    group_handles = _build_group_legend_handles(
        sorted_keys,
        group_fields,
        filtered_group_fields,
        key_to_linestyle,
        linewidth=legend_group_linewidth,
    )
    if common_style.get("show_legend", True):
        _apply_window_group_legends(
            ax_scatter,
            window_spans=window_spans if common_style.get("show_windows", True) else (),
            group_handles=group_handles,
            legend_fontsize=com_style["legend_fontsize"],
            framealpha=common_style["legend_framealpha"],
            loc=common_style["legend_loc"],
        )

    if common_style.get("show_subplot_titles", True):
        ax_scatter.set_title(
            "COMxy",
            fontsize=common_style["title_fontsize"],
            fontweight=common_style["title_fontweight"],
            pad=common_style["title_pad"],
        )
    if common_style.get("show_xlabel", True):
        ax_scatter.set_xlabel(com_style.get("x_label", comx_name), fontsize=common_style["label_fontsize"])
    if common_style.get("show_ylabel", True):
        ax_scatter.set_ylabel(com_style.get("y_label", comy_name), fontsize=common_style["label_fontsize"])
    ax_scatter.set_aspect("equal", adjustable="datalim")

    overlay_by = ", ".join(group_fields) if group_fields else "all"
    if common_style.get("show_suptitle", True):
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


