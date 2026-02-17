from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .plotly_annotation_legend import add_subplot_legend_annotation, build_legend_html
from .plotly_color import normalize_plotly_color
from .plotly_legacy_style import (
    apply_legacy_layout,
    apply_subplot_title_font,
    apply_time_axes_style,
    resolve_subplot_layout,
)


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            f = float(value)
        except Exception:
            return None
        return f if np.isfinite(f) else None
    text = str(value).strip()
    if not text:
        return None
    try:
        f = float(text)
    except Exception:
        return None
    return f if np.isfinite(f) else None


def _html_path_for_output_path(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_suffix(".html")
    return output_path.parent / f"{output_path.name}.html"


def _mpl_linestyle_to_plotly_dash(style: Any) -> str:
    if style is None:
        return "solid"
    if isinstance(style, str):
        s = style.strip()
        if s in ("-", "solid"):
            return "solid"
        if s == "--":
            return "dash"
        if s == ":":
            return "dot"
        if s == "-.":
            return "dashdot"
        return "solid"
    return "solid"


def _resolve_forceplate_line_color(channel: str, line_colors: Any) -> str:
    if not isinstance(line_colors, dict) or not line_colors:
        return "gray"

    direct = line_colors.get(channel)
    if direct is not None and str(direct).strip():
        return str(direct).strip()

    base = channel[:-5] if channel.endswith("_zero") else channel
    base_color = line_colors.get(base)
    if base_color is not None and str(base_color).strip():
        return str(base_color).strip()

    return "gray"


def _x_norm_to_frames(
    x_norm: np.ndarray,
    *,
    time_start_frame: Optional[float],
    time_end_frame: Optional[float],
    time_zero_frame: float,
) -> np.ndarray:
    if time_start_frame is None or time_end_frame is None:
        return np.asarray(x_norm, dtype=float)
    start = float(time_start_frame)
    end = float(time_end_frame)
    if not (np.isfinite(start) and np.isfinite(end)):
        return np.asarray(x_norm, dtype=float)
    span = end - start
    if span == 0:
        return np.asarray(x_norm, dtype=float)
    return start + (np.asarray(x_norm, dtype=float) * span) - float(time_zero_frame)


def _span_norm_to_frames(
    x_norm: float,
    *,
    time_start_frame: Optional[float],
    time_end_frame: Optional[float],
    time_zero_frame: float,
) -> float:
    if time_start_frame is None or time_end_frame is None:
        return float(x_norm)
    start = float(time_start_frame)
    end = float(time_end_frame)
    if not (np.isfinite(start) and np.isfinite(end)):
        return float(x_norm)
    span = end - start
    if span == 0:
        return float(x_norm)
    return float(start + float(x_norm) * span - float(time_zero_frame))


def _overlay_vline_event_names(overlay_cfg: Any) -> List[str]:
    if not isinstance(overlay_cfg, dict):
        return []
    if not bool(overlay_cfg.get("enabled", False)):
        return []
    raw_cols = overlay_cfg.get("columns")
    if not isinstance(raw_cols, (list, tuple)):
        return []
    out: List[str] = []
    for c in raw_cols:
        if c is None:
            continue
        name = str(c).strip()
        if not name or name in out:
            continue
        out.append(name)
    return out


def _format_overlay_label(
    key: Tuple[Any, ...],
    group_fields: Sequence[str],
    filtered_group_fields: Sequence[str],
) -> Optional[str]:
    if not group_fields or key == ("all",):
        return "all"

    # filtered_group_fields == [] means "hide labels" (same as matplotlib pipeline behavior)
    if filtered_group_fields is not None:
        if not filtered_group_fields:
            return None
        field_to_value = dict(zip(group_fields, key))
        if len(filtered_group_fields) == 1:
            return str(field_to_value.get(filtered_group_fields[0]))
        parts = []
        for f in filtered_group_fields:
            if f in field_to_value:
                parts.append(f"{f}={field_to_value[f]}")
        return ", ".join(parts) if parts else None

    if len(group_fields) == 1:
        return str(key[0])
    return ", ".join(f"{field}={value}" for field, value in zip(group_fields, key))


def _resolve_cop_channel_names(channels: Sequence[str]) -> Tuple[str, str]:
    if len(channels) < 2:
        raise ValueError("COP requires at least 2 channels.")
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
        raise ValueError("COM requires at least 2 channels.")
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


def _add_windows_and_events(
    fig: Any,
    *,
    row: int,
    col: int,
    window_spans: Sequence[Dict[str, Any]],
    window_alpha: float,
    event_vlines: Sequence[Dict[str, Any]],
    vline_dash: str,
    vline_width: float,
    vline_alpha: float,
    time_start_frame: Optional[float],
    time_end_frame: Optional[float],
    time_zero_frame: float,
) -> None:
    for span in window_spans or []:
        x0_raw = _coerce_float(span.get("start"))
        x1_raw = _coerce_float(span.get("end"))
        if x0_raw is None or x1_raw is None:
            continue
        x0 = _span_norm_to_frames(
            float(x0_raw),
            time_start_frame=time_start_frame,
            time_end_frame=time_end_frame,
            time_zero_frame=time_zero_frame,
        )
        x1 = _span_norm_to_frames(
            float(x1_raw),
            time_start_frame=time_start_frame,
            time_end_frame=time_end_frame,
            time_zero_frame=time_zero_frame,
        )
        color = normalize_plotly_color(span.get("color"), default="#cccccc")
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=color,
            opacity=float(window_alpha),
            line_width=0,
            layer="below",
            row=row,
            col=col,
        )

    for v in event_vlines or []:
        x_raw = _coerce_float(v.get("x"))
        if x_raw is None:
            continue
        x = _span_norm_to_frames(
            float(x_raw),
            time_start_frame=time_start_frame,
            time_end_frame=time_end_frame,
            time_zero_frame=time_zero_frame,
        )
        color = normalize_plotly_color(v.get("color"), default="#d62728")
        fig.add_vline(
            x=x,
            line_color=color,
            line_dash=vline_dash,
            line_width=float(vline_width),
            opacity=float(vline_alpha),
            layer="above",
            row=row,
            col=col,
        )


def _plotly_palette() -> List[str]:
    try:
        from plotly.colors import qualitative
    except Exception:
        return ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    colors = getattr(qualitative, "Plotly", None) or []
    return [str(c) for c in colors] if colors else ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def export_task_html(task: Dict[str, Any], *, output_path: Path) -> Optional[Path]:
    """
    Write Plotly HTML for a visualizer plot task.

    Returns the written HTML path, or None when disabled.
    """
    if not bool(task.get("plotly_html", False)):
        return None

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    kind = str(task.get("kind") or "").strip()
    png_path = Path(output_path)
    html_path = _html_path_for_output_path(png_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)

    common_style = task.get("common_style", {}) if isinstance(task.get("common_style"), dict) else {}
    show_windows = bool(common_style.get("show_windows", True))
    show_event_vlines = bool(common_style.get("show_event_vlines", True))
    show_legend = bool(common_style.get("show_legend", True))
    show_grid = bool(common_style.get("show_grid", True))
    grid_alpha = common_style.get("grid_alpha", 0.5)

    event_vline_style = task.get("event_vline_style", {}) if isinstance(task.get("event_vline_style"), dict) else {}
    vline_dash = _mpl_linestyle_to_plotly_dash(event_vline_style.get("linestyle", "--"))
    vline_width = float(_coerce_float(event_vline_style.get("linewidth")) or 1.5)
    vline_alpha = float(_coerce_float(event_vline_style.get("alpha")) or 0.9)

    palette = _plotly_palette()
    key_dashes = ("solid", "dash", "dot", "dashdot")
    x_tick_dtick = task.get("x_tick_dtick")
    if _coerce_float(x_tick_dtick) is None:
        x_tick_dtick = 25

    def _base_layout_title() -> str:
        mode_name = str(task.get("mode_name") or "").strip()
        signal_group = str(task.get("signal_group") or kind).strip()
        return f"{mode_name} | {signal_group}".strip(" |")

    if kind in ("emg", "forceplate"):
        channels = list(task.get("channels") or [])
        grid_layout = list(task.get("grid_layout") or [])
        rows = int(grid_layout[0]) if len(grid_layout) == 2 else 1
        cols = int(grid_layout[1]) if len(grid_layout) == 2 else max(1, len(channels))
        layout_spec = resolve_subplot_layout(rows=rows, cols=cols)

        aggregated = task.get("aggregated") or {}
        x_val = task.get("x")
        if x_val is None:
            first = next(iter(aggregated.values()), None)
            n = int(getattr(first, "size", 0) or 0) if first is not None else 0
            x_val = np.linspace(0.0, 1.0, num=max(1, n))
        x_norm = np.asarray(x_val, dtype=float)

        time_start_frame = _coerce_float(task.get("time_start_frame"))
        time_end_frame = _coerce_float(task.get("time_end_frame"))
        time_zero_frame = float(_coerce_float(task.get("time_zero_frame")) or 0.0)
        time_zero_frame_by_channel = task.get("time_zero_frame_by_channel")
        if not isinstance(time_zero_frame_by_channel, dict):
            time_zero_frame_by_channel = None

        window_spans = task.get("window_spans") or []
        window_spans_by_channel = task.get("window_spans_by_channel")
        if not isinstance(window_spans_by_channel, dict):
            window_spans_by_channel = None

        event_vlines = task.get("event_vlines") or []
        event_vlines_by_channel = task.get("event_vlines_by_channel")
        if not isinstance(event_vlines_by_channel, dict):
            event_vlines_by_channel = None

        window_alpha = float(_coerce_float(task.get("window_span_alpha")) or 0.15)

        style = task.get("style", {}) if isinstance(task.get("style"), dict) else {}
        line_width = float(_coerce_float(style.get("line_width")) or 1.2)
        line_alpha = float(_coerce_float(style.get("line_alpha")) or 0.85)
        line_colors = style.get("line_colors")

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[str(ch) for ch in channels] + [""] * max(0, rows * cols - len(channels)),
            horizontal_spacing=layout_spec.horizontal_spacing,
            vertical_spacing=layout_spec.vertical_spacing,
        )
        apply_subplot_title_font(fig, size=11)

        for idx, ch in enumerate(channels):
            y = aggregated.get(ch)
            if y is None:
                continue
            r = (idx // cols) + 1
            c = (idx % cols) + 1
            axis_idx = (r - 1) * cols + c
            ch_zero = time_zero_frame
            if time_zero_frame_by_channel is not None and ch in time_zero_frame_by_channel:
                try:
                    ch_zero = float(time_zero_frame_by_channel[ch])
                except Exception:
                    ch_zero = time_zero_frame
            x = _x_norm_to_frames(
                x_norm,
                time_start_frame=time_start_frame,
                time_end_frame=time_end_frame,
                time_zero_frame=ch_zero,
            )

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=np.asarray(y, dtype=float),
                    mode="lines",
                    name=str(ch),
                    showlegend=False,
                    opacity=float(line_alpha),
                    line=dict(
                        color=normalize_plotly_color(
                            style.get("line_color", "gray") if kind == "emg" else _resolve_forceplate_line_color(str(ch), line_colors),
                            default="#7f7f7f",
                        ),
                        width=float(line_width),
                    ),
                ),
                row=r,
                col=c,
            )

            if show_windows:
                spans = window_spans_by_channel.get(ch, window_spans) if window_spans_by_channel else window_spans
            else:
                spans = []
            if show_event_vlines:
                vlines = event_vlines_by_channel.get(ch, event_vlines) if event_vlines_by_channel else event_vlines
            else:
                vlines = []
            _add_windows_and_events(
                fig,
                row=r,
                col=c,
                window_spans=spans,
                window_alpha=window_alpha,
                event_vlines=vlines,
                vline_dash=vline_dash,
                vline_width=vline_width,
                vline_alpha=vline_alpha,
                time_start_frame=time_start_frame,
                time_end_frame=time_end_frame,
                time_zero_frame=ch_zero,
            )

            if show_legend:
                legend_text = build_legend_html(window_spans=spans if show_windows else (), event_vlines=vlines if show_event_vlines else ())
                add_subplot_legend_annotation(fig, axis_idx=axis_idx, legend_text=legend_text)

        apply_legacy_layout(fig, title=_base_layout_title(), layout_spec=layout_spec, showlegend=False)
        apply_time_axes_style(fig, show_grid=show_grid, grid_alpha=grid_alpha, x_tick_dtick=x_tick_dtick)
        fig.write_html(html_path, include_plotlyjs="cdn")
        return html_path

    if kind in ("cop", "com"):
        x_val = task.get("x_axis")
        if x_val is None:
            x_val = task.get("x")
        if x_val is None:
            x_val = np.linspace(0.0, 1.0, num=500)
        x_norm = np.asarray(x_val, dtype=float)
        time_start_frame = _coerce_float(task.get("time_start_frame"))
        time_end_frame = _coerce_float(task.get("time_end_frame"))
        time_zero_frame = float(_coerce_float(task.get("time_zero_frame")) or 0.0)
        x_frames = _x_norm_to_frames(
            x_norm,
            time_start_frame=time_start_frame,
            time_end_frame=time_end_frame,
            time_zero_frame=time_zero_frame,
        )

        aggregated = task.get("aggregated") or {}
        window_spans = task.get("window_spans") or []
        event_vlines = task.get("event_vlines") or []

        legend_text = ""
        if show_legend:
            legend_text = build_legend_html(
                window_spans=window_spans if show_windows else (),
                event_vlines=event_vlines if show_event_vlines else (),
            )

        if kind == "cop":
            cop_channels = list(task.get("cop_channels") or [])
            cx_name, cy_name = _resolve_cop_channel_names(cop_channels)
            cx = aggregated.get(cx_name)
            cy = aggregated.get(cy_name)
            if cx is None or cy is None:
                raise ValueError(f"[plotly_html] COP missing channels: {cx_name!r}, {cy_name!r}")
            cop_style = task.get("cop_style") if isinstance(task.get("cop_style"), dict) else {}
            y_invert = bool(cop_style.get("y_invert", False))

            ml_vals = -np.asarray(cy, dtype=float) if y_invert else np.asarray(cy, dtype=float)
            ap_vals = np.asarray(cx, dtype=float)

            fig = make_subplots(rows=1, cols=3, subplot_titles=[cx_name, cy_name, "Cxy"])
            fig.add_trace(go.Scatter(x=x_frames, y=ap_vals, mode="lines", name=cx_name, showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=x_frames, y=ml_vals, mode="lines", name=cy_name, showlegend=False), row=1, col=2)
            fig.add_trace(go.Scatter(x=ml_vals, y=ap_vals, mode="markers", name="Cxy", showlegend=False), row=1, col=3)

            if show_windows:
                alpha = float(_coerce_float(cop_style.get("window_span_alpha")) or 0.15)
                for col in (1, 2):
                    _add_windows_and_events(
                        fig,
                        row=1,
                        col=col,
                        window_spans=window_spans,
                        window_alpha=alpha,
                        event_vlines=[],
                        vline_dash=vline_dash,
                        vline_width=vline_width,
                        vline_alpha=vline_alpha,
                        time_start_frame=time_start_frame,
                        time_end_frame=time_end_frame,
                        time_zero_frame=time_zero_frame,
                    )
            if show_event_vlines:
                for col in (1, 2):
                    _add_windows_and_events(
                        fig,
                        row=1,
                        col=col,
                        window_spans=[],
                        window_alpha=0.0,
                        event_vlines=event_vlines,
                        vline_dash=vline_dash,
                        vline_width=vline_width,
                        vline_alpha=vline_alpha,
                        time_start_frame=time_start_frame,
                        time_end_frame=time_end_frame,
                        time_zero_frame=time_zero_frame,
                    )

            # Window coloring on scatter (closest to matplotlib behavior).
            if show_windows and window_spans:
                for span in window_spans:
                    x0 = _coerce_float(span.get("start"))
                    x1 = _coerce_float(span.get("end"))
                    if x0 is None or x1 is None:
                        continue
                    mask = (x_norm >= float(x0)) & (x_norm <= float(x1))
                    if not mask.any():
                        continue
                    fig.add_trace(
                        go.Scatter(
                            x=ml_vals[mask],
                            y=ap_vals[mask],
                            mode="markers",
                            marker=dict(color=normalize_plotly_color(span.get("color"), default="#cccccc"), size=4),
                            name=str(span.get("label") or span.get("name") or "window"),
                            showlegend=False,
                        ),
                        row=1,
                        col=3,
                    )

            # Use an annotation legend (legacy pattern) for window/event labels.
            if legend_text:
                add_subplot_legend_annotation(fig, axis_idx=1, legend_text=legend_text)

            fig.update_layout(
                title=_base_layout_title(),
                template="plotly_white",
                margin=dict(l=30, r=30, t=60, b=30),
            )
            fig.write_html(html_path, include_plotlyjs="cdn")
            return html_path

        # COM
        com_channels = list(task.get("com_channels") or [])
        comx_name, comy_name, comz_name = _resolve_com_channel_names(com_channels)
        comx = aggregated.get(comx_name)
        comy = aggregated.get(comy_name)
        if comx is None or comy is None:
            raise ValueError(f"[plotly_html] COM missing channels: {comx_name!r}, {comy_name!r}")
        com_style = task.get("com_style") if isinstance(task.get("com_style"), dict) else {}
        y_invert = bool(com_style.get("y_invert", False))

        ml_vals = -np.asarray(comy, dtype=float) if y_invert else np.asarray(comy, dtype=float)
        ap_vals = np.asarray(comx, dtype=float)

        time_panels = 3 if (comz_name is not None and aggregated.get(comz_name) is not None) else 2
        titles = [comx_name, comy_name] + ([comz_name] if time_panels == 3 else []) + ["COMxy"]
        fig = make_subplots(rows=1, cols=time_panels + 1, subplot_titles=titles)
        fig.add_trace(go.Scatter(x=x_frames, y=ap_vals, mode="lines", name=comx_name, showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_frames, y=ml_vals, mode="lines", name=comy_name, showlegend=False), row=1, col=2)
        if time_panels == 3 and comz_name is not None:
            comz = aggregated.get(comz_name)
            if comz is not None:
                fig.add_trace(
                    go.Scatter(x=x_frames, y=np.asarray(comz, dtype=float), mode="lines", name=comz_name, showlegend=False),
                    row=1,
                    col=3,
                )
        fig.add_trace(go.Scatter(x=ml_vals, y=ap_vals, mode="markers", name="COMxy", showlegend=False), row=1, col=time_panels + 1)

        if show_windows:
            alpha = float(_coerce_float(com_style.get("window_span_alpha")) or 0.15)
            for col in range(1, time_panels + 1):
                _add_windows_and_events(
                    fig,
                    row=1,
                    col=col,
                    window_spans=window_spans,
                    window_alpha=alpha,
                    event_vlines=[],
                    vline_dash=vline_dash,
                    vline_width=vline_width,
                    vline_alpha=vline_alpha,
                    time_start_frame=time_start_frame,
                    time_end_frame=time_end_frame,
                    time_zero_frame=time_zero_frame,
                )
        if show_event_vlines:
            for col in range(1, time_panels + 1):
                _add_windows_and_events(
                    fig,
                    row=1,
                    col=col,
                    window_spans=[],
                    window_alpha=0.0,
                    event_vlines=event_vlines,
                    vline_dash=vline_dash,
                    vline_width=vline_width,
                    vline_alpha=vline_alpha,
                    time_start_frame=time_start_frame,
                    time_end_frame=time_end_frame,
                    time_zero_frame=time_zero_frame,
                )

        if show_windows and window_spans:
            for span in window_spans:
                x0 = _coerce_float(span.get("start"))
                x1 = _coerce_float(span.get("end"))
                if x0 is None or x1 is None:
                    continue
                mask = (x_norm >= float(x0)) & (x_norm <= float(x1))
                if not mask.any():
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=ml_vals[mask],
                        y=ap_vals[mask],
                        mode="markers",
                        marker=dict(color=normalize_plotly_color(span.get("color"), default="#cccccc"), size=4),
                        name=str(span.get("label") or span.get("name") or "window"),
                        showlegend=False,
                    ),
                    row=1,
                    col=time_panels + 1,
                )

        # Use an annotation legend (legacy pattern) for window/event labels.
        if legend_text:
            add_subplot_legend_annotation(fig, axis_idx=1, legend_text=legend_text)

        fig.update_layout(
            title=_base_layout_title(),
            template="plotly_white",
            margin=dict(l=30, r=30, t=60, b=30),
        )
        fig.write_html(html_path, include_plotlyjs="cdn")
        return html_path

    if kind == "overlay":
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        signal_group = str(task.get("signal_group") or "").strip()
        if not signal_group:
            raise ValueError("[plotly_html] overlay task missing signal_group")

        group_fields = list(task.get("group_fields") or [])
        filtered_group_fields = list(task.get("filtered_group_fields") or [])
        sorted_keys = [tuple(k) for k in (task.get("sorted_keys") or [])]
        aggregated_by_key = task.get("aggregated_by_key") or {}

        time_start_frame = _coerce_float(task.get("time_start_frame"))
        time_end_frame = _coerce_float(task.get("time_end_frame"))
        time_zero_frame = float(_coerce_float(task.get("time_zero_frame")) or 0.0)
        time_zero_frame_by_channel = task.get("time_zero_frame_by_channel")
        if not isinstance(time_zero_frame_by_channel, dict):
            time_zero_frame_by_channel = None

        x_val = task.get("x")
        if x_val is None:
            x_val = task.get("x_axis")
        if x_val is None:
            x_val = np.linspace(0.0, 1.0, num=500)
        x_norm = np.asarray(x_val, dtype=float)

        window_spans = task.get("window_spans") or []
        window_spans_by_channel = task.get("window_spans_by_channel")
        if not isinstance(window_spans_by_channel, dict):
            window_spans_by_channel = None

        pooled_event_vlines = task.get("pooled_event_vlines") or []
        pooled_event_vlines_by_channel = task.get("pooled_event_vlines_by_channel")
        if not isinstance(pooled_event_vlines_by_channel, dict):
            pooled_event_vlines_by_channel = None

        event_vlines_by_key = task.get("event_vlines_by_key") or {}
        event_vlines_by_key_by_channel = task.get("event_vlines_by_key_by_channel")
        if not isinstance(event_vlines_by_key_by_channel, dict):
            event_vlines_by_key_by_channel = None

        overlay_cfg = task.get("event_vline_overlay_cfg")
        overlay_events = _overlay_vline_event_names(overlay_cfg)

        window_alpha = float(_coerce_float(task.get("window_span_alpha")) or 0.15)

        if signal_group in ("emg", "forceplate"):
            channels = list(task.get("channels") or [])
            grid_layout = list(task.get("grid_layout") or [])
            rows = int(grid_layout[0]) if len(grid_layout) == 2 else 1
            cols = int(grid_layout[1]) if len(grid_layout) == 2 else max(1, len(channels))
            layout_spec = resolve_subplot_layout(rows=rows, cols=cols)

            style = task.get("style") if isinstance(task.get("style"), dict) else {}
            use_group_colors = bool(common_style.get("use_group_colors", False))
            base_line_width = float(_coerce_float(style.get("line_width")) or 1.2)
            base_line_alpha = float(_coerce_float(style.get("line_alpha")) or 0.85)
            line_colors = style.get("line_colors")
            base_emg_color = normalize_plotly_color(style.get("line_color", "gray"), default="#7f7f7f")

            fig = make_subplots(
                rows=rows,
                cols=cols,
                subplot_titles=[str(ch) for ch in channels] + [""] * max(0, rows * cols - len(channels)),
                horizontal_spacing=layout_spec.horizontal_spacing,
                vertical_spacing=layout_spec.vertical_spacing,
            )
            apply_subplot_title_font(fig, size=11)

            for ch_idx, ch in enumerate(channels):
                r = (ch_idx // cols) + 1
                c = (ch_idx % cols) + 1

                ch_zero = time_zero_frame
                if time_zero_frame_by_channel is not None and ch in time_zero_frame_by_channel:
                    try:
                        ch_zero = float(time_zero_frame_by_channel[ch])
                    except Exception:
                        ch_zero = time_zero_frame

                x = _x_norm_to_frames(
                    x_norm,
                    time_start_frame=time_start_frame,
                    time_end_frame=time_end_frame,
                    time_zero_frame=ch_zero,
                )

                for key_idx, key in enumerate(sorted_keys):
                    series = (aggregated_by_key.get(tuple(key)) or {}).get(ch)
                    if series is None:
                        continue
                    label = _format_overlay_label(tuple(key), group_fields, filtered_group_fields)
                    showlegend = show_legend and bool(label) and (ch_idx == 0)

                    if use_group_colors:
                        color = palette[key_idx % len(palette)]
                    elif signal_group == "forceplate":
                        color = _resolve_forceplate_line_color(str(ch), line_colors)
                    else:
                        color = base_emg_color

                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=np.asarray(series, dtype=float),
                            mode="lines",
                            name=str(label) if label is not None else "",
                            showlegend=showlegend,
                            opacity=float(base_line_alpha),
                            line=dict(
                                color=normalize_plotly_color(color, default="#7f7f7f"),
                                dash=key_dashes[key_idx % len(key_dashes)],
                                width=float(base_line_width),
                            ),
                        ),
                        row=r,
                        col=c,
                    )

                if show_windows:
                    spans = window_spans_by_channel.get(ch, window_spans) if window_spans_by_channel else window_spans
                else:
                    spans = []
                if show_event_vlines:
                    vlines = pooled_event_vlines_by_channel.get(ch, pooled_event_vlines) if pooled_event_vlines_by_channel else pooled_event_vlines
                else:
                    vlines = []

                _add_windows_and_events(
                    fig,
                    row=r,
                    col=c,
                    window_spans=spans,
                    window_alpha=window_alpha,
                    event_vlines=vlines,
                    vline_dash=vline_dash,
                    vline_width=vline_width,
                    vline_alpha=vline_alpha,
                    time_start_frame=time_start_frame,
                    time_end_frame=time_end_frame,
                    time_zero_frame=ch_zero,
                )

                # Overlay-group-specific events (linestyle mode): draw per key.
                if show_event_vlines and overlay_events:
                    for key_idx, key in enumerate(sorted_keys):
                        if event_vlines_by_key_by_channel is not None:
                            vlist = (
                                event_vlines_by_key_by_channel.get(tuple(key), {}).get(ch, [])  # type: ignore[union-attr]
                            )
                        else:
                            vlist = event_vlines_by_key.get(tuple(key), [])
                        if not vlist:
                            continue
                        for v in vlist:
                            name = str(v.get("name") or "").strip()
                            if name and name in overlay_events:
                                x_raw = _coerce_float(v.get("x"))
                                if x_raw is None:
                                    continue
                                x_ev = _span_norm_to_frames(
                                    float(x_raw),
                                    time_start_frame=time_start_frame,
                                    time_end_frame=time_end_frame,
                                    time_zero_frame=ch_zero,
                                )
                                fig.add_vline(
                                    x=x_ev,
                                    line_color=normalize_plotly_color(v.get("color"), default="#d62728"),
                                    line_dash=key_dashes[key_idx % len(key_dashes)],
                                    line_width=float(vline_width),
                                    opacity=float(vline_alpha),
                                    layer="above",
                                    row=r,
                                    col=c,
                                )

                if show_legend:
                    legend_text = build_legend_html(
                        window_spans=spans if show_windows else (),
                        event_vlines=vlines if show_event_vlines else (),
                    )
                    add_subplot_legend_annotation(fig, axis_idx=(r - 1) * cols + c, legend_text=legend_text)

            overlay_by = ", ".join(group_fields) if group_fields else "all"
            apply_legacy_layout(
                fig,
                title=f"{_base_layout_title()} | overlay by {overlay_by}",
                layout_spec=layout_spec,
                showlegend=show_legend,
            )
            apply_time_axes_style(fig, show_grid=show_grid, grid_alpha=grid_alpha, x_tick_dtick=x_tick_dtick)
            fig.write_html(html_path, include_plotlyjs="cdn")
            return html_path

        if signal_group == "cop":
            cop_channels = list(task.get("cop_channels") or [])
            cx_name, cy_name = _resolve_cop_channel_names(cop_channels)
            style = task.get("style") if isinstance(task.get("style"), dict) else {}
            y_invert = bool(style.get("y_invert", False))

            fig = make_subplots(rows=1, cols=3, subplot_titles=[cx_name, cy_name, "Cxy"])

            # Time series panels
            x_frames = _x_norm_to_frames(
                x_norm,
                time_start_frame=time_start_frame,
                time_end_frame=time_end_frame,
                time_zero_frame=time_zero_frame,
            )
            for key_idx, key in enumerate(sorted_keys):
                mapping = aggregated_by_key.get(tuple(key), {})
                cx = mapping.get(cx_name)
                cy = mapping.get(cy_name)
                if cx is None or cy is None:
                    continue
                ml_vals = -np.asarray(cy, dtype=float) if y_invert else np.asarray(cy, dtype=float)
                ap_vals = np.asarray(cx, dtype=float)
                label = _format_overlay_label(tuple(key), group_fields, filtered_group_fields)
                showlegend = show_legend and bool(label)
                dash = key_dashes[key_idx % len(key_dashes)]
                color = palette[key_idx % len(palette)]
                fig.add_trace(
                    go.Scatter(x=x_frames, y=ap_vals, mode="lines", name=str(label) if label else "", showlegend=showlegend, line=dict(color=color, dash=dash)),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(x=x_frames, y=ml_vals, mode="lines", name=str(label) if label else "", showlegend=False, line=dict(color=color, dash=dash)),
                    row=1,
                    col=2,
                )

                # Scatter window segments (colored by window)
                if show_windows and window_spans:
                    for span in window_spans:
                        x0 = _coerce_float(span.get("start"))
                        x1 = _coerce_float(span.get("end"))
                        if x0 is None or x1 is None:
                            continue
                        mask = (x_norm >= float(x0)) & (x_norm <= float(x1))
                        if not mask.any():
                            continue
                        fig.add_trace(
                            go.Scatter(
                                x=ml_vals[mask],
                                y=ap_vals[mask],
                                mode="lines",
                                line=dict(color=normalize_plotly_color(span.get("color"), default="#cccccc"), dash=dash, width=1.2),
                                name="",
                                showlegend=False,
                            ),
                            row=1,
                            col=3,
                        )

            if show_legend:
                legend_text = build_legend_html(
                    window_spans=window_spans if show_windows else (),
                    event_vlines=pooled_event_vlines if show_event_vlines else (),
                )
                add_subplot_legend_annotation(fig, axis_idx=1, legend_text=legend_text)

            if show_windows:
                for col in (1, 2):
                    _add_windows_and_events(
                        fig,
                        row=1,
                        col=col,
                        window_spans=window_spans,
                        window_alpha=window_alpha,
                        event_vlines=[],
                        vline_dash=vline_dash,
                        vline_width=vline_width,
                        vline_alpha=vline_alpha,
                        time_start_frame=time_start_frame,
                        time_end_frame=time_end_frame,
                        time_zero_frame=time_zero_frame,
                    )
            if show_event_vlines:
                _add_windows_and_events(
                    fig,
                    row=1,
                    col=1,
                    window_spans=[],
                    window_alpha=0.0,
                    event_vlines=pooled_event_vlines,
                    vline_dash=vline_dash,
                    vline_width=vline_width,
                    vline_alpha=vline_alpha,
                    time_start_frame=time_start_frame,
                    time_end_frame=time_end_frame,
                    time_zero_frame=time_zero_frame,
                )
                _add_windows_and_events(
                    fig,
                    row=1,
                    col=2,
                    window_spans=[],
                    window_alpha=0.0,
                    event_vlines=pooled_event_vlines,
                    vline_dash=vline_dash,
                    vline_width=vline_width,
                    vline_alpha=vline_alpha,
                    time_start_frame=time_start_frame,
                    time_end_frame=time_end_frame,
                    time_zero_frame=time_zero_frame,
                )

                if overlay_events:
                    for key_idx, key in enumerate(sorted_keys):
                        vlist = event_vlines_by_key.get(tuple(key), [])
                        if not vlist:
                            continue
                        dash = key_dashes[key_idx % len(key_dashes)]
                        for v in vlist:
                            name = str(v.get("name") or "").strip()
                            if not name or name not in overlay_events:
                                continue
                            x_raw = _coerce_float(v.get("x"))
                            if x_raw is None:
                                continue
                            x_ev = _span_norm_to_frames(
                                float(x_raw),
                                time_start_frame=time_start_frame,
                                time_end_frame=time_end_frame,
                                time_zero_frame=time_zero_frame,
                            )
                            for col in (1, 2):
                                fig.add_vline(
                                    x=x_ev,
                                    line_color=normalize_plotly_color(v.get("color"), default="#d62728"),
                                    line_dash=dash,
                                    line_width=float(vline_width),
                                    opacity=float(vline_alpha),
                                    layer="above",
                                    row=1,
                                    col=col,
                                )

            fig.update_layout(
                title=f"{_base_layout_title()} | cop | overlay",
                template="plotly_white",
                margin=dict(l=30, r=30, t=60, b=30),
            )
            fig.write_html(html_path, include_plotlyjs="cdn")
            return html_path

        if signal_group == "com":
            com_channels = list(task.get("cop_channels") or task.get("com_channels") or [])
            comx_name, comy_name, comz_name = _resolve_com_channel_names(com_channels)
            style = task.get("style") if isinstance(task.get("style"), dict) else {}
            y_invert = bool(style.get("y_invert", False))

            # Decide panels
            has_z = False
            if comz_name is not None:
                for key in sorted_keys:
                    if (aggregated_by_key.get(tuple(key), {}) or {}).get(comz_name) is not None:
                        has_z = True
                        break
            time_panels = 3 if (has_z and comz_name is not None) else 2
            titles = [comx_name, comy_name] + ([comz_name] if time_panels == 3 else []) + ["COMxy"]
            fig = make_subplots(rows=1, cols=time_panels + 1, subplot_titles=titles)

            x_frames = _x_norm_to_frames(
                x_norm,
                time_start_frame=time_start_frame,
                time_end_frame=time_end_frame,
                time_zero_frame=time_zero_frame,
            )

            for key_idx, key in enumerate(sorted_keys):
                mapping = aggregated_by_key.get(tuple(key), {})
                comx = mapping.get(comx_name)
                comy = mapping.get(comy_name)
                if comx is None or comy is None:
                    continue
                ml_vals = -np.asarray(comy, dtype=float) if y_invert else np.asarray(comy, dtype=float)
                ap_vals = np.asarray(comx, dtype=float)
                dash = key_dashes[key_idx % len(key_dashes)]
                color = palette[key_idx % len(palette)]
                label = _format_overlay_label(tuple(key), group_fields, filtered_group_fields)
                showlegend = show_legend and bool(label)
                fig.add_trace(
                    go.Scatter(x=x_frames, y=ap_vals, mode="lines", name=str(label) if label else "", showlegend=showlegend, line=dict(color=color, dash=dash)),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(x=x_frames, y=ml_vals, mode="lines", name=str(label) if label else "", showlegend=False, line=dict(color=color, dash=dash)),
                    row=1,
                    col=2,
                )
                if time_panels == 3 and comz_name is not None:
                    comz = mapping.get(comz_name)
                    if comz is not None:
                        fig.add_trace(
                            go.Scatter(x=x_frames, y=np.asarray(comz, dtype=float), mode="lines", name=str(label) if label else "", showlegend=False, line=dict(color=color, dash=dash)),
                            row=1,
                            col=3,
                        )

                if show_windows and window_spans:
                    for span in window_spans:
                        x0 = _coerce_float(span.get("start"))
                        x1 = _coerce_float(span.get("end"))
                        if x0 is None or x1 is None:
                            continue
                        mask = (x_norm >= float(x0)) & (x_norm <= float(x1))
                        if not mask.any():
                            continue
                        fig.add_trace(
                            go.Scatter(
                                x=ml_vals[mask],
                                y=ap_vals[mask],
                                mode="lines",
                                line=dict(color=normalize_plotly_color(span.get("color"), default="#cccccc"), dash=dash, width=1.2),
                                name="",
                                showlegend=False,
                            ),
                            row=1,
                            col=time_panels + 1,
                        )

            if show_legend:
                legend_text = build_legend_html(
                    window_spans=window_spans if show_windows else (),
                    event_vlines=pooled_event_vlines if show_event_vlines else (),
                )
                add_subplot_legend_annotation(fig, axis_idx=1, legend_text=legend_text)

            if show_windows:
                for col in range(1, time_panels + 1):
                    _add_windows_and_events(
                        fig,
                        row=1,
                        col=col,
                        window_spans=window_spans,
                        window_alpha=window_alpha,
                        event_vlines=[],
                        vline_dash=vline_dash,
                        vline_width=vline_width,
                        vline_alpha=vline_alpha,
                        time_start_frame=time_start_frame,
                        time_end_frame=time_end_frame,
                        time_zero_frame=time_zero_frame,
                    )
            if show_event_vlines:
                for col in range(1, time_panels + 1):
                    _add_windows_and_events(
                        fig,
                        row=1,
                        col=col,
                        window_spans=[],
                        window_alpha=0.0,
                        event_vlines=pooled_event_vlines,
                        vline_dash=vline_dash,
                        vline_width=vline_width,
                        vline_alpha=vline_alpha,
                        time_start_frame=time_start_frame,
                        time_end_frame=time_end_frame,
                        time_zero_frame=time_zero_frame,
                    )

                if overlay_events:
                    for key_idx, key in enumerate(sorted_keys):
                        vlist = event_vlines_by_key.get(tuple(key), [])
                        if not vlist:
                            continue
                        dash = key_dashes[key_idx % len(key_dashes)]
                        for v in vlist:
                            name = str(v.get("name") or "").strip()
                            if not name or name not in overlay_events:
                                continue
                            x_raw = _coerce_float(v.get("x"))
                            if x_raw is None:
                                continue
                            x_ev = _span_norm_to_frames(
                                float(x_raw),
                                time_start_frame=time_start_frame,
                                time_end_frame=time_end_frame,
                                time_zero_frame=time_zero_frame,
                            )
                            for col in range(1, time_panels + 1):
                                fig.add_vline(
                                    x=x_ev,
                                    line_color=normalize_plotly_color(v.get("color"), default="#d62728"),
                                    line_dash=dash,
                                    line_width=float(vline_width),
                                    opacity=float(vline_alpha),
                                    layer="above",
                                    row=1,
                                    col=col,
                                )

            fig.update_layout(
                title=f"{_base_layout_title()} | com | overlay",
                template="plotly_white",
                margin=dict(l=30, r=30, t=60, b=30),
            )
            fig.write_html(html_path, include_plotlyjs="cdn")
            return html_path

        raise ValueError(f"[plotly_html] unsupported overlay signal_group: {signal_group!r}")

    raise ValueError(f"[plotly_html] unsupported task kind: {kind!r}")
