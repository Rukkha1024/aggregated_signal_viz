from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .annotation_legend import add_subplot_legend_annotation, build_legend_html
from .color import normalize_plotly_color
from .legacy_style import (
    apply_legacy_layout,
    apply_subplot_title_font,
    apply_time_axes_style,
    resolve_subplot_layout,
)


def _safe_filename(text: Any) -> str:
    out = str(text)
    for ch in ("/", "\\", ":", "\n", "\r", "\t"):
        out = out.replace(ch, "_")
    out = out.strip()
    return out if out else "untitled"


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


def write_emg_trial_grid_html(
    *,
    html_path: Path,
    title: str,
    x: np.ndarray,
    series_by_trial: Sequence[np.ndarray],
    subplot_titles: Sequence[str],
    max_cols: int,
    window_spans: Sequence[Dict[str, Any]] = (),
    event_vlines: Sequence[Dict[str, Any]] = (),
    window_span_alpha: float = 0.15,
    event_vline_style: Optional[Dict[str, Any]] = None,
    line_color: Any = "gray",
    line_width: float = 1.2,
    line_alpha: float = 0.85,
    show_grid: bool = True,
    grid_alpha: Any = 0.5,
    x_tick_dtick: Optional[Any] = 25,
) -> Path:
    """
    Render a single-EMG-channel trial-grid as Plotly HTML.

    Inputs are already trial-aligned; this module intentionally does not depend
    on the repository's plotting/task pipeline.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(series_by_trial)
    if n == 0:
        raise ValueError("series_by_trial is empty")

    cols = max(1, int(max_cols))
    cols = min(cols, n)
    rows = int(ceil(n / cols))

    titles = [str(t) for t in subplot_titles]
    if len(titles) < n:
        titles = titles + [""] * (n - len(titles))

    layout_spec = resolve_subplot_layout(rows=rows, cols=cols)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=titles,
        horizontal_spacing=layout_spec.horizontal_spacing,
        vertical_spacing=layout_spec.vertical_spacing,
    )
    apply_subplot_title_font(fig, size=11)
    legend_text = build_legend_html(window_spans=window_spans or (), event_vlines=event_vlines or ())
    for i, y in enumerate(series_by_trial):
        r = (i // cols) + 1
        c = (i % cols) + 1
        axis_idx = i + 1
        fig.add_trace(
            go.Scatter(
                x=x,
                y=np.asarray(y, dtype=float),
                mode="lines",
                showlegend=False,
                opacity=float(line_alpha),
                line=dict(color=normalize_plotly_color(line_color, default="#7f7f7f"), width=float(line_width)),
            ),
            row=r,
            col=c,
        )

        for span in window_spans or []:
            x0 = _coerce_float(span.get("start"))
            x1 = _coerce_float(span.get("end"))
            if x0 is None or x1 is None:
                continue
            fig.add_vrect(
                x0=float(x0),
                x1=float(x1),
                fillcolor=normalize_plotly_color(span.get("color"), default="#cccccc"),
                opacity=float(window_span_alpha),
                line_width=0,
                layer="below",
                row=r,
                col=c,
            )

        style = event_vline_style or {}
        vline_dash = _mpl_linestyle_to_plotly_dash(style.get("linestyle", "--"))
        vline_width = float(_coerce_float(style.get("linewidth")) or 1.5)
        vline_alpha = float(_coerce_float(style.get("alpha")) or 0.9)
        for v in event_vlines or []:
            x_raw = _coerce_float(v.get("x"))
            if x_raw is None:
                continue
            fig.add_vline(
                x=float(x_raw),
                line_color=normalize_plotly_color(v.get("color"), default="#d62728"),
                line_dash=vline_dash,
                line_width=vline_width,
                opacity=vline_alpha,
                layer="above",
                row=r,
                col=c,
            )

        if legend_text:
            add_subplot_legend_annotation(fig, axis_idx=axis_idx, legend_text=legend_text)

    apply_legacy_layout(fig, title=_safe_filename(title), layout_spec=layout_spec, showlegend=False)
    apply_time_axes_style(fig, show_grid=bool(show_grid), grid_alpha=grid_alpha, x_tick_dtick=x_tick_dtick)
    fig.write_html(html_path, include_plotlyjs="cdn")
    return html_path
