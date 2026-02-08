from __future__ import annotations

import html as _html
from typing import Any, Dict, Sequence, Tuple


def build_legend_html(
    *,
    window_spans: Sequence[Dict[str, Any]] = (),
    event_vlines: Sequence[Dict[str, Any]] = (),
) -> str:
    """
    Build an HTML snippet for a per-subplot legend annotation in Plotly.

    Plotly vrect/vline "shapes" do not participate in the standard legend. The
    legacy onset Plotly exporter in this repo used subplot annotations to show
    window/event legends; we replicate that pattern here for the main pipeline.
    """

    lines: list[str] = []
    seen_labels: set[str] = set()

    for span in window_spans or []:
        raw_label = span.get("label") or span.get("name")
        label = str(raw_label or "").strip()
        if not label or label in seen_labels:
            continue
        color = span.get("color")
        color_str = str(color).strip() if color is not None else ""
        if not color_str:
            continue
        seen_labels.add(label)
        # Keep source ASCII-only by using HTML entities.
        lines.append(
            f"<span style='color:{_html.escape(color_str)}'>&#9608;</span> {_html.escape(label)}"
        )

    for v in event_vlines or []:
        raw_label = v.get("label") or v.get("name")
        label = str(raw_label or "").strip()
        if not label or label in seen_labels:
            continue
        color = v.get("color")
        color_str = str(color).strip() if color is not None and str(color).strip() else "#000000"
        seen_labels.add(label)
        lines.append(
            f"<span style='color:{_html.escape(color_str)}'>&#9474;</span> {_html.escape(label)}"
        )

    return "<br>".join(lines)


def add_subplot_legend_annotation(
    fig: Any,
    *,
    axis_idx: int,
    legend_text: str,
    font_size: int = 10,
    x: float = 0.98,
    y: float = 0.98,
) -> None:
    """
    Add a legend-like annotation anchored to a subplot domain.

    axis_idx is the 1-based Plotly axis number in row-major order:
      1 -> x/y
      2 -> x2/y2
      ...
    """
    if not legend_text:
        return
    try:
        idx = int(axis_idx)
    except Exception:
        return
    if idx < 1:
        return

    xref_base = "x" if idx == 1 else f"x{idx}"
    yref_domain = "y domain" if idx == 1 else f"y{idx} domain"

    fig.add_annotation(
        x=float(x),
        y=float(y),
        xref=f"{xref_base} domain",
        yref=yref_domain,
        xanchor="right",
        yanchor="top",
        text=str(legend_text),
        showarrow=False,
        align="left",
        font=dict(size=int(font_size), color="#222222"),
        bgcolor="rgba(255,255,255,0.75)",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1,
    )

