from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SubplotLayoutSpec:
    rows: int
    cols: int
    width_px: int
    height_px: int
    margin_l: int
    margin_r: int
    margin_t: int
    margin_b: int
    horizontal_spacing: float
    vertical_spacing: float


def _as_int(value: Any, *, default: int) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return int(default)


def resolve_subplot_layout(
    rows: int,
    cols: int,
    *,
    subplot_cell_width_px: Any = 750,
    subplot_cell_height_px: Any = 375,
) -> SubplotLayoutSpec:
    """
    Legacy-safe Plotly subplot sizing.

    This mirrors the sizing logic used in `scripts/plotting/plotly/plotly_emg_sample.py` so
    that Plotly HTML outputs in the main pipeline match the legacy plot design.
    """
    rows_i = max(1, int(rows))
    cols_i = max(1, int(cols))

    cell_w_raw = _as_int(subplot_cell_width_px, default=750)
    cell_h_raw = _as_int(subplot_cell_height_px, default=375)
    cell_w = max(320, int(cell_w_raw))
    cell_h = max(220, int(cell_h_raw))

    gap_h = max(36, int(round(float(cell_w) * 0.08)))
    gap_v = max(72, int(round(float(cell_h) * 0.22)))

    panel_w = cols_i * cell_w + max(0, cols_i - 1) * gap_h
    panel_h = rows_i * cell_h + max(0, rows_i - 1) * gap_v

    margin_l = max(64, int(round(float(cell_w) * 0.11)))
    margin_r = max(24, int(round(float(cell_w) * 0.03)))
    margin_t = max(72, int(round(float(cell_h) * 0.18)))
    margin_b = max(64, int(round(float(cell_h) * 0.18)))

    width_px = margin_l + margin_r + panel_w
    height_px = margin_t + margin_b + panel_h

    horizontal_spacing = 0.0
    if cols_i > 1:
        horizontal_spacing = float(gap_h) / float(panel_w) if panel_w > 0 else 0.0
        max_h = max(0.0, (1.0 / float(cols_i - 1)) - 1e-6)
        horizontal_spacing = min(horizontal_spacing, max_h)

    vertical_spacing = 0.0
    if rows_i > 1:
        vertical_spacing = float(gap_v) / float(panel_h) if panel_h > 0 else 0.0
        max_v = max(0.0, (1.0 / float(rows_i - 1)) - 1e-6)
        vertical_spacing = min(vertical_spacing, max_v)

    return SubplotLayoutSpec(
        rows=rows_i,
        cols=cols_i,
        width_px=int(width_px),
        height_px=int(height_px),
        margin_l=int(margin_l),
        margin_r=int(margin_r),
        margin_t=int(margin_t),
        margin_b=int(margin_b),
        horizontal_spacing=float(horizontal_spacing),
        vertical_spacing=float(vertical_spacing),
    )


def apply_subplot_title_font(fig: Any, *, size: int = 11) -> None:
    """
    Match the legacy compact subplot-title styling.

    Call this immediately after `make_subplots(...)` and before adding any other
    annotations (e.g., per-subplot legend boxes).
    """
    try:
        anns = list(getattr(getattr(fig, "layout", None), "annotations", []) or [])
    except Exception:
        return
    for ann in anns:
        try:
            ann.font = dict(size=int(size))
        except Exception:
            continue


def apply_legacy_layout(
    fig: Any,
    *,
    title: str,
    layout_spec: SubplotLayoutSpec,
    template: str = "plotly_white",
    showlegend: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Apply legacy layout sizing/margins for Plotly subplot grids.
    """
    kwargs: Dict[str, Any] = {
        "title": title,
        "width": int(layout_spec.width_px),
        "height": int(layout_spec.height_px),
        "margin": dict(
            l=int(layout_spec.margin_l),
            r=int(layout_spec.margin_r),
            t=int(layout_spec.margin_t),
            b=int(layout_spec.margin_b),
        ),
        "template": str(template),
        "showlegend": bool(showlegend),
    }
    if extra:
        kwargs.update(extra)
    try:
        fig.update_layout(**kwargs)
    except Exception:
        return


def _clamp_alpha(alpha: Any, *, default: float) -> float:
    try:
        a = float(alpha)
    except Exception:
        return float(default)
    if a != a:  # nan
        return float(default)
    return max(0.0, min(1.0, a))


def apply_time_axes_style(
    fig: Any,
    *,
    show_grid: bool = True,
    grid_alpha: Any = 0.5,
    x_tick_dtick: Optional[Any] = 25,
) -> None:
    """
    Apply legacy axis styling for time-series subplots.
    """
    if not show_grid:
        try:
            fig.update_xaxes(showgrid=False, automargin=True)
            fig.update_yaxes(showgrid=False, automargin=True)
        except Exception:
            return
        return

    alpha = _clamp_alpha(grid_alpha, default=0.5)
    grid_color = f"rgba(0,0,0,{float(alpha):.3f})"

    xaxes_kwargs: Dict[str, Any] = {"showgrid": True, "gridcolor": grid_color}
    if x_tick_dtick is not None:
        try:
            dtick_value = float(x_tick_dtick)
        except Exception:
            dtick_value = None
        if dtick_value is not None and dtick_value > 0:
            xaxes_kwargs.update({"tickmode": "linear", "dtick": float(dtick_value)})

    try:
        fig.update_xaxes(automargin=True, **xaxes_kwargs)
        fig.update_yaxes(showgrid=True, gridcolor=grid_color, automargin=True)
    except Exception:
        return
