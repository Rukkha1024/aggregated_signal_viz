from __future__ import annotations

import re
from typing import Any, Optional, Sequence


_TAB10_HEX = (
    "#1f77b4",  # C0 / tab:blue
    "#ff7f0e",  # C1 / tab:orange
    "#2ca02c",  # C2 / tab:green
    "#d62728",  # C3 / tab:red
    "#9467bd",  # C4 / tab:purple
    "#8c564b",  # C5 / tab:brown
    "#e377c2",  # C6 / tab:pink
    "#7f7f7f",  # C7 / tab:gray
    "#bcbd22",  # C8 / tab:olive
    "#17becf",  # C9 / tab:cyan
)

_TAB_COLOR_NAME_TO_HEX = {
    "tab:blue": _TAB10_HEX[0],
    "tab:orange": _TAB10_HEX[1],
    "tab:green": _TAB10_HEX[2],
    "tab:red": _TAB10_HEX[3],
    "tab:purple": _TAB10_HEX[4],
    "tab:brown": _TAB10_HEX[5],
    "tab:pink": _TAB10_HEX[6],
    "tab:gray": _TAB10_HEX[7],
    "tab:olive": _TAB10_HEX[8],
    "tab:cyan": _TAB10_HEX[9],
}

_MPL_SHORT_TO_HEX = {
    "k": "#000000",
    "r": "#ff0000",
    "g": "#008000",
    "b": "#0000ff",
    "c": "#00ffff",
    "m": "#ff00ff",
    "y": "#ffff00",
    "w": "#ffffff",
}

_HEX3_RE = re.compile(r"^#?[0-9a-fA-F]{3}$")
_HEX6_RE = re.compile(r"^#?[0-9a-fA-F]{6}$")
_MPL_CYCLE_RE = re.compile(r"^C(\d+)$")


def _clamp_int(value: float, lo: int, hi: int) -> int:
    try:
        v = int(round(float(value)))
    except Exception:
        return lo
    return max(lo, min(hi, v))


def _clamp_float(value: float, lo: float, hi: float) -> float:
    try:
        v = float(value)
    except Exception:
        return lo
    return max(lo, min(hi, v))


def _sequence_to_rgb_or_rgba(value: Sequence[Any]) -> Optional[str]:
    if len(value) not in (3, 4):
        return None
    comps: list[float] = []
    for comp in value:
        try:
            comps.append(float(comp))
        except Exception:
            return None
    # Heuristic: 0..1 floats (matplotlib RGBA) vs 0..255 ints.
    use_255 = any(c > 1.0 for c in comps[:3])
    if use_255:
        r, g, b = (_clamp_int(c, 0, 255) for c in comps[:3])
    else:
        r, g, b = (_clamp_int(c * 255.0, 0, 255) for c in comps[:3])
    if len(comps) == 4:
        a = _clamp_float(comps[3], 0.0, 1.0)
        return f"rgba({r},{g},{b},{a:.4g})"
    return f"rgb({r},{g},{b})"


def normalize_plotly_color(value: Any, *, default: str = "#000000") -> str:
    """
    Normalize common matplotlib color shorthands into Plotly-compatible colors.

    Plotly rejects values like "C0" (matplotlib color-cycle tokens). This helper
    maps those to the tab10 palette hex codes and passes through valid CSS-like
    colors (e.g., "#rrggbb", "red", "rgb(...)").
    """
    if value is None:
        return default

    if isinstance(value, (list, tuple)):
        seq_color = _sequence_to_rgb_or_rgba(value)
        if seq_color is not None:
            return seq_color

    if isinstance(value, (int, float)):
        # Matplotlib can accept 0..1 grayscale floats; Plotly does not.
        f = float(value)
        if 0.0 <= f <= 1.0:
            gray = _clamp_int(f * 255.0, 0, 255)
            return f"#{gray:02x}{gray:02x}{gray:02x}"
        return default

    text = str(value).strip()
    if not text:
        return default

    if text.lower() in ("none", "null"):
        return default

    m = _MPL_CYCLE_RE.match(text)
    if m:
        try:
            idx = int(m.group(1))
        except Exception:
            return default
        if idx < 0:
            return default
        return _TAB10_HEX[idx % len(_TAB10_HEX)]

    lower = text.lower()
    if lower in _TAB_COLOR_NAME_TO_HEX:
        return _TAB_COLOR_NAME_TO_HEX[lower]
    if lower in _MPL_SHORT_TO_HEX:
        return _MPL_SHORT_TO_HEX[lower]

    if _HEX6_RE.match(text) or _HEX3_RE.match(text):
        return text if text.startswith("#") else f"#{text}"

    # Allow CSS-like functional forms.
    if lower.startswith(("rgb(", "rgba(", "hsl(", "hsla(")):
        return text

    # Fall back to whatever Plotly/CSS can interpret (e.g., "red", "black").
    return text

