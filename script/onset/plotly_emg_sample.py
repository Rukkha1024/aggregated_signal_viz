"""Plotly EMG onset sample visualizer.

CLI modes:
1. Default mode (no flags)
   - Runs with current RULES/config behavior and writes all configured outputs.
2. `--sample` mode
   - Writes exactly one output pair (`.html` + `.png`) for quick validation.
   - Saves into a `sample` subfolder under the mode output directory.
   - Intended for fast AI/code-change verification to avoid full-batch runtime.

Examples:
- `conda run -n module python script/onset/plotly_emg_sample.py`
- `conda run -n module python script/onset/plotly_emg_sample.py --sample`
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

# ============================================================
# RULES (edit here; no CLI options)
# ============================================================

RULES: Dict[str, Any] = {
    # repo-root config
    "config_path": None,  # None -> <repo_root>/config.yaml
    # which aggregation_modes to run (None -> all enabled modes)
    "selected_modes": ["diff_step_TF_subject"],
    # mode overrides (same schema as config.yaml: aggregation_modes.<mode>)
    "mode_overrides": {},
    # output path policy:
    # - default: <config.output.base_dir>/plotly_check_onset
    # - optional override: set an absolute/relative path here
    "output_base_dir": None,
    "output_subdir": "plotly_check_onset",
    # export options
    "export_html": True,
    "export_png": True,
    "figure_width": 3000,
    "figure_height": 1500,
    # subplot figure sizing policy
    # - fixed_cell: rows/cols에 맞춰 figure width/height를 자동 계산 (subplot 셀 크기 고정)
    # - legacy_figure: 기존처럼 figure_width/figure_height를 그대로 사용
    "subplot_size_mode": "fixed_cell",
    # fixed_cell mode: per-subplot pixel size (사용자 조정 권장)
    "subplot_cell_width_px": 750,
    "subplot_cell_height_px": 375,
    # fixed_cell mode: gaps between subplot panels (pixel)
    "subplot_gap_h_px": 24,
    "subplot_gap_v_px": 28,
    # shared margins for both fixed_cell/legacy_figure
    "subplot_margin_l_px": 40,
    "subplot_margin_r_px": 20,
    "subplot_margin_t_px": 60,
    "subplot_margin_b_px": 40,
    # When provided, these override auto spacing derived from pixel gaps.
    "subplot_horizontal_spacing": None,
    "subplot_vertical_spacing": None,
    # x-axis tick interval (None -> Plotly auto). e.g. 25 for 25-frame steps.
    "x_tick_dtick": 25,
    # grid transparency (0~1). None -> use config.yaml plot_style.common.grid_alpha.
    "grid_alpha": 0.5,
    # safety limits
    "max_files_per_mode": None,  # None -> all
    "max_trials_per_file": None,  # e.g. 5 (when groupby groups multiple trials)
    # ------------------------------------------------------------
    # New output variant: trial-grid per single EMG channel
    #
    # 목적:
    # - "한 근육(채널)만" 한 번에 보기 위해, 채널(16개) 그리드를 뒤집어서
    #   "trial 그리드"를 만들어 파일을 저장합니다.
    # - 예: subject=가윤호, emg_channel=TA -> 해당 subject의 모든 trial이 subplot으로 배치된 1개 파일
    #
    # 설계 원칙:
    # - config.yaml과 연결될 필요 없음 (옵션은 RULES로만 제어)
    # - 기본값(enabled=false)에서는 기존 출력이 1도 바뀌지 않아야 함
    # - --sample 모드에서는 항상 output 1쌍(.html+.png)만 생성
    # ------------------------------------------------------------
    "trial_grid_by_channel": {
        "enabled": True,   # 활성화 여부 (False면 기존 channel-grid(16채널) 출력만 수행)
        "also_emit_channel_grid": False,    # True면 trial-grid 출력 후에도 기존 channel-grid 출력도 같이 생성합니다.
        # 출력 대상 subject 제한 (None이면 mode filter를 통과한 모든 subject)
        # - 문자열 1개: 단일 subject
        # - 리스트: 여러 subject
        "subjects": None,
        # 출력 대상 EMG 채널 제한 (None이면 config.yaml signal_groups.emg.columns 순서대로 전부)
        # - 문자열 1개: 단일 채널(예: "TA")
        # - 리스트: 여러 채널
        "channels": None,
        "max_cols": 4,  # trial subplot 배치 최대 열 수 (행 수는 trial 개수에 맞춰 자동 계산)
        "max_trials": None,  # trial 개수 상한 (None이면 전부). trial이 너무 많을 때만 사용.
        # subplot title 템플릿 (trial별로 표시).
        # 사용 가능한 placeholder: {velocity}, {trial_num}, {trial}, {step_TF}
        "subplot_title_template": "v={velocity} | trial={trial_num} | {step_TF}",
        # 파일명 템플릿 (확장자 제외). 최소 {subject}, {emg_channel} 포함 권장.
        # 사용 가능한 placeholder: {mode}, {subject}, {emg_channel}, {signal_group}, {velocity}
        # - velocity: 파일 내 trial들이 여러 velocity면 'mix'로 기록됩니다.
        "filename_pattern": "{mode}_{subject}_{emg_channel}_trial_grid_{signal_group}",
        # mode별 output_dir 아래에 추가로 들어갈 폴더명
        "output_dir": "trial_grid_by_channel",
    },
}


_EMG_CHANNEL_SPECIFIC_EVENTS = {"TKEO_AGLR_emg_onset_timing"}


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "config.yaml").exists():
            return candidate
    return here.parents[2]


def _import_repo_utils() -> Tuple[Any, Any, Any]:
    import sys

    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from script.config_utils import get_frame_ratio, load_config, resolve_path

    return load_config, resolve_path, get_frame_ratio


_MATPLOTLIB_TAB10: Tuple[str, ...] = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


def _mpl_color_to_hex(color: Any) -> str:
    if color is None:
        return "#000000"
    text = str(color).strip()
    if not text:
        return "#000000"
    if text.startswith("#") and len(text) in (4, 7):
        return text
    if text.startswith("C") and text[1:].isdigit():
        idx = int(text[1:])
        return _MATPLOTLIB_TAB10[idx % len(_MATPLOTLIB_TAB10)]
    named = {
        "black": "#000000",
        "white": "#ffffff",
        "gray": "#808080",
        "red": "#d62728",
        "green": "#2ca02c",
        "blue": "#1f77b4",
        "orange": "#ff7f0e",
        "purple": "#9467bd",
    }
    return named.get(text.lower(), text)


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
    if isinstance(style, (list, tuple)) and len(style) == 2 and isinstance(style[1], (list, tuple)):
        # matplotlib custom dash: (offset, (on, off, ...))
        pattern = style[1]
        try:
            parts = []
            for v in pattern:
                f = float(v)
                if f <= 0:
                    continue
                parts.append(f"{int(round(f))}px")
            return ",".join(parts) if parts else "solid"
        except Exception:
            return "solid"
    return "solid"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _normalize_field(name: str, *, id_cfg: Dict[str, Any]) -> str:
    text = str(name).strip()
    if not text:
        return text
    if text == "subject":
        return str(id_cfg.get("subject") or "subject")
    if text == "velocity":
        return str(id_cfg.get("velocity") or "velocity")
    if text in ("trial", "trial_num"):
        return str(id_cfg.get("trial") or "trial_num")
    return text


def _is_reference_event_zero(value: Any) -> bool:
    if value is None:
        return False
    try:
        f = float(value)
    except Exception:
        return False
    return abs(f) <= 1e-9


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


def _safe_filename(text: str) -> str:
    out = str(text)
    for ch in ("/", "\\", ":", "\n", "\r", "\t"):
        out = out.replace(ch, "_")
    return out.strip() or "untitled"


def _format_pattern(pattern: str, values: Dict[str, Any]) -> str:
    fmt = Formatter()
    needed = [field for _, field, _, _ in fmt.parse(pattern) if field]
    mapping = dict(values)
    for field in needed:
        if field not in mapping:
            mapping[field] = "all"
    try:
        return pattern.format(**mapping)
    except Exception:
        return pattern


def _format_template(template: str, values: Dict[str, Any], *, missing: str = "") -> str:
    """
    Safe `str.format()` for labels/titles.
    - Missing keys are filled with `missing` (default: empty string).
    - On any formatting error, returns the raw template.
    """
    fmt = Formatter()
    needed = [field for _, field, _, _ in fmt.parse(template) if field]
    mapping = dict(values)
    for field in needed:
        if field not in mapping:
            mapping[field] = missing
    try:
        return template.format(**mapping)
    except Exception:
        return template


def _build_event_color_map(event_cfg: Any, event_cols: Sequence[str]) -> Dict[str, str]:
    cfg = event_cfg if isinstance(event_cfg, dict) else {}
    overrides = cfg.get("colors") if isinstance(cfg.get("colors"), dict) else {}
    palette = cfg.get("palette") if isinstance(cfg.get("palette"), list) else None
    if not palette:
        palette = [f"C{i}" for i in range(10)]

    out: Dict[str, str] = {}
    for idx, col in enumerate(event_cols):
        if col in overrides and str(overrides[col]).strip():
            out[col] = _mpl_color_to_hex(overrides[col])
        else:
            out[col] = _mpl_color_to_hex(palette[idx % len(palette)])
    return out


def _as_float(value: Any, *, default: float) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


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


def _resolve_subplot_layout(rows: int, cols: int) -> SubplotLayoutSpec:
    rows_i = max(1, int(rows))
    cols_i = max(1, int(cols))

    margin_l = max(0, _as_int(RULES.get("subplot_margin_l_px"), default=40))
    margin_r = max(0, _as_int(RULES.get("subplot_margin_r_px"), default=20))
    margin_t = max(0, _as_int(RULES.get("subplot_margin_t_px"), default=60))
    margin_b = max(0, _as_int(RULES.get("subplot_margin_b_px"), default=40))

    # Optional direct override for Plotly spacing values.
    h_override = RULES.get("subplot_horizontal_spacing")
    v_override = RULES.get("subplot_vertical_spacing")

    mode = str(RULES.get("subplot_size_mode") or "fixed_cell").strip().lower()
    if mode != "legacy_figure":
        cell_w = max(1, _as_int(RULES.get("subplot_cell_width_px"), default=750))
        cell_h = max(1, _as_int(RULES.get("subplot_cell_height_px"), default=375))
        gap_h = max(0, _as_int(RULES.get("subplot_gap_h_px"), default=24))
        gap_v = max(0, _as_int(RULES.get("subplot_gap_v_px"), default=28))

        panel_w = cols_i * cell_w + max(0, cols_i - 1) * gap_h
        panel_h = rows_i * cell_h + max(0, rows_i - 1) * gap_v
        width_px = margin_l + margin_r + panel_w
        height_px = margin_t + margin_b + panel_h

        if cols_i > 1:
            horizontal_spacing = float(gap_h) / float(panel_w) if panel_w > 0 else 0.0
        else:
            horizontal_spacing = 0.0
        if rows_i > 1:
            vertical_spacing = float(gap_v) / float(panel_h) if panel_h > 0 else 0.0
        else:
            vertical_spacing = 0.0
    else:
        width_px = max(1, _as_int(RULES.get("figure_width"), default=1800))
        height_px = max(1, _as_int(RULES.get("figure_height"), default=900))
        horizontal_spacing = 0.03
        vertical_spacing = 0.07

    if h_override is not None:
        horizontal_spacing = max(0.0, float(h_override))
    if v_override is not None:
        vertical_spacing = max(0.0, float(v_override))

    # Plotly requires spacing < 1/(n-1) when n > 1.
    if cols_i > 1:
        max_h = max(0.0, (1.0 / float(cols_i - 1)) - 1e-6)
        horizontal_spacing = min(horizontal_spacing, max_h)
    else:
        horizontal_spacing = 0.0
    if rows_i > 1:
        max_v = max(0.0, (1.0 / float(rows_i - 1)) - 1e-6)
        vertical_spacing = min(vertical_spacing, max_v)
    else:
        vertical_spacing = 0.0

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


def _write_png_via_kaleido(
    fig: Any,
    out_png: Path,
    *,
    timeout_s: Optional[float] = None,
) -> None:
    """
    Write PNG using kaleido directly (instead of fig.write_image).

    Why:
    - With kaleido v1+, the timeout is configured via `kopts={"timeout": ...}`.
    - Plotly's high-level API doesn't expose this knob directly.
    """
    import plotly.io as pio

    import kaleido

    fig_dict = fig.to_dict() if hasattr(fig, "to_dict") else fig
    width = (
        getattr(getattr(fig, "layout", None), "width", None)
        if hasattr(fig, "layout")
        else None
    )
    height = (
        getattr(getattr(fig, "layout", None), "height", None)
        if hasattr(fig, "layout")
        else None
    )

    kopts: Dict[str, Any] = {}
    if getattr(pio.defaults, "plotlyjs", None):
        kopts["plotlyjs"] = pio.defaults.plotlyjs
    if getattr(pio.defaults, "mathjax", None):
        kopts["mathjax"] = pio.defaults.mathjax
    if timeout_s is not None:
        kopts["timeout"] = float(timeout_s)

    img_bytes = kaleido.calc_fig_sync(
        fig_dict,
        opts=dict(
            format="png",
            width=int(width) if width is not None else None,
            height=int(height) if height is not None else None,
            scale=pio.defaults.default_scale,
        ),
        topojson=pio.defaults.topojson,
        kopts=kopts,
    )
    out_png.write_bytes(img_bytes)


def _window_colors_from_config(cfg: Dict[str, Any]) -> Dict[str, str]:
    # Prefer existing centralized window_colors (used by matplotlib visualizer) for consistent semantics.
    common = ((cfg.get("plot_style") or {}).get("common") or {})
    raw = common.get("window_colors")
    if isinstance(raw, dict):
        out: Dict[str, str] = {}
        for k, v in raw.items():
            if k is None or v is None:
                continue
            kk = str(k).strip()
            vv = str(v).strip()
            if kk and vv:
                out[kk] = vv
        if out:
            return out
    return {"p1": "#4E79A7", "p2": "#F28E2B", "p3": "#E15759", "p4": "#59A14F"}


def _is_emg_channel_specific_event(value: Any) -> bool:
    name = str(value or "").strip()
    return bool(name) and name in _EMG_CHANNEL_SPECIFIC_EVENTS


def _parse_window_boundary_spec(value: Any) -> Optional[Tuple[str, Any]]:
    """
    Returns one of:
      - ("offset", float_ms)
      - ("event", event_name)
      - ("event_offset", (event_name, float_ms))
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        try:
            return ("offset", float(value))
        except Exception:
            return None

    text = str(value).strip()
    if not text:
        return None

    try:
        return ("offset", float(text))
    except Exception:
        pass

    for op in ("+", "-"):
        if op in text:
            left, right = text.split(op, 1)
            left = left.strip()
            right = right.strip()
            if not left or not right:
                continue
            try:
                ms = float(right)
            except Exception:
                continue
            if op == "-":
                ms = -ms
            return ("event_offset", (left, ms))

    return ("event", text)


def _nanmean(values: Sequence[Optional[float]]) -> Optional[float]:
    arr = np.asarray([np.nan if v is None else float(v) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.mean())


@dataclass(frozen=True)
class WindowSpan:
    name: str
    start_x: float
    end_x: float
    color: str
    duration_ms: float

    @property
    def label(self) -> str:
        dur = int(round(self.duration_ms))
        return f"{self.name} ({dur} ms)"


def _event_ms_from_trial(
    *,
    event_name: str,
    platform_onset_mocap: float,
    mocap_rate: float,
    trial_row: Dict[str, Any],
    tkeo_ms: Optional[float],
) -> Optional[float]:
    name = str(event_name).strip()
    if not name:
        return None
    if name in ("platform_onset",):
        return 0.0
    if name == "step_onset":
        raw = trial_row.get("step_onset")
        if raw is None:
            return None
        return (float(raw) - float(platform_onset_mocap)) * 1000.0 / float(mocap_rate)
    if _is_emg_channel_specific_event(name):
        return None if tkeo_ms is None else float(tkeo_ms)

    raw = trial_row.get(name)
    if raw is None:
        return None
    try:
        # event columns in merged.parquet are in the same mocap-frame domain as platform_onset.
        return (float(raw) - float(platform_onset_mocap)) * 1000.0 / float(mocap_rate)
    except Exception:
        return None


def _event_abs_x_from_trial(
    *,
    event_name: str,
    platform_onset_mocap: float,
    mocap_rate: float,
    trial_row: Dict[str, Any],
    tkeo_ms: Optional[float],
    onset_device_abs: float,
    device_rate: float,
) -> Optional[float]:
    event_ms = _event_ms_from_trial(
        event_name=event_name,
        platform_onset_mocap=platform_onset_mocap,
        mocap_rate=mocap_rate,
        trial_row=trial_row,
        tkeo_ms=tkeo_ms,
    )
    if event_ms is None:
        return None
    return float(onset_device_abs) + float(event_ms) * float(device_rate) / 1000.0


def _compute_window_spans(
    *,
    windows_cfg: Dict[str, Any],
    reference_event: Optional[str],
    window_colors: Dict[str, str],
    device_rate: float,
    mocap_rate: float,
    platform_onset_mocap: float,
    onset_device_abs: float,
    trial_row: Dict[str, Any],
    tkeo_ms: Optional[float],
    crop_start_x: float,
    crop_end_x: float,
) -> List[WindowSpan]:
    definitions = windows_cfg.get("definitions", {})
    if not isinstance(definitions, dict):
        return []

    if _is_reference_event_zero(reference_event):
        ref_name = "platform_onset"
    else:
        ref_name = str(reference_event or "").strip() or "platform_onset"
    ref_ms = _event_ms_from_trial(
        event_name=ref_name,
        platform_onset_mocap=platform_onset_mocap,
        mocap_rate=mocap_rate,
        trial_row=trial_row,
        tkeo_ms=tkeo_ms,
    )
    if ref_ms is None:
        ref_ms = 0.0

    spans: List[WindowSpan] = []
    for key, cfg in definitions.items():
        if not isinstance(cfg, dict):
            continue
        name = str(key).strip()
        if not name:
            continue

        start_spec = _parse_window_boundary_spec(cfg.get("start_ms"))
        end_spec = _parse_window_boundary_spec(cfg.get("end_ms"))
        if start_spec is None or end_spec is None:
            continue

        def _resolve(spec: Tuple[str, Any]) -> Optional[float]:
            kind, value = spec
            if kind == "offset":
                return float(ref_ms) + float(value)
            if kind == "event":
                return _event_ms_from_trial(
                    event_name=str(value),
                    platform_onset_mocap=platform_onset_mocap,
                    mocap_rate=mocap_rate,
                    trial_row=trial_row,
                    tkeo_ms=tkeo_ms,
                )
            if kind == "event_offset":
                ev = _event_ms_from_trial(
                    event_name=str(value[0]),
                    platform_onset_mocap=platform_onset_mocap,
                    mocap_rate=mocap_rate,
                    trial_row=trial_row,
                    tkeo_ms=tkeo_ms,
                )
                return None if ev is None else float(ev) + float(value[1])
            return None

        start_ms = _resolve(start_spec)
        end_ms = _resolve(end_spec)
        if start_ms is None or end_ms is None:
            continue

        start_x = float(onset_device_abs) + float(start_ms) * float(device_rate) / 1000.0
        end_x = float(onset_device_abs) + float(end_ms) * float(device_rate) / 1000.0

        start_x = max(start_x, float(crop_start_x))
        end_x = min(end_x, float(crop_end_x))
        if start_x >= end_x:
            continue

        spans.append(
            WindowSpan(
                name=name,
                start_x=start_x,
                end_x=end_x,
                color=str(window_colors.get(name, "#cccccc")),
                duration_ms=(end_x - start_x) * (1000.0 / float(device_rate)),
            )
        )

    return spans


def _legend_html(
    *,
    window_spans: Sequence[WindowSpan],
    event_items: Sequence[Tuple[str, str]],
) -> str:
    lines: List[str] = []
    for span in window_spans:
        lines.append(f"<span style='color:{span.color}'>█</span> {span.label}")
    for label, color in event_items:
        lines.append(f"<span style='color:{color}'>│</span> {label}")
    return "<br>".join(lines)


def _load_tkeo_by_trial_channel(
    *,
    features_path: Path,
    key_cols: Sequence[str],
    channels: Sequence[str],
    trials_df: pl.DataFrame,
    tkeo_col: str = "TKEO_AGLR_emg_onset_timing",
) -> Dict[Tuple[Any, ...], Dict[str, float]]:
    if trials_df.is_empty():
        return {}
    df = pl.read_csv(features_path)
    if "emg_channel" not in df.columns:
        return {}
    if tkeo_col not in df.columns:
        return {}

    df = df.join(trials_df.select(list(key_cols)).unique(), on=list(key_cols), how="inner")
    if df.is_empty():
        return {}

    df = df.with_columns(
        [
            pl.col("emg_channel").cast(pl.Utf8, strict=False).alias("emg_channel"),
            pl.col(tkeo_col).cast(pl.Float64, strict=False).fill_nan(None).alias("__tkeo"),
        ]
    )

    grouped = df.group_by([*key_cols, "emg_channel"], maintain_order=False).agg(pl.col("__tkeo").mean().alias("__tkeo"))

    out: Dict[Tuple[Any, ...], Dict[str, float]] = {}
    valid_channels = {str(c) for c in channels}
    for row in grouped.iter_rows(named=True):
        ch = row.get("emg_channel")
        val = row.get("__tkeo")
        if ch is None or val is None:
            continue
        ch_name = str(ch)
        if ch_name not in valid_channels:
            continue
        try:
            fval = float(val)
        except Exception:
            continue
        if not np.isfinite(fval):
            continue
        key = tuple(row.get(c) for c in key_cols)
        out.setdefault(key, {})[ch_name] = fval
    return out


def _collect_required_event_columns(
    *,
    windows_cfg: Dict[str, Any],
    event_cfg: Dict[str, Any],
    x_axis_zeroing_enabled: bool = False,
    x_axis_zeroing_reference_event: Optional[str] = None,
) -> List[str]:
    needed: List[str] = []

    raw_event_cols = (event_cfg or {}).get("columns")
    if isinstance(raw_event_cols, list):
        for c in raw_event_cols:
            if c is None:
                continue
            name = str(c).strip()
            if name and name not in needed and not _is_emg_channel_specific_event(name):
                needed.append(name)

    ref = (windows_cfg or {}).get("reference_event")
    if ref is not None and not _is_reference_event_zero(ref):
        name = str(ref).strip()
        if name and name not in needed and not _is_emg_channel_specific_event(name):
            needed.append(name)

    defs = (windows_cfg or {}).get("definitions")
    if isinstance(defs, dict):
        for cfg in defs.values():
            if not isinstance(cfg, dict):
                continue
            for key in ("start_ms", "end_ms"):
                spec = _parse_window_boundary_spec(cfg.get(key))
                if spec is None:
                    continue
                kind, value = spec
                if kind == "event":
                    name = str(value).strip()
                elif kind == "event_offset":
                    try:
                        name = str(value[0]).strip()
                    except Exception:
                        continue
                else:
                    continue
                if name and name not in needed and not _is_emg_channel_specific_event(name):
                    needed.append(name)

    if x_axis_zeroing_enabled:
        ref_name = str(x_axis_zeroing_reference_event or "").strip()
        if ref_name and ref_name not in needed and not _is_emg_channel_specific_event(ref_name):
            needed.append(ref_name)

    return needed


def _collect_trials_series(
    *,
    lf: pl.LazyFrame,
    key_cols: Sequence[str],
    x_abs_col: str,
    channels: Sequence[str],
    meta_cols: Sequence[str],
    device_rate: float,
    time_start_ms: float,
    time_end_ms: float,
) -> pl.DataFrame:
    start_rel = float(time_start_ms) * float(device_rate) / 1000.0
    end_rel = float(time_end_ms) * float(device_rate) / 1000.0

    lf = lf.with_columns((pl.col(x_abs_col) - pl.col("onset_device")).alias("__rel_from_onset"))
    lf = lf.filter((pl.col("__rel_from_onset") >= start_rel) & (pl.col("__rel_from_onset") <= end_rel))

    select_cols: List[str] = []
    for c in [*key_cols, x_abs_col, "onset_device", "platform_onset", "step_onset", *meta_cols, *channels]:
        if c not in select_cols:
            select_cols.append(c)

    lf = lf.select([pl.col(c) for c in select_cols])

    agg_exprs: List[pl.Expr] = [
        pl.col(x_abs_col).sort().alias("__x_abs"),
        pl.col("onset_device").first().alias("onset_device"),
        pl.col("platform_onset").first().alias("platform_onset"),
        pl.col("step_onset").max().alias("step_onset"),
    ]
    for c in meta_cols:
        agg_exprs.append(pl.col(c).first().alias(c))
    for ch in channels:
        agg_exprs.append(pl.col(ch).sort_by(x_abs_col).alias(ch))

    df = lf.group_by(list(key_cols), maintain_order=False).agg(agg_exprs).collect()
    return df


def _resolve_time_window_ms(
    *,
    lf: pl.LazyFrame,
    x_abs_col: str,
    device_rate: float,
    interp_cfg: Dict[str, Any],
) -> Tuple[float, float]:
    start_cfg = interp_cfg.get("start_ms", -100)
    end_cfg = interp_cfg.get("end_ms", 800)

    data_start_ms: Optional[float] = None
    data_end_ms: Optional[float] = None
    if start_cfg is None or end_cfg is None:
        stats = (
            lf.select(
                [
                    (pl.col(x_abs_col) - pl.col("onset_device")).min().alias("min_rel"),
                    (pl.col(x_abs_col) - pl.col("onset_device")).max().alias("max_rel"),
                ]
            ).collect()
        )
        if stats.is_empty():
            raise ValueError("No data available to resolve interpolation time window.")

        min_rel = stats["min_rel"].item()
        max_rel = stats["max_rel"].item()
        if min_rel is None or max_rel is None:
            raise ValueError("No valid onset-aligned frames found to resolve interpolation time window.")

        data_start_ms = float(min_rel) * 1000.0 / float(device_rate)
        data_end_ms = float(max_rel) * 1000.0 / float(device_rate)

    time_start_ms = float(start_cfg) if start_cfg is not None else float(data_start_ms)
    time_end_ms = float(end_cfg) if end_cfg is not None else float(data_end_ms)
    if time_end_ms <= time_start_ms:
        raise ValueError("interpolation.end_ms must be greater than interpolation.start_ms")
    return time_start_ms, time_end_ms


def _mode_cfgs(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    raw_modes = cfg.get("aggregation_modes") or {}
    if not isinstance(raw_modes, dict):
        return {}

    overrides = RULES.get("mode_overrides") or {}
    if not isinstance(overrides, dict):
        overrides = {}

    selected = RULES.get("selected_modes")
    if selected is None:
        names = [n for n, mc in raw_modes.items() if isinstance(mc, dict) and bool(mc.get("enabled", True))]
    else:
        names = [str(n) for n in selected]

    out: Dict[str, Dict[str, Any]] = {}
    for name in names:
        base = raw_modes.get(name)
        if not isinstance(base, dict):
            continue
        if selected is None and not bool(base.get("enabled", True)):
            continue
        merged = _deep_merge(base, overrides.get(name, {}) if isinstance(overrides.get(name), dict) else {})
        out[str(name)] = merged
    return out


def _emit_emg_figure(
    *,
    out_html: Optional[Path],
    out_png: Optional[Path],
    title: str,
    channels: Sequence[str],
    grid_layout: Sequence[int],
    trials: Sequence[Dict[str, Any]],
    key_cols: Sequence[str],
    tkeo_by_trial_channel: Dict[Tuple[Any, ...], Dict[str, float]],
    device_rate: float,
    mocap_rate: float,
    time_start_ms: float,
    time_end_ms: float,
    event_cfg: Dict[str, Any],
    windows_cfg: Dict[str, Any],
    window_colors: Dict[str, str],
    grid_alpha: float,
    x_axis_zeroing_enabled: bool,
    x_axis_zeroing_reference_event: str,
) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    rows, cols = int(grid_layout[0]), int(grid_layout[1])
    layout_spec = _resolve_subplot_layout(rows=rows, cols=cols)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=list(channels),
        horizontal_spacing=layout_spec.horizontal_spacing,
        vertical_spacing=layout_spec.vertical_spacing,
    )

    raw_event_cols = list((event_cfg or {}).get("columns") or [])
    event_cols: List[str] = []
    for raw_name in raw_event_cols:
        name = str(raw_name).strip()
        if name and name not in event_cols:
            event_cols.append(name)
    event_color_map = _build_event_color_map(event_cfg, event_cols)
    event_labels = (event_cfg or {}).get("event_labels") if isinstance(event_cfg, dict) else {}

    style = (event_cfg or {}).get("style") if isinstance(event_cfg, dict) else {}
    vline_dash = _mpl_linestyle_to_plotly_dash((style or {}).get("linestyle", "--"))
    vline_width = float((style or {}).get("linewidth", 1.5)) if (style or {}).get("linewidth") is not None else 1.5
    vline_alpha = float((style or {}).get("alpha", 0.9)) if (style or {}).get("alpha") is not None else 0.9

    ref_event = str((windows_cfg or {}).get("reference_event") or "platform_onset").strip() or "platform_onset"
    zero_abs_x = 0.0
    zero_abs_x_by_channel: Dict[str, float] = {}
    if x_axis_zeroing_enabled:
        zero_ref_event = str(x_axis_zeroing_reference_event or "").strip() or "platform_onset"
        channel_specific_zero = _is_emg_channel_specific_event(zero_ref_event)
        zero_values_global: List[float] = []
        zero_values_by_channel: Dict[str, List[float]] = {str(ch): [] for ch in channels}
        for trial in trials:
            onset_device_raw = trial.get("onset_device")
            platform_onset_raw = trial.get("platform_onset")
            if onset_device_raw is None or platform_onset_raw is None:
                continue
            onset_device_abs = float(onset_device_raw)
            platform_onset_mocap = float(platform_onset_raw)
            trial_key = tuple(trial.get(c) for c in key_cols)

            if channel_specific_zero:
                for ch_name in channels:
                    tkeo_ms = tkeo_by_trial_channel.get(trial_key, {}).get(str(ch_name))
                    event_abs = _event_abs_x_from_trial(
                        event_name=zero_ref_event,
                        platform_onset_mocap=platform_onset_mocap,
                        mocap_rate=mocap_rate,
                        trial_row=trial,
                        tkeo_ms=tkeo_ms,
                        onset_device_abs=onset_device_abs,
                        device_rate=device_rate,
                    )
                    if event_abs is not None and np.isfinite(float(event_abs)):
                        event_abs_f = float(event_abs)
                        zero_values_global.append(event_abs_f)
                        zero_values_by_channel.setdefault(str(ch_name), []).append(event_abs_f)
            else:
                event_abs = _event_abs_x_from_trial(
                    event_name=zero_ref_event,
                    platform_onset_mocap=platform_onset_mocap,
                    mocap_rate=mocap_rate,
                    trial_row=trial,
                    tkeo_ms=None,
                    onset_device_abs=onset_device_abs,
                    device_rate=device_rate,
                )
                if event_abs is not None and np.isfinite(float(event_abs)):
                    zero_values_global.append(float(event_abs))

        if not zero_values_global:
            raise ValueError(
                f"[x_axis_zeroing] reference_event '{zero_ref_event}' has no valid values in this file."
            )
        zero_abs_x = float(np.mean(np.asarray(zero_values_global, dtype=float)))
        if channel_specific_zero:
            for ch_name, values in zero_values_by_channel.items():
                if not values:
                    continue
                zero_abs_x_by_channel[str(ch_name)] = float(np.mean(np.asarray(values, dtype=float)))

    # Keep legend content stable: derive window/event labels from the first trial.
    legend_trial = trials[0] if trials else None
    legend_spans_by_channel: Dict[str, List[WindowSpan]] = {}
    legend_event_items_by_channel: Dict[str, List[Tuple[str, str]]] = {}
    warned_missing_events: set[str] = set()

    if legend_trial is not None:
        onset_device_raw = legend_trial.get("onset_device")
        platform_onset_raw = legend_trial.get("platform_onset")

        if onset_device_raw is not None and platform_onset_raw is not None:
            onset_device_abs = float(onset_device_raw)
            platform_onset_mocap = float(platform_onset_raw)

            crop_start_x = onset_device_abs + float(time_start_ms) * float(device_rate) / 1000.0
            crop_end_x = onset_device_abs + float(time_end_ms) * float(device_rate) / 1000.0

            for ch in channels:
                tkeo_ms = tkeo_by_trial_channel.get(tuple(legend_trial.get(c) for c in key_cols), {}).get(str(ch))
                spans = _compute_window_spans(
                    windows_cfg=windows_cfg,
                    reference_event=ref_event,
                    window_colors=window_colors,
                    device_rate=device_rate,
                    mocap_rate=mocap_rate,
                    platform_onset_mocap=platform_onset_mocap,
                    onset_device_abs=onset_device_abs,
                    trial_row=legend_trial,
                    tkeo_ms=tkeo_ms,
                    crop_start_x=crop_start_x,
                    crop_end_x=crop_end_x,
                )
                legend_spans_by_channel[str(ch)] = spans
                event_items: List[Tuple[str, str]] = []
                for event_name in event_cols:
                    event_abs = _event_abs_x_from_trial(
                        event_name=event_name,
                        platform_onset_mocap=platform_onset_mocap,
                        mocap_rate=mocap_rate,
                        trial_row=legend_trial,
                        tkeo_ms=tkeo_ms,
                        onset_device_abs=onset_device_abs,
                        device_rate=device_rate,
                    )
                    if event_abs is None:
                        continue
                    if float(event_abs) < float(crop_start_x) or float(event_abs) > float(crop_end_x):
                        continue
                    label = str((event_labels or {}).get(event_name, event_name))
                    color = event_color_map.get(event_name, "#000000")
                    event_items.append((label, color))
                legend_event_items_by_channel[str(ch)] = event_items

    for idx_ch, ch in enumerate(channels):
        r = idx_ch // cols + 1
        c = idx_ch % cols + 1
        axis_idx = idx_ch + 1
        xref = "x" if axis_idx == 1 else f"x{axis_idx}"
        yref_domain = "y domain" if axis_idx == 1 else f"y{axis_idx} domain"
        axis_min_x: Optional[float] = None
        axis_max_x: Optional[float] = None
        ch_zero_abs_x = float(zero_abs_x_by_channel.get(str(ch), zero_abs_x))

        for t_idx, trial in enumerate(trials):
            x = np.asarray(trial["__x_abs"], dtype=float) - ch_zero_abs_x
            y = np.asarray(trial[str(ch)], dtype=float)
            finite_x = x[np.isfinite(x)]
            if finite_x.size > 0:
                cur_min = float(finite_x.min())
                cur_max = float(finite_x.max())
                axis_min_x = cur_min if axis_min_x is None else min(axis_min_x, cur_min)
                axis_max_x = cur_max if axis_max_x is None else max(axis_max_x, cur_max)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=str(trial.get("step_TF", "")) or f"trial{t_idx + 1}",
                    line=dict(color="gray", width=1.2, dash="solid" if t_idx == 0 else "dash"),
                    opacity=0.85,
                    showlegend=False,
                ),
                row=r,
                col=c,
            )

            # Draw per-trial window spans + vlines (absolute frame).
            onset_device_raw = trial.get("onset_device")
            platform_onset_raw = trial.get("platform_onset")
            if onset_device_raw is None or platform_onset_raw is None:
                continue
            onset_device_abs = float(onset_device_raw)
            platform_onset_mocap = float(platform_onset_raw)
            crop_start_x = onset_device_abs + float(time_start_ms) * float(device_rate) / 1000.0
            crop_end_x = onset_device_abs + float(time_end_ms) * float(device_rate) / 1000.0

            tkeo_ms = tkeo_by_trial_channel.get(tuple(trial.get(k) for k in key_cols), {}).get(str(ch))

            spans = _compute_window_spans(
                windows_cfg=windows_cfg,
                reference_event=ref_event,
                window_colors=window_colors,
                device_rate=device_rate,
                mocap_rate=mocap_rate,
                platform_onset_mocap=platform_onset_mocap,
                onset_device_abs=onset_device_abs,
                trial_row=trial,
                tkeo_ms=tkeo_ms,
                crop_start_x=crop_start_x,
                crop_end_x=crop_end_x,
            )
            span_alpha = 0.12 if t_idx == 0 else 0.06
            for span in spans:
                fig.add_shape(
                    type="rect",
                    xref=xref,
                    yref=yref_domain,
                    x0=float(span.start_x) - ch_zero_abs_x,
                    x1=float(span.end_x) - ch_zero_abs_x,
                    y0=0,
                    y1=1,
                    fillcolor=span.color,
                    opacity=span_alpha,
                    line=dict(width=0),
                    layer="below",
                )

            def _add_vline(name: str, xval: float, *, dash: Optional[str] = None) -> None:
                fig.add_shape(
                    type="line",
                    xref=xref,
                    yref=yref_domain,
                    x0=float(xval) - ch_zero_abs_x,
                    x1=float(xval) - ch_zero_abs_x,
                    y0=0,
                    y1=1,
                    line=dict(
                        color=event_color_map.get(name, "#000000"),
                        width=vline_width,
                        dash=(dash or vline_dash),
                    ),
                    opacity=vline_alpha if t_idx == 0 else min(vline_alpha, 0.6),
                    layer="above",
                )

            for event_name in event_cols:
                event_abs = _event_abs_x_from_trial(
                    event_name=event_name,
                    platform_onset_mocap=platform_onset_mocap,
                    mocap_rate=mocap_rate,
                    trial_row=trial,
                    tkeo_ms=tkeo_ms,
                    onset_device_abs=onset_device_abs,
                    device_rate=device_rate,
                )
                if event_abs is None:
                    if event_name not in warned_missing_events:
                        print(
                            f"[event_vlines] Warning: event '{event_name}' is missing/invalid in some trials or channels; "
                            "skipping unavailable vlines"
                        )
                        warned_missing_events.add(event_name)
                    continue
                _add_vline(event_name, float(event_abs))

        if axis_min_x is not None and axis_max_x is not None and axis_max_x > axis_min_x:
            fig.update_xaxes(range=[axis_min_x, axis_max_x], row=r, col=c)

        # per-subplot legend (annotation): show stable window/event labels (from first trial)
        spans_for_legend = legend_spans_by_channel.get(str(ch), [])
        legend_event_items = legend_event_items_by_channel.get(str(ch), [])
        legend_text = _legend_html(window_spans=spans_for_legend, event_items=legend_event_items)
        fig.add_annotation(
            x=0.98,
            y=0.98,
            xref=xref + " domain",
            yref=yref_domain,
            xanchor="right",
            yanchor="top",
            text=legend_text,
            showarrow=False,
            align="left",
            font=dict(size=10, color="#222222"),
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        )

    fig.update_layout(
        title=title,
        width=layout_spec.width_px,
        height=layout_spec.height_px,
        margin=dict(
            l=layout_spec.margin_l,
            r=layout_spec.margin_r,
            t=layout_spec.margin_t,
            b=layout_spec.margin_b,
        ),
        template="plotly_white",
        showlegend=False,
    )
    grid_color = f"rgba(0,0,0,{float(grid_alpha):.3f})"
    xaxes_kwargs: Dict[str, Any] = {"showgrid": True, "gridcolor": grid_color}
    x_tick_dtick = RULES.get("x_tick_dtick")
    if x_tick_dtick is not None:
        try:
            dtick_value = float(x_tick_dtick)
        except Exception as exc:
            raise ValueError("RULES['x_tick_dtick'] must be numeric or None") from exc
        if dtick_value <= 0:
            raise ValueError("RULES['x_tick_dtick'] must be > 0 when provided")
        xaxes_kwargs.update({"tickmode": "linear", "dtick": dtick_value})

    fig.update_xaxes(**xaxes_kwargs)
    fig.update_yaxes(showgrid=True, gridcolor=grid_color)

    if out_html is not None:
        out_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_html, include_plotlyjs="cdn", div_id="emg-onset-figure")
        print(f"Wrote: {out_html}")
    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(out_png)
        print(f"Wrote: {out_png}")


def _emit_emg_trial_grid_single_channel(
    *,
    out_html: Optional[Path],
    out_png: Optional[Path],
    title: str,
    emg_channel: str,
    trials: Sequence[Dict[str, Any]],
    key_cols: Sequence[str],
    velocity_col: str,
    trial_col: str,
    tkeo_by_trial_channel: Dict[Tuple[Any, ...], Dict[str, float]],
    device_rate: float,
    mocap_rate: float,
    time_start_ms: float,
    time_end_ms: float,
    event_cfg: Dict[str, Any],
    windows_cfg: Dict[str, Any],
    window_colors: Dict[str, str],
    grid_alpha: float,
    x_axis_zeroing_enabled: bool,
    x_axis_zeroing_reference_event: str,
    max_cols: int,
    subplot_title_template: str,
) -> None:
    import math

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ch = str(emg_channel).strip()
    if not ch:
        raise ValueError("emg_channel is empty")
    if not trials:
        raise ValueError("No trials provided for trial-grid rendering.")

    n_trials = len(trials)
    cols = int(max(1, min(int(max_cols), n_trials)))
    rows = int(math.ceil(float(n_trials) / float(cols)))
    layout_spec = _resolve_subplot_layout(rows=rows, cols=cols)

    subplot_titles: List[str] = []
    for t in trials:
        values = {
            "velocity": t.get(velocity_col),
            "trial_num": t.get(trial_col),
            "trial": t.get(trial_col),
            "step_TF": t.get("step_TF") or "",
        }
        subplot_titles.append(_format_template(str(subplot_title_template), values, missing=""))

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=layout_spec.horizontal_spacing,
        vertical_spacing=layout_spec.vertical_spacing,
    )

    raw_event_cols = list((event_cfg or {}).get("columns") or [])
    event_cols: List[str] = []
    for raw_name in raw_event_cols:
        name = str(raw_name).strip()
        if name and name not in event_cols:
            event_cols.append(name)
    event_color_map = _build_event_color_map(event_cfg, event_cols)
    event_labels = (event_cfg or {}).get("event_labels") if isinstance(event_cfg, dict) else {}

    style = (event_cfg or {}).get("style") if isinstance(event_cfg, dict) else {}
    vline_dash = _mpl_linestyle_to_plotly_dash((style or {}).get("linestyle", "--"))
    vline_width = float((style or {}).get("linewidth", 1.5)) if (style or {}).get("linewidth") is not None else 1.5
    vline_alpha = float((style or {}).get("alpha", 0.9)) if (style or {}).get("alpha") is not None else 0.9

    ref_event = str((windows_cfg or {}).get("reference_event") or "platform_onset").strip() or "platform_onset"

    # Compute x-axis zeroing (single channel).
    # - Trial-grid uses per-channel mean and per-trial values.
    # - If a channel has no valid zeroing-event value in all filtered trials,
    #   do not fail; keep original x-axis (unzeroed) and mark it in title/log.
    zero_abs_x = 0.0
    zero_ref_event = str(x_axis_zeroing_reference_event or "").strip() or "platform_onset"
    channel_specific_zero = _is_emg_channel_specific_event(zero_ref_event)
    zeroing_note = ""
    if x_axis_zeroing_enabled:
        zero_values: List[float] = []
        for trial in trials:
            onset_device_raw = trial.get("onset_device")
            platform_onset_raw = trial.get("platform_onset")
            if onset_device_raw is None or platform_onset_raw is None:
                continue
            onset_device_abs = float(onset_device_raw)
            platform_onset_mocap = float(platform_onset_raw)
            trial_key = tuple(trial.get(c) for c in key_cols)

            tkeo_ms = tkeo_by_trial_channel.get(trial_key, {}).get(ch) if channel_specific_zero else None
            event_abs = _event_abs_x_from_trial(
                event_name=zero_ref_event,
                platform_onset_mocap=platform_onset_mocap,
                mocap_rate=mocap_rate,
                trial_row=trial,
                tkeo_ms=tkeo_ms,
                onset_device_abs=onset_device_abs,
                device_rate=device_rate,
            )
            if event_abs is None:
                continue
            try:
                f = float(event_abs)
            except Exception:
                continue
            if np.isfinite(f):
                zero_values.append(f)

        if not zero_values:
            if channel_specific_zero:
                print(
                    f"[x_axis_zeroing] Warning: reference_event '{zero_ref_event}' has no valid values "
                    f"for emg_channel={ch}; zeroing skipped for this channel."
                )
                zeroing_note = f" | {zero_ref_event}=N/A for {ch} (zeroing skipped)"
            else:
                raise ValueError(
                    f"[x_axis_zeroing] reference_event '{zero_ref_event}' has no valid values for emg_channel={ch}."
                )
        else:
            zero_abs_x = float(np.mean(np.asarray(zero_values, dtype=float)))

    # Stable legend content: use the first trial in the file.
    legend_trial = trials[0]
    legend_spans: List[WindowSpan] = []
    legend_event_items: List[Tuple[str, str]] = []
    warned_missing_events: set[str] = set()

    onset_device_raw = legend_trial.get("onset_device")
    platform_onset_raw = legend_trial.get("platform_onset")
    if onset_device_raw is not None and platform_onset_raw is not None:
        onset_device_abs = float(onset_device_raw)
        platform_onset_mocap = float(platform_onset_raw)
        crop_start_x = onset_device_abs + float(time_start_ms) * float(device_rate) / 1000.0
        crop_end_x = onset_device_abs + float(time_end_ms) * float(device_rate) / 1000.0
        trial_key = tuple(legend_trial.get(c) for c in key_cols)
        tkeo_ms = tkeo_by_trial_channel.get(trial_key, {}).get(ch)

        legend_spans = _compute_window_spans(
            windows_cfg=windows_cfg,
            reference_event=ref_event,
            window_colors=window_colors,
            device_rate=device_rate,
            mocap_rate=mocap_rate,
            platform_onset_mocap=platform_onset_mocap,
            onset_device_abs=onset_device_abs,
            trial_row=legend_trial,
            tkeo_ms=tkeo_ms,
            crop_start_x=crop_start_x,
            crop_end_x=crop_end_x,
        )
        for event_name in event_cols:
            event_abs = _event_abs_x_from_trial(
                event_name=event_name,
                platform_onset_mocap=platform_onset_mocap,
                mocap_rate=mocap_rate,
                trial_row=legend_trial,
                tkeo_ms=tkeo_ms,
                onset_device_abs=onset_device_abs,
                device_rate=device_rate,
            )
            if event_abs is None:
                continue
            if float(event_abs) < float(crop_start_x) or float(event_abs) > float(crop_end_x):
                continue
            label = str((event_labels or {}).get(event_name, event_name))
            color = event_color_map.get(event_name, "#000000")
            legend_event_items.append((label, color))

    legend_text = _legend_html(window_spans=legend_spans, event_items=legend_event_items)

    for idx_trial, trial in enumerate(trials):
        r = idx_trial // cols + 1
        c = idx_trial % cols + 1
        axis_idx = idx_trial + 1
        xref = "x" if axis_idx == 1 else f"x{axis_idx}"
        yref_domain = "y domain" if axis_idx == 1 else f"y{axis_idx} domain"
        trial_zero_abs_x = float(zero_abs_x)

        onset_device_raw = trial.get("onset_device")
        platform_onset_raw = trial.get("platform_onset")
        onset_device_abs: Optional[float] = None
        platform_onset_mocap: Optional[float] = None
        crop_start_x: Optional[float] = None
        crop_end_x: Optional[float] = None
        trial_key = tuple(trial.get(k) for k in key_cols)
        tkeo_ms = tkeo_by_trial_channel.get(trial_key, {}).get(ch)
        if onset_device_raw is not None and platform_onset_raw is not None:
            onset_device_abs = float(onset_device_raw)
            platform_onset_mocap = float(platform_onset_raw)
            crop_start_x = onset_device_abs + float(time_start_ms) * float(device_rate) / 1000.0
            crop_end_x = onset_device_abs + float(time_end_ms) * float(device_rate) / 1000.0

            if x_axis_zeroing_enabled:
                tkeo_ms_for_zero = tkeo_ms if channel_specific_zero else None
                event_abs = _event_abs_x_from_trial(
                    event_name=zero_ref_event,
                    platform_onset_mocap=platform_onset_mocap,
                    mocap_rate=mocap_rate,
                    trial_row=trial,
                    tkeo_ms=tkeo_ms_for_zero,
                    onset_device_abs=onset_device_abs,
                    device_rate=device_rate,
                )
                if event_abs is not None:
                    try:
                        f = float(event_abs)
                    except Exception:
                        f = None
                    if f is not None and np.isfinite(f):
                        trial_zero_abs_x = float(f)

        x = np.asarray(trial["__x_abs"], dtype=float) - float(trial_zero_abs_x)
        y = np.asarray(trial.get(ch), dtype=float)

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=str(ch),
                line=dict(color="gray", width=1.2, dash="solid"),
                opacity=0.85,
                showlegend=False,
            ),
            row=r,
            col=c,
        )

        finite_x = x[np.isfinite(x)]
        if finite_x.size > 0:
            fig.update_xaxes(range=[float(finite_x.min()), float(finite_x.max())], row=r, col=c)

        if onset_device_abs is None or platform_onset_mocap is None or crop_start_x is None or crop_end_x is None:
            continue

        spans = _compute_window_spans(
            windows_cfg=windows_cfg,
            reference_event=ref_event,
            window_colors=window_colors,
            device_rate=device_rate,
            mocap_rate=mocap_rate,
            platform_onset_mocap=platform_onset_mocap,
            onset_device_abs=onset_device_abs,
            trial_row=trial,
            tkeo_ms=tkeo_ms,
            crop_start_x=crop_start_x,
            crop_end_x=crop_end_x,
        )
        for span in spans:
            fig.add_shape(
                type="rect",
                xref=xref,
                yref=yref_domain,
                x0=float(span.start_x) - float(trial_zero_abs_x),
                x1=float(span.end_x) - float(trial_zero_abs_x),
                y0=0,
                y1=1,
                fillcolor=span.color,
                opacity=0.12,
                line=dict(width=0),
                layer="below",
            )

        def _add_vline(name: str, xval: float, *, dash: Optional[str] = None) -> None:
            fig.add_shape(
                type="line",
                xref=xref,
                yref=yref_domain,
                x0=float(xval) - float(trial_zero_abs_x),
                x1=float(xval) - float(trial_zero_abs_x),
                y0=0,
                y1=1,
                line=dict(
                    color=event_color_map.get(name, "#000000"),
                    width=vline_width,
                    dash=(dash or vline_dash),
                ),
                opacity=vline_alpha,
                layer="above",
            )

        for event_name in event_cols:
            event_abs = _event_abs_x_from_trial(
                event_name=event_name,
                platform_onset_mocap=platform_onset_mocap,
                mocap_rate=mocap_rate,
                trial_row=trial,
                tkeo_ms=tkeo_ms,
                onset_device_abs=onset_device_abs,
                device_rate=device_rate,
            )
            if event_abs is None:
                if event_name not in warned_missing_events:
                    print(
                        f"[event_vlines] Warning: event '{event_name}' is missing/invalid in some trials or channels; "
                        "skipping unavailable vlines"
                    )
                    warned_missing_events.add(event_name)
                continue
            _add_vline(event_name, float(event_abs))

        # Repeat the legend annotation in each subplot for parity with the existing channel-grid output.
        fig.add_annotation(
            x=0.98,
            y=0.98,
            xref=xref + " domain",
            yref=yref_domain,
            xanchor="right",
            yanchor="top",
            text=legend_text,
            showarrow=False,
            align="left",
            font=dict(size=10, color="#222222"),
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        )

    fig.update_layout(
        title=f"{title}{zeroing_note}",
        width=layout_spec.width_px,
        height=layout_spec.height_px,
        margin=dict(
            l=layout_spec.margin_l,
            r=layout_spec.margin_r,
            t=layout_spec.margin_t,
            b=layout_spec.margin_b,
        ),
        template="plotly_white",
        showlegend=False,
    )

    grid_color = f"rgba(0,0,0,{float(grid_alpha):.3f})"
    xaxes_kwargs: Dict[str, Any] = {"showgrid": True, "gridcolor": grid_color}
    x_tick_dtick = RULES.get("x_tick_dtick")
    if x_tick_dtick is not None:
        try:
            dtick_value = float(x_tick_dtick)
        except Exception as exc:
            raise ValueError("RULES['x_tick_dtick'] must be numeric or None") from exc
        if dtick_value <= 0:
            raise ValueError("RULES['x_tick_dtick'] must be > 0 when provided")
        xaxes_kwargs.update({"tickmode": "linear", "dtick": dtick_value})

    fig.update_xaxes(**xaxes_kwargs)
    fig.update_yaxes(showgrid=True, gridcolor=grid_color)

    if out_html is not None:
        out_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_html, include_plotlyjs="cdn", div_id="emg-onset-trial-grid")
        print(f"Wrote: {out_html}")
    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        # hardcoded timeout for dense trial-grid export (kaleido default can timeout)
        _write_png_via_kaleido(fig, out_png, timeout_s=180.0)
        print(f"Wrote: {out_png}")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Plotly EMG onset figures from merged parquet/config."
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Generate exactly one HTML+PNG pair under a 'sample' output subfolder.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    sample_mode = bool(args.sample)

    load_config, resolve_path, _ = _import_repo_utils()

    cfg_path = RULES.get("config_path")
    config_path = Path(cfg_path) if cfg_path else (_repo_root() / "config.yaml")
    cfg = load_config(config_path)
    base_dir = config_path.parent

    data_cfg = cfg.get("data", {})
    id_cfg = data_cfg.get("id_columns", {})
    device_rate = float(data_cfg.get("device_sample_rate", 1000))
    mocap_rate = float(data_cfg.get("mocap_sample_rate", 100))

    input_path = resolve_path(base_dir, data_cfg.get("input_file", "data/merged.parquet"))
    features_path_raw = data_cfg.get("features_file")
    features_path = resolve_path(base_dir, features_path_raw) if features_path_raw else None

    subject_col = str(id_cfg.get("subject") or "subject")
    velocity_col = str(id_cfg.get("velocity") or "velocity")
    trial_col = str(id_cfg.get("trial") or "trial_num")
    key_cols = (subject_col, velocity_col, trial_col)

    emg_cfg = ((cfg.get("signal_groups") or {}).get("emg") or {})
    channels = list(emg_cfg.get("columns") or [])
    if not channels:
        raise ValueError("signal_groups.emg.columns is empty")
    grid_layout = emg_cfg.get("grid_layout") or [4, 4]

    interp_cfg = cfg.get("interpolation", {})
    lf = pl.scan_parquet(str(input_path))
    schema = lf.collect_schema()
    available = set(schema.keys())

    # Absolute x-axis requirement
    if "original_DeviceFrame" not in available:
        raise ValueError("This script requires 'original_DeviceFrame' to plot absolute device frames.")
    x_abs_col = "original_DeviceFrame"

    task_col = str(id_cfg.get("task") or "").strip()
    task_filter = data_cfg.get("task_filter")
    if task_filter and task_col and task_col in available:
        lf = lf.filter(pl.col(task_col) == task_filter)

    event_cfg = cfg.get("event_vlines", {}) if isinstance(cfg.get("event_vlines"), dict) else {}
    windows_cfg = cfg.get("windows", {}) if isinstance(cfg.get("windows"), dict) else {}
    window_colors = _window_colors_from_config(cfg)
    common_style_cfg = ((cfg.get("plot_style") or {}).get("common") or {})
    x_zero_cfg = cfg.get("x_axis_zeroing", {}) if isinstance(cfg.get("x_axis_zeroing"), dict) else {}
    onset_col = str(id_cfg.get("onset") or "platform_onset").strip() or "platform_onset"
    x_axis_zeroing_enabled = _coerce_bool(x_zero_cfg.get("enabled"), False)
    x_axis_zeroing_reference_event = str(x_zero_cfg.get("reference_event") or onset_col).strip() or onset_col
    rule_grid_alpha = RULES.get("grid_alpha")
    grid_alpha_raw = common_style_cfg.get("grid_alpha", 0.08) if rule_grid_alpha is None else rule_grid_alpha
    try:
        grid_alpha = float(grid_alpha_raw)
    except Exception as exc:
        raise ValueError(
            "grid_alpha must be numeric (RULES['grid_alpha'] or config plot_style.common.grid_alpha)"
        ) from exc
    if not (0.0 <= grid_alpha <= 1.0):
        raise ValueError("grid_alpha must be in [0, 1]")

    mode_cfgs = _mode_cfgs(cfg)
    if not mode_cfgs:
        raise ValueError("No aggregation_modes selected/found.")

    output_base_cfg = (cfg.get("output") or {}).get("base_dir", "output")
    output_base_dir = resolve_path(base_dir, output_base_cfg)
    output_subdir = str(RULES.get("output_subdir") or "plotly_check_onset").strip() or "plotly_check_onset"
    out_base_override = RULES.get("output_base_dir")
    if out_base_override:
        out_base = resolve_path(base_dir, out_base_override)
    else:
        out_base = output_base_dir / output_subdir
    max_files = RULES.get("max_files_per_mode")
    max_trials_per_file = RULES.get("max_trials_per_file")
    export_html = bool(RULES.get("export_html", True))
    export_png = bool(RULES.get("export_png", True))
    if sample_mode:
        # Sample validation should always emit one comparable output pair.
        export_html = True
        export_png = True

    emitted_outputs = 0
    for mode_name, mode_cfg in mode_cfgs.items():
        mode_filter = mode_cfg.get("filter") if isinstance(mode_cfg.get("filter"), dict) else {}
        raw_groupby = list(mode_cfg.get("groupby") or [])
        groupby = [_normalize_field(g, id_cfg=id_cfg) for g in raw_groupby]
        groupby = [g for g in groupby if g]
        # Primary processing/grouping unit is subject-velocity-trial (enforce).
        for required in (subject_col, velocity_col, trial_col):
            if required not in groupby:
                groupby.append(required)

        overlay = bool(mode_cfg.get("overlay", False))
        overlay_within_raw = list(mode_cfg.get("overlay_within") or [])
        overlay_within = [_normalize_field(g, id_cfg=id_cfg) for g in overlay_within_raw]
        overlay_within = [g for g in overlay_within if g]

        output_dir_name = str(mode_cfg.get("output_dir") or mode_name).strip() or mode_name
        filename_pattern = str(
            mode_cfg.get("filename_pattern") or "{subject}_v{velocity}_{trial_num}_{signal_group}.png"
        )

        lf_mode = lf
        for col, value in mode_filter.items():
            if col in available:
                lf_mode = lf_mode.filter(pl.col(col) == value)

        # Ensure required columns exist for the mode.
        needed_cols = set(key_cols) | {x_abs_col, "onset_device", "platform_onset", "step_onset"} | set(channels)
        meta_cols = set(groupby)
        meta_cols |= set(_normalize_field(c, id_cfg=id_cfg) for c in (mode_cfg.get("color_by") or []) if c)
        meta_cols |= set(mode_filter.keys())
        meta_cols |= set(
            _collect_required_event_columns(
                windows_cfg=windows_cfg,
                event_cfg=event_cfg,
                x_axis_zeroing_enabled=x_axis_zeroing_enabled,
                x_axis_zeroing_reference_event=x_axis_zeroing_reference_event,
            )
        )
        reserved = set(key_cols) | {x_abs_col, "onset_device", "platform_onset", "step_onset", "__x_abs"}
        meta_cols = {c for c in meta_cols if c in available and c not in reserved and c not in channels}
        needed_cols |= meta_cols

        missing = [c for c in needed_cols if c not in available]
        if missing:
            raise ValueError(f"[{mode_name}] Missing required columns in merged.parquet: {missing}")

        time_start_ms, time_end_ms = _resolve_time_window_ms(
            lf=lf_mode,
            x_abs_col=x_abs_col,
            device_rate=device_rate,
            interp_cfg=interp_cfg,
        )

        df_trials = _collect_trials_series(
            lf=lf_mode,
            key_cols=key_cols,
            x_abs_col=x_abs_col,
            channels=channels,
            meta_cols=sorted(meta_cols),
            device_rate=device_rate,
            time_start_ms=time_start_ms,
            time_end_ms=time_end_ms,
        )
        if df_trials.is_empty():
            print(f"[{mode_name}] skip: no trials after filters")
            continue

        # Load per-trial, per-channel TKEO (ms) from features file.
        tkeo_by_trial_channel: Dict[Tuple[Any, ...], Dict[str, float]] = {}
        if features_path is not None and features_path.exists():
            tkeo_by_trial_channel = _load_tkeo_by_trial_channel(
                features_path=features_path,
                key_cols=key_cols,
                channels=channels,
                trials_df=df_trials.select(list(key_cols)),
            )

        trial_grid_cfg = RULES.get("trial_grid_by_channel") if isinstance(RULES.get("trial_grid_by_channel"), dict) else {}
        trial_grid_enabled = bool(trial_grid_cfg.get("enabled", False))
        also_emit_channel_grid = bool(trial_grid_cfg.get("also_emit_channel_grid", False))

        if trial_grid_enabled:
            # Render "one muscle at a time": one file per (subject x emg_channel),
            # with a subplot grid over trials.
            raw_subjects = trial_grid_cfg.get("subjects")
            if isinstance(raw_subjects, str) and raw_subjects.strip():
                selected_subjects = [raw_subjects.strip()]
            elif isinstance(raw_subjects, list):
                selected_subjects = [str(s).strip() for s in raw_subjects if s is not None and str(s).strip()]
            else:
                selected_subjects = []

            raw_channels = trial_grid_cfg.get("channels")
            if isinstance(raw_channels, str) and raw_channels.strip():
                selected_channels = [raw_channels.strip()]
            elif isinstance(raw_channels, list):
                selected_channels = [str(c).strip() for c in raw_channels if c is not None and str(c).strip()]
            else:
                selected_channels = []

            if not selected_subjects:
                selected_subjects = (
                    df_trials.select(pl.col(subject_col).unique().sort())
                    .get_column(subject_col)
                    .to_list()
                )
            if not selected_channels:
                selected_channels = [str(c) for c in channels]

            # Sample mode contract: emit exactly one HTML+PNG pair.
            if sample_mode:
                selected_subjects = selected_subjects[:1]
                selected_channels = selected_channels[:1]

            max_cols = _as_int(trial_grid_cfg.get("max_cols"), default=4)
            if max_cols <= 0:
                raise ValueError("RULES['trial_grid_by_channel']['max_cols'] must be > 0")
            max_trials = trial_grid_cfg.get("max_trials")
            max_trials_int = None if max_trials is None else _as_int(max_trials, default=0)
            if max_trials_int is not None and max_trials_int <= 0:
                raise ValueError("RULES['trial_grid_by_channel']['max_trials'] must be > 0 or None")

            out_dir_name = str(trial_grid_cfg.get("output_dir") or "trial_grid_by_channel").strip() or "trial_grid_by_channel"
            file_pattern = str(trial_grid_cfg.get("filename_pattern") or "{mode}_{subject}_{emg_channel}_trial_grid_{signal_group}").strip()
            subplot_title_template = str(trial_grid_cfg.get("subplot_title_template") or "v={velocity} | trial={trial_num}").strip()

            stop_now = False
            file_count = 0
            for subject in selected_subjects:
                if stop_now:
                    break

                subset_sub = df_trials.filter(pl.col(subject_col) == subject)
                if subset_sub.is_empty():
                    continue

                # Collect + sort trials for stable grid layout.
                trials_rows = subset_sub.to_dicts()
                trials_rows.sort(
                    key=lambda r: (
                        _as_float(r.get(velocity_col), default=float("inf")),
                        _as_int(r.get(trial_col), default=10**9),
                        str(r.get("step_TF") or ""),
                    )
                )
                if max_trials_int is not None:
                    trials_rows = trials_rows[:max_trials_int]

                velocities = (
                    subset_sub.select(pl.col(velocity_col).unique().sort()).get_column(velocity_col).to_list()
                )
                velocity_label = velocities[0] if len(velocities) == 1 else "mix"

                for emg_channel in selected_channels:
                    if max_files is not None and file_count >= int(max_files):
                        stop_now = True
                        break
                    file_count += 1

                    format_values: Dict[str, Any] = {
                        "signal_group": "emg",
                        "mode": mode_name,
                        "subject": subject,
                        "emg_channel": emg_channel,
                        "velocity": velocity_label,
                    }
                    filename_raw = _safe_filename(_format_pattern(file_pattern, format_values))
                    filename_path = Path(filename_raw)
                    stem = filename_path.stem if filename_path.suffix else filename_raw
                    png_name = f"{stem}.png"
                    html_name = f"{stem}.html"

                    out_dir = out_base / out_dir_name
                    if sample_mode:
                        out_dir = out_dir / "sample"

                    out_png = (out_dir / png_name) if export_png else None
                    out_html = (out_dir / html_name) if export_html else None

                    axis_desc = f"absolute frames ({x_abs_col})"
                    if x_axis_zeroing_enabled:
                        if _is_emg_channel_specific_event(x_axis_zeroing_reference_event):
                            axis_desc = (
                                f"zeroed by {x_axis_zeroing_reference_event} "
                                "(channel mean)"
                            )
                        else:
                            axis_desc = f"zeroed by {x_axis_zeroing_reference_event} (mean)"

                    title = f"{mode_name} | emg:{emg_channel} | trial-grid | {axis_desc} | {subject}"
                    _emit_emg_trial_grid_single_channel(
                        out_html=out_html,
                        out_png=out_png,
                        title=title,
                        emg_channel=emg_channel,
                        trials=trials_rows,
                        key_cols=key_cols,
                        velocity_col=velocity_col,
                        trial_col=trial_col,
                        tkeo_by_trial_channel=tkeo_by_trial_channel,
                        device_rate=device_rate,
                        mocap_rate=mocap_rate,
                        time_start_ms=time_start_ms,
                        time_end_ms=time_end_ms,
                        event_cfg=event_cfg,
                        windows_cfg=windows_cfg,
                        window_colors=window_colors,
                        grid_alpha=grid_alpha,
                        x_axis_zeroing_enabled=x_axis_zeroing_enabled,
                        x_axis_zeroing_reference_event=x_axis_zeroing_reference_event,
                        max_cols=max_cols,
                        subplot_title_template=subplot_title_template,
                    )
                    emitted_outputs += 1
                    if sample_mode and emitted_outputs >= 1:
                        stop_now = True
                        break

                if stop_now:
                    break

            if not also_emit_channel_grid:
                if sample_mode and emitted_outputs >= 1:
                    break
                continue

        # Determine file grouping (mimic aggregation_modes overlay semantics).
        file_fields: List[str]
        if overlay:
            if overlay_within:
                file_fields = [f for f in groupby if f not in overlay_within]
            else:
                file_fields = []
        else:
            file_fields = list(groupby)

        if not file_fields:
            file_groups = [("all",)]
        else:
            file_groups = [
                tuple(row) for row in df_trials.select(file_fields).unique().sort(file_fields).iter_rows()
            ]

        file_count = 0
        for file_key in file_groups:
            if max_files is not None and file_count >= int(max_files):
                break
            file_count += 1

            subset = df_trials
            if file_fields and file_key != ("all",):
                for f, v in zip(file_fields, file_key):
                    subset = subset.filter(pl.col(f) == v)

            # Determine which trials to draw in this output file.
            # - overlay=True: multiple group keys can be drawn in one file (subset already constrained by file_fields)
            # - overlay=False: subset should represent one group key; still guard for duplicates.
            if overlay and groupby:
                # one representative per groupby key
                keys_df = subset.select(groupby).unique().sort(groupby)
                chosen_rows: List[Dict[str, Any]] = []
                for key_vals in keys_df.iter_rows():
                    part = subset
                    for f, v in zip(groupby, key_vals):
                        part = part.filter(pl.col(f) == v)
                    part_limited = part.head(int(max_trials_per_file) if max_trials_per_file is not None else part.height)
                    chosen_rows.extend(part_limited.to_dicts())
                trials_rows = chosen_rows
            else:
                part_limited = subset.head(int(max_trials_per_file) if max_trials_per_file is not None else subset.height)
                trials_rows = part_limited.to_dicts()

            if not trials_rows:
                continue

            first_row = trials_rows[0]
            format_values: Dict[str, Any] = {
                "signal_group": "emg",
                "subject": first_row.get(subject_col),
                "velocity": first_row.get(velocity_col),
                "trial": first_row.get(trial_col),
                "trial_num": first_row.get(trial_col),
            }
            for col in groupby:
                format_values[str(col)] = first_row.get(col)

            filename_raw = _safe_filename(_format_pattern(filename_pattern, format_values))
            filename_path = Path(filename_raw)
            stem = filename_path.stem if filename_path.suffix else filename_raw
            png_name = f"{stem}.png"
            html_name = f"{stem}.html"
            out_dir = out_base / output_dir_name
            if sample_mode:
                out_dir = out_dir / "sample"
            out_png = (out_dir / png_name) if export_png else None
            out_html = (out_dir / html_name) if export_html else None

            axis_desc = f"absolute frames ({x_abs_col})"
            if x_axis_zeroing_enabled:
                if _is_emg_channel_specific_event(x_axis_zeroing_reference_event):
                    axis_desc = (
                        f"zeroed by {x_axis_zeroing_reference_event} "
                        "(channel mean; fallback=global mean)"
                    )
                else:
                    axis_desc = f"zeroed by {x_axis_zeroing_reference_event} (mean)"
            title = (
                f"{mode_name} | emg | {axis_desc} | "
                f"{format_values.get('subject')} | v={format_values.get('velocity')} | trial={format_values.get('trial')}"
            )
            _emit_emg_figure(
                out_html=out_html,
                out_png=out_png,
                title=title,
                channels=channels,
                grid_layout=grid_layout,
                trials=trials_rows,
                key_cols=key_cols,
                tkeo_by_trial_channel=tkeo_by_trial_channel,
                device_rate=device_rate,
                mocap_rate=mocap_rate,
                time_start_ms=time_start_ms,
                time_end_ms=time_end_ms,
                event_cfg=event_cfg,
                windows_cfg=windows_cfg,
                window_colors=window_colors,
                grid_alpha=grid_alpha,
                x_axis_zeroing_enabled=x_axis_zeroing_enabled,
                x_axis_zeroing_reference_event=x_axis_zeroing_reference_event,
            )
            emitted_outputs += 1
            if sample_mode and emitted_outputs >= 1:
                break

        if sample_mode and emitted_outputs >= 1:
            break

    if sample_mode:
        if emitted_outputs == 0:
            print("[sample] No output was generated (check filters/input data).")
        else:
            print("[sample] Completed: wrote exactly one HTML+PNG pair in a 'sample' subfolder.")


if __name__ == "__main__":
    main()
