from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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


def _parse_window_boundary_spec(value: Any) -> Optional[Tuple[str, Any]]:
    """
    Returns one of:
      - ("offset", float_ms)
      - ("event", event_col)
      - ("event_offset", (event_col, float_ms))
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


def _nanmean(values: Sequence[Optional[float]]) -> Optional[float]:
    arr = np.asarray([np.nan if v is None else float(v) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.mean())


def _interp_trial(x_list: Sequence[float], ys_lists: Sequence[Sequence[float]], x_target: np.ndarray) -> np.ndarray:
    x_all = np.asarray(x_list, dtype=float)
    n_channels = len(ys_lists)
    out = np.full((n_channels, x_target.size), np.nan, dtype=float)

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

        uniq_x, inv = np.unique(x, return_inverse=True)
        if uniq_x.size != x.size:
            sums = np.bincount(inv, weights=y)
            counts = np.bincount(inv)
            y = sums / counts
            x = uniq_x

        out[i] = np.interp(x_target, x, y, left=np.nan, right=np.nan).astype(float, copy=False)

    return out


def _subject_mean_then_group_mean(
    tensor: np.ndarray,
    subjects: Sequence[Any],
    indices: np.ndarray,
) -> np.ndarray:
    subj_vals = np.asarray(subjects, dtype=object)[indices]
    unique_subjects = list(dict.fromkeys(subj_vals.tolist()))
    if len(unique_subjects) <= 1:
        return np.nanmean(tensor[indices], axis=0)

    subj_means: List[np.ndarray] = []
    for subj in unique_subjects:
        subj_idx = indices[subj_vals == subj]
        subj_means.append(np.nanmean(tensor[subj_idx], axis=0))
    return np.nanmean(np.stack(subj_means, axis=0), axis=0)


def _sorted_overlay_keys(step_values: Sequence[Any]) -> List[str]:
    vals = ["" if v is None else str(v) for v in step_values]
    uniq = sorted(set(vals))
    order = {"nonstep": 0, "step": 1}
    return sorted(uniq, key=lambda v: (order.get(v, 99), v))


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


def _window_colors_default() -> Dict[str, str]:
    return {"p1": "#4E79A7", "p2": "#F28E2B", "p3": "#E15759", "p4": "#59A14F"}


def _resolve_window_bound_ms(
    spec: Tuple[str, Any],
    *,
    event_means: Dict[str, float],
    device_rate: float,
) -> Optional[float]:
    kind, value = spec
    if kind == "offset":
        try:
            # ms offset from reference_event (converted later)
            return float(value)
        except Exception:
            return None
    if kind == "event":
        name = str(value).strip()
        return event_means.get(name)
    if kind == "event_offset":
        try:
            name = str(value[0]).strip()
            delta = float(value[1])
        except Exception:
            return None
        base = event_means.get(name)
        return None if base is None else float(base) + float(delta) * float(device_rate) / 1000.0
    return None


def _compute_window_spans_for_channel(
    *,
    windows_cfg: Dict[str, Any],
    channel_event_frames: Dict[str, float],
    reference_event: Optional[str],
    time_start_x: float,
    time_end_x: float,
    device_rate: float,
    window_colors: Dict[str, str],
) -> List[WindowSpan]:
    definitions = windows_cfg.get("definitions", {})
    if not isinstance(definitions, dict):
        return []

    spans: List[WindowSpan] = []
    ref_name = str(reference_event or "").strip()
    ref_x = channel_event_frames.get(ref_name) if ref_name else None
    if ref_x is None:
        ref_x = channel_event_frames.get("platform_onset", 0.0)
    for name, cfg in definitions.items():
        if not isinstance(cfg, dict):
            continue
        wname = str(name).strip()
        if not wname:
            continue
        start_spec = _parse_window_boundary_spec(cfg.get("start_ms"))
        end_spec = _parse_window_boundary_spec(cfg.get("end_ms"))
        if start_spec is None or end_spec is None:
            continue
        start = _resolve_window_bound_ms(start_spec, event_means=channel_event_frames, device_rate=device_rate)
        end = _resolve_window_bound_ms(end_spec, event_means=channel_event_frames, device_rate=device_rate)
        if start is None or end is None:
            continue
        # Interpret numeric offsets as ms offsets from the configured reference event.
        if start_spec[0] == "offset":
            start = float(ref_x) + float(start) * float(device_rate) / 1000.0
        if end_spec[0] == "offset":
            end = float(ref_x) + float(end) * float(device_rate) / 1000.0

        start = max(float(start), float(time_start_x))
        end = min(float(end), float(time_end_x))
        if start >= end:
            continue
        spans.append(
            WindowSpan(
                name=wname,
                start_x=float(start),
                end_x=float(end),
                color=str(window_colors.get(wname, "#cccccc")),
                duration_ms=(float(end) - float(start)) * (1000.0 / float(device_rate)),
            )
        )
    return spans


def _legend_html(
    *,
    window_spans: Sequence[WindowSpan],
    event_items: Sequence[Tuple[str, str]],
    group_items: Sequence[Tuple[str, str]],
) -> str:
    lines: List[str] = []
    for span in window_spans:
        lines.append(f"<span style='color:{span.color}'>█</span> {span.label}")
    for label, color in event_items:
        lines.append(f"<span style='color:{color}'>│</span> {label}")
    for label, style_text in group_items:
        lines.append(f"<span style='color:gray'>{style_text}</span> {label}")
    return "<br>".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plotly EMG overlay sample (check_onset)")
    parser.add_argument("--config", type=str, default=str(_repo_root() / "config.yaml"))
    parser.add_argument("--mode", type=str, default="step_TF_mean")
    parser.add_argument("--html", type=str, default=str(Path(__file__).with_name("output") / "emg_plotly_sample.html"))
    parser.add_argument("--png", type=str, default=str(Path(__file__).with_name("output") / "emg_plotly_sample.png"))
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    load_config, resolve_path, get_frame_ratio = _import_repo_utils()

    args = parse_args()
    config_path = Path(args.config)
    cfg = load_config(config_path)
    base_dir = config_path.parent

    data_cfg = cfg.get("data", {})
    id_cfg = data_cfg.get("id_columns", {})

    input_path = resolve_path(base_dir, data_cfg.get("input_file", "data/merged.parquet"))
    if not input_path.exists():
        raise FileNotFoundError(f"input_file not found: {input_path}")

    mode_cfg = (cfg.get("aggregation_modes") or {}).get(args.mode)
    if not isinstance(mode_cfg, dict):
        raise KeyError(f"aggregation_modes.{args.mode} not found in config.yaml")

    filter_cfg = mode_cfg.get("filter") if isinstance(mode_cfg.get("filter"), dict) else {}
    group_fields = list(mode_cfg.get("groupby") or [])
    if group_fields != ["step_TF"]:
        raise ValueError(f"This sample expects groupby ['step_TF'], got: {group_fields}")

    emg_cfg = (cfg.get("signal_groups") or {}).get("emg") or {}
    channels = list(emg_cfg.get("columns") or [])
    if not channels:
        raise ValueError("signal_groups.emg.columns is empty")
    grid_layout = emg_cfg.get("grid_layout") or [4, 4]
    rows, cols = int(grid_layout[0]), int(grid_layout[1])

    interp_cfg = cfg.get("interpolation", {})
    time_start_ms = float(interp_cfg.get("start_ms", -100))
    time_end_ms = float(interp_cfg.get("end_ms", 800))
    if time_end_ms <= time_start_ms:
        raise ValueError("interpolation.end_ms must be greater than interpolation.start_ms")

    device_rate = float(data_cfg.get("device_sample_rate", 1000))
    frame_ratio = int(get_frame_ratio(data_cfg))

    subject_col = str(id_cfg.get("subject", "subject"))
    velocity_col = str(id_cfg.get("velocity", "velocity"))
    trial_col = str(id_cfg.get("trial", "trial_num"))
    frame_col = str(id_cfg.get("frame", "DeviceFrame"))
    mocap_col = str(id_cfg.get("mocap_frame", "MocapFrame"))
    onset_col = str(id_cfg.get("onset", "platform_onset"))
    task_col = str(id_cfg.get("task", "task"))

    key_cols = [subject_col, velocity_col, trial_col]

    lf = pl.scan_parquet(str(input_path))
    available = set(lf.collect_schema().keys())

    task_filter = data_cfg.get("task_filter")
    if task_filter and task_col in available:
        lf = lf.filter(pl.col(task_col) == task_filter)

    for k, v in (filter_cfg or {}).items():
        if k in available:
            lf = lf.filter(pl.col(k) == v)

    # We render a single subject-velocity-trial in absolute (original) device frames.
    required_cols = set([*key_cols, frame_col, mocap_col, onset_col, "step_onset", "onset_device"])
    if "original_DeviceFrame" in available:
        required_cols.add("original_DeviceFrame")
    required_cols.update(channels)
    missing = [c for c in required_cols if c not in available]
    if missing:
        raise ValueError(f"Missing required columns in merged.parquet: {missing}")

    x_col_abs = "original_DeviceFrame" if "original_DeviceFrame" in available else frame_col

    # Select one trial (first row after filters), then render only that trial.
    first = lf.select(key_cols).limit(1).collect()
    if first.is_empty():
        raise ValueError("No data available after applying filters.")
    first_subject, first_velocity, first_trial = first.row(0)

    lf_trial = lf.filter(
        (pl.col(subject_col) == first_subject) & (pl.col(velocity_col) == first_velocity) & (pl.col(trial_col) == first_trial)
    )

    meta_row = lf_trial.select([subject_col, velocity_col, trial_col, onset_col, "step_onset", "onset_device"]).limit(1).collect()
    if meta_row.is_empty():
        raise ValueError("No data available for the selected trial.")
    _, _, _, platform_onset_mocap, step_onset_mocap, onset_device_abs = meta_row.row(0)
    if platform_onset_mocap is None or step_onset_mocap is None or onset_device_abs is None:
        raise ValueError("Missing platform_onset/step_onset/onset_device for the selected trial.")

    platform_onset_abs = float(onset_device_abs)
    step_onset_abs = float(onset_device_abs) + (float(step_onset_mocap) - float(platform_onset_mocap)) * float(frame_ratio)

    crop_start_x = platform_onset_abs + float(time_start_ms) * float(device_rate) / 1000.0
    crop_end_x = platform_onset_abs + float(time_end_ms) * float(device_rate) / 1000.0
    if crop_end_x <= crop_start_x:
        raise ValueError("Invalid crop window after converting ms to frames.")

    df_plot = (
        lf_trial.select([x_col_abs, *channels])
        .filter((pl.col(x_col_abs) >= crop_start_x) & (pl.col(x_col_abs) <= crop_end_x))
        .collect()
        .sort(x_col_abs)
    )
    if df_plot.is_empty():
        raise ValueError("No rows remain after applying the crop window in absolute frames.")

    x_abs = df_plot[x_col_abs].to_numpy()

    # --- Features: channel-specific TKEO means (overall + per overlay group) ---
    features_path_raw = data_cfg.get("features_file")
    features_path = resolve_path(base_dir, features_path_raw) if features_path_raw else None
    if features_path is None or not features_path.exists():
        raise FileNotFoundError(f"features_file not found: {features_path_raw}")

    feat = pl.read_csv(features_path)
    if "emg_channel" not in feat.columns:
        raise ValueError("features_file must contain 'emg_channel'")
    if "TKEO_AGLR_emg_onset_timing" not in feat.columns:
        raise ValueError("features_file must contain 'TKEO_AGLR_emg_onset_timing'")

    trial_keys = pl.DataFrame({subject_col: [first_subject], velocity_col: [first_velocity], trial_col: [first_trial]})
    feat = feat.join(trial_keys, on=key_cols, how="inner")
    feat = feat.with_columns(
        [
            pl.col("emg_channel").cast(pl.Utf8, strict=False).alias("emg_channel"),
            pl.col("TKEO_AGLR_emg_onset_timing").cast(pl.Float64, strict=False).fill_nan(None).alias("__tkeo"),
        ]
    )

    tkeo_mean_by_channel: Dict[str, float] = {}
    tkeo_grouped = feat.group_by("emg_channel", maintain_order=False).agg(pl.col("__tkeo").mean().alias("__tkeo"))
    for row in tkeo_grouped.iter_rows(named=True):
        ch = row.get("emg_channel")
        val = row.get("__tkeo")
        if ch is None or val is None:
            continue
        f = float(val)
        if np.isfinite(f):
            tkeo_mean_by_channel[str(ch)] = f

    # --- Plotly figure ---
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    window_colors = _window_colors_default()
    window_alpha = 0.15
    event_cfg = cfg.get("event_vlines", {})
    event_cols = list((event_cfg or {}).get("columns") or ["platform_onset", "step_onset", "TKEO_AGLR_emg_onset_timing"])
    event_color_map = _build_event_color_map(event_cfg, event_cols)
    event_labels = (event_cfg or {}).get("event_labels") if isinstance(event_cfg, dict) else {}
    tkeo_label = str((event_labels or {}).get("TKEO_AGLR_emg_onset_timing", "TKEO")).strip() or "TKEO"

    windows_cfg = cfg.get("windows", {}) if isinstance(cfg.get("windows"), dict) else {}
    reference_event = windows_cfg.get("reference_event")

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=channels, horizontal_spacing=0.03, vertical_spacing=0.07)

    for ch_idx, ch in enumerate(channels):
        r = ch_idx // cols + 1
        c = ch_idx % cols + 1
        axis_idx = ch_idx + 1
        xref = "x" if axis_idx == 1 else f"x{axis_idx}"
        yref_domain = "y domain" if axis_idx == 1 else f"y{axis_idx} domain"

        fig.add_trace(
            go.Scatter(
                x=x_abs,
                y=df_plot[str(ch)].to_numpy(),
                mode="lines",
                name=str(ch),
                line=dict(color="gray", width=1.3, dash="solid"),
                opacity=0.85,
                showlegend=False,
            ),
            row=r,
            col=c,
        )

        # per-subplot legend (annotation)
        tkeo_ms = tkeo_mean_by_channel.get(str(ch))
        tkeo_abs = None
        if tkeo_ms is not None and np.isfinite(float(tkeo_ms)):
            tkeo_abs = float(platform_onset_abs) + float(tkeo_ms) * float(device_rate) / 1000.0

        # Resolve event positions in absolute device frames for window definitions.
        channel_event_frames: Dict[str, float] = {
            "platform_onset": float(platform_onset_abs),
            onset_col: float(platform_onset_abs),
            "step_onset": float(step_onset_abs),
        }
        if tkeo_abs is not None:
            channel_event_frames["TKEO_AGLR_emg_onset_timing"] = float(tkeo_abs)

        # Generic support: if the merged table contains other mocap-domain event columns referenced by windows,
        # interpret them as mocap frames and convert to absolute device frames.
        for maybe_event in event_cols:
            if maybe_event in channel_event_frames:
                continue
            if maybe_event in available:
                try:
                    raw = (
                        lf_trial.select(pl.col(maybe_event).max().alias("_v")).limit(1).collect()["_v"].item()
                    )
                except Exception:
                    raw = None
                if raw is not None:
                    try:
                        channel_event_frames[maybe_event] = float(platform_onset_abs) + (float(raw) - float(platform_onset_mocap)) * float(frame_ratio)
                    except Exception:
                        pass

        spans = _compute_window_spans_for_channel(
            windows_cfg=windows_cfg,
            channel_event_frames=channel_event_frames,
            reference_event=str(reference_event) if reference_event is not None else None,
            time_start_x=crop_start_x,
            time_end_x=crop_end_x,
            device_rate=device_rate,
            window_colors=window_colors,
        )

        # window spans as shapes
        for span in spans:
            fig.add_shape(
                type="rect",
                xref=xref,
                yref=yref_domain,
                x0=span.start_x,
                x1=span.end_x,
                y0=0,
                y1=1,
                fillcolor=span.color,
                opacity=window_alpha,
                line=dict(width=0),
                layer="below",
            )

        # vlines as shapes
        for event_name, xval in [
            ("platform_onset", float(platform_onset_abs)),
            ("step_onset", float(step_onset_abs)),
        ]:
            fig.add_shape(
                type="line",
                xref=xref,
                yref=yref_domain,
                x0=xval,
                x1=xval,
                y0=0,
                y1=1,
                line=dict(color=event_color_map.get(event_name, "#000000"), width=1.6, dash="dash"),
                layer="above",
            )

        if tkeo_abs is not None:
            fig.add_shape(
                type="line",
                xref=xref,
                yref=yref_domain,
                x0=float(tkeo_abs),
                x1=float(tkeo_abs),
                y0=0,
                y1=1,
                line=dict(color=event_color_map.get("TKEO_AGLR_emg_onset_timing", "#2ca02c"), width=1.6, dash="dash"),
                layer="above",
            )

        event_items = [
            ("platform_onset", event_color_map.get("platform_onset", "#1f77b4")),
            ("step_onset", event_color_map.get("step_onset", "#ff7f0e")),
        ]
        if tkeo_abs is not None:
            event_items.append((tkeo_label, event_color_map.get("TKEO_AGLR_emg_onset_timing", "#2ca02c")))

        legend_text = _legend_html(window_spans=spans, event_items=event_items, group_items=[])

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
        title=f"{args.mode} | emg | {first_subject} | v={first_velocity} | trial={first_trial} | absolute frames ({x_col_abs})",
        width=int(args.width),
        height=int(args.height),
        margin=dict(l=40, r=20, t=60, b=40),
        template="plotly_white",
        showlegend=False,
    )

    # x/y axis cosmetics
    fig.update_xaxes(range=[crop_start_x, crop_end_x], showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")

    out_html = Path(args.html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")

    out_png = Path(args.png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(out_png)
    except Exception as e:
        print(f"[warn] PNG export failed (kaleido needed?): {e}")

    print(f"Wrote: {out_html}")
    print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()
