from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl
import yaml
from scipy import stats


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_path(base_dir: Path, maybe_path: str) -> Path:
    path = Path(maybe_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out


def _moving_average_nan(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y.astype(float, copy=False)
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float)
    pad = window // 2

    y = y.astype(float, copy=False)
    valid = ~np.isnan(y)
    y_filled = np.where(valid, y, 0.0)
    w = valid.astype(float)

    y_pad = np.pad(y_filled, (pad, pad), mode="edge")
    w_pad = np.pad(w, (pad, pad), mode="edge")

    y_sum = np.convolve(y_pad, kernel, mode="valid")
    w_sum = np.convolve(w_pad, kernel, mode="valid")
    out = np.full_like(y_sum, np.nan, dtype=float)
    ok = w_sum > 0
    out[ok] = y_sum[ok] / w_sum[ok]
    return out


def _find_zero_crossings(time_ms: np.ndarray, y: np.ndarray) -> List[float]:
    crossings: List[float] = []
    if time_ms.size < 2:
        return crossings
    for i in range(time_ms.size - 1):
        t1 = float(time_ms[i])
        t2 = float(time_ms[i + 1])
        y1 = float(y[i])
        y2 = float(y[i + 1])
        if any(math.isnan(v) for v in (t1, t2, y1, y2)):
            continue
        if y1 == 0.0 or y2 == 0.0:
            continue
        if y1 * y2 > 0.0:
            continue
        denom = (y2 - y1)
        if denom == 0.0:
            continue
        t0 = t1 + (0.0 - y1) * (t2 - t1) / denom
        crossings.append(float(t0))
    return crossings


def _sign_segments(
    time_ms: np.ndarray,
    y: np.ndarray,
    *,
    eps: float,
    min_len_ms: int,
) -> List[Tuple[int, int, int]]:
    """
    Returns list of (sign, start_ms, end_ms) for segments with constant sign and length>=min_len_ms.
    sign: +1 (nonstep>step), -1 (step>nonstep)
    """
    if time_ms.size == 0:
        return []
    t = time_ms.astype(int, copy=False)
    y = y.astype(float, copy=False)

    def to_sign(val: float) -> int:
        if math.isnan(val):
            return 0
        if val > eps:
            return 1
        if val < -eps:
            return -1
        return 0

    s = np.array([to_sign(v) for v in y], dtype=int)
    segments: List[Tuple[int, int, int]] = []

    cur_sign = 0
    seg_start: Optional[int] = None
    seg_end: Optional[int] = None
    prev_t: Optional[int] = None

    def close_segment() -> None:
        nonlocal cur_sign, seg_start, seg_end
        if cur_sign == 0 or seg_start is None or seg_end is None:
            cur_sign = 0
            seg_start = None
            seg_end = None
            return
        length = seg_end - seg_start + 1
        if length >= min_len_ms:
            segments.append((int(cur_sign), int(seg_start), int(seg_end)))
        cur_sign = 0
        seg_start = None
        seg_end = None

    for ti, si in zip(t.tolist(), s.tolist()):
        if prev_t is not None and ti != prev_t + 1:
            close_segment()
        if si == 0:
            close_segment()
            prev_t = ti
            continue

        if cur_sign == 0:
            cur_sign = si
            seg_start = ti
            seg_end = ti
            prev_t = ti
            continue

        if si != cur_sign:
            close_segment()
            cur_sign = si
            seg_start = ti
            seg_end = ti
            prev_t = ti
            continue

        seg_end = ti
        prev_t = ti

    close_segment()
    return segments


@dataclass(frozen=True)
class CrossingSummary:
    subject: str
    velocity: float
    axis: str
    primary_crossing_ms: Optional[float]
    crossings_ms: List[float]
    n_pos_ms: int
    n_neg_ms: int


def _build_phase_table_for_axis(
    diff_df: pl.DataFrame,
    *,
    axis: str,
    time_col: str,
    subject_col: str,
    velocity_col: str,
    smoothing_ms: int,
    eps: float,
    min_segment_ms: int,
    primary_crossing_policy: str,
    crossing_window_ms: int,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    required = {subject_col, velocity_col, time_col, "diff"}
    missing = [c for c in required if c not in diff_df.columns]
    if missing:
        raise ValueError(f"Missing columns in diff_df: {missing}")

    phase_rows: List[Dict[str, Any]] = []
    summaries: List[CrossingSummary] = []

    for (subject, velocity), subdf in diff_df.group_by([subject_col, velocity_col], maintain_order=False):
        sdf = subdf.sort(time_col)
        time_ms = sdf[time_col].to_numpy()
        y = sdf["diff"].to_numpy()

        y_smooth = _moving_average_nan(y, int(max(1, smoothing_ms)))
        segments = _sign_segments(time_ms, y_smooth, eps=eps, min_len_ms=int(min_segment_ms))

        # Build A phases from segments
        pos_count = 0
        neg_count = 0
        for seg_sign, start_ms, end_ms in segments:
            phase = "A_nonstep_gt_step" if seg_sign > 0 else "A_step_gt_nonstep"
            for t in range(start_ms, end_ms + 1):
                phase_rows.append(
                    {
                        subject_col: subject,
                        velocity_col: velocity,
                        time_col: int(t),
                        "axis": axis,
                        "phase": phase,
                    }
                )
            if seg_sign > 0:
                pos_count += (end_ms - start_ms + 1)
            else:
                neg_count += (end_ms - start_ms + 1)

        # Crossings computed on the smoothed diff curve
        crossings = _find_zero_crossings(time_ms.astype(float, copy=False), y_smooth)
        crossings = sorted(dict.fromkeys(round(x, 6) for x in crossings))

        primary: Optional[float] = None
        if crossings:
            if primary_crossing_policy == "first_after_onset":
                primary = next((c for c in crossings if c >= 0.0), None)
            else:
                primary = crossings[0]

        # Build B phases around the primary crossing
        if primary is not None and not math.isnan(primary):
            pre_start = int(math.floor(primary - crossing_window_ms))
            pre_end = int(math.floor(primary - 1))
            post_start = int(math.ceil(primary))
            post_end = int(math.ceil(primary + crossing_window_ms - 1))

            if pre_end >= pre_start:
                for t in range(pre_start, pre_end + 1):
                    phase_rows.append(
                        {
                            subject_col: subject,
                            velocity_col: velocity,
                            time_col: int(t),
                            "axis": axis,
                            "phase": "B_pre_cross",
                        }
                    )
            if post_end >= post_start:
                for t in range(post_start, post_end + 1):
                    phase_rows.append(
                        {
                            subject_col: subject,
                            velocity_col: velocity,
                            time_col: int(t),
                            "axis": axis,
                            "phase": "B_post_cross",
                        }
                    )

        summaries.append(
            CrossingSummary(
                subject=str(subject),
                velocity=float(velocity),
                axis=axis,
                primary_crossing_ms=(None if primary is None else float(primary)),
                crossings_ms=[float(c) for c in crossings],
                n_pos_ms=int(pos_count),
                n_neg_ms=int(neg_count),
            )
        )

    phase_df = pl.DataFrame(phase_rows) if phase_rows else pl.DataFrame(
        schema={subject_col: pl.Utf8, velocity_col: pl.Float64, time_col: pl.Int32, "axis": pl.Utf8, "phase": pl.Utf8}
    )

    summary_df = pl.DataFrame(
        [
            {
                "subject": s.subject,
                "velocity": s.velocity,
                "axis": s.axis,
                "primary_crossing_ms": s.primary_crossing_ms,
                "crossings_ms": json.dumps(s.crossings_ms, ensure_ascii=False),
                "n_pos_ms": s.n_pos_ms,
                "n_neg_ms": s.n_neg_ms,
            }
            for s in summaries
        ]
    ).sort(["axis", "subject", "velocity"])

    return phase_df, summary_df


def _plot_crossing_hist(out_dir: Path, summary_df: pl.DataFrame, *, axis: str) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return None

    vals = summary_df.filter(pl.col("axis") == axis)["primary_crossing_ms"].drop_nulls().to_numpy()
    if vals.size == 0:
        return None

    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.hist(vals, bins=20, color="#4E79A7", alpha=0.85, edgecolor="white")
    ax.set_title(f"Primary CoP Crossing Time ({axis})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Crossing time (ms, relative to onset=0)")
    ax.set_ylabel("Count (subject-velocity pairs)")
    ax.grid(True, alpha=0.25)
    out_path = out_dir / f"crossing_hist_{axis}.png"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def _plot_effect_heatmap(
    out_dir: Path,
    stats_df: pl.DataFrame,
    *,
    axis: str,
    metric: str,
    muscles: Sequence[str],
) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return None

    phases = ["A_nonstep_gt_step", "A_step_gt_nonstep", "B_pre_cross", "B_post_cross"]
    sub = stats_df.filter((pl.col("axis") == axis) & (pl.col("metric") == metric))
    if sub.is_empty():
        return None

    # Build dz matrix (phase x muscle)
    mat = np.full((len(phases), len(muscles)), np.nan, dtype=float)
    qmat = np.full((len(phases), len(muscles)), np.nan, dtype=float)
    lookup = {
        (r["phase"], r["muscle"]): (r["dz"], r["q"])
        for r in sub.select(["phase", "muscle", "dz", "q"]).iter_rows(named=True)
    }
    for i, ph in enumerate(phases):
        for j, m in enumerate(muscles):
            val = lookup.get((ph, m))
            if val is None:
                continue
            mat[i, j] = float(val[0]) if val[0] is not None else np.nan
            qmat[i, j] = float(val[1]) if val[1] is not None else np.nan

    fig, ax = plt.subplots(figsize=(max(10, 0.6 * len(muscles)), 3.6), dpi=300)
    vmax = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1.0
    vmax = max(0.5, float(vmax))
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title(f"EMG Effect Size (dz) | {axis} | {metric}", fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(phases)))
    ax.set_yticklabels(phases, fontsize=8)
    ax.set_xticks(range(len(muscles)))
    ax.set_xticklabels(muscles, rotation=45, ha="right", fontsize=8)

    # Mark significance (q<=0.05) with a dot
    for i in range(len(phases)):
        for j in range(len(muscles)):
            q = qmat[i, j]
            if np.isfinite(q) and q <= 0.05:
                ax.text(j, i, "•", ha="center", va="center", color="black", fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("dz (nonstep - step)", rotation=90)
    ax.set_xlabel("EMG muscle/channel")
    ax.set_ylabel("Phase")
    fig.tight_layout()
    out_path = out_dir / f"effect_heatmap_{axis}_{metric}.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def _compute_stats(
    paired_df: pl.DataFrame,
    *,
    axes: Sequence[str],
    phases: Sequence[str],
    metrics: Sequence[str],
    muscles: Sequence[str],
) -> pl.DataFrame:
    rows: List[Dict[str, Any]] = []

    for axis in axes:
        for phase in phases:
            sub = paired_df.filter((pl.col("axis") == axis) & (pl.col("phase") == phase))
            n_pairs = int(sub.height)
            if n_pairs < 3:
                continue
            for metric in metrics:
                pvals: List[float] = []
                row_idxs: List[int] = []
                for muscle in muscles:
                    col = f"{metric}_{muscle}"
                    col_step = f"{col}_step"
                    if col not in sub.columns or col_step not in sub.columns:
                        continue

                    non_vals = sub[col].to_numpy()
                    step_vals = sub[col_step].to_numpy()
                    diff = non_vals - step_vals
                    diff = diff[~np.isnan(diff)]
                    if diff.size < 3:
                        continue

                    t, p = stats.ttest_1samp(diff, popmean=0.0, nan_policy="omit")
                    mean_diff = float(np.nanmean(diff))
                    sd_diff = float(np.nanstd(diff, ddof=1)) if diff.size > 1 else float("nan")
                    dz = mean_diff / sd_diff if sd_diff and not math.isnan(sd_diff) else float("nan")

                    rows.append(
                        {
                            "axis": axis,
                            "phase": phase,
                            "metric": metric,
                            "muscle": muscle,
                            "n_pairs": int(n_pairs),
                            "mean_nonstep": float(np.nanmean(non_vals)),
                            "mean_step": float(np.nanmean(step_vals)),
                            "mean_diff": mean_diff,
                            "t": float(t),
                            "p": float(p),
                            "dz": float(dz),
                        }
                    )
                    pvals.append(float(p))
                    row_idxs.append(len(rows) - 1)

                # FDR per (axis, phase, metric)
                if pvals:
                    qvals = _bh_fdr(np.asarray(pvals, dtype=float))
                    for idx, q in zip(row_idxs, qvals):
                        rows[idx]["q"] = float(q)

    out = pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={
            "axis": pl.Utf8,
            "phase": pl.Utf8,
            "metric": pl.Utf8,
            "muscle": pl.Utf8,
            "n_pairs": pl.Int64,
            "mean_nonstep": pl.Float64,
            "mean_step": pl.Float64,
            "mean_diff": pl.Float64,
            "t": pl.Float64,
            "p": pl.Float64,
            "dz": pl.Float64,
            "q": pl.Float64,
        }
    )
    if out.is_empty():
        return out
    # Ensure q exists for all rows (fill with null if missing)
    if "q" not in out.columns:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("q"))
    return out.sort(["axis", "metric", "phase", "q", "muscle"])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze EMG differences using CoP(step/nonstep) crossing-based windows.")
    p.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    p.add_argument("--no-plots", action="store_true", help="Disable plotting (png outputs).")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config_path = Path(args.config)
    cfg = _load_config(config_path)
    base_dir = config_path.parent.resolve()

    analysis_cfg = cfg.get("cop_crossing_emg_analysis", {})
    if not analysis_cfg.get("enabled", True):
        raise SystemExit("cop_crossing_emg_analysis.enabled is false")

    # Core config
    data_cfg = cfg["data"]
    id_cfg = data_cfg["id_columns"]
    subject_col = str(id_cfg["subject"])
    velocity_col = str(id_cfg["velocity"])
    trial_col = str(id_cfg["trial"])
    frame_col = str(id_cfg["frame"])
    mocap_col = str(id_cfg["mocap_frame"])
    onset_col = str(id_cfg["onset"])

    step_col = "step_TF"
    if step_col not in pl.scan_parquet(str(_resolve_path(base_dir, data_cfg["input_file"]))).collect_schema():
        raise ValueError("Expected 'step_TF' column in merged.parquet.")

    device_rate = float(data_cfg.get("device_sample_rate", 1000))
    mocap_rate = float(data_cfg.get("mocap_sample_rate", 100))
    frame_ratio = int(data_cfg.get("frame_ratio") or int(device_rate / mocap_rate))
    dt_ms = 1000.0 / device_rate

    # Time window
    interp_cfg = cfg.get("interpolation", {})
    win_cfg = analysis_cfg.get("time_window_ms", {}) or {}
    start_ms = win_cfg.get("start_ms")
    end_ms = win_cfg.get("end_ms")
    if start_ms is None:
        start_ms = float(interp_cfg.get("start_ms", -100))
    if end_ms is None:
        end_ms = float(interp_cfg.get("end_ms", 800))
    start_ms = float(start_ms)
    end_ms = float(end_ms)

    # Analysis settings
    cop_cols = list(analysis_cfg.get("cop_columns") or [])
    if not cop_cols:
        raise ValueError("cop_crossing_emg_analysis.cop_columns is empty")

    crossing_cfg = analysis_cfg.get("crossing", {}) or {}
    smoothing_ms = int(crossing_cfg.get("smoothing_ms", 11))
    eps = float(crossing_cfg.get("diff_epsilon", 0.0))
    min_segment_ms = int(crossing_cfg.get("min_segment_ms", 10))
    primary_policy = str(crossing_cfg.get("primary_crossing_policy", "first_after_onset"))
    crossing_window_ms = int(analysis_cfg.get("crossing_window_ms", 50))

    metrics = list(analysis_cfg.get("metrics") or ["mean", "iemg", "peak"])
    allowed_metrics = {"mean", "iemg", "peak"}
    unknown = [m for m in metrics if m not in allowed_metrics]
    if unknown:
        raise ValueError(f"Unknown metrics in config: {unknown}. Allowed: {sorted(allowed_metrics)}")

    out_base = _resolve_path(base_dir, cfg["output"]["base_dir"])
    out_dir = out_base / str(analysis_cfg.get("output_subdir", "cop_crossing_emg"))
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = _resolve_path(base_dir, str(analysis_cfg.get("report_path", "report/cop_crossing_emg_report.md")))
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # EMG channels (order preserved from config)
    muscles = list(cfg["signal_groups"]["emg"]["columns"])
    group_cols = [subject_col, velocity_col, trial_col]

    # Build base lazyframe: align time to onset, filter, select needed columns
    input_path = _resolve_path(base_dir, data_cfg["input_file"])
    lf = pl.scan_parquet(str(input_path))

    task_col = id_cfg.get("task")
    task_filter = data_cfg.get("task_filter")
    if task_filter and task_col and task_col in lf.collect_schema():
        lf = lf.filter(pl.col(str(task_col)) == task_filter)

    filter_cfg = analysis_cfg.get("filter", {}) or {}
    for col, val in filter_cfg.items():
        if col not in lf.collect_schema():
            raise ValueError(f"Filter column '{col}' not found in input parquet.")
        lf = lf.filter(pl.col(col) == val)

    mocap_start = pl.col(mocap_col).min().over(group_cols)
    onset_device = (pl.col(onset_col).first().over(group_cols) - mocap_start) * frame_ratio
    aligned_frame = (pl.col(frame_col) - onset_device).alias("aligned_frame")
    time_ms = (aligned_frame * (1000.0 / device_rate)).round(0).cast(pl.Int32).alias("time_ms")

    needed_cols = (
        [subject_col, velocity_col, trial_col, step_col]
        + ["time_ms"]
        + cop_cols
        + muscles
    )
    available = set(lf.collect_schema().names())
    missing_needed = [c for c in needed_cols if c not in available and c != "time_ms"]
    if missing_needed:
        raise ValueError(f"Missing required columns in input parquet: {missing_needed}")

    lf_base = (
        lf.with_columns([aligned_frame, time_ms])
        .filter((pl.col("time_ms") >= int(math.floor(start_ms))) & (pl.col("time_ms") <= int(math.ceil(end_ms))))
        .select([pl.col(c) for c in needed_cols if c in available or c == "time_ms"])
    )

    # Build mean CoP curves per subject-velocity-step_TF-time_ms
    cop_mean_exprs = [pl.mean(c).alias(c) for c in cop_cols]
    cop_mean = (
        lf_base.group_by([subject_col, velocity_col, step_col, "time_ms"])
        .agg(cop_mean_exprs)
        .collect()
        .sort([subject_col, velocity_col, step_col, "time_ms"])
    )

    # Build phase table across axes
    phase_tables: List[pl.DataFrame] = []
    summary_tables: List[pl.DataFrame] = []

    for axis in cop_cols:
        wide = (
            cop_mean.select([subject_col, velocity_col, "time_ms", step_col, axis])
            .pivot(index=[subject_col, velocity_col, "time_ms"], on=step_col, values=axis)
            .sort([subject_col, velocity_col, "time_ms"])
        )
        if "nonstep" not in wide.columns or "step" not in wide.columns:
            raise ValueError(f"Missing step/nonstep columns after pivot for axis '{axis}'.")
        diff_df = wide.with_columns((pl.col("nonstep") - pl.col("step")).alias("diff")).select(
            [subject_col, velocity_col, "time_ms", "diff"]
        )

        phase_df, summary_df = _build_phase_table_for_axis(
            diff_df,
            axis=axis,
            time_col="time_ms",
            subject_col=subject_col,
            velocity_col=velocity_col,
            smoothing_ms=smoothing_ms,
            eps=eps,
            min_segment_ms=min_segment_ms,
            primary_crossing_policy=primary_policy,
            crossing_window_ms=crossing_window_ms,
        )
        phase_tables.append(phase_df)
        summary_tables.append(summary_df)

    phase_table = pl.concat(phase_tables, how="vertical") if phase_tables else pl.DataFrame()
    crossing_summary = pl.concat(summary_tables, how="vertical") if summary_tables else pl.DataFrame()

    # Limit phase table to analysis window
    if not phase_table.is_empty():
        phase_table = phase_table.filter(
            (pl.col("time_ms") >= int(math.floor(start_ms))) & (pl.col("time_ms") <= int(math.ceil(end_ms)))
        )

    # Join phases onto base rows
    lf_labeled = lf_base.join(
        phase_table.lazy(),
        on=[subject_col, velocity_col, "time_ms"],
        how="inner",
    )

    # Per-trial metrics per axis/phase
    metric_exprs: List[pl.Expr] = [pl.len().alias("n_samples")]
    for m in muscles:
        if "mean" in metrics:
            metric_exprs.append(pl.mean(m).alias(f"mean_{m}"))
        if "iemg" in metrics:
            metric_exprs.append((pl.sum(m) * dt_ms).alias(f"iemg_{m}"))
        if "peak" in metrics:
            metric_exprs.append(pl.max(m).alias(f"peak_{m}"))

    trial_metrics = (
        lf_labeled.group_by(["axis", "phase", subject_col, velocity_col, trial_col, step_col])
        .agg(metric_exprs)
        .collect()
        .sort(["axis", "phase", subject_col, velocity_col, trial_col, step_col])
    )

    # Subject-velocity aggregation (minimum comparison unit)
    metric_cols = [c for c in trial_metrics.columns if c.startswith(("mean_", "iemg_", "peak_"))]
    sv_metrics = (
        trial_metrics.group_by(["axis", "phase", subject_col, velocity_col, step_col])
        .agg([pl.mean(c).alias(c) for c in metric_cols] + [pl.len().alias("n_trials")])
        .sort(["axis", "phase", subject_col, velocity_col, step_col])
    )

    non = sv_metrics.filter(pl.col(step_col) == "nonstep").drop(step_col)
    step = sv_metrics.filter(pl.col(step_col) == "step").drop(step_col)
    paired = non.join(step, on=["axis", "phase", subject_col, velocity_col], how="inner", suffix="_step")

    # Stats across subject-velocity pairs
    phases = ["A_nonstep_gt_step", "A_step_gt_nonstep", "B_pre_cross", "B_post_cross"]
    axes = cop_cols
    stats_df = _compute_stats(
        paired,
        axes=axes,
        phases=phases,
        metrics=metrics,
        muscles=muscles,
    )

    # Save outputs
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    run_meta = {
        "timestamp": now,
        "config_path": str(config_path),
        "input_path": str(input_path),
        "filter": filter_cfg,
        "cop_columns": cop_cols,
        "time_window_ms": {"start_ms": start_ms, "end_ms": end_ms},
        "crossing": {
            "smoothing_ms": smoothing_ms,
            "diff_epsilon": eps,
            "min_segment_ms": min_segment_ms,
            "primary_crossing_policy": primary_policy,
            "crossing_window_ms": crossing_window_ms,
        },
        "metrics": metrics,
        "n_trials_total": int(
            lf_base.select(pl.struct([subject_col, velocity_col, trial_col]).n_unique()).collect().item()
        ),
        "n_subject_velocity_pairs": int(paired.select(pl.struct([subject_col, velocity_col]).n_unique()).item()),
    }

    (out_dir / "run_metadata.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    crossing_summary.write_csv(out_dir / "crossing_summary.csv")
    trial_metrics.write_parquet(out_dir / "trial_metrics.parquet")
    sv_metrics.write_parquet(out_dir / "sv_metrics.parquet")
    stats_df.write_csv(out_dir / "emg_stats.csv")

    if not args.no_plots:
        for axis in axes:
            _plot_crossing_hist(out_dir, crossing_summary, axis=axis)
        for axis in axes:
            for metric in metrics:
                _plot_effect_heatmap(out_dir, stats_df, axis=axis, metric=metric, muscles=muscles)

    # Minimal console summary
    print(f"[done] outputs saved under: {out_dir}")
    if not stats_df.is_empty():
        sig = stats_df.filter(pl.col("q").is_not_null() & (pl.col("q") <= 0.05)).sort(["q", "axis", "phase", "metric"])
        print(f"[stats] significant tests (q<=0.05): {sig.height}")
        if sig.height:
            print(sig.select(["axis", "phase", "metric", "muscle", "mean_diff", "dz", "q"]).head(30))
    else:
        print("[stats] no statistics computed (insufficient pairs or missing data)")

    # Touch report path placeholder (report is written in a separate step)
    if not report_path.exists():
        report_path.write_text(
            "# CoP crossing-based EMG report\n\nRun the analyzer to populate results.\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
