from __future__ import annotations

"""Process-safe Matplotlib plot task entrypoints.

This module provides top-level functions that can be pickled and executed in a
`concurrent.futures.ProcessPoolExecutor`. Heavy rendering helpers remain in
`src.plotting.matplotlib.common`, while this module owns dispatch and worker
bootstrap.
"""

from typing import Any, Dict, Optional

from . import common as _common
from .line import plot_emg, plot_forceplate
from .scatter import plot_com, plot_cop
from .shared import as_ndarray, as_optional_ndarray, as_output_path, as_tuple_key, as_tuple_keys


def plot_worker_init(font_family: Optional[str]) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # noqa: F401

    setattr(_common, "_PLOT_FONT_FAMILY", font_family)
    plt.rcParams["axes.unicode_minus"] = False
    if font_family:
        plt.rcParams["font.family"] = font_family


def plot_task(task: Dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    kind = task["kind"]
    common_style = task["common_style"]
    output_path = as_output_path(task["output_path"])
    event_vline_style = task.get("event_vline_style", {})
    event_vline_order = task.get("event_vline_order", [])

    if kind == "overlay":
        _common._plot_overlay_generic(
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
            sorted_keys=as_tuple_keys(task["sorted_keys"]),
            x=as_ndarray(task["x"]),
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
        _common._maybe_export_plotly_html(task, output_path)
        return

    if kind == "emg":
        plot_emg(
            aggregated=task["aggregated"],
            output_path=output_path,
            key=as_tuple_key(task["key"]),
            mode_name=task["mode_name"],
            group_fields=task["group_fields"],
            markers=task["markers"],
            event_vlines=task.get("event_vlines", []),
            event_vlines_by_channel=task.get("event_vlines_by_channel"),
            event_vline_style=event_vline_style,
            event_vline_order=event_vline_order,
            x=as_ndarray(task["x"]),
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
        _common._maybe_export_plotly_html(task, output_path)
        return

    if kind == "forceplate":
        plot_forceplate(
            aggregated=task["aggregated"],
            output_path=output_path,
            key=as_tuple_key(task["key"]),
            mode_name=task["mode_name"],
            group_fields=task["group_fields"],
            markers=task["markers"],
            event_vlines=task.get("event_vlines", []),
            event_vline_style=event_vline_style,
            event_vline_order=event_vline_order,
            x=as_ndarray(task["x"]),
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
        _common._maybe_export_plotly_html(task, output_path)
        return

    if kind == "cop":
        plot_cop(
            aggregated=task["aggregated"],
            output_path=output_path,
            key=as_tuple_key(task["key"]),
            mode_name=task["mode_name"],
            group_fields=task["group_fields"],
            markers=task["markers"],
            event_vlines=task.get("event_vlines", []),
            event_vline_style=event_vline_style,
            event_vline_order=event_vline_order,
            x_axis=as_optional_ndarray(task.get("x_axis")),
            target_axis=as_optional_ndarray(task.get("target_axis")),
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
        _common._maybe_export_plotly_html(task, output_path)
        return

    if kind == "com":
        plot_com(
            aggregated=task["aggregated"],
            output_path=output_path,
            key=as_tuple_key(task["key"]),
            mode_name=task["mode_name"],
            group_fields=task["group_fields"],
            markers=task["markers"],
            event_vlines=task.get("event_vlines", []),
            event_vline_style=event_vline_style,
            event_vline_order=event_vline_order,
            x_axis=as_optional_ndarray(task.get("x_axis")),
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
        _common._maybe_export_plotly_html(task, output_path)
        return

    plt.close("all")
    raise ValueError(f"Unknown plot task kind: {kind!r}")


__all__ = ["plot_task", "plot_worker_init"]
