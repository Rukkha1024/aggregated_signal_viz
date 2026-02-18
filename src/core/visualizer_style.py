from __future__ import annotations

from typing import Any, Dict

from ..plotting.matplotlib.common import _parse_group_linestyles


class VisualizerStyleMixin:
    # All plot style parameters are configured via config.yaml under `plot_style`.
    # Module-level style constants were removed; do not reintroduce hard-coded styles.
    def _build_common_style(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "dpi": cfg["dpi"],
            "grid_alpha": cfg["grid_alpha"],
            "tick_labelsize": cfg["tick_labelsize"],
            "title_fontsize": cfg["title_fontsize"],
            "title_fontweight": cfg["title_fontweight"],
            "title_pad": cfg["title_pad"],
            "label_fontsize": cfg["label_fontsize"],
            "legend_loc": cfg["legend_loc"],
            "legend_framealpha": cfg["legend_framealpha"],
            "tight_layout_rect": cfg["tight_layout_rect"],
            "savefig_bbox_inches": cfg["savefig_bbox_inches"],
            "savefig_facecolor": cfg["savefig_facecolor"],
            "font_family": cfg["font_family"],
            "show_suptitle": bool(cfg.get("show_suptitle", True)),
            "show_subplot_titles": bool(cfg.get("show_subplot_titles", True)),
            "show_grid": bool(cfg.get("show_grid", True)),
            "show_legend": bool(cfg.get("show_legend", True)),
            "show_xlabel": bool(cfg.get("show_xlabel", True)),
            "show_ylabel": bool(cfg.get("show_ylabel", True)),
            "show_xtick_labels": bool(cfg.get("show_xtick_labels", True)),
            "show_ytick_labels": bool(cfg.get("show_ytick_labels", True)),
            "show_event_vlines": bool(cfg.get("show_event_vlines", True)),
            "show_windows": bool(cfg.get("show_windows", True)),
            "show_max_marker": cfg["show_max_marker"],
            "use_group_colors": bool(cfg.get("use_group_colors", False)),
            "group_linestyles": _parse_group_linestyles(cfg.get("group_linestyles")),
        }

    def _build_emg_style(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "subplot_size": tuple(cfg["subplot_size"]),
            "line_color": cfg["line_color"],
            "line_width": cfg["line_width"],
            "line_alpha": cfg["line_alpha"],
            "window_span_alpha": cfg["window_span_alpha"],
            "max_marker_color": cfg["max_marker_color"],
            "max_marker_linestyle": cfg["max_marker_linestyle"],
            "max_marker_linewidth": cfg["max_marker_linewidth"],
            "legend_fontsize": cfg["legend_fontsize"],
            "x_label": cfg["x_label"],
            "y_label": cfg["y_label"],
            "max_marker": {
                "color": cfg["max_marker_color"],
                "linestyle": cfg["max_marker_linestyle"],
                "linewidth": cfg["max_marker_linewidth"],
            },
        }

    def _build_forceplate_style(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "subplot_size": tuple(cfg["subplot_size"]),
            "line_colors": cfg["line_colors"],
            "axis_labels": dict(cfg.get("axis_labels", {})),
            "line_width": cfg["line_width"],
            "line_alpha": cfg["line_alpha"],
            "window_span_alpha": cfg["window_span_alpha"],
            "legend_fontsize": cfg["legend_fontsize"],
            "x_label": cfg["x_label"],
            "y_label": cfg["y_label"],
        }

    def _build_cop_style(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        out = {
            "subplot_size": tuple(cfg["subplot_size"]),
            "scatter_size": cfg["scatter_size"],
            "scatter_alpha": cfg["scatter_alpha"],
            "background_color": cfg["background_color"],
            "background_alpha": cfg["background_alpha"],
            "background_size": cfg["background_size"],
            "window_span_alpha": cfg.get("window_span_alpha", 0.15),
            "line_colors": cfg.get("line_colors", {"Cx": "gray", "Cy": "gray"}),
            "line_width": cfg.get("line_width", 0.8),
            "line_alpha": cfg.get("line_alpha", 0.8),
            "max_marker_color": cfg["max_marker_color"],
            "max_marker_size": cfg["max_marker_size"],
            "max_marker_symbol": cfg["max_marker_symbol"],
            "max_marker_edgecolor": cfg["max_marker_edgecolor"],
            "max_marker_linewidth": cfg["max_marker_linewidth"],
            "max_marker_zorder": cfg["max_marker_zorder"],
            "legend_fontsize": cfg["legend_fontsize"],
            "x_label_time": cfg.get("x_label_time", "Normalized time (0-1)"),
            "y_label_cx": cfg.get("y_label_cx", "Cx"),
            "y_label_cy": cfg.get("y_label_cy", "Cy"),
            "x_label": cfg["x_label"],
            "y_label": cfg["y_label"],
            "y_invert": bool(cfg["y_invert"]),
            "max_marker": {
                "size": cfg["max_marker_size"],
                "marker": cfg["max_marker_symbol"],
                "color": cfg["max_marker_color"],
                "edgecolor": cfg["max_marker_edgecolor"],
                "linewidth": cfg["max_marker_linewidth"],
                "zorder": cfg["max_marker_zorder"],
            },
            "overlay_scatter_edgewidth": cfg.get("overlay_scatter_edgewidth", 0.6),
        }
        return out

    @staticmethod
    def _build_com_style(com_cfg: Any, cop_style: Dict[str, Any]) -> Dict[str, Any]:
        base_line_colors = dict(cop_style.get("line_colors", {}) or {})
        comx_color = base_line_colors.get("Cx", "gray")
        comy_color = base_line_colors.get("Cy", "gray")
        comz_color = base_line_colors.get("Cz", "gray")

        if not isinstance(com_cfg, dict):
            com_cfg = {}

        def _cop_to_com(label: Any, fallback: str) -> str:
            if label is None:
                return fallback
            text = str(label)
            if not text:
                return fallback
            return text.replace("COP", "COM")

        default_y_label_comx = _cop_to_com(cop_style.get("y_label_cx", cop_style.get("x_label")), "COMx")
        default_y_label_comy = _cop_to_com(cop_style.get("y_label_cy", cop_style.get("y_label")), "COMy")
        default_x_label = _cop_to_com(cop_style.get("x_label"), "COMx")
        default_y_label = _cop_to_com(cop_style.get("y_label"), "COMy")

        def _cfg_or_default(key: str, default: str) -> str:
            val = com_cfg.get(key)
            if val is None:
                return default
            text = str(val)
            return text if text else default

        out = dict(cop_style)
        out["line_colors"] = {
            "COMx": comx_color,
            "COMx_zero": comx_color,
            "COMy": comy_color,
            "COMy_zero": comy_color,
            "COMz": comz_color,
            "COMz_zero": comz_color,
        }
        out["y_label_comx"] = _cfg_or_default("y_label_comx", default_y_label_comx)
        out["y_label_comy"] = _cfg_or_default("y_label_comy", default_y_label_comy)
        out["y_label_comz"] = _cfg_or_default("y_label_comz", "COMz")
        out["x_label"] = _cfg_or_default("x_label", default_x_label)
        out["y_label"] = _cfg_or_default("y_label", default_y_label)
        out.setdefault("x_label_time", "Normalized time (0-1)")
        out.setdefault("y_invert", False)
        return out

