---
name: aggregated-signal-figure
description: Create and refactor figure-generating scripts in aggregated_signal_viz (script/vis_*.py) using Polars (no pandas). Use config.yaml for channel grid_layout (signal_groups.<group>.grid_layout) and summary max_cols (figure_layout.summary_plots.<plot_type>.max_cols). Save PNG (300dpi) under Path(output.base_dir)/<plot_type>/ and keep EMG muscle order from config.yaml signal_groups.emg.columns. Triggers on onset plot, boxplot, summary plot, grid plot, aggregated signal figure, legend dashed line not visible, step/nonstep legend dashed, event vline legend dashed, legend linestyle looks solid due to linewidth. Triggers on windows.definition legend mismatch, COP/COM window legend parity, and min-max vs range (duration) labels in ms.
---

# Aggregated Signal Figure Skill

## Rules (project-specific)

- Create one plot per script: `script/vis_<plotname>.py`.
- Use `polars` for all data IO/processing (do not use pandas).
- Keep visualization style parameters inside each script (top-level `VizConfig`/`CONSTANTS`).
- Read only the following from `config.yaml`:
  - data paths / join keys: `data.*`
  - EMG muscle/channel order: `signal_groups.emg.columns`
  - channel-grid layout: `signal_groups.<group>.grid_layout`
  - analysis windows: `windows.definitions` (when needed)
  - summary-grid layout: `figure_layout.summary_plots.<plot_type>.max_cols`
  - output base dir: `output.base_dir`
- Do not read `config.yaml: plot_style` in new `vis_*.py` scripts (keep style in the script).
- Always include a legend inside each subplot (use `ax.legend(...)`, not `fig.legend(...)`).

## Grid policy

- Channel plots (EMG/forceplate):
  - Build `rows, cols = config["signal_groups"][group]["grid_layout"]`.
  - Create `plt.subplots(rows, cols, ...)`, flatten axes, fill per channel order.
  - Hide unused subplots with `ax.axis("off")`.
  - Include subplot legend in each subplot.
- Summary plots (onset/boxplot/etc):
  - Build a panel list (e.g., facet values).
  - Read `max_cols = config["figure_layout"]["summary_plots"][plot_type]["max_cols"]`.
  - Compute `cols = min(max_cols, n_panels)` and `rows = ceil(n_panels / cols)`.
  - Create a grid and hide unused axes (same off-policy).
  - Include subplot legend for each panel when labels exist.

## Output policy

- Save under `Path(config["output"]["base_dir"]) / <plot_type> / <filename>.png`.
- Always save as `png` with `dpi=300`, `bbox_inches="tight"`, `facecolor="white"`.
- Use deterministic naming (include the main grouping/facet/hue conditions in filename).

## Refactor validation (MD5)

- Before refactoring an existing script, generate a reference output and record its MD5.
- After refactoring, rerun with the same inputs and compare MD5.
- If MD5 differs unexpectedly, treat it as a regression and fix (unless an intentional change is approved).

## Troubleshooting: Dashed Lines in Legends

- Symptom: Group lines (e.g., `step/nonstep`) look solid in the legend because the plot linewidth is thick and the legend sample is short.
- Root cause: Matplotlib legends reuse plot handles and can visually compress dash patterns at larger linewidths.
- Fix pattern: Keep the plot linewidth, but provide custom legend handles with a thinner legend-only linewidth cap (for readability).
- Reference implementation in this repo: search `script/visualizer.py` for `legend_group_linewidth`, `_build_group_legend_handles`, and `_style_timeseries_axis(group_handles=...)`.

- Related symptom: Event vline legend entries (`--`, `:`, `-.`) do not look dashed.
- Fix pattern: Build legend-only vline handles that map common linestyles to legend-friendly dash tuples and cap legend linewidth.
- Reference implementation in this repo: search `script/visualizer.py` for `_build_event_vline_legend_handles`.

## Troubleshooting: Window Definition Legend Labels (Range vs min-max)

- Symptom: `windows.definitions` legend entries differ across plots (EMG/forceplate vs COP/COM), or labels show `start-end` (min-max) instead of a range/duration.
- Root cause: Some plot functions do not pass `window_spans` into legend assembly (via `_style_timeseries_axis`/`_apply_window_group_legends`), and window labels are formatted from clamped `start_ms`/`end_ms` instead of duration.
- Fix pattern:
  - Legend inclusion: pass `window_spans=window_spans` for time-series axes so window handles are included consistently.
  - Label format: format `span["label"]` as a duration (e.g., `p1 (200 ms)`) when constructing window spans.
  - Configuration source: window boundaries live in `config.yaml` under `windows.definitions` and colors are typically set under `plot_style.common.window_colors`.
- Reference implementation in this repo: search `script/visualizer.py` for `_compute_window_spans`, `_apply_window_group_legends`, `_style_timeseries_axis`, `_plot_cop`, and `_plot_com`.

## Templates

- Summary grid template: `templates/summary_grid_template.py`
- Channel grid template: `templates/channel_grid_template.py`
- Matplotlib style template (copy into each vis script): `templates/mpl_style_template.py`
- Vis script skeleton (copy/modify per plot): `templates/vis_script_skeleton.py`

## Example PNGs (visual check)

- Summary plot example: `assets/examples/example_onset_summary.png`
- Channel-grid example: `assets/examples/example_emg_channel_grid.png`
