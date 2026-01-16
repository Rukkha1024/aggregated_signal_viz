---
name: aggregated-signal-figure
description: Create and refactor figure-generating scripts in aggregated_signal_viz (script/vis_*.py) using Polars (no pandas). Use config.yaml for channel grid_layout (signal_groups.<group>.grid_layout) and summary max_cols (figure_layout.summary_plots.<plot_type>.max_cols). Save PNG (300dpi) under Path(output.base_dir)/<plot_type>/ and keep EMG muscle order from config.yaml signal_groups.emg.columns. Triggers on onset plot, boxplot, summary plot, grid plot, aggregated signal figure.
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

## Templates

- Summary grid template: `templates/summary_grid_template.py`
- Channel grid template: `templates/channel_grid_template.py`
- Matplotlib style template (copy into each vis script): `templates/mpl_style_template.py`
- Vis script skeleton (copy/modify per plot): `templates/vis_script_skeleton.py`

## Example PNGs (visual check)

- Summary plot example: `assets/examples/example_onset_summary.png`
- Channel-grid example: `assets/examples/example_emg_channel_grid.png`
