---
name: aggregated-signal-figure
description: >-
  Create/refactor figure scripts in aggregated_signal_viz using Polars and
  config-driven layouts/outputs. Triggers: figure grids/summary plots, legend
  readability, and windows.definitions legend labels (COP/COM parity; duration
  vs min-max).
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

## X-axis normalization policy

- The x-axis must always be normalized to **0–100 %** of the minimum analysis unit (window/segment duration).
- Raw frame or time values must never appear on the x-axis; always display percent-normalized ticks (0, 20, 40, 60, 80, 100).

## Output policy

- Save under `Path(config["output"]["base_dir"]) / <plot_type> / <filename>.png`.
- Always save as `png` with `dpi=300`, `bbox_inches="tight"`, `facecolor="white"`.
- Use deterministic naming (include the main grouping/facet/hue conditions in filename).

## Refactor validation (MD5)

- Before refactoring an existing script, generate a reference output and record its MD5.
- After refactoring, rerun with the same inputs and compare MD5.
- If MD5 differs unexpectedly, treat it as a regression and fix (unless an intentional change is approved).

## Troubleshooting Notes

- Symptom: Legends are inconsistent (dash styles look solid, or `windows.definitions` labels/parity differ across plots).
- Root cause: Legends reuse plot handles; some plot functions omit `window_spans`; window labels may be formatted as start-end (min-max).
- Fix pattern: Use custom legend handles with a legend-only linewidth cap; always pass `window_spans` into `_style_timeseries_axis`/`_apply_window_group_legends`; format window labels as duration (e.g., `p1 (200 ms)`), with boundaries in `config.yaml: windows.definitions`.
- Reference implementation in this repo: `script/visualizer.py` (`legend_group_linewidth`, `_build_group_legend_handles`, `_build_event_vline_legend_handles`, `_compute_window_spans`, `_apply_window_group_legends`, `_style_timeseries_axis`, `_plot_cop`, `_plot_com`) and `config.yaml` (`windows.definitions`).

- Symptom: `windows.reference_event: 0`인데 x축 tick 라벨에 `0`이 안 보임.
- Root cause: xticks가 윈도우 경계/끝점 기반으로만 잡혀서, 시간 0이 자동 포함되지 않을 수 있음(또는 `0`을 이벤트 컬럼명으로 오해).
- Fix pattern: `script/visualizer.py`에서 0 tick을 강제로 포함(`_ensure_time_zero_xtick`), `reference_event=0`은 “shift 없음”으로 처리.
- Reference implementation in this repo: `script/visualizer.py`, `script/check_zero_tick.py`.

- Symptom: Figure script마다 x tick 간격/격자 투명도 같은 axis style을 매번 수동 수정해야 함.
- Root cause: 축 스타일 우선순위가 불명확하거나 하드코딩되어 `RULES`와 `config.yaml` fallback이 분리됨.
- Fix pattern: Precedence는 script-level `RULES`가 first override. 해당 값이 `None`일 때만 `config.yaml`의 `plot_style.common.*`를 fallback으로 사용.
- Reference (generic): Figure scripts에서는 공통으로 `RULES -> (if None) plot_style.common.*` 패턴을 유지.

## Templates

- Summary grid template: `templates/summary_grid_template.py`
- Channel grid template: `templates/channel_grid_template.py`
- Matplotlib style template (copy into each vis script): `templates/mpl_style_template.py`
- Vis script skeleton (copy/modify per plot): `templates/vis_script_skeleton.py`

## Example PNGs (visual check)

- Summary plot example: `assets/examples/example_onset_summary.png`
- Channel-grid example: `assets/examples/example_emg_channel_grid.png`

## Piecewise x-axis normalization
- The plot x-axis is split into three segments, each independently normalized: [onset − pad, onset], [onset, offset], [offset, offset + pad].
- Output data is never modified; only the display x-axis is warped for visualization.
- This aligns onset and offset across subjects/trials even when perturbation durations differ.

## Context: Subject × Variable Category Grid

- Generate one grid-subplot figure per subject × variable category combination
  (e.g. MOS/BOS, COM family, joint angles lower/upper, ankle torque, GRF/COP).
- Each category declares its own nrows×ncols layout and subplot mapping
  (data column → grid position) in a `CATEGORIES` dict; the script iterates
  over all subjects and emits one PNG per subject-category pair.
- Reference implementation: `scripts/plot_grid_timeseries.py` — `CATEGORIES`
  list and `plot_subject_category()` function.
