# aggregation_modes + event_vlines specification (English)

This document describes safe, reproducible rules for config-driven time-series visualization pipelines that support:

- `aggregation_modes`: filter / groupby / overlay / file splitting / filename formatting
- `event_vlines`: event marker vertical lines, legends, and per-overlay-group styling

This repository (`aggregated_signal_viz`) is one concrete implementation. The doc is split into:

1) **Generic rules** (portable across projects)
2) **Repo-specific notes** (how this repo currently behaves)

---

## 1) Generic concepts (portable rules)

### 1.1 Processing unit
Define the minimum processing/caching/file-naming unit first (non-negotiable). Examples:
- `subject-velocity-trial`
- `session-trial`
- `recording-id`

All aggregation, filtering, grouping, and output partitioning should be understandable in terms of this unit.

### 1.2 Time axis policy
Event vlines and windows are only meaningful if the time axis domain is explicit:
- absolute time/frame, or
- event-locked (e.g., onset = 0)

If you resample, your plotting x-axis is usually a derived axis (e.g., 0..1). Event locations must be converted to the same domain (e.g., ms → normalized x).

---

## 2) `aggregation_modes` schema (generic)

```yaml
aggregation_modes:
  <mode_name>:
    enabled: true|false
    filter: {col1: value1, col2: value2, ...}
    groupby: [colA, colB, ...]
    overlay: true|false
    overlay_within: [colX, colY, ...]
    color_by: [colK, ...]
    output_dir: "..."
    filename_pattern: "....png"
```

### 2.1 `enabled`
- Default: treated as `true`
- If `false`, the mode is skipped.

### 2.2 `filter` (AND semantics)
- Use a dict only: `filter: { mixed: 1, age_group: "young" }`
- Semantics: `col == value` for all items, combined with AND.

Operational tips:
- Define your “missing column” policy explicitly: warn+skip, fail fast, or silently ignore.
- Be careful with type mismatches (`10` vs `10.0`, `"1"` vs `1`).

### 2.3 `groupby` (aggregation key)
- If empty, treat as a single group (commonly `("all",)`).
- Fields used for grouping must be stable within your processing unit; otherwise aggregation becomes ambiguous and many implementations will raise.

### 2.4 `overlay`
- `overlay: false` (common default)
  - One output file per group key.
- `overlay: true`
  - Multiple group keys can be plotted as overlaid lines in a single output file (depending on file splitting rules).

### 2.5 `overlay_within` (file splitting when overlaying)
Only meaningful when `overlay=true`.

#### OLD behavior (legacy pattern)
If `overlay_within` is missing or empty:
- Overlay all group keys into a single output file.
- The file key is effectively “all”.

#### NEW behavior (recommended pattern)
If `overlay_within` is provided:
- `file_fields = groupby - overlay_within`
- Split output files by `file_fields`
- Within each file, overlay lines vary by `overlay_within` fields.

Recommended invariant:
- Keep `overlay_within ⊆ groupby`.

Example:
- `groupby: ["subject", "step_TF"]`
- `overlay: true`
- `overlay_within: ["step_TF"]`
→ files split by `subject`, with `step_TF` overlaid within each file.

### 2.6 `color_by`
Common pattern:
- Colors are derived from the group key, so `color_by ⊆ groupby` is typically safest.
- If `color_by` references fields not present in the group key, many implementations will end up assigning a single color (or grouping into `None`), which looks “not working” but may not error.

### 2.7 `output_dir`
Use a consistent base output directory policy (e.g., `output.base_dir`) and resolve mode output paths relative to it.

### 2.8 `filename_pattern`
- `str.format()`-style template.
- Always safe to include `{signal_group}` if your implementation provides it.

High-risk area:
- Under overlay NEW behavior, the formatting mapping commonly includes **only file-level fields** (`file_fields`) plus `{signal_group}`.
- Using placeholders that are not present in that mapping can raise `KeyError`.

---

## 3) `event_vlines` schema (generic)

Recommended structure:

```yaml
event_vlines:
  columns: ["event_a", "event_b"]
  event_labels:
    event_a: "Event A"
  # Optional:
  # colors:
  #   event_a: "black"
  # Optional:
  # palette: ["C0","C1",...,"C9"]
  style:
    linestyle: "--"
    linewidth: 1.5
    alpha: 0.9
  overlay_group:
    enabled: true
    mode: "linestyle"
    columns: ["event_a"]
    linestyles: ["-","--",":","-."]
```

Minimal shorthand (if the implementation supports it):

```yaml
event_vlines: ["event_a", "event_b"]
```

### 3.1 `columns`
- List of event column names to render as vertical lines.
- De-duplicate and drop empty strings.

### 3.2 Event time-domain rules
You must make event timing semantics explicit:
- Where do event values come from? (input table vs external features table)
- Are values frames or milliseconds?
- Are values absolute or relative to a reference event (onset)?

Safe approaches:
1) Convert and store events into the chosen domain early (e.g., `event_ms_from_onset`)
2) Keep raw provenance and convert only for plotting (do not overwrite raw pointers)

Events typically are not drawn if their mean timing falls outside the plotted window.

### 3.3 `event_labels`
Legend label mapping for event names.

### 3.4 `palette` / `colors`
- Use a palette to assign default colors.
- Allow explicit overrides per event (preferred for key reference events).

### 3.5 `style`
Base matplotlib-style kwargs used for drawing vlines; per-vline overrides may apply.

### 3.6 `overlay_group`
Purpose: when overlaying multiple group keys (e.g., step vs nonstep), draw selected events **per overlay group** using distinct linestyles (or other supported style channels).

Common duplicate-avoidance rule:
- Events listed in `overlay_group.columns` are removed from pooled (global) vlines so they appear only as per-group vlines.

---

## 4) Combined behavior (generic)

### 4.1 `overlay=false`
Each group gets its own file; event vlines are computed per group.

### 4.2 `overlay=true`
Often two event layers exist:
- pooled vlines (mean over everything included in the file)
- per-group vlines (for selected events under `event_vlines.overlay_group`)

### 4.3 Optional: channel-/sensor-specific events
Some pipelines support channel-specific event timing (e.g., EMG). Decide whether to:
- show a single global event per file, or
- show per-channel events per subplot.

---

## 5) Troubleshooting checklist (generic)

### 5.1 “I expected multiple files, but got only one”
- Likely `overlay=true` with `overlay_within` missing/empty (OLD behavior).
- Fix: set `overlay_within` so `file_fields = groupby - overlay_within` matches your intended split.

### 5.2 “KeyError during file generation”
- `filename_pattern` references placeholders not present in the format mapping.
- Fix:
  - Under overlay NEW, use only `file_fields` placeholders (plus `{signal_group}` if available).
  - Under overlay OLD, keep filenames simple.

### 5.3 “Filters don’t apply”
- Missing columns (implementation-dependent: might warn+skip).
- Type mismatch (int/float/string).

### 5.4 “Event vlines don’t show”
- Event columns unavailable in both primary input and features table.
- Mean event timing outside the plotted window.

---

## 6) Repo-specific notes: `aggregated_signal_viz`

This section documents the current behavior of this repository (source of truth: `config.yaml` + `script/visualizer.py`).

### 6.1 Processing unit
The resampling (trial tensor) unit is `subject-velocity-trial`, derived from:
- `data.id_columns.subject`
- `data.id_columns.velocity`
- `data.id_columns.trial`

### 6.2 Time axis (resampling)
The pipeline crops to `interpolation.start_ms .. interpolation.end_ms` and resamples to `interpolation.target_length`.
Plotting uses a normalized x-axis (`x_norm`) spanning 0..1.

### 6.3 Event domain: input parquet vs `features_file`
Internally, events are represented as `__event_<event>_ms` metadata columns.

1) If the event column exists in the input parquet:
- It is interpreted in the same domain as the onset reference (mocap-frame domain).
- It is converted into “ms relative to onset”.

2) If the event column does not exist in the input parquet but exists only in `features_file`:
- It is interpreted as “ms relative to onset”.
- It is joined into metadata and coalesced into `__event_<event>_ms`.

Events are not drawn if their mean timing falls outside `interpolation.start_ms/end_ms`.

### 6.4 EMG channel-specific events
If `features_file` contains an `emg_channel` column and an event varies by channel within the same processing unit, the implementation may treat it as channel-specific and draw vlines per subplot.

---

## 7) Porting/extension checklist (to another project)

Answer these before copying the rules:

1) What is the processing unit?
2) Is the time axis absolute or event-locked?
3) Where do events come from (input vs features), and what domain are they in (frame vs ms)?
4) Under overlay, what is the file-splitting rule? (Is there an `overlay_within`-like concept?)
5) What placeholders are valid in `filename_pattern` for each mode?
6) What is the policy when filter columns are missing (skip vs fail)?
7) Do you need channel-specific events/windows? If yes, what is the display policy?

