---
name: aggregation-modes-and-event-vlines
description: Interpret and debug config-driven time-series visualization pipelines that use aggregation modes (filter → groupby → overlay → file splitting → filename formatting) and event vlines (event markers, legends, per-overlay-group styling). Use when diagnosing overlay/overlay_within behavior, filename_pattern KeyError issues, filters not applying (missing columns or type mismatches), or event_vlines.overlay_group (group-specific linestyles). In this repo, the source of truth is `config.yaml` plus the implementation in `script/visualizer.py`.
---

# Aggregation Modes And Event Vlines

## Overview

This skill provides rules and checklists to safely design and validate `aggregation_modes` (aggregation/overlay/output) and `event_vlines` (event marker vlines and overlay-group behavior) in config-driven time-series visualization workflows.

## Quick Start (작업 절차)

1) Inspect the `aggregation_modes` / `event_vlines` blocks in `config.yaml`.
2) When you need to confirm “what really happens”, search the implementation for the keywords below to find the exact branch (OLD vs NEW):
   - `overlay_within`, `filename_pattern`, `_apply_filter_indices`, `_collect_event_vlines`, `overlay_group`
3) Before changing configs, make `groupby` / `overlay_within` / `filename_pattern` consistent using the safety rules below.
4) Open `references/aggregation-modes-and-event-vlines.en.md` for the full spec and examples.

## 핵심 안전 규칙(요약)

- `aggregation_modes.<mode>.filter` must be a dict and is applied as an AND of all conditions. (If a column is missing, the implementation may warn and skip that condition.)
- With `overlay=true`, file splitting depends on whether `overlay_within` is set:
  - If `overlay_within` is missing/empty (**OLD behavior**): all group keys are overlaid into a single output file.
  - If `overlay_within` is provided (**NEW behavior**): `file_fields = groupby - overlay_within`; files split by `file_fields`, with overlays within each file.
- `filename_pattern` uses `str.format()`. Under overlay NEW behavior, using placeholders that are not present in `file_fields` can raise `KeyError`.
- `color_by` typically derives its values from the group key; in practice, `color_by ⊆ groupby` is the safest rule.
- `event_vlines.overlay_group` is for drawing selected events per overlay group (typically via different linestyles). Events listed in `overlay_group.columns` are usually removed from pooled vlines to avoid duplicates.

## 자주 나는 문제(증상 → 원인 → 해결)

- “I expected multiple files, but only one file was produced”
  - Cause: `overlay=true` with `overlay_within` missing/empty → OLD behavior overlays everything into one file.
  - Fix: set `overlay_within` and ensure your intended split criteria remains in `file_fields = groupby - overlay_within`.
- “KeyError while generating png”
  - Cause: `filename_pattern` includes a `{placeholder}` that is not present in the formatting mapping (common under overlay NEW behavior).
  - Fix: in overlay NEW, only use placeholders from `file_fields`, or simplify to `{signal_group}`.
- “Filters don’t seem to apply”
  - Cause: (1) missing column (condition is skipped), (2) type mismatch (e.g., `10` vs `10.0`).
  - Fix: verify column existence/types in the dataset and match the config value types accordingly.
- “Event vlines don’t show up”
  - Cause: (1) event column missing in input/features, (2) mean event timing falls outside the plotted time window.
  - Fix: check `event_vlines.columns`, the event domain source (input parquet vs features file), and the `interpolation.start_ms/end_ms` range.

## Reference

- Full spec, schema, and example YAML: `references/aggregation-modes-and-event-vlines.en.md`
