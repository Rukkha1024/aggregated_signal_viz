### Current behavior (important context)

1. `config.yaml` defines the input files and ID columns:

```yaml
data:
  input_file: "data/normalized_data.csv"
  features_file: "data/final_dataset.csv"
  id_columns:
    subject: "subject"
    velocity: "velocity"
    trial: "trial_num"
    frame: "DeviceFrame"
    mocap_frame: "MocapFrame"
    onset: "platform_onset"
    offset: "platform_offset"
    task: "task"
  task_filter: "perturb"
```

and aggregation modes:

```yaml
aggregation_modes:
  subject_mean:
    enabled: true
    groupby: ["subject"]
    output_dir: "output/subject_mean"
    filename_pattern: "{subject}_mean_{signal_group}.png"
  grand_mean:
    enabled: true
    groupby: []
    output_dir: "output/grand_mean"
    filename_pattern: "grand_mean_{signal_group}.png"
  filtered_mean:
    enabled: true
    filter:
      column: "velocity"
      value: 10.0
    groupby: []
    output_dir: "output/filtered_mean"
    filename_pattern: "vel10_mean_{signal_group}.png"
```

2. `normalized_data.csv` is a frame-level table: many rows per trial, with columns including:

* ID / alignment: `subject`, `velocity`, `trial_num`, `DeviceFrame`, `MocapFrame`, `platform_onset`, `platform_offset`, `task`, `aligned_frame`, etc.
* Time-series channels: EMG, forceplate, COP, as configured in `signal_groups`.
* Per-trial scalar metadata columns such as `T/F_step`, `T/F`, age, etc. These are constant within each `(subject, velocity, trial_num)` group but currently are **not** exposed to the aggregation logic.

3. In `visualizer.py` there is:

```py
@dataclass
class AggregatedRecord:
    subject: str
    velocity: float
    trial: int
    data: Dict[str, np.ndarray]

    def get(self, key: str) -> Optional[object]:
        return getattr(self, key, None)
```

and the resampling step:

```py
def _resample_all(
    self, df: pl.DataFrame, selected_groups: Optional[Iterable[str]]
) -> Dict[str, List[AggregatedRecord]]:
    subject_col = self.id_cfg["subject"]
    velocity_col = self.id_cfg["velocity"]
    trial_col = self.id_cfg["trial"]
    group_cols = [subject_col, velocity_col, trial_col]

    records: Dict[str, List[AggregatedRecord]] = {k: [] for k in self.config["signal_groups"]}
    groups = df.group_by(group_cols, maintain_order=True)
    for key, subdf in groups:
        subdf_sorted = subdf.sort("aligned_frame")
        meta = {
            subject_col: key[0],
            velocity_col: float(key[1]),
            trial_col: int(key[2]),
        }
        for group_name, cfg in self.config["signal_groups"].items():
            if selected_groups and group_name not in selected_groups:
                continue
            data = self._interpolate_group(subdf_sorted, cfg["columns"])
            records[group_name].append(
                AggregatedRecord(
                    subject=meta[subject_col],
                    velocity=meta[velocity_col],
                    trial=meta[trial_col],
                    data=data,
                )
            )
    return records
```

4. Aggregation uses simple equality on `AggregatedRecord` attributes:

```py
def _apply_filter(
    self, records: List[AggregatedRecord], filter_cfg: Optional[Dict]
) -> List[AggregatedRecord]:
    if not filter_cfg:
        return records
    col = filter_cfg["column"]
    value = filter_cfg["value"]
    return [r for r in records if getattr(r, col, None) == value]

def _group_records(
    self, records: List[AggregatedRecord], group_fields: List[str]
) -> Dict[Tuple, List[AggregatedRecord]]:
    if not group_fields:
        return {("all",): records}

    grouped: Dict[Tuple, List[AggregatedRecord]] = {}
    for rec in records:
        key = tuple(getattr(rec, f) for f in group_fields)
        grouped.setdefault(key, []).append(rec)
    return grouped
```

This means only `subject`, `velocity`, and `trial` can currently be used safely in `aggregation_modes.*.filter.column` and `aggregation_modes.*.groupby`.

### Goal

I want to treat **all trial-level scalar metadata columns from `normalized_data.csv` as first-class fields for aggregation**, without modifying `config.yaml` when adding new metadata.

Concretely:

* Any column in `normalized_data.csv` that is **constant within a `(subject, velocity, trial_num)` group** should be automatically attached to each `AggregatedRecord` as metadata when `_resample_all` runs.
* Then `aggregation_modes` in `config.yaml` should be able to:

  * `groupby` on those metadata columns, e.g. `"T/F_step"`, `"T/F"`, or new derived columns such as `"age_group"`, `"step_group"`.
  * `filter` on those metadata columns in the `filter.column` field.

Examples of desired use cases (after your changes):

1. In `normalized_data.csv` I add a column `age_years` and a derived column `age_group` such that:

   * `age_group = "young"` if `age_years < 30`
   * `age_group = "old"` if `age_years >= 30`

   Then I set an aggregation mode like:

   ```yaml
   aggregation_modes:
     age_group_mean:
       enabled: true
       groupby: ["age_group"]
       output_dir: "output/age_group_mean"
       filename_pattern: "{age_group}_mean_{signal_group}.png"
   ```

   This should produce separate mean EMG / forceplate / COP plots for `young` and `old` across all trials, with no additional code or config changes needed.

2. I have a column `T/F` taking values `"step"` or `"nonstep"`. I want:

   ```yaml
   aggregation_modes:
     step_type_mean:
       enabled: true
       groupby: ["T/F"]
       output_dir: "output/step_type_mean"
       filename_pattern: "{T/F}_mean_{signal_group}.png"
   ```

   and optionally:

   ```yaml
   filtered_nonstep:
     enabled: true
     filter:
       column: "T/F"
       value: "nonstep"
     groupby: []
     output_dir: "output/nonstep_only"
     filename_pattern: "nonstep_mean_{signal_group}.png"
   ```

   These should work out of the box as soon as those columns exist in `normalized_data.csv`.

The key requirement: **“opt-out” design**. All per-trial scalar columns from `normalized_data.csv` are automatically available for grouping and filtering; I do not want to maintain a separate allow-list in `config.yaml`.

### Required code changes

Please implement the following changes in `aggregated_signal_viz/visualizer.py`:

1. **Extend `AggregatedRecord` to hold arbitrary metadata.**

Change the dataclass to include a `meta` dict and improve `get`:

```py
from typing import Any, Dict, Optional

@dataclass
class AggregatedRecord:
    subject: str
    velocity: float
    trial: int
    data: Dict[str, np.ndarray]
    meta: Optional[Dict[str, Any]] = None

    def get(self, key: str) -> Optional[object]:
        """
        Return the value for a given metadata key.

        Priority:
        1) Explicit attributes (subject, velocity, trial, etc.)
        2) meta dict (per-trial scalar columns from normalized_data.csv)
        """
        if hasattr(self, key):
            return getattr(self, key, None)
        if self.meta is not None:
            return self.meta.get(key)
        return None
```

2. **Populate `meta` with all per-trial scalar columns from `normalized_data.csv` in `_resample_all`.**

Inside `_resample_all`, after computing `meta` for `subject`, `velocity`, `trial`, add logic that scans the group dataframe and collects all columns that are constant within that `(subject, velocity, trial)` group.

* “Constant within the group” means: `n_unique() == 1` when computed on the group’s sub-dataframe.
* Exclude obvious time-series / index columns that are not meaningful metadata:

  * The frame IDs: `id_columns.frame`, `id_columns.mocap_frame`, and `"aligned_frame"`.
  * All signal channels listed in `self.config["signal_groups"][...]["columns"]` (EMG, forceplate, COP).
* Do **not** exclude `subject`, `velocity`, `trial`, `task`, `platform_onset`, `platform_offset`, or custom metadata columns; these should be available in `meta`.

Pseudo-code for what you should actually implement:

```py
def _resample_all(
    self, df: pl.DataFrame, selected_groups: Optional[Iterable[str]]
) -> Dict[str, List[AggregatedRecord]]:
    subject_col = self.id_cfg["subject"]
    velocity_col = self.id_cfg["velocity"]
    trial_col = self.id_cfg["trial"]
    group_cols = [subject_col, velocity_col, trial_col]

    records: Dict[str, List[AggregatedRecord]] = {k: [] for k in self.config["signal_groups"]}

    # Build a set of columns to ignore when collecting scalar metadata
    ignore_cols: set[str] = set(group_cols)
    frame_col = self.id_cfg.get("frame")
    if frame_col:
        ignore_cols.add(frame_col)
    mocap_frame_col = self.id_cfg.get("mocap_frame")
    if mocap_frame_col:
        ignore_cols.add(mocap_frame_col)
    ignore_cols.add("aligned_frame")

    # Also ignore all time-series channels listed under signal_groups
    for sg_cfg in self.config["signal_groups"].values():
        for col in sg_cfg.get("columns", []):
            ignore_cols.add(col)

    groups = df.group_by(group_cols, maintain_order=True)
    for key, subdf in groups:
        subdf_sorted = subdf.sort("aligned_frame")
        meta_base = {
            subject_col: key[0],
            velocity_col: float(key[1]),
            trial_col: int(key[2]),
        }

        # Collect per-trial scalar metadata from all columns
        meta_all: Dict[str, Any] = dict(meta_base)
        for col in subdf.columns:
            if col in ignore_cols:
                continue
            # Skip if we already have the field (e.g., subject / velocity / trial)
            if col in meta_all:
                continue
            # Only keep columns that are constant within this group
            n_unique = subdf.select(pl.col(col).n_unique()).item()
            if n_unique != 1:
                continue
            value = subdf.select(pl.col(col).first()).item()
            meta_all[col] = value

        for group_name, cfg in self.config["signal_groups"].items():
            if selected_groups and group_name not in selected_groups:
                continue
            data = self._interpolate_group(subdf_sorted, cfg["columns"])
            records[group_name].append(
                AggregatedRecord(
                    subject=meta_base[subject_col],
                    velocity=meta_base[velocity_col],
                    trial=meta_base[trial_col],
                    data=data,
                    meta=meta_all,
                )
            )
    return records
```

The exact syntax can differ, but the core behavior must match:

* `meta_all` holds **all per-trial scalar fields**, keyed by column name.
* `AggregatedRecord.meta` gets that dict.

3. **Make `filter` and `groupby` use `AggregatedRecord.get` instead of `getattr`.**

Update `_apply_filter` to use `r.get(col)`:

```py
def _apply_filter(
    self, records: List[AggregatedRecord], filter_cfg: Optional[Dict]
) -> List[AggregatedRecord]:
    if not filter_cfg:
        return records
    col = filter_cfg["column"]
    value = filter_cfg["value"]
    return [r for r in records if r.get(col) == value]
```

Update `_group_records` similarly:

```py
def _group_records(
    self, records: List[AggregatedRecord], group_fields: List[str]
) -> Dict[Tuple, List[AggregatedRecord]]:
    if not group_fields:
        return {("all",): records}

    grouped: Dict[Tuple, List[AggregatedRecord]] = {}
    for rec in records:
        key = tuple(rec.get(f) for f in group_fields)
        grouped.setdefault(key, []).append(rec)
    return grouped
```

`_render_filename` and `_format_title` already receive `key` and `group_fields`; that logic can remain unchanged. When group_fields includes new metadata columns (e.g. `"T/F_step"` or `"age_group"`), the corresponding values in `key` will be taken from `AggregatedRecord.meta`.

4. **Do not change external behavior or config structure.**

* Keep the CLI, function signatures, and `config.yaml` schema unchanged.
* Existing configurations that only use `subject` / `velocity` should continue to work exactly as before.
* `features_file` (`final_dataset.csv`) and marker collection `_collect_markers` do not need to change. They already filter by any group field only if that column exists in `features_df`; for new metadata columns that are not in `final_dataset.csv`, they will simply be ignored, which is acceptable.

### Acceptance criteria

After your changes:

1. I can add arbitrary per-trial scalar columns to `data/normalized_data.csv` (e.g. `T/F_step`, `T/F`, `age_group`, `BMI_group`, etc.) and they will automatically be available as metadata fields on `AggregatedRecord`, without touching `config.yaml` except for using these names in `aggregation_modes.*.groupby` or `.filter.column`.

2. The following examples should work:

* Group by age group:

  ```yaml
  aggregation_modes:
    age_group_mean:
      enabled: true
      groupby: ["age_group"]
      output_dir: "output/age_group_mean"
      filename_pattern: "{age_group}_mean_{signal_group}.png"
  ```

* Group by `T/F_step`:

  ```yaml
  aggregation_modes:
    step_group_mean:
      enabled: true
      groupby: ["T/F_step"]
      output_dir: "output/step_group_mean"
      filename_pattern: "{T/F_step}_mean_{signal_group}.png"
  ```

* Filter by `T/F`:

  ```yaml
  aggregation_modes:
    nonstep_only:
      enabled: true
      filter:
        column: "T/F"
        value: "nonstep"
      groupby: []
      output_dir: "output/nonstep_only"
      filename_pattern: "nonstep_mean_{signal_group}.png"
  ```

3. Plots for EMG, forceplate, and COP continue to generate correctly for both existing modes and new modes that use these metadata fields.

Please implement the above changes in `aggregated_signal_viz/visualizer.py`, keep the style consistent with the existing code (Polars, NumPy, type hints), and ensure the code runs without errors.
