<prompt>
You are modifying the codebase in `aggregated_signal_viz/` (files: `config.yaml`, `visualizer.py`) to add a new “overlay plotting” capability for aggregation modes. The goal is: when an aggregation mode has `overlay: true`, the visualizer must aggregate each group separately but plot ALL groups together in a single output image (one image per signal_group), using Matplotlib’s auto color cycle (do NOT set any line colors).

Hard constraints (must satisfy):

* Overlay applies ONLY to `signal_group in {"emg","forceplate"}`. If `signal_group == "cop"` AND `overlay: true`, skip (do not output any COP overlay file). Non-overlay COP behavior must remain unchanged.
* Legends must show EVERYTHING: group lines + all window spans (p1..p4 with their ms ranges) + all markers that are enabled (onset/max). No “hide duplicates” logic that removes window labels.
* Overlay must work for ANY `groupby` definition (e.g., ["step_TF"], ["subject"], ["age_group"], ["subject","velocity"], etc.). It must not be step_TF-specific.
* Key ordering must be stable:

  * If `groupby == ["step_TF"]` exactly, enforce `nonstep` then `step`.
  * Otherwise, sort keys lexicographically by their string representations, so overlay line color assignment remains stable run-to-run.

Part 1) Update `config.yaml`

* Add a new mode example under `aggregation_modes` that demonstrates overlay for step_TF comparison:

  * name: `step_TF_comparison`
  * fields: `enabled: true`, `groupby: ["step_TF"]`, `overlay: true`, `output_dir: "output/comparison"`, `filename_pattern: "comparison_by_step_TF_{signal_group}.png"`
* Do NOT remove or alter existing modes.
* Do NOT change existing plot_style keys; overlay logic will simply ignore `line_color` / `line_colors` when overlay is active.

Part 2) Update `visualizer.py`
A) Modify `_run_mode(self, signal_group: str, mode_name: str, mode_cfg: Dict) -> None`

* Current behavior loops groups and saves one file per group. Keep that for non-overlay.
* Add an overlay branch:

  1. Read `overlay = bool(mode_cfg.get("overlay", False))`
  2. If `overlay` is True:

     * If `signal_group == "cop"`: print a short warning (e.g., “[overlay] skip COP …”) and `return`
     * Filter records using existing `_apply_filter`
     * Group using existing `_group_records(records, group_fields)`
     * For each group key, compute:

       * `aggregated_by_key[key] = self._aggregate_group(recs)`
       * `markers_by_key[key] = self._collect_markers(signal_group, key, group_fields, mode_cfg.get("filter"))`
     * Sort keys using a NEW helper `_sort_overlay_keys(keys, group_fields)`
     * Render ONE output filename and path:

       * Prefer using key `("all",)` so filename patterns that only require `{signal_group}` still work:
         `filename = self._render_filename(mode_cfg["filename_pattern"], ("all",), signal_group, group_fields=[])`
       * Additionally, make this robust if a user accidentally includes group placeholders:
         try formatting with `signal_group` only; on KeyError, format again by providing each group field with the literal `"overlay"` so it won’t crash (and print a warning once).
     * Call a NEW method `_plot_overlay(...)` that plots all groups into a single figure and saves ONE file.
  3. If `overlay` is False: keep existing behavior exactly.

B) Add new helpers/methods

1. `_sort_overlay_keys(self, keys: List[Tuple], group_fields: List[str]) -> List[Tuple]`

   * If `group_fields == ["step_TF"]`:
     order using mapping `{ "nonstep": 0, "step": 1 }`, unknowns go after (e.g., 99), and keep stable fallback sort.
   * Else:
     sort by `tuple(str(v) for v in key)`.

2. `_format_group_label(self, key: Tuple, group_fields: List[str]) -> str`

   * If `not group_fields` or `key == ("all",)`: return "all"
   * If len(group_fields)==1: return `str(key[0])`
   * Else: return `", ".join(f"{f}={v}" for f, v in zip(group_fields, key))`

3. `_plot_overlay(self, signal_group: str, aggregated_by_key: Dict[Tuple, Dict[str,np.ndarray]], markers_by_key: Dict[Tuple, Dict[str,Any]], output_path: Path, mode_name: str, group_fields: List[str], sorted_keys: List[Tuple]) -> None`

   * Dispatch:

     * emg -> `_plot_emg_overlay(...)`
     * forceplate -> `_plot_forceplate_overlay(...)`
     * (cop should never reach here)

C) Implement overlay plotting (MUST use auto color cycle)

1. `_plot_emg_overlay(...)`

   * Layout must match `_plot_emg` (grid_layout, figsize, dpi, suptitle, supxlabel, supylabel, tight_layout, savefig).
   * For each channel subplot:

     * Add window spans ONCE per window with label from `_format_window_label(name)` (use existing `self.window_norm_ranges` and `self.window_colors`).
     * For each group in `sorted_keys`:

       * Get y = aggregated_by_key[key].get(channel). If None -> skip.
       * Plot: `ax.plot(x, y, linewidth=self.emg_style["line_width"], alpha=self.emg_style["line_alpha"], label=<group_label>)`
         IMPORTANT: do NOT pass any color or format string like `"blue"`; do NOT pass `color=...`.
     * Markers (legend must show all):

       * For each group in `sorted_keys`, if markers_by_key[key] has this channel:

         * If `show_onset_marker` enabled and onset exists:

           * draw axvline using `self.emg_style["onset_marker"]`
           * label must be unique per group, e.g. `f"{group_label} onset"`
         * If `show_max_marker` enabled and max exists:

           * draw axvline using `self.emg_style["max_marker"]`
           * label must be unique per group, e.g. `f"{group_label} max"`
     * Call `ax.legend(...)` (do not filter or deduplicate labels; legends must include windows + groups + markers).
   * Title:

     * Use a new overlay title format: `f"{mode_name} | {signal_group} | overlay by {', '.join(group_fields) if group_fields else 'all'}"`

2. `_plot_forceplate_overlay(...)`

   * Layout must match `_plot_forceplate` (grid_layout, figsize, dpi, suptitle, per-axes xlabel/ylabel, tight_layout, savefig).
   * For each force channel subplot (Fx/Fy/Fz):

     * Add window spans ONCE per window (label included).
     * For each group in `sorted_keys`:

       * y = aggregated_by_key[key][channel]
       * Plot WITHOUT specifying color:
         `ax.plot(x, y, linewidth=self.forceplate_style["line_width"], alpha=self.forceplate_style["line_alpha"], label=<group_label>)`
     * Markers:

       * For each group, if markers_by_key has onset for this channel, draw axvline with label `f"{group_label} onset"`.
     * Call legend without dedupe.

D) Preserve existing non-overlay behavior

* Do NOT change `_plot_emg`, `_plot_forceplate`, `_plot_cop` output logic (except if you must refactor shared code, but outputs must remain identical for non-overlay modes).
* Do NOT reintroduce hard-coded styling outside config; overlay just avoids specifying colors.

Part 3) Verification / acceptance checks

* Run:

  * `python main.py --modes step_TF_comparison --groups emg forceplate --sample`
* Expected:

  * Exactly ONE EMG overlay PNG saved under `output/comparison/` (per filename_pattern), containing each EMG channel subplot with two lines (nonstep/step) in different auto-cycle colors, window spans labeled, and legend showing windows + both group lines (and markers if enabled).
  * Exactly ONE forceplate overlay PNG saved similarly, each subplot containing multiple group lines with auto-cycle colors and legend containing windows + group lines (and markers if enabled).
  * No COP overlay file is produced for overlay modes.
* Also run at least one existing non-overlay mode (e.g., `step_TF_mean`) to confirm it still creates per-group files and looks unchanged.

Deliverables:

* Patch `config.yaml` with the new overlay mode example.
* Patch `visualizer.py` implementing overlay logic and helpers as described.

</prompt>
