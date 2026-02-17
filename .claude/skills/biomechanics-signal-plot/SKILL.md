---
name: biomechanics-signal-plot
description: Visualize biomechanics signal data including EMG, Forceplate (Fx/Fy/Fz), and CoP/CoM trajectories. Use when creating grid plots, onset timing markers, window highlights, TKEO pipeline visualizations, or trajectory scatter plots for biomechanics research. Triggers on EMG plot, forceplate visualization, CoP trajectory, CoM trajectory, TKEO onset, signal grid, biomechanics chart.
---

# Biomechanics Signal Plot Skill

> Professional guidelines for biomechanics signal data visualization

## Overview

This Skill provides guidelines and code templates for visualizing signal data used in biomechanics research.

**Supported Data Types**:
- **EMG Signals**: TKEO pipeline, onset detection markers
- **Forceplate Signals**: Fx, Fy, Fz channel visualization
- **CoP/CoM Trajectories**: X-Y coordinate trajectory visualization, window color coding

## When to Use

- Visualizing EMG signal analysis results
- Visualizing Forceplate data
- Visualizing CoP (Center of Pressure) or CoM (Center of Mass) trajectories
- Creating velocity-trial combination grid plots
- Displaying onset timing markers
- Highlighting analysis windows

## File Structure

| File | Purpose |
|------|---------|
| `SKILL.md` | Main Skill overview and usage instructions |
| `emg-plot-guide.md` | EMG signal visualization guidelines |
| `forceplate-guide.md` | Forceplate signal visualization guidelines |
| `trajectory-guide.md` | CoP/CoM trajectory visualization guidelines |
| `templates/grid_plot_template.py` | Grid plot code template |

## Core Principles

### 1. Grid Plot Default Behavior
- All visualizations are generated as grid plots by default
- Number of columns = `ceil(sqrt(number of plots))`
- Empty subplots are hidden
- Each subplot includes individual title and legend

### 2. Velocity-Trial Sorting
Data is sorted by velocity-trial combination and arranged in grid:
```
10-1, 10-2, 10-3
15-1, 15-2, 15-3
20-1, 20-2, 20-3
```

### 3. Quality Settings
- DPI: 300 (publication quality)
- Korean font setup required

## Guide File References

### EMG Visualization
Refer to [emg-plot-guide.md](./emg-plot-guide.md) for EMG signal visualization:
- TKEO pipeline visualization
- Onset timing markers (vertical dashed lines)
- Window highlights (p1, p2, p3, p4)

### Forceplate Visualization
Refer to [forceplate-guide.md](./forceplate-guide.md) for Forceplate signal visualization:
- Fx, Fy, Fz channel visualization
- Onset timing markers
- Channel-specific color settings

### Trajectory Visualization
Refer to [trajectory-guide.md](./trajectory-guide.md) for CoP/CoM trajectory visualization:
- X vs Y scatter plot
- Y-axis flip (anterior-positive)
- Window-based color coding
- Maximum value markers (star markers)

## Using Code Templates

Import basic grid plot generation functions from `templates/grid_plot_template.py`:

```python
from templates.grid_plot_template import (
    calculate_grid_dimensions,
    setup_korean_font,
    create_grid_figure,
    save_figure
)

# Calculate grid dimensions
rows, cols = calculate_grid_dimensions(n_plots)

# Setup Korean font
setup_korean_font()

# Create figure
fig, axes = create_grid_figure(rows, cols, figsize=(16, 12))

# Save (DPI 300)
save_figure(fig, output_path)
```

## Common Color Settings

### Window Colors
| Window | Color |
|--------|-------|
| p1 | `#1f77b4` (Blue) |
| p2 | `#ff7f0e` (Orange) |
| p3 | `#2ca02c` (Green) |
| p4 | `#d62728` (Red) |

### Marker Styles
| Marker Type | Style |
|-------------|-------|
| onset timing | Vertical dashed line (`linestyle='--'`) |
| maximum value | Star marker (`marker='*'`, `markersize=10`) |
