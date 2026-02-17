# X-Y Trajectory Visualization Guide

> Detailed guidelines for CoP (Center of Pressure) and CoM (Center of Mass) trajectory data visualization

## Overview

This guide defines visualization methods for CoP and CoM X-Y coordinate trajectory data. It includes Y-axis flip (anterior-positive), window-based color coding, and maximum value marker display.

## Data Types

### CoP (Center of Pressure)

| Column | Direction | Description |
|--------|-----------|-------------|
| Cx | Medio-Lateral | Left-right direction (Right +, Left -) |
| Cy | Anterior-Posterior | Front-back direction (Original: Posterior +, Visualization: Anterior +) |

### CoM (Center of Mass)

| Column | Direction | Description |
|--------|-----------|-------------|
| COM_X | Medio-Lateral | Left-right direction (Right +, Left -) |
| COM_Y | Anterior-Posterior | Front-back direction (Original: Posterior +, Visualization: Anterior +) |

## Y-Axis Flip (Anterior-Positive) Guidelines

In biomechanics data, the Y-axis typically has Posterior as the positive direction. For intuitive interpretation during visualization, the Y-axis is flipped so that Anterior becomes the positive direction.

### Y-Axis Flip Rules

```python
# Apply Y-axis flip (anterior-positive)
flip_ap_sign = True  # Default: True

# CoP data
x = df['Cx']  # ML axis: Right +, Left -
y = -df['Cy'] if flip_ap_sign else df['Cy']  # AP axis: Anterior + (flipped)

# CoM data
x = df['COM_X']  # ML axis: Right +, Left -
y = -df['COM_Y'] if flip_ap_sign else df['COM_Y']  # AP axis: Anterior + (flipped)
```

### Axis Label Settings

```python
# Labels when Y-axis flip is applied
xlabel = 'X (Right +, Left -)'
ylabel = 'Y (Anterior +)'  # When flip is applied

# Labels when Y-axis flip is not applied
ylabel = 'Y (Posterior +, Anterior -)'  # When flip is not applied
```

## Window-Based Color Coding Guidelines

Data points are distinguished by color according to analysis windows.

### Window Color Palette

| Window | Color Code | Color Name | Alpha |
|--------|------------|------------|-------|
| p1 | `#1f77b4` | Blue | 0.85 |
| p2 | `#ff7f0e` | Orange | 0.85 |
| p3 | `#2ca02c` | Green | 0.85 |
| p4 | `#d62728` | Red | 0.85 |
| Other | `lightgray` | Gray | 0.4 |

### Window Color Definition Code

```python
# Window color definitions
WINDOW_COLORS = {
    'p1': '#1f77b4',  # Blue
    'p2': '#ff7f0e',  # Orange
    'p3': '#2ca02c',  # Green
    'p4': '#d62728',  # Red
}

# Other frames color
BACKGROUND_COLOR = 'lightgray'
BACKGROUND_ALPHA = 0.4
```

### Window Mask Creation Code

```python
import numpy as np

def create_window_masks(df, window_boundaries):
    """
    Create data masks for each window
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing DeviceFrame column
    window_boundaries : dict
        Window boundary information {'p1': (start, end), 'p2': (start, end), ...}
    
    Returns:
    --------
    dict : Boolean masks for each window
    """
    window_masks = {}
    for window_name, (ws, we) in window_boundaries.items():
        window_masks[window_name] = (df['DeviceFrame'] >= ws) & (df['DeviceFrame'] <= we)
    return window_masks

def get_combined_mask(window_masks):
    """
    Create combined mask for all windows
    
    Parameters:
    -----------
    window_masks : dict
        Boolean masks for each window
    
    Returns:
    --------
    np.ndarray : Combined boolean mask
    """
    combined = None
    for mask in window_masks.values():
        if combined is None:
            combined = np.asarray(mask)
        else:
            combined |= np.asarray(mask)
    return combined
```

## Maximum Value Marker Display Method

Maximum value points of trajectories (CoP Max, CoM Max) are highlighted with star markers.

### Maximum Value Marker Style

| Property | Value |
|----------|-------|
| Marker | `'*'` (star) |
| Size | 200 |
| Color | `#ED1C24` (Red) |
| Edge Color | `white` |
| Edge Width | 2 |
| Alpha | 0.9 |
| Z-order | 10 (topmost) |

### Maximum Value Marker Code Example

```python
import matplotlib.pyplot as plt
import numpy as np

def add_trajectory_max_marker(ax, df, x_col, y_col, max_timing, flip_ap_sign=True, series_type='cop'):
    """
    Add star marker at trajectory maximum value point
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Target axes
    df : pd.DataFrame
        DataFrame
    x_col : str
        X-axis column name ('Cx' or 'COM_X')
    y_col : str
        Y-axis column name ('Cy' or 'COM_Y')
    max_timing : int
        DeviceFrame number of maximum value point
    flip_ap_sign : bool
        Whether to apply Y-axis flip
    series_type : str
        Data type ('cop' or 'com')
    """
    if max_timing is None:
        return
    
    # Find maximum value point
    max_point_mask = df['DeviceFrame'] == max_timing
    
    if np.sum(max_point_mask) > 0:
        max_x = df[np.asarray(max_point_mask)][x_col].iloc[0]
        max_y = -df[np.asarray(max_point_mask)][y_col].iloc[0] if flip_ap_sign else df[np.asarray(max_point_mask)][y_col].iloc[0]
        
        # Set label
        label_prefix = 'CoP Max' if series_type == 'cop' else 'CoM Max'
        
        ax.scatter(max_x, max_y, 
                   c='#ED1C24',           # Red
                   s=200,                  # Size
                   marker='*',             # Star marker
                   edgecolor='white',      # Edge color
                   linewidth=2,            # Edge width
                   alpha=0.9,
                   label=f'{label_prefix} (Frame {int(max_timing)})',
                   zorder=10)              # Topmost layer
```

## Single Trajectory Visualization

Method for visualizing individual CoP or CoM trajectories.

### Single Trajectory Plot Code Example

```python
import matplotlib.pyplot as plt
import numpy as np

def create_single_trajectory_plot(df, subject, velocity, trial_num, 
                                   series_type='cop', flip_ap_sign=True,
                                   window_boundaries=None, max_timing=None):
    """
    Visualize single CoP/CoM trajectory
    
    Parameters:
    -----------
    df : pd.DataFrame
        Trajectory data (containing DeviceFrame, Cx/Cy or COM_X/COM_Y)
    subject : str
        Subject ID
    velocity : float
        Velocity condition
    trial_num : int
        Trial number
    series_type : str
        Data type ('cop' or 'com')
    flip_ap_sign : bool
        Whether to apply Y-axis flip
    window_boundaries : dict
        Window boundary information (optional)
    max_timing : int
        Maximum value DeviceFrame number (optional)
    """
    # Set column names and labels
    if series_type.lower() == 'cop':
        x_col, y_col = 'Cx', 'Cy'
        title_prefix = 'CoP Trajectory: Cx vs Cy'
        xlabel = 'Cx (Right +, Left -)'
        ylabel = 'Cy (Anterior +)' if flip_ap_sign else 'Cy (Posterior +, Anterior -)'
    else:  # com
        x_col, y_col = 'COM_X', 'COM_Y'
        title_prefix = 'CoM Trajectory: COM_X vs COM_Y'
        xlabel = 'COM_X (Right +, Left -)'
        ylabel = 'COM_Y (Anterior +)' if flip_ap_sign else 'COM_Y (Posterior +, Anterior -)'
    
    # Prepare data
    x = df[x_col]
    y = -df[y_col] if flip_ap_sign else df[y_col]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Window-based color coding
    if window_boundaries:
        window_masks = create_window_masks(df, window_boundaries)
        combined_mask = get_combined_mask(window_masks)
        
        # Scatter plot for each window
        for w_name, mask in window_masks.items():
            color = WINDOW_COLORS.get(w_name, '#999999')
            ws, we = window_boundaries[w_name]
            ax.scatter(x[np.asarray(mask)], y[np.asarray(mask)], 
                       c=color, s=24, alpha=0.85,
                       label=f'{w_name} ({ws}-{we})')
        
        # Data outside windows
        ax.scatter(x[~np.asarray(combined_mask)], y[~np.asarray(combined_mask)],
                   c=BACKGROUND_COLOR, alpha=BACKGROUND_ALPHA, s=18,
                   label='Other frames')
    else:
        # Display all data without windows
        ax.scatter(x, y, c='#1f77b4', s=24, alpha=0.85)
    
    # Add maximum value marker
    if max_timing is not None:
        add_trajectory_max_marker(ax, df, x_col, y_col, max_timing, 
                                   flip_ap_sign, series_type)
    
    # Plot settings
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'{title_prefix}\n{subject} | Vel: {velocity} | Trial: {trial_num}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
```

## Grid Plot Configuration

Multiple velocity-trial combinations are arranged in a single grid plot.

### Grid Layout Rules

1. Number of columns = `ceil(sqrt(number of plots))`
2. Number of rows = `ceil(number of plots / columns)`
3. Empty subplots are hidden
4. Each subplot includes individual title and legend

### Velocity-Trial Sorting Order

```
10-1, 10-2, 10-3
15-1, 15-2, 15-3
20-1, 20-2, 20-3
```

### Grid Plot Code Example

```python
import math
import matplotlib.pyplot as plt
import numpy as np

def calculate_grid_dimensions(n_plots):
    """Calculate grid dimensions"""
    if n_plots <= 0:
        return (0, 0)
    cols = math.ceil(math.sqrt(n_plots))
    rows = math.ceil(n_plots / cols)
    return (rows, cols)

def create_trajectory_grid_plot(data_dict, subject, series_type='cop', flip_ap_sign=True):
    """
    Create CoP/CoM trajectory grid plot
    
    Parameters:
    -----------
    data_dict : dict
        {(velocity, trial_num): {'df': DataFrame, 'windows': dict, 'max_timing': int}}
    subject : str
        Subject ID
    series_type : str
        Data type ('cop' or 'com')
    flip_ap_sign : bool
        Whether to apply Y-axis flip
    """
    n_plots = len(data_dict)
    rows, cols = calculate_grid_dimensions(n_plots)
    
    # Set column names and labels
    if series_type.lower() == 'cop':
        x_col, y_col = 'Cx', 'Cy'
        title_prefix = 'CoP Trajectory Grid'
        xlabel = 'Cx (Right +, Left -)'
        ylabel = 'Cy (Anterior +)' if flip_ap_sign else 'Cy (Posterior +)'
    else:
        x_col, y_col = 'COM_X', 'COM_Y'
        title_prefix = 'CoM Trajectory Grid'
        xlabel = 'COM_X (Right +, Left -)'
        ylabel = 'COM_Y (Anterior +)' if flip_ap_sign else 'COM_Y (Posterior +)'
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    fig.suptitle(f'{title_prefix} | {subject}', fontsize=16, fontweight='bold')
    
    # Flatten axes for easy iteration
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Sort by velocity, then trial_num
    sorted_keys = sorted(data_dict.keys(), key=lambda x: (x[0], x[1]))
    
    for idx, (velocity, trial_num) in enumerate(sorted_keys):
        ax = axes[idx]
        data = data_dict[(velocity, trial_num)]
        df = data['df']
        
        # Prepare data
        x = df[x_col]
        y = -df[y_col] if flip_ap_sign else df[y_col]
        
        # Window-based color coding
        if 'windows' in data and data['windows']:
            window_boundaries = data['windows']
            window_masks = create_window_masks(df, window_boundaries)
            combined_mask = get_combined_mask(window_masks)
            
            for w_name, mask in window_masks.items():
                color = WINDOW_COLORS.get(w_name, '#999999')
                ws, we = window_boundaries[w_name]
                ax.scatter(x[np.asarray(mask)], y[np.asarray(mask)],
                           c=color, s=20, alpha=0.85,
                           label=f'{w_name}')
            
            ax.scatter(x[~np.asarray(combined_mask)], y[~np.asarray(combined_mask)],
                       c=BACKGROUND_COLOR, alpha=BACKGROUND_ALPHA, s=14,
                       label='Other')
        else:
            ax.scatter(x, y, c='#1f77b4', s=20, alpha=0.85)
        
        # Add maximum value marker
        if 'max_timing' in data and data['max_timing'] is not None:
            add_trajectory_max_marker(ax, df, x_col, y_col, 
                                       data['max_timing'], flip_ap_sign, series_type)
        
        ax.set_title(f'Vel: {velocity} | Trial: {trial_num}', fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='upper right')
    
    # Hide empty subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig
```

## Time Progression Gradient Visualization

When highlighting a single window, a color gradient can be applied based on time progression.

### Gradient Color Settings

| Property | Value |
|----------|-------|
| Start Color | `#B9DDF1` (Light Blue) |
| End Color | `#173049` (Dark Blue) |
| Colorbar Label | 'Time Progression (Normalized DeviceFrame)' |

### Gradient Visualization Code Example

```python
from matplotlib.colors import LinearSegmentedColormap

def create_gradient_trajectory_plot(df, x_col, y_col, frame_start, frame_end, 
                                     flip_ap_sign=True):
    """
    Visualize trajectory with time progression gradient
    
    Parameters:
    -----------
    df : pd.DataFrame
        Trajectory data
    x_col : str
        X-axis column name
    y_col : str
        Y-axis column name
    frame_start : int
        Highlight start frame
    frame_end : int
        Highlight end frame
    flip_ap_sign : bool
        Whether to apply Y-axis flip
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data
    x = df[x_col]
    y = -df[y_col] if flip_ap_sign else df[y_col]
    
    # Highlight mask
    highlight_mask = (df['DeviceFrame'] >= frame_start) & (df['DeviceFrame'] <= frame_end)
    
    # Background data (outside highlight)
    ax.scatter(x[~np.asarray(highlight_mask)], y[~np.asarray(highlight_mask)],
               c='lightgray', alpha=0.6, s=20, label='Other frames')
    
    # Highlight data (with gradient)
    if np.sum(highlight_mask) > 0:
        highlight_data = df[np.asarray(highlight_mask)].copy()
        
        # Calculate normalized frame values
        norm_frames = (highlight_data['DeviceFrame'] - highlight_data['DeviceFrame'].min()) / \
                      (highlight_data['DeviceFrame'].max() - highlight_data['DeviceFrame'].min())
        
        # Create custom colormap
        colors = ['#B9DDF1', '#173049']  # Light blue -> Dark blue
        custom_cmap = LinearSegmentedColormap.from_list('custom_blue', colors, N=256)
        
        # Gradient scatter plot
        scatter = ax.scatter(
            x[np.asarray(highlight_mask)], 
            y[np.asarray(highlight_mask)],
            c=norm_frames, 
            cmap=custom_cmap, 
            s=30, 
            alpha=0.8,
            label=f'Highlight ({frame_start}-{frame_end})'
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Time Progression (Normalized DeviceFrame)', 
                       rotation=270, labelpad=20)
        
        # Set colorbar tick labels (display as actual DeviceFrame values)
        actual_frames = highlight_data['DeviceFrame'].values
        cbar_ticks = np.linspace(0, 1, 5)
        cbar_tick_labels = [
            f"{int(np.interp(t, [0, 1], [actual_frames.min(), actual_frames.max()]))}" 
            for t in cbar_ticks
        ]
        cbar.set_ticks(cbar_ticks.tolist())
        cbar.set_ticklabels(cbar_tick_labels)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
```

### Colorbar Configuration Details

Follow these guidelines when configuring time progression gradient colorbar.

#### Colorbar Basic Settings

| Property | Value | Description |
|----------|-------|-------------|
| Label | `'Time Progression (Normalized DeviceFrame)'` | Label text |
| Label rotation | `270` | Vertical orientation |
| Label padding | `20` | Gap between label and colorbar |

#### Colorbar Tick Settings

```python
def configure_colorbar_ticks(cbar, highlight_data):
    """
    Set colorbar ticks to actual DeviceFrame values
    
    Parameters:
    -----------
    cbar : matplotlib.colorbar.Colorbar
        Colorbar object
    highlight_data : pd.DataFrame
        Highlighted data (containing DeviceFrame column)
    """
    actual_frames = highlight_data['DeviceFrame'].values
    frame_min = actual_frames.min()
    frame_max = actual_frames.max()
    
    # Create 5 evenly spaced ticks (0.0, 0.25, 0.5, 0.75, 1.0)
    cbar_ticks = np.linspace(0, 1, 5)
    
    # Convert normalized tick values to actual DeviceFrame values
    cbar_tick_labels = [
        f"{int(np.interp(t, [0, 1], [frame_min, frame_max]))}" 
        for t in cbar_ticks
    ]
    
    # Set tick positions and labels
    cbar.set_ticks(cbar_ticks.tolist())
    cbar.set_ticklabels(cbar_tick_labels)
```

#### Tick Label Conversion Logic

1. **Normalized range**: `[0, 1]` (colormap input range)
2. **Actual frame range**: `[frame_min, frame_max]` (DeviceFrame values)
3. **Conversion formula**: `actual_frame = frame_min + t * (frame_max - frame_min)`
4. **Number of ticks**: 5 (start, 1/4, middle, 3/4, end)
5. **Label format**: Display as integer (`int()` applied)

#### Example Output

When highlight range is DeviceFrame 3027-3087:

| Tick Position (Normalized) | Actual DeviceFrame |
|---------------------------|-------------------|
| 0.00 | 3027 |
| 0.25 | 3042 |
| 0.50 | 3057 |
| 0.75 | 3072 |
| 1.00 | 3087 |

## Quality Settings

### Save Settings

```python
# DPI setting (publication quality)
DPI = 300

# Save figure
fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
plt.close(fig)
```

### Korean Font Setup

```python
import matplotlib.pyplot as plt
import platform

def setup_korean_font():
    """Setup Korean font"""
    system = platform.system()
    
    if system == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    elif system == 'Darwin':  # macOS
        plt.rcParams['font.family'] = 'AppleGothic'
    else:  # Linux
        plt.rcParams['font.family'] = 'NanumGothic'
    
    # Prevent minus sign display issues
    plt.rcParams['axes.unicode_minus'] = False
```

## Checklist

Verify the following items when creating X-Y trajectory visualizations:

- [ ] Is Y-axis flipped to display anterior-positive direction?
- [ ] Do axis labels include direction information? (Right +, Left -, Anterior +)
- [ ] Are data points for each window distinguished with correct colors (p1-p4)?
- [ ] Is data outside windows displayed in gray (lightgray)?
- [ ] Is the maximum value point highlighted with a star marker (*)?
- [ ] Is the maximum value marker red (#ED1C24) with white edge?
- [ ] Is the maximum value frame number included in the legend?
- [ ] Is the grid plot sorted by velocity-trial order?
- [ ] Are empty subplots hidden?
- [ ] Is the figure saved at DPI 300?
- [ ] Is Korean font properly configured?
- [ ] Is the same visualization logic applied to both CoP and CoM data?
