# EMG Signal Visualization Guide

> Detailed guidelines for EMG (Electromyography) signal data visualization

## Overview

This guide defines visualization methods for EMG signal data. It includes TKEO (Teager-Kaiser Energy Operator) pipeline visualization, onset timing markers, and window highlights.

## TKEO Pipeline Visualization

The TKEO signal processing pipeline consists of 5 stages. Each stage is visualized as an individual panel to verify the signal transformation process.

### Pipeline Stages

| Stage | Processing | Visualization Color |
|-------|------------|---------------------|
| 1 | Original EMG Signal | Blue (`'b-'`) |
| 2 | Bandpass Filter (30-300 Hz) | Green (`'g-'`) |
| 3 | TKEO Operator | Orange (`'orange'`) |
| 4 | Rectification (Absolute Value) | Purple (`'purple'`) |
| 5 | Lowpass Filter (50 Hz) | Red (`'red'`) |

### Pipeline Visualization Code Example

```python
import matplotlib.pyplot as plt
import numpy as np

def create_tkeo_pipeline_plot(processed_signals, device_frames, channel, subject, velocity, trial_num):
    """
    Visualize 5-stage TKEO pipeline
    
    Parameters:
    -----------
    processed_signals : dict
        Processed signals for each stage {'original', 'bandpass', 'tkeo', 'rectified', 'final'}
    device_frames : np.ndarray
        X-axis frame numbers
    channel : str
        EMG channel name
    subject : str
        Subject ID
    velocity : float
        Velocity condition
    trial_num : int
        Trial number
    """
    fig, axes = plt.subplots(5, 1, figsize=(15, 12))
    fig.suptitle(f'TKEO Processing Pipeline: {channel} | {subject} | Vel: {velocity} | Trial: {trial_num}', 
                 fontsize=16, fontweight='bold')
    
    # Panel 1: Original EMG signal
    axes[0].plot(device_frames, processed_signals['original'], 'b-', linewidth=1, alpha=0.8)
    axes[0].set_title('1. Original EMG Signal', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Amplitude', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: After bandpass filter
    axes[1].plot(device_frames, processed_signals['bandpass'], 'g-', linewidth=1, alpha=0.8)
    axes[1].set_title('2. After Bandpass Filter (30-300 Hz)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Amplitude', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: After TKEO operator
    axes[2].plot(device_frames, processed_signals['tkeo'], 'orange', linewidth=1, alpha=0.8)
    axes[2].set_title('3. After TKEO Operator', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('TKEO Energy', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    # Panel 4: After rectification
    axes[3].plot(device_frames, processed_signals['rectified'], 'purple', linewidth=1, alpha=0.8)
    axes[3].set_title('4. After Rectification (Absolute Value)', fontsize=12, fontweight='bold')
    axes[3].set_ylabel('Rectified Energy', fontsize=10)
    axes[3].grid(True, alpha=0.3)
    
    # Panel 5: Final processed signal
    axes[4].plot(device_frames, processed_signals['final'], 'red', linewidth=1.5, alpha=0.9)
    axes[4].set_title('5. Final Processed Signal (50 Hz Lowpass)', fontsize=12, fontweight='bold')
    axes[4].set_xlabel('DeviceFrame', fontsize=10)
    axes[4].set_ylabel('Final TKEO', fontsize=10)
    axes[4].grid(True, alpha=0.3)
    
    # Maintain consistent X-axis range
    x_min, x_max = np.nanmin(device_frames), np.nanmax(device_frames)
    for ax in axes:
        ax.set_xlim(x_min, x_max)
    
    plt.tight_layout()
    return fig
```

## Onset Timing Marker Display

EMG onset timing is displayed as vertical dashed lines, with frame numbers included in the legend.

### Onset Marker Styles

| Onset Type | Color | Style | Line Width |
|------------|-------|-------|------------|
| TKEO-TH | Blue (`'blue'`) | Dashed (`'--'`) | 2 |
| TKEO-AGLR | Green (`'green'`) | Dashed (`'--'`) | 2 |
| Non-TKEO | Orange (`'orange'`) | Dashed (`'--'`) | 2 |

### Onset Marker Code Example

```python
def add_onset_markers(ax, timing_data):
    """
    Add EMG onset timing markers
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Target axes
    timing_data : dict
        Onset timing information {'TKEO_TH': frame, 'TKEO_AGLR': frame, 'non_TKEO': frame}
    """
    # TKEO-TH onset marker
    if timing_data.get('TKEO_TH') is not None:
        th_onset = timing_data['TKEO_TH']
        ax.axvline(x=th_onset, color='blue', linestyle='--', linewidth=2,
                   label=f'TKEO-TH Onset (Frame {int(th_onset)})')
    
    # TKEO-AGLR onset marker
    if timing_data.get('TKEO_AGLR') is not None:
        aglr_onset = timing_data['TKEO_AGLR']
        ax.axvline(x=aglr_onset, color='green', linestyle='--', linewidth=2,
                   label=f'TKEO-AGLR Onset (Frame {int(aglr_onset)})')
    
    # Non-TKEO onset marker
    if timing_data.get('non_TKEO') is not None:
        non_tkeo_onset = timing_data['non_TKEO']
        ax.axvline(x=non_tkeo_onset, color='orange', linestyle='--', linewidth=2,
                   label=f'Non-TKEO Onset (Frame {int(non_tkeo_onset)})')
    
    ax.legend()
```

## Window Highlight Color Settings

Analysis windows are highlighted with semi-transparent colored regions. Each window (p1, p2, p3, p4) uses a distinct color.

### Window Color Palette

| Window | Color Code | Color Name | Alpha |
|--------|------------|------------|-------|
| p1 | `#1f77b4` | Blue | 0.15 |
| p2 | `#ff7f0e` | Orange | 0.15 |
| p3 | `#2ca02c` | Green | 0.15 |
| p4 | `#d62728` | Red | 0.15 |

### Window Highlight Code Example

```python
# Window color definitions
WINDOW_COLORS = {
    'p1': '#1f77b4',  # Blue
    'p2': '#ff7f0e',  # Orange
    'p3': '#2ca02c',  # Green
    'p4': '#d62728',  # Red
}

def draw_window_spans(ax, window_boundaries, onset_offset=0):
    """
    Draw analysis window highlight regions
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Target axes
    window_boundaries : dict
        Window boundary information {'p1': (start, end), 'p2': (start, end), ...}
    onset_offset : float
        Onset reference offset (for X-axis shift)
    """
    for window_name, (ws, we) in window_boundaries.items():
        color = WINDOW_COLORS.get(window_name, '#999999')
        
        # Shift based on onset
        ws_shifted = ws - onset_offset
        we_shifted = we - onset_offset
        
        ax.axvspan(ws_shifted, we_shifted, 
                   color=color, 
                   alpha=0.15,
                   label=f'{window_name}: {int(ws_shifted)}-{int(we_shifted)}')
```

## Maximum Value Marker Display

The maximum value point of the signal is displayed with a star marker, with frame number included in the legend.

### Maximum Value Marker Style

```python
def add_max_marker(ax, x_data, y_data, color='red'):
    """
    Add star marker at maximum value point
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Target axes
    x_data : np.ndarray
        X-axis data (frame numbers)
    y_data : np.ndarray
        Y-axis data (signal values)
    color : str
        Marker color
    """
    max_idx = np.argmax(y_data)
    max_x = x_data[max_idx]
    max_y = y_data[max_idx]
    
    ax.plot(max_x, max_y, 
            marker='*', 
            markersize=10, 
            color=color,
            label=f'Max (Frame {int(max_x)})')
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

def calculate_grid_dimensions(n_plots):
    """Calculate grid dimensions"""
    if n_plots <= 0:
        return (0, 0)
    cols = math.ceil(math.sqrt(n_plots))
    rows = math.ceil(n_plots / cols)
    return (rows, cols)

def create_emg_grid_plot(data_dict, channel, subject):
    """
    Create EMG signal grid plot
    
    Parameters:
    -----------
    data_dict : dict
        {(velocity, trial_num): {'signal': array, 'frames': array, 'timing': dict}}
    channel : str
        EMG channel name
    subject : str
        Subject ID
    """
    n_plots = len(data_dict)
    rows, cols = calculate_grid_dimensions(n_plots)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    fig.suptitle(f'EMG Signal Grid: {channel} | {subject}', fontsize=16, fontweight='bold')
    
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
        
        # Plot signal
        ax.plot(data['frames'], data['signal'], 'b-', linewidth=1, alpha=0.8)
        
        # Add onset markers if available
        if 'timing' in data:
            add_onset_markers(ax, data['timing'])
        
        # Add window highlights if available
        if 'windows' in data:
            draw_window_spans(ax, data['windows'])
        
        ax.set_title(f'Vel: {velocity} | Trial: {trial_num}', fontsize=10)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    # Hide empty subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig
```

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

Verify the following items when creating EMG visualizations:

- [ ] Are the 5 TKEO pipeline stages displayed with correct colors?
- [ ] Are onset timing markers displayed as vertical dashed lines?
- [ ] Are onset frame numbers included in the legend?
- [ ] Are window highlights displayed with correct colors (p1-p4)?
- [ ] Are window regions semi-transparent (alpha=0.15)?
- [ ] Is the maximum value point displayed with a star marker?
- [ ] Is the grid plot sorted by velocity-trial order?
- [ ] Are empty subplots hidden?
- [ ] Is the figure saved at DPI 300?
- [ ] Is Korean font properly configured?
