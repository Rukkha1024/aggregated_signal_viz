# Forceplate Signal Visualization Guide

> Detailed guidelines for Forceplate (Ground Reaction Force) signal data visualization

## Overview

This guide defines visualization methods for Forceplate signal data (Fx, Fy, Fz). It includes channel-specific color settings, onset timing markers, and window highlights.

## Forceplate Channel Configuration

Forceplate measures ground reaction forces in 3 axes.

| Channel | Direction | Description | Unit |
|---------|-----------|-------------|------|
| Fx | Medio-Lateral | Left-right force | N (Newton) |
| Fy | Anterior-Posterior | Front-back force | N (Newton) |
| Fz | Vertical | Vertical force | N (Newton) |

## Channel Color Settings

Each Forceplate channel uses a unique color for distinction.

### Channel Color Palette

| Channel | Color Code | Color Name | Line Style |
|---------|------------|------------|------------|
| Fx | `#1f77b4` | Blue | Solid (`'-'`) |
| Fy | `#ff7f0e` | Orange | Solid (`'-'`) |
| Fz | `#2ca02c` | Green | Solid (`'-'`) |

### Channel Color Definition Code

```python
# Forceplate channel color definitions
FORCEPLATE_COLORS = {
    'Fx': '#1f77b4',  # Blue - Medio-Lateral
    'Fy': '#ff7f0e',  # Orange - Anterior-Posterior
    'Fz': '#2ca02c',  # Green - Vertical
}

FORCEPLATE_LABELS = {
    'Fx': 'Fx (Medio-Lateral)',
    'Fy': 'Fy (Anterior-Posterior)',
    'Fz': 'Fz (Vertical)',
}
```

## Single Channel Visualization

Method for visualizing individual Forceplate channels.

### Single Channel Plot Code Example

```python
import matplotlib.pyplot as plt
import numpy as np

def create_single_channel_plot(frames, signal, channel, subject, velocity, trial_num):
    """
    Visualize single Forceplate channel
    
    Parameters:
    -----------
    frames : np.ndarray
        X-axis frame numbers
    signal : np.ndarray
        Forceplate signal values
    channel : str
        Channel name ('Fx', 'Fy', 'Fz')
    subject : str
        Subject ID
    velocity : float
        Velocity condition
    trial_num : int
        Trial number
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    color = FORCEPLATE_COLORS.get(channel, '#333333')
    label = FORCEPLATE_LABELS.get(channel, channel)
    
    ax.plot(frames, signal, color=color, linewidth=1.5, alpha=0.9, label=label)
    
    ax.set_title(f'Forceplate {channel} | {subject} | Vel: {velocity} | Trial: {trial_num}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Force (N)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    return fig
```

## Multi-Channel Visualization

Method for displaying Fx, Fy, Fz three channels in a single figure.

### Multi-Channel Plot Code Example

```python
def create_multi_channel_plot(frames, signals, subject, velocity, trial_num):
    """
    Visualize 3 Forceplate channels simultaneously
    
    Parameters:
    -----------
    frames : np.ndarray
        X-axis frame numbers
    signals : dict
        Channel signals {'Fx': array, 'Fy': array, 'Fz': array}
    subject : str
        Subject ID
    velocity : float
        Velocity condition
    trial_num : int
        Trial number
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Forceplate Signals | {subject} | Vel: {velocity} | Trial: {trial_num}',
                 fontsize=14, fontweight='bold')
    
    channels = ['Fx', 'Fy', 'Fz']
    
    for idx, channel in enumerate(channels):
        ax = axes[idx]
        signal = signals.get(channel)
        
        if signal is not None:
            color = FORCEPLATE_COLORS[channel]
            label = FORCEPLATE_LABELS[channel]
            
            ax.plot(frames, signal, color=color, linewidth=1.5, alpha=0.9, label=label)
            ax.set_ylabel('Force (N)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)
            ax.set_title(f'{label}', fontsize=11, fontweight='bold')
    
    axes[-1].set_xlabel('Frame', fontsize=12)
    
    plt.tight_layout()
    return fig
```

## Onset Timing Marker Display

Forceplate onset timing is displayed as vertical dashed lines, with frame numbers included in the legend.

### Onset Marker Styles

| Onset Type | Color | Style | Line Width |
|------------|-------|-------|------------|
| Fz Onset | Red (`'red'`) | Dashed (`'--'`) | 2 |
| Threshold Onset | Purple (`'purple'`) | Dashed (`'--'`) | 2 |
| Manual Onset | Black (`'black'`) | Dashed (`'--'`) | 2 |

### Onset Marker Code Example

```python
def add_forceplate_onset_markers(ax, timing_data):
    """
    Add Forceplate onset timing markers
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Target axes
    timing_data : dict
        Onset timing information {'Fz_onset': frame, 'threshold_onset': frame, 'manual_onset': frame}
    """
    # Fz onset marker (primary onset)
    if timing_data.get('Fz_onset') is not None:
        fz_onset = timing_data['Fz_onset']
        ax.axvline(x=fz_onset, color='red', linestyle='--', linewidth=2,
                   label=f'Fz Onset (Frame {int(fz_onset)})')
    
    # Threshold onset marker
    if timing_data.get('threshold_onset') is not None:
        th_onset = timing_data['threshold_onset']
        ax.axvline(x=th_onset, color='purple', linestyle='--', linewidth=2,
                   label=f'Threshold Onset (Frame {int(th_onset)})')
    
    # Manual onset marker
    if timing_data.get('manual_onset') is not None:
        manual_onset = timing_data['manual_onset']
        ax.axvline(x=manual_onset, color='black', linestyle='--', linewidth=2,
                   label=f'Manual Onset (Frame {int(manual_onset)})')
    
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

def draw_forceplate_window_spans(ax, window_boundaries, onset_offset=0):
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
def add_forceplate_max_marker(ax, x_data, y_data, channel='Fz'):
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
    channel : str
        Channel name (for marker color)
    """
    max_idx = np.argmax(np.abs(y_data))  # Maximum by absolute value
    max_x = x_data[max_idx]
    max_y = y_data[max_idx]
    
    # Marker color by channel
    marker_colors = {
        'Fx': '#1f77b4',
        'Fy': '#ff7f0e',
        'Fz': '#2ca02c',
    }
    color = marker_colors.get(channel, 'red')
    
    ax.plot(max_x, max_y, 
            marker='*', 
            markersize=12, 
            color=color,
            markeredgecolor='black',
            markeredgewidth=0.5,
            label=f'{channel} Max (Frame {int(max_x)}, {max_y:.1f} N)')
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

def create_forceplate_grid_plot(data_dict, channel, subject):
    """
    Create Forceplate signal grid plot
    
    Parameters:
    -----------
    data_dict : dict
        {(velocity, trial_num): {'signal': array, 'frames': array, 'timing': dict}}
    channel : str
        Forceplate channel name ('Fx', 'Fy', 'Fz')
    subject : str
        Subject ID
    """
    n_plots = len(data_dict)
    rows, cols = calculate_grid_dimensions(n_plots)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    fig.suptitle(f'Forceplate {channel} Grid | {subject}', fontsize=16, fontweight='bold')
    
    # Flatten axes for easy iteration
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Sort by velocity, then trial_num
    sorted_keys = sorted(data_dict.keys(), key=lambda x: (x[0], x[1]))
    
    color = FORCEPLATE_COLORS.get(channel, '#333333')
    
    for idx, (velocity, trial_num) in enumerate(sorted_keys):
        ax = axes[idx]
        data = data_dict[(velocity, trial_num)]
        
        # Plot signal
        ax.plot(data['frames'], data['signal'], color=color, linewidth=1.5, alpha=0.9)
        
        # Add onset markers if available
        if 'timing' in data:
            add_forceplate_onset_markers(ax, data['timing'])
        
        # Add window highlights if available
        if 'windows' in data:
            draw_forceplate_window_spans(ax, data['windows'])
        
        # Add max marker if requested
        if data.get('show_max', False):
            add_forceplate_max_marker(ax, data['frames'], data['signal'], channel)
        
        ax.set_title(f'Vel: {velocity} | Trial: {trial_num}', fontsize=10)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Force (N)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')
    
    # Hide empty subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig
```

## EMG and Forceplate Combined Visualization

Visualization for comparing EMG onset and Forceplate onset together.

### EMG-Forceplate Combined Plot Code Example

```python
def create_emg_forceplate_combined_plot(frames, emg_signal, fz_signal, 
                                         emg_timing, fz_timing,
                                         subject, velocity, trial_num):
    """
    Visualize EMG and Forceplate Fz simultaneously
    
    Parameters:
    -----------
    frames : np.ndarray
        X-axis frame numbers
    emg_signal : np.ndarray
        EMG signal (normalized)
    fz_signal : np.ndarray
        Forceplate Fz signal
    emg_timing : dict
        EMG onset timing information
    fz_timing : dict
        Forceplate onset timing information
    subject : str
        Subject ID
    velocity : float
        Velocity condition
    trial_num : int
        Trial number
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f'EMG & Forceplate | {subject} | Vel: {velocity} | Trial: {trial_num}',
                 fontsize=14, fontweight='bold')
    
    # Panel 1: EMG Signal
    axes[0].plot(frames, emg_signal, 'b-', linewidth=1.5, alpha=0.9, label='EMG')
    if emg_timing.get('TKEO_TH') is not None:
        axes[0].axvline(x=emg_timing['TKEO_TH'], color='blue', linestyle='--', 
                        linewidth=2, label=f"EMG Onset (Frame {int(emg_timing['TKEO_TH'])})")
    axes[0].set_ylabel('EMG Amplitude', fontsize=11)
    axes[0].set_title('EMG Signal', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)
    
    # Panel 2: Forceplate Fz
    axes[1].plot(frames, fz_signal, color='#2ca02c', linewidth=1.5, alpha=0.9, label='Fz')
    if fz_timing.get('Fz_onset') is not None:
        axes[1].axvline(x=fz_timing['Fz_onset'], color='red', linestyle='--', 
                        linewidth=2, label=f"Fz Onset (Frame {int(fz_timing['Fz_onset'])})")
    axes[1].set_xlabel('Frame', fontsize=12)
    axes[1].set_ylabel('Force (N)', fontsize=11)
    axes[1].set_title('Forceplate Fz (Vertical)', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)
    
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

Verify the following items when creating Forceplate visualizations:

- [ ] Are Fx, Fy, Fz channels distinguished with correct colors?
- [ ] Are direction labels accurate for each channel? (Fx: Medio-Lateral, Fy: Anterior-Posterior, Fz: Vertical)
- [ ] Is Y-axis unit displayed as Force (N)?
- [ ] Are onset timing markers displayed as vertical dashed lines?
- [ ] Are onset frame numbers included in the legend?
- [ ] Are window highlights displayed with correct colors (p1-p4)?
- [ ] Are window regions semi-transparent (alpha=0.15)?
- [ ] Is the maximum value point displayed with a star marker?
- [ ] Is the grid plot sorted by velocity-trial order?
- [ ] Are empty subplots hidden?
- [ ] Is the figure saved at DPI 300?
- [ ] Is Korean font properly configured?
