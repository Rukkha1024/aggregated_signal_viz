#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid Plot Template for Biomechanics Signal Visualization

This template provides basic functions for creating grid plots of biomechanics signal data.

Key Features:
- Grid dimension calculation (ceil(sqrt(n)))
- Korean font setup
- Basic grid plot creation
- DPI 300 save

Requirements: 3.2, 4.1, 5.4
"""

import math
import platform
from typing import Tuple, Optional, List

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np


# =============================================================================
# Constants
# =============================================================================
DEFAULT_DPI = 300
DEFAULT_SUBPLOT_WIDTH = 12  # inches
DEFAULT_SUBPLOT_HEIGHT = 6  # inches


# =============================================================================
# Grid Dimension Calculation
# =============================================================================
def calculate_grid_dimensions(n_plots: int) -> Tuple[int, int]:
    """
    Calculate optimal grid dimensions (rows, cols) for n_plots.
    
    Number of columns = ceil(sqrt(n_plots))
    Number of rows = ceil(n_plots / cols)
    
    Args:
        n_plots: Number of plots to arrange in grid
        
    Returns:
        Tuple of (rows, cols)
        
    Examples:
        >>> calculate_grid_dimensions(9)
        (3, 3)
        >>> calculate_grid_dimensions(10)
        (4, 4)
        >>> calculate_grid_dimensions(1)
        (1, 1)
    """
    if n_plots <= 0:
        return (0, 0)
    
    cols = math.ceil(math.sqrt(n_plots))
    rows = math.ceil(n_plots / cols)
    
    return (rows, cols)


# =============================================================================
# Korean Font Setup
# =============================================================================
def setup_korean_font() -> str:
    """
    Set up Korean font for matplotlib.
    
    Sets appropriate Korean font based on operating system.
    - Windows: Malgun Gothic
    - macOS: AppleGothic
    - Linux: NanumGothic (fallback: DejaVu Sans)
    
    Returns:
        Name of the font that was set
    """
    system = platform.system()
    
    if system == 'Windows':
        font_name = 'Malgun Gothic'
    elif system == 'Darwin':  # macOS
        font_name = 'AppleGothic'
    else:  # Linux
        # Try NanumGothic first, fallback to DejaVu Sans
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        if 'NanumGothic' in available_fonts:
            font_name = 'NanumGothic'
        else:
            font_name = 'DejaVu Sans'
    
    plt.rcParams['font.family'] = font_name
    plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display
    
    return font_name


# =============================================================================
# Grid Figure Creation
# =============================================================================
def create_grid_figure(
    rows: int,
    cols: int,
    subplot_width: float = DEFAULT_SUBPLOT_WIDTH,
    subplot_height: float = DEFAULT_SUBPLOT_HEIGHT,
    dpi: int = DEFAULT_DPI,
    suptitle: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a grid figure with specified dimensions.
    
    Args:
        rows: Number of rows in grid
        cols: Number of columns in grid
        subplot_width: Width of each subplot in inches
        subplot_height: Height of each subplot in inches
        dpi: Dots per inch for figure resolution
        suptitle: Optional super title for the figure
        
    Returns:
        Tuple of (figure, axes_array)
        axes_array is always 2D for consistent indexing
    """
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive integers")
    
    fig_width = subplot_width * cols
    fig_height = subplot_height * rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), dpi=dpi)
    
    # Normalize axes to 2D array for consistent indexing
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    if suptitle:
        fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=0.995)
    
    return fig, axes


def hide_unused_subplots(axes: np.ndarray, n_used: int) -> None:
    """
    Hide unused subplots in a grid.
    
    Args:
        axes: 2D array of axes from create_grid_figure
        n_used: Number of subplots actually used
    """
    axes_flat = axes.flatten()
    for idx in range(n_used, len(axes_flat)):
        axes_flat[idx].set_visible(False)


# =============================================================================
# Figure Saving
# =============================================================================
def save_figure(
    fig: plt.Figure,
    output_path: str,
    dpi: int = DEFAULT_DPI,
    bbox_inches: str = 'tight',
    facecolor: str = 'white',
    close_after: bool = True,
) -> str:
    """
    Save figure with publication-quality settings.
    
    Args:
        fig: Matplotlib figure to save
        output_path: Path to save the figure
        dpi: Dots per inch (default: 300 for publication quality)
        bbox_inches: Bounding box setting
        facecolor: Background color
        close_after: Whether to close the figure after saving
        
    Returns:
        The output path where figure was saved
    """
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches=bbox_inches,
        facecolor=facecolor,
    )
    
    if close_after:
        plt.close(fig)
    
    return output_path


# =============================================================================
# Velocity-Trial Sorting
# =============================================================================
def sort_velocity_trial_combos(
    velocities: List[float],
    trial_nums: List[int],
) -> List[Tuple[float, int]]:
    """
    Sort velocity-trial combinations for grid arrangement.
    
    Sorting order: velocity ascending, then trial_num ascending within same velocity
    Example: [(10, 1), (10, 2), (10, 3), (15, 1), (15, 2), (20, 1), ...]
    
    Args:
        velocities: List of velocity values
        trial_nums: List of trial numbers (same length as velocities)
        
    Returns:
        Sorted list of (velocity, trial_num) tuples
    """
    if len(velocities) != len(trial_nums):
        raise ValueError("velocities and trial_nums must have same length")
    
    combos = list(zip(velocities, trial_nums))
    unique_combos = list(set(combos))
    sorted_combos = sorted(unique_combos, key=lambda x: (x[0], x[1]))
    
    return sorted_combos


# =============================================================================
# Subplot Styling Utilities
# =============================================================================
def style_subplot(
    ax: plt.Axes,
    title: str,
    xlabel: str = 'Frame',
    ylabel: str = 'Value',
    grid: bool = True,
    grid_alpha: float = 0.3,
    title_fontsize: int = 10,
    label_fontsize: int = 8,
    tick_fontsize: int = 7,
) -> None:
    """
    Apply consistent styling to a subplot.
    
    Args:
        ax: Matplotlib axes to style
        title: Subplot title
        xlabel: X-axis label
        ylabel: Y-axis label
        grid: Whether to show grid
        grid_alpha: Grid transparency
        title_fontsize: Font size for title
        label_fontsize: Font size for axis labels
        tick_fontsize: Font size for tick labels
    """
    ax.set_title(title, fontsize=title_fontsize, pad=5)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    
    if grid:
        ax.grid(True, alpha=grid_alpha)


def add_legend(
    ax: plt.Axes,
    loc: str = 'best',
    fontsize: int = 6,
    framealpha: float = 0.8,
) -> None:
    """
    Add legend to subplot if there are labeled elements.
    
    Args:
        ax: Matplotlib axes
        loc: Legend location
        fontsize: Legend font size
        framealpha: Legend frame transparency
    """
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc=loc, fontsize=fontsize, framealpha=framealpha)


# =============================================================================
# Window Colors (Biomechanics Standard)
# =============================================================================
WINDOW_COLORS = {
    'p1': '#1f77b4',  # Blue
    'p2': '#ff7f0e',  # Orange
    'p3': '#2ca02c',  # Green
    'p4': '#d62728',  # Red
}

FORCEPLATE_COLORS = {
    'Fx': 'purple',
    'Fy': 'brown',
    'Fz': 'green',
}


# =============================================================================
# Example Usage
# =============================================================================
if __name__ == '__main__':
    # Example: Create a sample grid plot
    setup_korean_font()
    
    # Calculate grid dimensions for 10 plots
    n_plots = 10
    rows, cols = calculate_grid_dimensions(n_plots)
    print(f"Grid for {n_plots} plots: {rows} rows x {cols} cols")
    
    # Create figure
    fig, axes = create_grid_figure(rows, cols, suptitle='Sample Grid Plot')
    
    # Fill subplots with sample data
    axes_flat = axes.flatten()
    for i in range(n_plots):
        ax = axes_flat[i]
        x = np.linspace(0, 10, 100)
        y = np.sin(x + i)
        ax.plot(x, y)
        style_subplot(ax, title=f'Plot {i+1}', xlabel='Time', ylabel='Signal')
    
    # Hide unused subplots
    hide_unused_subplots(axes, n_plots)
    
    # Save (commented out for template)
    # save_figure(fig, 'sample_grid.png')
    
    plt.show()
