#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMG and Signal Grid Visualization Module (V5)

Subject별로 모든 velocity-trial 조합을 하나의 grid plot에 배치.
- EMG: TKEO-AGLR onset timing 마커 + Window span 하이라이트
- Forceplate: Fx, Fy, Fz 각각 grid plot
- CoP: Cx vs Cy trajectory grid plot
"""

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List

# Import from utils module
from emg_pipeline.utils import (
    setup_korean_font,
    setup_logger,
    load_config,
    get_window_boundaries,
    resolve_windowed_column_name,
    EMGSignalPipelineProcessor,
)

# Initialize
setup_korean_font()
logger = setup_logger(__name__)

# Constants
SUBPLOT_WIDTH = 12  # inches
SUBPLOT_HEIGHT = 6  # inches
DPI = 300
PLOT_EMG_INPUT_PATH = os.path.join('data', 'processed_emg_data.csv')


def get_velocity_trial_combos(df: pd.DataFrame, subject: str) -> list:
    """Get sorted velocity-trial combinations for a subject."""
    subj_df = df[df['subject'] == subject][['velocity', 'trial_num']].drop_duplicates()
    subj_df = subj_df.sort_values(['velocity', 'trial_num'])
    return list(subj_df.itertuples(index=False, name=None))


def calculate_grid_dimensions(n_plots: int) -> tuple:
    """Calculate optimal grid dimensions (rows, cols) for n_plots."""
    if n_plots <= 0:
        return (0, 0)
    cols = math.ceil(math.sqrt(n_plots))
    rows = math.ceil(n_plots / cols)
    return (rows, cols)


def compute_onset_offset(trial_df, timings_df, subject, velocity, trial_num, config):
    """Compute platform onset offset for x-axis alignment."""
    try:
        dev_hz = float(config.get('windowing', {}).get('sampling_rate', 1000))
        mocap_hz = float(config.get('mocap', {}).get('sampling_rate', 100))
        ratio = dev_hz / mocap_hz if mocap_hz else 1.0

        trial_rows = timings_df[
            (timings_df['subject'] == subject) &
            (timings_df['velocity'] == velocity) &
            (timings_df['trial_num'] == trial_num)
        ]
        dataset_row = trial_rows.iloc[0] if not trial_rows.empty else None

        if dataset_row is not None and 'platform_onset' in dataset_row.index:
            odf_min = float(pd.to_numeric(trial_df['original_DeviceFrame'], errors='coerce').min())
            df_min = float(pd.to_numeric(trial_df['DeviceFrame'], errors='coerce').min())
            odf_onset = float(dataset_row['platform_onset']) * ratio
            onset_df_abs = (odf_onset - odf_min) + df_min
            return onset_df_abs, dataset_row
    except Exception:
        pass
    return 0.0, None


def draw_window_spans(ax, dataset_row, config, onset_df_abs, trial_df):
    """Draw window spans on the axes."""
    if dataset_row is None:
        return
    try:
        vis_cfg = config.get('visualization', {})
        window_colors = vis_cfg.get('window_colors', {})
        default_fallback = vis_cfg.get('plot_colors', {}).get('default_fallback', '#999999')

        wb = get_window_boundaries(dataset_row, config_data=config, domain='DeviceFrame', trial_df=trial_df)
        all_bounds = wb.get('boundaries', {})
        palette = wb.get('palette', window_colors)

        for wname, (ws, we) in all_bounds.items():
            color = palette.get(wname, default_fallback)
            ws_shifted = ws - onset_df_abs
            we_shifted = we - onset_df_abs
            ax.axvspan(ws_shifted, we_shifted, color=color, alpha=0.15,
                       label=f'{wname}: {int(ws_shifted)}-{int(we_shifted)}')
    except Exception as e:
        logger.warning(f"Failed to draw window spans: {e}")



def create_emg_subplot(
    ax,
    trial_df: pd.DataFrame,
    channel: str,
    velocity: float,
    trial_num: int,
    subject: str,
    timings_df: pd.DataFrame,
    config: dict,
    emg_processor=None,
):
    """Create a single EMG signal subplot with TKEO-AGLR onset timing."""
    if channel not in trial_df.columns:
        ax.text(0.5, 0.5, f'{channel} not found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Vel:{velocity} Trial:{int(trial_num)}', fontsize=20, fontweight='bold')
        return

    trial_df = trial_df.sort_values('DeviceFrame').copy()
    vis_cfg = config.get('visualization', {})

    # Compute onset offset
    onset_df_abs, dataset_row = compute_onset_offset(trial_df, timings_df, subject, velocity, trial_num, config)

    # Process EMG signal
    if emg_processor is not None:
        try:
            processed_data = emg_processor.process_emg_data(trial_df.copy(), [channel])
            y_vals = processed_data[channel].values
        except Exception:
            y_vals = trial_df[channel].values
    else:
        y_vals = trial_df[channel].values

    x_vals = trial_df['DeviceFrame'].values.astype(float) - float(onset_df_abs)

    # Plot signal
    ax.plot(x_vals, y_vals, 'b-', linewidth=0.8, alpha=0.8)

    # Draw window spans
    draw_window_spans(ax, dataset_row, config, onset_df_abs, trial_df)

    # Add timing markers - TKEO-AGLR onset
    channel_timings = timings_df[
        (timings_df['subject'] == subject) &
        (timings_df['velocity'] == velocity) &
        (timings_df['trial_num'] == trial_num) &
        (timings_df['emg_channel'] == channel)
    ]

    if not channel_timings.empty:
        timing_row = channel_timings.iloc[0]

        # TKEO-AGLR Onset timing marker
        onset_col = 'TKEO_AGLR_emg_onset_timing'
        if pd.notna(timing_row.get(onset_col)):
            onset_time = timing_row[onset_col]
            ax.axvline(x=onset_time, color='red', linestyle='--', linewidth=1.5,
                       label=f'TKEO-AGLR Onset ({int(onset_time)})')

        # Max amplitude timing marker
        highlight_window = vis_cfg.get('highlight_window', None)
        max_col = resolve_windowed_column_name(list(timing_row.index), 'max_amp_timing', highlight_window)
        if max_col and pd.notna(timing_row.get(max_col)):
            max_time = timing_row[max_col]
            ax.axvline(x=max_time, color='orange', linestyle='--', linewidth=1.5,
                       label=f'Max Amp ({int(max_time)})')

    # Title and labels
    ax.set_title(f'Vel:{velocity} Trial:{int(trial_num)}', fontsize=20, fontweight='bold', pad=5)
    ax.set_xlabel('Frame (onset=0)', fontsize=8)
    ax.set_ylabel(channel, fontsize=8)
    ax.tick_params(axis='both', labelsize=7)
    ax.grid(True, alpha=0.3)

    try:
        ax.set_xlim(float(np.nanmin(x_vals)), float(np.nanmax(x_vals)))
    except Exception:
        pass

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='best', fontsize=6, framealpha=0.8)


def create_forceplate_subplot(
    ax,
    trial_df: pd.DataFrame,
    channel: str,
    velocity: float,
    trial_num: int,
    subject: str,
    timings_df: pd.DataFrame,
    config: dict,
):
    """Create a single forceplate signal subplot (Fx, Fy, Fz)."""
    if channel not in trial_df.columns:
        ax.text(0.5, 0.5, f'{channel} not found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Vel:{velocity} Trial:{int(trial_num)}', fontsize=20, fontweight='bold')
        return

    trial_df = trial_df.sort_values('DeviceFrame').copy()

    # Compute onset offset
    onset_df_abs, dataset_row = compute_onset_offset(trial_df, timings_df, subject, velocity, trial_num, config)

    x_vals = trial_df['DeviceFrame'].values.astype(float) - float(onset_df_abs)
    y_vals = trial_df[channel].values

    # Plot signal with different colors per channel
    colors = {'Fx': 'purple', 'Fy': 'brown', 'Fz': 'green'}
    ax.plot(x_vals, y_vals, color=colors.get(channel, 'black'), linewidth=0.8, alpha=0.8)

    # Draw window spans
    draw_window_spans(ax, dataset_row, config, onset_df_abs, trial_df)

    # Add onset timing marker
    trial_timings = timings_df[
        (timings_df['subject'] == subject) &
        (timings_df['velocity'] == velocity) &
        (timings_df['trial_num'] == trial_num)
    ]

    if not trial_timings.empty:
        timing_row = trial_timings.iloc[0]
        onset_col = f'{channel.lower()}_onset_timing'
        if onset_col in timing_row.index and pd.notna(timing_row.get(onset_col)):
            onset_time = timing_row[onset_col]
            ax.axvline(x=onset_time, color='red', linestyle='--', linewidth=1.5,
                       label=f'{channel} Onset ({int(onset_time)})')

    # Title and labels
    ax.set_title(f'Vel:{velocity} Trial:{int(trial_num)}', fontsize=20, fontweight='bold', pad=5)
    ax.set_xlabel('Frame (onset=0)', fontsize=8)
    ax.set_ylabel(f'{channel} Value', fontsize=8)
    ax.tick_params(axis='both', labelsize=7)
    ax.grid(True, alpha=0.3)

    try:
        ax.set_xlim(float(np.nanmin(x_vals)), float(np.nanmax(x_vals)))
    except Exception:
        pass

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='best', fontsize=6, framealpha=0.8)


def create_cop_subplot(
    ax,
    trial_df: pd.DataFrame,
    velocity: float,
    trial_num: int,
    subject: str,
    timings_df: pd.DataFrame,
    config: dict,
):
    """Create a single CoP scatter plot (Cx vs Cy)."""
    if 'Cx' not in trial_df.columns or 'Cy' not in trial_df.columns:
        ax.text(0.5, 0.5, 'Cx/Cy not found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Vel:{velocity} Trial:{int(trial_num)}', fontsize=20, fontweight='bold')
        return

    trial_df = trial_df.sort_values('DeviceFrame').copy()
    vis_cfg = config.get('visualization', {})
    plot_colors = vis_cfg.get('plot_colors', {})

    # Compute onset offset
    onset_df_abs, dataset_row = compute_onset_offset(trial_df, timings_df, subject, velocity, trial_num, config)

    # Prepare data (flip Cy for Anterior +)
    x = trial_df['Cx'].values
    y = -trial_df['Cy'].values

    # Get window boundaries
    all_boundaries = {}
    if dataset_row is not None:
        try:
            wb = get_window_boundaries(dataset_row, config_data=config, domain='DeviceFrame', trial_df=trial_df)
            all_boundaries = wb.get('boundaries', {})
        except Exception:
            pass

    # Create window masks
    window_masks = {}
    for w_name, (ws, we) in all_boundaries.items():
        window_masks[w_name] = (trial_df['DeviceFrame'] >= ws) & (trial_df['DeviceFrame'] <= we)

    # Plot with window colors
    window_colors = vis_cfg.get('window_colors', {})
    if window_masks:
        for w_name, mask in window_masks.items():
            color = window_colors.get(w_name, None)
            mask_arr = np.asarray(mask)
            if np.sum(mask_arr) > 0:
                ax.scatter(x[mask_arr], y[mask_arr], c=color, s=8, alpha=0.7, label=f'{w_name}')

        # Plot remaining points
        combined = np.zeros(len(trial_df), dtype=bool)
        for m in window_masks.values():
            combined |= np.asarray(m)
        if np.sum(~combined) > 0:
            ax.scatter(x[~combined], y[~combined], c=plot_colors.get('background', 'lightgray'),
                       alpha=0.3, s=6)
    else:
        ax.scatter(x, y, c='blue', s=8, alpha=0.7)

    # Add CoP max marker
    trial_timings = timings_df[
        (timings_df['subject'] == subject) &
        (timings_df['velocity'] == velocity) &
        (timings_df['trial_num'] == trial_num)
    ]
    if not trial_timings.empty:
        timing_row = trial_timings.iloc[0]
        highlight_window = vis_cfg.get('highlight_window', 'p4')
        cop_max_col = f'cop_max_timing_{highlight_window}'
        if cop_max_col not in timing_row.index:
            cop_max_col = 'cop_max_timing'

        if cop_max_col in timing_row.index and pd.notna(timing_row.get(cop_max_col)):
            max_timing = timing_row[cop_max_col]
            max_mask = trial_df['DeviceFrame'] == max_timing
            if np.sum(max_mask) > 0:
                max_x = trial_df[max_mask]['Cx'].iloc[0]
                max_y = -trial_df[max_mask]['Cy'].iloc[0]
                ax.scatter(max_x, max_y, c=plot_colors.get('max_highlight', '#ED1C24'),
                           s=80, marker='*', edgecolor='white', linewidth=1, zorder=10,
                           label=f'Max ({int(max_timing)})')

    # Title and labels
    ax.set_title(f'Vel:{velocity} Trial:{int(trial_num)}', fontsize=20, fontweight='bold', pad=5)
    ax.set_xlabel('Cx (R+/L-)', fontsize=8)
    ax.set_ylabel('Cy (A+)', fontsize=8)
    ax.tick_params(axis='both', labelsize=7)
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='best', fontsize=5, framealpha=0.8)



def generate_grid_plot(
    raw_data: pd.DataFrame,
    timings_df: pd.DataFrame,
    subject: str,
    channel: str,
    output_dir: str,
    config: dict,
    plot_type: str = 'emg',
) -> str:
    """
    Generate a grid plot for a single subject and channel.

    Args:
        raw_data: Raw signal data
        timings_df: Timing data from final_dataset
        subject: Subject name
        channel: Channel name (EMG channel, Fx/Fy/Fz, or 'CoP')
        output_dir: Output directory
        config: Configuration dictionary
        plot_type: 'emg', 'forceplate', or 'cop'

    Returns:
        Output file path or None
    """
    combos = get_velocity_trial_combos(raw_data, subject)
    n_plots = len(combos)

    if n_plots == 0:
        logger.warning(f"No data for subject {subject}")
        return None

    logger.info(f"Generating {plot_type} grid plot for {subject} - {channel}: {n_plots} plots")

    rows, cols = calculate_grid_dimensions(n_plots)

    # Adjust subplot size for CoP (square-ish)
    if plot_type == 'cop':
        fig_width = 8 * cols
        fig_height = 8 * rows
    else:
        fig_width = SUBPLOT_WIDTH * cols
        fig_height = SUBPLOT_HEIGHT * rows

    # Initialize EMG processor for EMG plots
    emg_processor = None
    if plot_type == 'emg':
        try:
            emg_processor = EMGSignalPipelineProcessor(config)
        except Exception as e:
            logger.warning(f"Failed to initialize EMG processor: {e}")

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), dpi=DPI)

    title_map = {
        'emg': f'{subject} - {channel} EMG Signal Grid',
        'forceplate': f'{subject} - {channel} Forceplate Signal Grid',
        'cop': f'{subject} - CoP Trajectory Grid',
    }
    fig.suptitle(title_map.get(plot_type, f'{subject} - {channel}'), fontsize=16, fontweight='bold', y=0.995)

    # Flatten axes
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    axes_flat = axes.flatten()

    # Plot each velocity-trial combination
    for idx, (velocity, trial_num) in enumerate(combos):
        ax = axes_flat[idx]

        trial_df = raw_data[
            (raw_data['subject'] == subject) &
            (raw_data['velocity'] == velocity) &
            (raw_data['trial_num'] == trial_num)
        ]

        if plot_type == 'emg':
            create_emg_subplot(ax, trial_df, channel, velocity, trial_num, subject, timings_df, config, emg_processor)
        elif plot_type == 'forceplate':
            create_forceplate_subplot(ax, trial_df, channel, velocity, trial_num, subject, timings_df, config)
        elif plot_type == 'cop':
            create_cop_subplot(ax, trial_df, velocity, trial_num, subject, timings_df, config)

    # Hide unused subplots
    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save
    os.makedirs(output_dir, exist_ok=True)
    if plot_type == 'cop':
        output_path = os.path.join(output_dir, f'{subject}_CoP_grid.png')
    else:
        output_path = os.path.join(output_dir, f'{subject}_{channel}_grid.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Saved: {output_path}")
    return output_path


def generate_all_signal_plots(
    raw_data: Optional[pd.DataFrame] = None,
    timing_data: Optional[pd.DataFrame] = None,
    limit_emg: Optional[int] = None,
    limit_fz: Optional[int] = None,
):
    """
    Generate grid plots for all subjects.

    Args:
        raw_data: Raw signal data (loaded from config if None)
        timing_data: Timing data (loaded from config if None)
        limit_emg: If provided, limit to first N subjects for EMG
        limit_fz: If provided, limit to first N subjects for Forceplate/CoP
    """
    logger.info("Starting grid plot generation...")

    # Load configuration
    try:
        config = load_config('config.yaml')
    except Exception as e:
        logger.error(f"Could not load configuration: {e}")
        return

    # Load data if not provided
    if raw_data is None:
        raw_data_path = PLOT_EMG_INPUT_PATH  # Hardcode plot input to avoid config interference
        if not os.path.exists(raw_data_path):
            logger.error(f"Raw data file for plotting not found: {raw_data_path}")
            return
        raw_data = pd.read_csv(raw_data_path)
        logger.info(f"Loaded raw data for plotting from {raw_data_path}: {len(raw_data)} rows")

    if timing_data is None:
        timing_path = config['data_paths']['final_dataset_filename']
        if not os.path.exists(timing_path):
            logger.error(f"Timing data file not found: {timing_path}")
            return
        timing_data = pd.read_csv(timing_path)
        logger.info(f"Loaded timing data: {len(timing_data)} rows")

    # Output directory
    output_dir = config['data_paths']['signal_plot_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Get all subjects
    subjects = raw_data['subject'].unique().tolist()
    logger.info(f"Found {len(subjects)} subjects")

    # Get EMG channels from config
    emg_channels = config.get('emg_parameters', {}).get('channels', [])

    # Apply limits if in sample mode
    emg_subjects = subjects[:limit_emg] if limit_emg else subjects
    fp_subjects = subjects[:limit_fz] if limit_fz else subjects

    # Generate EMG grid plots
    logger.info("=" * 50)
    logger.info("Generating EMG grid plots...")
    for subject in emg_subjects:
        for channel in emg_channels:
            try:
                generate_grid_plot(
                    raw_data=raw_data,
                    timings_df=timing_data,
                    subject=subject,
                    channel=channel,
                    output_dir=output_dir,
                    config=config,
                    plot_type='emg',
                )
            except Exception as e:
                logger.warning(f"Failed EMG grid for {subject}-{channel}: {e}")

    # Generate Forceplate grid plots (Fx, Fy, Fz)
    logger.info("=" * 50)
    logger.info("Generating Forceplate grid plots...")
    for subject in fp_subjects:
        for fp_channel in ['Fx', 'Fy', 'Fz']:
            try:
                generate_grid_plot(
                    raw_data=raw_data,
                    timings_df=timing_data,
                    subject=subject,
                    channel=fp_channel,
                    output_dir=output_dir,
                    config=config,
                    plot_type='forceplate',
                )
            except Exception as e:
                logger.warning(f"Failed Forceplate grid for {subject}-{fp_channel}: {e}")

    # Generate CoP grid plots
    logger.info("=" * 50)
    logger.info("Generating CoP grid plots...")
    for subject in fp_subjects:
        try:
            generate_grid_plot(
                raw_data=raw_data,
                timings_df=timing_data,
                subject=subject,
                channel='CoP',
                output_dir=output_dir,
                config=config,
                plot_type='cop',
            )
        except Exception as e:
            logger.warning(f"Failed CoP grid for {subject}: {e}")

    logger.info("=" * 50)
    logger.info("Grid plot generation complete!")


def generate_sample_plots(
    merged_data: Optional[pd.DataFrame] = None,
    final_dataset: Optional[pd.DataFrame] = None,
    limit_per_type: int = 10,
):
    """
    Generate sample grid plots (limited subjects).

    Args:
        merged_data: Raw signal data
        final_dataset: Timing data
        limit_per_type: Number of subjects to process
    """
    logger.info(f"Generating sample grid plots (limit: {limit_per_type} subjects)...")
    generate_all_signal_plots(
        raw_data=merged_data,
        timing_data=final_dataset,
        limit_emg=limit_per_type,
        limit_fz=limit_per_type,
    )


# Legacy compatibility aliases
def generate_generic_signal_plots(*args, **kwargs):
    """Legacy compatibility - redirects to generate_all_signal_plots."""
    logger.info("generate_generic_signal_plots called - redirecting to grid plots")
    return generate_all_signal_plots(*args, **kwargs)


if __name__ == '__main__':
    generate_all_signal_plots()
