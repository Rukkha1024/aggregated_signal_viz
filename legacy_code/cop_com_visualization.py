import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, Dict, Any

# Import from new utils module (consolidated utilities)
from .utils import (
    setup_korean_font, 
    setup_logger, 
    run_parallel_plotting,
    EMGKoreanEncodingHandler,
    load_config,
    get_window_boundaries,
)

# Initialize Korean font support and logger
setup_korean_font()
logger = setup_logger(__name__)

# Default fallback frame values used throughout the module
DEFAULT_FRAME_START, DEFAULT_FRAME_END = 3027, 3087

def get_onset_offset_from_dataset(final_dataset, subject, velocity, trial_num=None):
    """
    Extract onset/offset values based on windowing configuration.
    """
    # Default fallback values
    fallback_start, fallback_end = DEFAULT_FRAME_START, DEFAULT_FRAME_END
    
    try:
        # Load configuration to get highlight window
        config = load_config('config.yaml')
        highlight_window = config.get('visualization', {}).get('highlight_window', 'full_window')
        windowing_config = config.get('windowing', {})
        
        # Check if windowing is enabled
        if not windowing_config.get('enabled', True):
            # Fallback to legacy behavior
            logger.info("Windowing disabled, using legacy platform_onset/platform_offset")
            return _get_legacy_onset_offset_from_dataset(final_dataset, subject, velocity, trial_num)
        
        # Get window configuration
        windows_config = windowing_config.get('windows', {})
        if highlight_window not in windows_config:
            logger.warning(f"Highlight window '{highlight_window}' not found in config. Using legacy approach.")
            return _get_legacy_onset_offset_from_dataset(final_dataset, subject, velocity, trial_num)
        
        window_config = windows_config[highlight_window]
        sampling_rate = windowing_config.get('sampling_rate', 1000)
        
        # Check if required columns exist
        required_cols = ['subject', 'velocity', 'trial_num', 'platform_onset', 'platform_offset']
        missing_cols = [col for col in required_cols if col not in final_dataset.columns]
        if missing_cols:
            logger.warning(f"Missing columns {missing_cols} in dataset. Using fallback values.")
            return fallback_start, fallback_end
        
        # Filter by subject and velocity
        subject_data = final_dataset[final_dataset['subject'] == subject]
        if len(subject_data) == 0:
            logger.warning(f"No data found for subject '{subject}'. Using fallback values.")
            return fallback_start, fallback_end
        
        # Try to find exact match with both velocity and trial if provided
        matching_data = pd.DataFrame()
        if trial_num is not None:
            matching_data = subject_data[
                (subject_data['velocity'] == velocity) & 
                (subject_data['trial_num'] == trial_num)
            ]
        
        # If no match with trial, try velocity only
        if len(matching_data) == 0:
            matching_data = subject_data[subject_data['velocity'] == velocity]
        
        if len(matching_data) > 0:
            row = matching_data.iloc[0]
            
            # Calculate window boundaries using centralized helper
            res = get_window_boundaries(
                row,
                config_data=config,
                domain='original_DeviceFrame',
                include_all=True,
                include_highlight=True,
            )
            hl = res.get('highlight_range')
            if hl is not None:
                frame_start, frame_end = hl
                logger.info(f"Found window boundaries for {subject}, velocity {velocity}, window '{highlight_window}': {frame_start}-{frame_end}")
                return frame_start, frame_end
            else:
                logger.warning(f"Highlight window '{highlight_window}' has no range; falling back to legacy approach.")
                return _get_legacy_onset_offset_from_dataset(final_dataset, subject, velocity, trial_num)
        else:
            logger.warning(f"No matching data found for subject '{subject}', velocity '{velocity}'. Using fallback values.")
            return fallback_start, fallback_end
            
    except Exception as e:
        logger.error(f"Error extracting onset/offset from dataset: {e}. Using fallback values.")
        return fallback_start, fallback_end

def _get_legacy_onset_offset_from_dataset(final_dataset, subject, velocity, trial_num=None):
    """
    Legacy function using platform_onset/platform_offset directly.
    """
    fallback_start, fallback_end = DEFAULT_FRAME_START, DEFAULT_FRAME_END
    
    try:
        # Filter by subject
        subject_data = final_dataset[final_dataset['subject'] == subject]
        if len(subject_data) == 0:
            return fallback_start, fallback_end
        
        # Try to find exact match
        matching_data = pd.DataFrame()
        if trial_num is not None:
            matching_data = subject_data[
                (subject_data['velocity'] == velocity) & 
                (subject_data['trial_num'] == trial_num)
            ]
        
        if len(matching_data) == 0:
            matching_data = subject_data[subject_data['velocity'] == velocity]
        
        if len(matching_data) > 0:
            row = matching_data.iloc[0]
            frame_start = int(row['platform_onset'])
            frame_end = int(row['platform_offset'])
            return frame_start, frame_end
        else:
            return fallback_start, fallback_end
    except:
        return fallback_start, fallback_end




def get_max_timing_from_dataset(final_dataset, subject, velocity, trial_num=None):
    """
    Extract cop_max_timing and com_max_timing values from final_dataset.
    """
    try:
        # Load configuration to get highlight window
        config = load_config('config.yaml')
        highlight_window = config.get('visualization', {}).get('highlight_window', 'full_window')
        windowing_config = config.get('windowing', {})
        
        # Determine column names based on windowing configuration
        if windowing_config.get('enabled', True) and highlight_window in windowing_config.get('windows', {}):
            # Use windowed column names
            cop_timing_col = f'cop_max_timing_{highlight_window}'
            com_timing_col = f'com_max_timing_{highlight_window}'
            logger.info(f"Looking for windowed timing columns: {cop_timing_col}, {com_timing_col}")
        else:
            # Use legacy column names
            cop_timing_col = 'cop_max_timing'
            com_timing_col = 'com_max_timing'
            logger.info(f"Looking for legacy timing columns: {cop_timing_col}, {com_timing_col}")
        
        # Check if the timing columns exist in the dataset
        timing_cols = [cop_timing_col, com_timing_col]
        available_timing_cols = [col for col in timing_cols if col in final_dataset.columns]
        
        if not available_timing_cols:
            # Try to find any timing columns (fallback)
            all_cop_cols = [col for col in final_dataset.columns if col.startswith('cop_max_timing')]
            all_com_cols = [col for col in final_dataset.columns if col.startswith('com_max_timing')]
            
            if all_cop_cols or all_com_cols:
                # Use the first available windowed columns
                cop_timing_col = all_cop_cols[0] if all_cop_cols else None
                com_timing_col = all_com_cols[0] if all_com_cols else None
                logger.info(f"Using fallback timing columns: {cop_timing_col}, {com_timing_col}")
            else:
                logger.warning(f"No timing columns found in dataset.")
                return None, None
        
        # Check if required columns exist
        required_cols = ['subject', 'velocity']
        missing_cols = [col for col in required_cols if col not in final_dataset.columns]
        if missing_cols:
            logger.warning(f"Missing columns {missing_cols} in dataset for max timing extraction.")
            return None, None
        
        # Filter by subject
        subject_data = final_dataset[final_dataset['subject'] == subject]
        if len(subject_data) == 0:
            logger.warning(f"No data found for subject '{subject}' for max timing extraction.")
            return None, None
        
        # Try to find exact match with both velocity and trial if provided
        matching_data = pd.DataFrame()
        if trial_num is not None:
            # Match both velocity and trial columns
            matching_data = subject_data[
                (subject_data['velocity'] == velocity) & 
                (subject_data['trial_num'] == trial_num)
            ]
        
        # If no match with trial, try velocity only
        if len(matching_data) == 0:
            matching_data = subject_data[subject_data['velocity'] == velocity]
        
        if len(matching_data) > 0:
            row = matching_data.iloc[0]
            
            # Extract timing values (handling None column names)
            cop_max_timing = None
            com_max_timing = None
            
            if cop_timing_col and cop_timing_col in row:
                cop_max_timing = row[cop_timing_col] if pd.notna(row[cop_timing_col]) else None
                
            if com_timing_col and com_timing_col in row:
                com_max_timing = row[com_timing_col] if pd.notna(row[com_timing_col]) else None
            
            logger.info(f"Found max timing for {subject}, velocity {velocity}, window '{highlight_window}': CoP={cop_max_timing}, CoM={com_max_timing}")
            return cop_max_timing, com_max_timing
        else:
            logger.warning(f"No matching data found for subject '{subject}', velocity '{velocity}' for max timing extraction.")
            return None, None
            
    except Exception as e:
        logger.error(f"Error extracting max timing from dataset: {e}")
        return None, None

def create_tableau_style_plot(trial_df, subject, velocity, trial_num=None, final_dataset=None, flip_ap_sign=True, series_type="cop"):
    """
    Unified Tableau-style scatter plot with dynamic range coloring.
    
    Args:
        series_type (str): "cop" for Center of Pressure (Cx/Cy) or "com" for Center of Mass (COM_X/COM_Y)
    """
    
    # Determine column names and labels based on series type
    if series_type.lower() == "cop":
        x_col, y_col = 'Cx', 'Cy'
        title_prefix = 'Force Plate Data: Cx vs Cy'
        xlabel = 'Cx (Right +, Left -)'
        ylabel = 'Cy (Anterior +)' if flip_ap_sign else 'Cy (Posterior +, Anterior -)'
        max_timing_type = 'cop'
        marker_label_prefix = 'CoP Max'
    elif series_type.lower() == "com":
        x_col, y_col = 'COM_X', 'COM_Y'
        title_prefix = 'Center of Mass Data: COM_X vs COM_Y'
        xlabel = 'COM_X (Right +, Left -)'
        ylabel = 'COM_Y (Anterior +)' if flip_ap_sign else 'COM_Y (Posterior +, Anterior -)'
        max_timing_type = 'com'
        marker_label_prefix = 'CoM Max'
    else:
        logger.error(f"Invalid series_type '{series_type}'. Must be 'cop' or 'com'")
        return None
    
    # Use the provided trial data
    df = trial_df
    
    # Check if required columns exist
    if x_col not in df.columns or y_col not in df.columns:
        logger.error(f"{x_col} and {y_col} columns not found in trial data")
        logger.info("Available columns: " + str(list(df.columns)))
        return None
    
    # Validation check: Ensure DeviceFrame column exists
    if 'DeviceFrame' not in df.columns:
        logger.error("DeviceFrame column not found in DataFrame")
        return None
    
    # Determine which dataset to use for window boundaries
    dataset_for_windows = None
    if final_dataset is not None:
        dataset_for_windows = final_dataset
    else:
        cfg = load_config('config.yaml')
        fallback_csv_path = cfg['data_paths']['final_dataset_filename']
        if os.path.exists(fallback_csv_path):
            logger.info(f"Loading final dataset from {fallback_csv_path} for window boundary extraction")
            dataset_for_windows = pd.read_csv(fallback_csv_path)

    # Resolve dataset row for this trial (subject/velocity/trial_num)
    dataset_row = None
    if dataset_for_windows is not None:
        subset = dataset_for_windows[dataset_for_windows['subject'] == subject]
        if trial_num is not None:
            subset = subset[(subset['velocity'] == velocity) & (subset['trial_num'] == trial_num)]
        else:
            subset = subset[subset['velocity'] == velocity]
        if len(subset) > 0:
            dataset_row = subset.iloc[0]

    # Load visualization/windowing configuration
    config_data = load_config('config.yaml')
    vis_cfg = config_data.get('visualization', {})
    windowing_cfg = config_data.get('windowing', {})
    highlight_window = vis_cfg.get('highlight_window', 'p4')
    plot_all_windows = vis_cfg.get('plot_all_windows', True)
    window_colors_cfg = vis_cfg.get('window_colors', {})

    # Compute window boundaries in DeviceFrame domain via centralized helper
    if dataset_row is not None:
        wb = get_window_boundaries(dataset_row, config_data=config_data, domain='DeviceFrame', trial_df=df)
        all_boundaries = wb.get('boundaries', {})
        # Determine highlight range
        if highlight_window in all_boundaries:
            frame_start, frame_end = all_boundaries[highlight_window]
        else:
            hlr = wb.get('highlight_range')
            if hlr is not None:
                frame_start, frame_end = hlr
            else:
                # Legacy fallback
                frame_start, frame_end = get_onset_offset_from_dataset(dataset_for_windows, subject, velocity, trial_num) if dataset_for_windows is not None else (DEFAULT_FRAME_START, DEFAULT_FRAME_END)
        # Use configured palette if available
        window_colors_cfg = vis_cfg.get('window_colors', {})
        palette = wb.get('palette', window_colors_cfg)
    else:
        # No dataset row available; fallback
        all_boundaries = {}
        frame_start, frame_end = DEFAULT_FRAME_START, DEFAULT_FRAME_END
        window_colors_cfg = vis_cfg.get('window_colors', {})
        palette = window_colors_cfg
    
    # Prepare x, y data with optional y flip for visualization
    x = df[x_col]  # ML axis: Right +, Left -
    y = -df[y_col] if flip_ap_sign else df[y_col]  # AP axis: conditional flip for Anterior +
    
    # Create figure and axis
    plt.figure(figsize=(12, 8))
    
    # Prepare masks per window if available
    window_masks: Dict[str, Any] = {}
    highlight_mask = None
    if all_boundaries:
        for w_name, (ws, we) in all_boundaries.items():
            window_masks[w_name] = (df['DeviceFrame'] >= ws) & (df['DeviceFrame'] <= we)

    # Get colors from configuration
    plot_colors = vis_cfg.get('plot_colors', {})
    
    # Use configured window colors with fallback to config defaults
    # palette was set earlier from helper; keep variable name for continuity

    if plot_all_windows and window_masks:
        # Plot each window in its own color
        for w_name, mask in window_masks.items():
            color = palette.get(w_name, None)
            plt.scatter(x[np.asarray(mask)], y[np.asarray(mask)], c=color if color else None, s=24, alpha=0.85, label=f'{w_name} ({all_boundaries[w_name][0]}-{all_boundaries[w_name][1]})')
        # Plot remaining points (outside all windows) lightly
        if window_masks:
            combined = np.zeros(len(df), dtype=bool)
            for m in window_masks.values():
                combined |= np.asarray(m)
            # Save combined mask for summary stats
            highlight_mask = combined
            plt.scatter(x[~np.asarray(combined)], y[~np.asarray(combined)], c=plot_colors.get('background', 'lightgray'), alpha=0.4, s=18, label='Other frames')
    else:
        # Legacy single highlight visualization
        highlight_mask = None
        if highlight_window in window_masks:
            highlight_mask = window_masks[highlight_window]
        else:
            highlight_mask = (df['DeviceFrame'] >= frame_start) & (df['DeviceFrame'] <= frame_end)

        plt.scatter(x[~np.asarray(highlight_mask)], y[~np.asarray(highlight_mask)], c=plot_colors.get('background', 'lightgray'), alpha=0.6, s=20, label='Other frames')

        if np.sum(highlight_mask) > 0:
            highlight_data = df[np.asarray(highlight_mask)].copy()
            norm_frames = (highlight_data['DeviceFrame'] - highlight_data['DeviceFrame'].min()) / (highlight_data['DeviceFrame'].max() - highlight_data['DeviceFrame'].min())
            colors = [plot_colors.get('time_gradient_start', '#B9DDF1'), plot_colors.get('time_gradient_end', '#173049')]
            custom_cmap = LinearSegmentedColormap.from_list('custom_blue', colors, N=256)
            scatter = plt.scatter(x[np.asarray(highlight_mask)], y[np.asarray(highlight_mask)], c=norm_frames, cmap=custom_cmap, s=30, alpha=0.8,
                                  label=f'{highlight_window} ({frame_start}-{frame_end})')
            cbar = plt.colorbar(scatter)
            cbar.set_label('Time Progression (Normalized DeviceFrame)', rotation=270, labelpad=20)
            actual_frames = highlight_data['DeviceFrame'].values
            cbar_ticks = np.linspace(0, 1, 5)
            cbar_tick_labels = [f"{int(np.interp(t, [0, 1], [actual_frames.min(), actual_frames.max()]))}" for t in cbar_ticks]
            cbar.set_ticks(cbar_ticks.tolist())
            cbar.set_ticklabels(cbar_tick_labels)
    
    # Add maximum value highlight
    max_timing_dataset = None
    
    # Check if final_dataset has max timing columns
    timing_col = f'{max_timing_type}_max_timing'
    if final_dataset is not None and timing_col in final_dataset.columns:
        max_timing_dataset = final_dataset
    else:
        # Fallback: load from CSV file for max timing data
        fallback_csv_path = load_config('config.yaml')['data_paths']['final_dataset_filename']
        if os.path.exists(fallback_csv_path):
            logger.info(f"Loading final dataset from {fallback_csv_path} for max timing extraction")
            max_timing_dataset = pd.read_csv(fallback_csv_path)
        else:
            logger.warning(f"Fallback file '{fallback_csv_path}' not found. {marker_label_prefix} max highlight skipped.")
    
    if max_timing_dataset is not None:
        if max_timing_type == 'cop':
            max_timing, _ = get_max_timing_from_dataset(max_timing_dataset, subject, velocity, trial_num)
        else:  # com
            _, max_timing = get_max_timing_from_dataset(max_timing_dataset, subject, velocity, trial_num)
            
        if max_timing is not None:
            # Find the data point at the max timing frame using DeviceFrame
            if 'DeviceFrame' in df.columns:
                max_point_mask = df['DeviceFrame'] == max_timing
                if np.sum(max_point_mask) > 0:
                    max_x = df[np.asarray(max_point_mask)][x_col].iloc[0]
                    max_y = -df[np.asarray(max_point_mask)][y_col].iloc[0] if flip_ap_sign else df[np.asarray(max_point_mask)][y_col].iloc[0]
                    plt.scatter(max_x, max_y, c=plot_colors.get('max_highlight', '#ED1C24'), s=200, marker='*', 
                               edgecolor=plot_colors.get('edge_color', 'white'), linewidth=2, alpha=0.9,
                               label=f'{marker_label_prefix} (DeviceFrame {int(max_timing)})', zorder=10)
                    logger.info(f"Added {marker_label_prefix} max highlight at DeviceFrame {max_timing}: ({max_x:.4f}, {max_y:.4f})")
                else:
                    logger.warning(f"{marker_label_prefix} max timing DeviceFrame {max_timing} not found in trial data")
            else:
                logger.warning(f"DeviceFrame column not found in trial data, {marker_label_prefix} max highlight skipped")
    
    # Customize the plot
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f'{title_prefix}\nHighlighted Range: DeviceFrame {frame_start}-{frame_end}', 
              fontsize=14, pad=20)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Make the plot look more professional
    plt.tight_layout()
    
    # Show statistics
    total_points = len(df)
    highlighted_points = int(np.sum(highlight_mask)) if highlight_mask is not None else 0
    logger.info(f"Total data points: {total_points}")
    logger.info(f"Highlighted points (frames {frame_start}-{frame_end}): {highlighted_points}")
    logger.info(f"DeviceFrame range in data: {df['DeviceFrame'].min()} - {df['DeviceFrame'].max()}")
    
    return plt


def create_tableau_style_com_plot(trial_df, subject, velocity, trial_num=None, final_dataset=None, flip_ap_sign=True):
    """
    Create Tableau-style scatter plot for COM data.
    """
    
    # Use the provided trial data
    df = trial_df
    
    # Check if COM_X and COM_Y columns exist
    if 'COM_X' not in df.columns or 'COM_Y' not in df.columns:
        logger.error("COM_X and COM_Y columns not found in trial data")
        logger.info("Available columns: " + str(list(df.columns)))
        return None
    
    
    # Validation check: Ensure DeviceFrame column exists
    if 'DeviceFrame' not in df.columns:
        logger.error("DeviceFrame column not found in DataFrame")
        return None
    # Determine which dataset to use for window boundaries
    dataset_for_windows = None
    if final_dataset is not None:
        dataset_for_windows = final_dataset
    else:
        cfg = load_config('config.yaml')
        fallback_csv_path = cfg['data_paths']['final_dataset_filename']
        if os.path.exists(fallback_csv_path):
            logger.info(f"Loading final dataset from {fallback_csv_path} for window boundary extraction")
            dataset_for_windows = pd.read_csv(fallback_csv_path)

    # Resolve dataset row for this trial
    dataset_row = None
    if dataset_for_windows is not None:
        subset = dataset_for_windows[dataset_for_windows['subject'] == subject]
        if trial_num is not None:
            subset = subset[(subset['velocity'] == velocity) & (subset['trial_num'] == trial_num)]
        else:
            subset = subset[subset['velocity'] == velocity]
        if len(subset) > 0:
            dataset_row = subset.iloc[0]

    # Load visualization/windowing configuration
    config_data = load_config('config.yaml')
    vis_cfg = config_data.get('visualization', {})
    windowing_cfg = config_data.get('windowing', {})
    highlight_window = vis_cfg.get('highlight_window', 'p4')
    plot_all_windows = vis_cfg.get('plot_all_windows', True)
    window_colors_cfg = vis_cfg.get('window_colors', {})

    # Compute window boundaries in DeviceFrame domain via centralized helper
    if dataset_row is not None:
        wb = get_window_boundaries(dataset_row, config_data=config_data, domain='DeviceFrame', trial_df=df)
        all_boundaries = wb.get('boundaries', {})
        if highlight_window in all_boundaries:
            frame_start, frame_end = all_boundaries[highlight_window]
        else:
            hlr = wb.get('highlight_range')
            if hlr is not None:
                frame_start, frame_end = hlr
            else:
                frame_start, frame_end = get_onset_offset_from_dataset(dataset_for_windows, subject, velocity, trial_num) if dataset_for_windows is not None else (DEFAULT_FRAME_START, DEFAULT_FRAME_END)
        window_colors_cfg = vis_cfg.get('window_colors', {})
        palette = wb.get('palette', window_colors_cfg)
    else:
        all_boundaries = {}
        frame_start, frame_end = DEFAULT_FRAME_START, DEFAULT_FRAME_END
        window_colors_cfg = vis_cfg.get('window_colors', {})
        palette = window_colors_cfg
    
    # Prepare x, y data with optional COM_Y flip for visualization
    x = df['COM_X']  # ML axis: Right +, Left -
    y = -df['COM_Y'] if flip_ap_sign else df['COM_Y']  # AP axis: conditional flip for Anterior +
    
    # Create figure and axis
    plt.figure(figsize=(12, 8))
    
    # Prepare masks per window if available
    window_masks: Dict[str, Any] = {}
    highlight_mask = None
    if all_boundaries:
        for w_name, (ws, we) in all_boundaries.items():
            window_masks[w_name] = (df['DeviceFrame'] >= ws) & (df['DeviceFrame'] <= we)

    # Use configured window colors
    palette = window_colors_cfg

    if plot_all_windows and window_masks:
        for w_name, mask in window_masks.items():
            color = palette.get(w_name, None)
            plt.scatter(x[np.asarray(mask)], y[np.asarray(mask)], c=color if color else None, s=24, alpha=0.85, label=f'{w_name} ({all_boundaries[w_name][0]}-{all_boundaries[w_name][1]})')
        if window_masks:
            combined = np.zeros(len(df), dtype=bool)
            for m in window_masks.values():
                combined |= np.asarray(m)
            highlight_mask = combined
            plt.scatter(x[~np.asarray(combined)], y[~np.asarray(combined)], c=plot_colors.get('background', 'lightgray'), alpha=0.4, s=18, label='Other frames')
    else:
        highlight_mask = None
        if highlight_window in window_masks:
            highlight_mask = window_masks[highlight_window]
        else:
            highlight_mask = (df['DeviceFrame'] >= frame_start) & (df['DeviceFrame'] <= frame_end)

        plt.scatter(x[~np.asarray(highlight_mask)], y[~np.asarray(highlight_mask)], c=plot_colors.get('background', 'lightgray'), alpha=0.6, s=20, label='Other frames')

        if np.sum(highlight_mask) > 0:
            highlight_data = df[np.asarray(highlight_mask)].copy()
            norm_frames = (highlight_data['DeviceFrame'] - highlight_data['DeviceFrame'].min()) / (highlight_data['DeviceFrame'].max() - highlight_data['DeviceFrame'].min())
            colors = [plot_colors.get('time_gradient_start', '#B9DDF1'), plot_colors.get('time_gradient_end', '#173049')]
            custom_cmap = LinearSegmentedColormap.from_list('custom_blue', colors, N=256)
            scatter = plt.scatter(x[np.asarray(highlight_mask)], y[np.asarray(highlight_mask)], c=norm_frames, cmap=custom_cmap, s=30, alpha=0.8,
                                  label=f'{highlight_window} ({frame_start}-{frame_end})')
            cbar = plt.colorbar(scatter)
            cbar.set_label('Time Progression (Normalized DeviceFrame)', rotation=270, labelpad=20)
            actual_frames = highlight_data['DeviceFrame'].values
            cbar_ticks = np.linspace(0, 1, 5)
            cbar_tick_labels = [f"{int(np.interp(t, [0, 1], [actual_frames.min(), actual_frames.max()]))}" for t in cbar_ticks]
            cbar.set_ticks(cbar_ticks.tolist())
            cbar.set_ticklabels(cbar_tick_labels)
    
    # Add CoM maximum value highlight
    max_timing_dataset = None
    
    # Check if final_dataset has max timing columns
    if final_dataset is not None and 'com_max_timing' in final_dataset.columns:
        max_timing_dataset = final_dataset
    else:
        # Fallback: load from CSV file for max timing data
        fallback_csv_path = load_config('config.yaml')['data_paths']['final_dataset_filename']
        if os.path.exists(fallback_csv_path):
            logger.info(f"Loading final dataset from {fallback_csv_path} for max timing extraction")
            max_timing_dataset = pd.read_csv(fallback_csv_path)
        else:
            logger.warning(f"Fallback file '{fallback_csv_path}' not found. CoM max highlight skipped.")
    
    if max_timing_dataset is not None:
        _, com_max_timing = get_max_timing_from_dataset(max_timing_dataset, subject, velocity, trial_num)
        if com_max_timing is not None:
            # Find the data point at the max timing frame
            if 'DeviceFrame' in df.columns:
                max_point_mask = df['DeviceFrame'] == com_max_timing
                if np.sum(max_point_mask) > 0:
                    max_x = df[np.asarray(max_point_mask)]['COM_X'].iloc[0]
                    max_y = -df[np.asarray(max_point_mask)]['COM_Y'].iloc[0] if flip_ap_sign else df[np.asarray(max_point_mask)]['COM_Y'].iloc[0]
                    plt.scatter(max_x, max_y, c=plot_colors.get('max_highlight', '#ED1C24'), s=200, marker='*', 
                               edgecolor=plot_colors.get('edge_color', 'white'), linewidth=2, alpha=0.9,
                               label=f'CoM Max (DeviceFrame {int(com_max_timing)})', zorder=10)
                    logger.info(f"Added CoM max highlight at DeviceFrame {com_max_timing}: ({max_x:.4f}, {max_y:.4f})")
                else:
                    logger.warning(f"CoM max timing DeviceFrame {com_max_timing} not found in trial data")
            else:
                logger.warning("DeviceFrame column not found in trial data, CoM max highlight skipped")

    # Add single onset point marker for CoP/CoM if onset timings exist in final_dataset
    try:
        if final_dataset is not None and 'DeviceFrame' in df.columns:
            onset_df = final_dataset[(final_dataset['subject'] == subject)
                                     & (final_dataset['velocity'] == velocity)
                                     & (final_dataset['trial_num'] == trial_num)]
            if len(onset_df) > 0:
                onset_frame = None
                if series_type.lower() == 'cop':
                    cand = []
                    if 'cx_onset_timing' in onset_df.columns:
                        v = onset_df.iloc[0].get('cx_onset_timing')
                        if pd.notna(v):
                            cand.append(float(v))
                    if 'cy_onset_timing' in onset_df.columns:
                        v = onset_df.iloc[0].get('cy_onset_timing')
                        if pd.notna(v):
                            cand.append(float(v))
                    if len(cand) > 0:
                        onset_frame = min(cand)
                else:  # com
                    cand = []
                    for cname in ('com_x_onset_timing', 'com_y_onset_timing', 'com_z_onset_timing'):
                        if cname in onset_df.columns:
                            v = onset_df.iloc[0].get(cname)
                            if pd.notna(v):
                                cand.append(float(v))
                    if len(cand) > 0:
                        onset_frame = min(cand)

                if onset_frame is not None:
                    mask_on = (df['DeviceFrame'] == onset_frame)
                    if np.sum(mask_on) > 0:
                        px = df[np.asarray(mask_on)][x_col].iloc[0]
                        py_raw = df[np.asarray(mask_on)][y_col].iloc[0]
                        py = -py_raw if flip_ap_sign else py_raw
                        plt.scatter(px, py, c=plot_colors.get('max_highlight', '#ED1C24'), s=120, marker='X',
                                    edgecolor=plot_colors.get('edge_color', 'white'), linewidth=1.5, alpha=0.9,
                                    label=f"Onset ({int(onset_frame)})", zorder=12)
    except Exception:
        pass
    
    # Customize the plot
    plt.xlabel('COM_X (Right +, Left -)', fontsize=12)
    plt.ylabel('COM_Y (Anterior +)' if flip_ap_sign else 'COM_Y (Posterior +, Anterior -)', fontsize=12)
    plt.title(f'Center of Mass Data: COM_X vs COM_Y\nHighlighted Range: DeviceFrame {frame_start}-{frame_end}', 
              fontsize=14, pad=20)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Make the plot look more professional
    plt.tight_layout()
    
    # Show statistics
    total_points = len(df)
    highlighted_points = int(np.sum(highlight_mask)) if highlight_mask is not None else 0
    logger.info(f"Total data points: {total_points}")
    logger.info(f"Highlighted points (frames {frame_start}-{frame_end}): {highlighted_points}")
    logger.info(f"DeviceFrame range in data: {df['DeviceFrame'].min()} - {df['DeviceFrame'].max()}")
    
    return plt


def create_plot_from_pipeline_data(final_dataset, subject='김윤자', velocity=20, trial_num=4, flip_ap_sign=True):
    """
    Create plot using final_dataset from pipeline execution.
    """
    # Filter for the specific trial data from final_dataset (which contains all data)
    trial_df = final_dataset[
        (final_dataset['subject'] == subject) & 
        (final_dataset['velocity'] == velocity) & 
        (final_dataset['trial_num'] == trial_num)
    ]
    
    if len(trial_df) == 0:
        logger.error(f"Error: No data found for subject '{subject}', velocity '{velocity}', trial {trial_num} in pipeline data")
        available = final_dataset.groupby(['subject', 'velocity', 'trial_num']).size().reset_index(name='count')
        logger.info("Available combinations in the pipeline data:")
        logger.info(str(available.head(10)))
        return None
    
    logger.info(f"Found {len(trial_df)} data points for {subject}, velocity {velocity}, trial {trial_num} from pipeline")
    
    # Create the plot using pipeline data
    plot = create_tableau_style_plot(trial_df, subject, velocity, trial_num, final_dataset, flip_ap_sign, series_type="cop")
    return plot


def create_com_plot_from_pipeline_data(final_dataset, subject='김윤자', velocity=20, trial_num=4, flip_ap_sign=True):
    """
    Create COM plot using final_dataset from pipeline execution.
    """
    # Filter for the specific trial data from final_dataset (which contains all data)
    trial_df = final_dataset[
        (final_dataset['subject'] == subject) & 
        (final_dataset['velocity'] == velocity) & 
        (final_dataset['trial_num'] == trial_num)
    ]
    
    if len(trial_df) == 0:
        logger.error(f"Error: No data found for subject '{subject}', velocity '{velocity}', trial {trial_num} in pipeline data")
        available = final_dataset.groupby(['subject', 'velocity', 'trial_num']).size().reset_index(name='count')
        logger.info("Available combinations in the pipeline data:")
        logger.info(str(available.head(10)))
        return None
    
    # Check if COM_X and COM_Y columns exist
    if 'COM_X' not in trial_df.columns or 'COM_Y' not in trial_df.columns:
        logger.error("Error: COM_X and COM_Y columns not found in pipeline data")
        logger.info("Available columns: " + str(list(trial_df.columns)))
        return None
    
    logger.info(f"Found {len(trial_df)} data points for {subject}, velocity {velocity}, trial {trial_num} from pipeline (COM data)")
    
    # Create the plot using pipeline data
    plot = create_tableau_style_plot(trial_df, subject, velocity, trial_num, final_dataset, flip_ap_sign, series_type="com")
    return plot


def _generate_cop_com_plot_worker(args):
    """
    Worker function to generate CoP and CoM plots for a single trial.
    """
    try:
        # Extract parameters from args
        subject = args['subject']
        velocity = args['velocity']
        trial_num = args['trial_num']
        trial_data = args['trial_data']
        timing_data = args.get('timing_data')
        mode = args.get('mode', 'full')
        config_data = args.get('config_data', {})
        output_base_dir = args.get('output_base_dir')
        has_cop_data = args.get('has_cop_data', False)
        has_com_data = args.get('has_com_data', False)
        combinations = args.get('combinations', timing_data)
        
        if output_base_dir is None:
            # Derive from config if not provided
            output_base_dir = config_data.get('data_paths', {}).get('cop_com_plot_dir', 'output/cop&com_plot')
        
        # Determine output directory based on mode
        if mode == "sample":
            base_dir = os.path.join(output_base_dir, 'sample')
        else:
            base_dir = output_base_dir
            # Create subject subfolder for full mode
            base_dir = os.path.join(base_dir, subject)
        
        os.makedirs(base_dir, exist_ok=True)
        
        if len(trial_data) == 0:
            return (False, f"No data found for {subject}, velocity {velocity}, trial {trial_num}")
        
        plots_generated = []
        
        # Generate CoP plot if data available
        if has_cop_data:
            try:
                plt.figure(figsize=(12, 8))  # Start fresh figure
                # Pass final dataset directly; avoid boolean evaluation on DataFrame
                cop_plot = create_tableau_style_plot(
                    trial_data, subject, velocity, trial_num, combinations, series_type="cop"
                )
                
                if cop_plot is not None:
                    # Use appropriate naming convention based on mode
                    if mode == "sample":
                        filename = f"sample_CoP_{subject}_vel{velocity}_trial{trial_num}.png"
                    else:
                        filename = f"{subject}_vel{velocity}_trial{trial_num}_CoP.png"
                    
                    filepath = os.path.join(base_dir, filename)
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    plots_generated.append(f"CoP: {filename}")
                
                plt.close()  # Critical: Close figure to free memory
                
            except Exception as e:
                plt.close()  # Clean up on error
                return (False, f"CoP plot error for {subject}_vel{velocity}_trial{trial_num}: {e}")
        
        # Generate CoM plot if data available
        if has_com_data:
            try:
                plt.figure(figsize=(12, 8))  # Start fresh figure
                # Pass final dataset directly; avoid boolean evaluation on DataFrame
                com_plot = create_tableau_style_plot(
                    trial_data, subject, velocity, trial_num, combinations, series_type="com"
                )
                
                if com_plot is not None:
                    # Use appropriate naming convention based on mode
                    if mode == "sample":
                        filename = f"sample_CoM_{subject}_vel{velocity}_trial{trial_num}.png"
                    else:
                        filename = f"{subject}_vel{velocity}_trial{trial_num}_CoM.png"
                    
                    filepath = os.path.join(base_dir, filename)
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    plots_generated.append(f"CoM: {filename}")
                
                plt.close()  # Critical: Close figure to free memory
                
            except Exception as e:
                plt.close()  # Clean up on error
                return (False, f"CoM plot error for {subject}_vel{velocity}_trial{trial_num}: {e}")
        
        if plots_generated:
            return (True, f"Generated plots for {subject}_vel{velocity}_trial{trial_num}: {', '.join(plots_generated)}")
        else:
            return (False, f"No plots generated for {subject}_vel{velocity}_trial{trial_num}")
            
    except Exception as e:
        try:
            plt.close()  # Ensure cleanup on any error
        except:
            pass
        return (False, f"Worker error: {str(e)}")




def _load_and_merge_data(config_data, merged_data=None, final_dataset=None):
    """
    Helper function to load and merge data if not provided.
    
    Returns:
        tuple: (merged_data, final_dataset, has_cop_data, has_com_data)
    """
    # Load data if not provided (backward compatibility)
    if merged_data is None:
        logger.info("Loading raw EMG data from config...")
        # Prefer processed_emg_file; fall back to legacy emg_file for compatibility
        raw_emg_data_path = (
            config_data.get('data_paths', {}).get('processed_emg_file')
            or config_data.get('data_paths', {}).get('emg_file')
        )
        if not raw_emg_data_path:
            logger.error("No EMG data path configured (processed_emg_file/emg_file)")
            return None, None, False, False
        if not os.path.exists(raw_emg_data_path):
            logger.error(f"Raw EMG data file '{raw_emg_data_path}' not found.")
            return None, None, False, False
        
        encoding_handler = EMGKoreanEncodingHandler()
        raw_emg_data = encoding_handler.read_csv_with_korean_text(raw_emg_data_path)
        
        # Load platform conditions
        platform_conditions_path = config_data['data_paths']['platform_file']
        if not os.path.exists(platform_conditions_path):
            logger.error(f"Platform conditions file '{platform_conditions_path}' not found.")
            return None, None, False, False
        
        platform_conditions = encoding_handler.read_csv_with_korean_text(platform_conditions_path, sheet_name='in')
        
        # Rename 'trial' to 'trial_num' if needed
        if 'trial' in platform_conditions.columns:
            platform_conditions = platform_conditions.rename(columns={'trial': 'trial_num'})
        
        # Merge raw EMG data with platform conditions
        merged_data = raw_emg_data.merge(
            platform_conditions[['subject', 'velocity', 'trial_num', 'platform_onset', 'platform_offset']],
            on=['subject', 'velocity', 'trial_num'],
            how='inner'
        )
    
    # Check if COP or COM data exists
    has_cop_data = 'Cx' in merged_data.columns and 'Cy' in merged_data.columns
    has_com_data = 'COM_X' in merged_data.columns and 'COM_Y' in merged_data.columns
    
    logger.info("Available data types:")
    if has_cop_data:
        logger.info("- COP data (Cx, Cy): Available")
    if has_com_data:
        logger.info("- COM data (COM_X, COM_Y): Available")
    
    if not has_cop_data and not has_com_data:
        logger.error("No required columns for plotting found in data.")
        logger.error("Need either: Cx & Cy (for COP) or COM_X & COM_Y (for COM)")
        logger.error(f"Available columns: {list(merged_data.columns)}")
        return None, None, False, False
    
    # Use merged_data as final_dataset if not provided
    if final_dataset is None:
        final_dataset = merged_data.groupby(['subject', 'velocity', 'trial_num']).first().reset_index()
    
    return merged_data, final_dataset, has_cop_data, has_com_data


def _generate_cop_com_plots_unified(
    merged_data: Optional[pd.DataFrame] = None,
    final_dataset: Optional[pd.DataFrame] = None,
    mode: str = "full",
    limit_per_type: int = 10
) -> None:
    """
    Unified function to generate CoP and CoM plots using shared helpers.
    
    Args:
        merged_data: Pre-loaded merged dataset (None to load from config)
        final_dataset: Pre-loaded final dataset (None to use merged_data)
        mode: "full" or "sample"
        limit_per_type: Limit for sample mode
    """
    mode_desc = "sample" if mode == "sample" else "all"
    logger.info(f"Starting CoP and CoM {mode_desc} plot generation...")
    
    # Load configuration
    config_data = load_config('config.yaml')
    
    # Load and validate data
    merged_data, final_dataset, has_cop_data, has_com_data = _load_and_merge_data(
        config_data, merged_data, final_dataset
    )
    
    if merged_data is None:
        return
    
    # Get output directory
    output_base_dir = config_data['data_paths']['cop_com_plot_dir']
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Generate plots in sample mode directly
    if mode == "sample":
        logger.info(f"Generating up to {limit_per_type} sample plots each for CoP and CoM")
        
        # Get sample data
        sample_combinations = final_dataset.head(limit_per_type)
        
        # Set output directory for sample mode
        sample_output_dir = os.path.join(output_base_dir, 'sample')
        os.makedirs(sample_output_dir, exist_ok=True)
        
        plots_generated = []
        
        for _, row in sample_combinations.iterrows():
            subject = row['subject']
            velocity = row['velocity']
            trial_num = row['trial_num']
            
            try:
                # Generate CoP plot if data available
                if has_cop_data:
                    trial_data = merged_data[
                        (merged_data['subject'] == subject) & 
                        (merged_data['velocity'] == velocity) & 
                        (merged_data['trial_num'] == trial_num)
                    ]
                    
                    if len(trial_data) > 0:
                        plt.figure(figsize=(12, 8))
                        cop_plot = create_tableau_style_plot(
                            trial_data, subject, velocity, trial_num, final_dataset, series_type="cop"
                        )
                        
                        if cop_plot is not None:
                            filename = f"sample_CoP_{subject}_vel{velocity}_trial{trial_num}.png"
                            filepath = os.path.join(sample_output_dir, filename)
                            plt.savefig(filepath, dpi=300, bbox_inches='tight')
                            plots_generated.append(f"CoP: {filename}")
                        
                        plt.close()
                
                # Generate CoM plot if data available
                if has_com_data:
                    trial_data = merged_data[
                        (merged_data['subject'] == subject) & 
                        (merged_data['velocity'] == velocity) & 
                        (merged_data['trial_num'] == trial_num)
                    ]
                    
                    if len(trial_data) > 0:
                        plt.figure(figsize=(12, 8))
                        com_plot = create_tableau_style_plot(
                            trial_data, subject, velocity, trial_num, final_dataset, series_type="com"
                        )
                        
                        if com_plot is not None:
                            filename = f"sample_CoM_{subject}_vel{velocity}_trial{trial_num}.png"
                            filepath = os.path.join(sample_output_dir, filename)
                            plt.savefig(filepath, dpi=300, bbox_inches='tight')
                            plots_generated.append(f"CoM: {filename}")
                        
                        plt.close()
                        
            except Exception as e:
                logger.error(f"Error generating plots for {subject}_vel{velocity}_trial{trial_num}: {e}")
                plt.close()  # Ensure cleanup
        
        logger.info(f"Generated {len(plots_generated)} sample plots in {sample_output_dir}")
        return
    
    # For full mode, generate all plots
    logger.info("Generating all CoP and CoM plots...")
    all_combinations = final_dataset
    plots_generated = []
    
    for _, row in all_combinations.iterrows():
        subject = row['subject']
        velocity = row['velocity']
        trial_num = row['trial_num']
        
        try:
            # Generate CoP plot if data available
            if has_cop_data:
                trial_data = merged_data[
                    (merged_data['subject'] == subject) & 
                    (merged_data['velocity'] == velocity) & 
                    (merged_data['trial_num'] == trial_num)
                ]
                
                if len(trial_data) > 0:
                    plt.figure(figsize=(12, 8))
                    cop_plot = create_tableau_style_plot(
                        trial_data, subject, velocity, trial_num, final_dataset, series_type="cop"
                    )
                    
                    if cop_plot is not None:
                        filename = f"{subject}_vel{velocity}_trial{trial_num}_CoP.png"
                        filepath = os.path.join(output_base_dir, filename)
                        plt.savefig(filepath, dpi=300, bbox_inches='tight')
                        plots_generated.append(f"CoP: {filename}")
                    
                    plt.close()
            
            # Generate CoM plot if data available
            if has_com_data:
                trial_data = merged_data[
                    (merged_data['subject'] == subject) & 
                    (merged_data['velocity'] == velocity) & 
                    (merged_data['trial_num'] == trial_num)
                ]
                
                if len(trial_data) > 0:
                    plt.figure(figsize=(12, 8))
                    com_plot = create_tableau_style_plot(
                        trial_data, subject, velocity, trial_num, final_dataset, series_type="com"
                    )
                    
                    if com_plot is not None:
                        filename = f"{subject}_vel{velocity}_trial{trial_num}_CoM.png"
                        filepath = os.path.join(output_base_dir, filename)
                        plt.savefig(filepath, dpi=300, bbox_inches='tight')
                        plots_generated.append(f"CoM: {filename}")
                    
                    plt.close()
                    
        except Exception as e:
            logger.error(f"Error generating plots for {subject}_vel{velocity}_trial{trial_num}: {e}")
            plt.close()  # Ensure cleanup
    
    logger.info(f"Generated {len(plots_generated)} total plots in {output_base_dir}")


def generate_all_plots(merged_data: Optional[pd.DataFrame] = None, final_dataset: Optional[pd.DataFrame] = None):
    """
    Generate CoP and CoM plots for all subject-velocity-trial combinations.
    
    Args:
        merged_data (pd.DataFrame, optional): Pre-loaded merged dataset. If None, loads from config.
        final_dataset (pd.DataFrame, optional): Pre-loaded final dataset. If None, loads from config.
    
    This function now uses the unified plotting system to eliminate code duplication.
    """
    _generate_cop_com_plots_unified(
        merged_data=merged_data,
        final_dataset=final_dataset,
        mode="full"
    )


def generate_sample_plots(
    merged_data: Optional[pd.DataFrame] = None,
    final_dataset: Optional[pd.DataFrame] = None,
    limit_per_type: int = 10,
):
    """
    Generate up to `limit_per_type` CoP and CoM sample plots.

    - Outputs under cop_com_plot_dir/sample
    - Uses unified plotting system to eliminate code duplication
    """
    _generate_cop_com_plots_unified(
        merged_data=merged_data,
        final_dataset=final_dataset,
        mode="sample",
        limit_per_type=limit_per_type
    )


def main():
    """
    Main function - calls generate_all_plots() for backward compatibility
    """
    generate_all_plots()

if __name__ == "__main__":
    main()
