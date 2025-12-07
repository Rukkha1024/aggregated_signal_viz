#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMG and Signal Grid Visualization Module (V5)

Subject별로 모든 velocity-trial 조합을 하나의 grid plot에 배치.
- EMG: TKEO-AGLR onset timing 마커 + Window span 하이라이트
- Fplot

"""

import os
import math
import pandas as pd
p
import matplotlib.pyplot as plt
from typing import O

# Import from util
from emg_pipeline.utils im (
    setup_koreant,
    setup_logger,
    load_config,
    get_windo,
 ,
    EMGSignalPipelineProcessor,
)

ialize
setup_korean_font()
logger = setup_logg_)

tants
 # inches
SUBPLOT_HEIGHT = 6  # inches
DPI = 300


def get_velocity_trial_combos(df: pd.DataFrame, sub
    """Get sorted velocity-trial combin"
    subj_df = df[df['subject'] == subject][['velocity', 'trial_num']].()
    subj_df = subj_])
    ))


def calculate_grid_dimensions(n_plots: int) -
    """Calculate optimal grid dimensions (rows, cols) for n_plots."""
    if n_plots <= 0:
, 0)
    cols = math.ceil(math.sqrt(n_plots))
    rows = math.ceil(n_plots / cols)
    return (rows, cols)


def comp:
    """Compute platform onset offset for x-"""
    try:
        dev_hz = float(config.get('windowing', {}).get('sa
        
        ratio = dev_hz / mocap_hz if mocap_hz else 1.0

        trial_rows = timings_df[
            (timings_df['subject'] == subject) &
        ocity) &
            (timings_df['trial_num'] == trial_num)
        ]
        dataset_row = trial_rows.iloc[0] if not trial_rows.empty else None

        if dataset_row is not None and 'platform_onset' inx:
            odf_min = float(pd.to_nu').min())
            df_min = float(pd.to_numeric(trial_df[())
            odf_onset = float(dataset_row['platform_ontio
            onset_df_abs = (odf_onset - odf_min) + df_min
            rw
    except Exception:
        pass
    return 0.0, None


def draw_window_spans(ax, dataset_row, config, onset_df_abs, trial_df):
    """Draw window spans on the axes."""
    if dataset_ro
        return
    try:
        vis_cfg = config.get(')
)
        default_fallback = vis_cfg.get('plot_colors', {}).get('default_fallback', '#999999')

        wb = get_window_boundaries(
        all_bounds = wb.get('boundaries', {})
        palette = wb.get('olors)

        for wname, (ws, we) in all_bounds.items():
            
            ws_shifted = ws - onset_df_abs
            we_shifted = we - onset_df_abs
            ax.axvspan(ws_shifted, we_shifted, color=color, alpha=0.15,
                       label=f'{wname}: {int(ws_shifted)}-fted)}')
    except Exception as e:
        logger.warning(f"Failed to draw window spa


