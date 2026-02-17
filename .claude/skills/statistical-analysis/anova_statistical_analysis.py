"""
Repeated-Measures ANOVA Function Library
=========================================
Primarily for biomechanics research (repeated-measures experimental data).

This is a PURE FUNCTION LIBRARY.
Each function takes all required parameters explicitly - NO hardcoded defaults.

DESIGN PRINCIPLE:
    - The AI/user determines which columns represent subject/unit/task/trial at analysis time
    - Functions receive all configuration as explicit parameters
    - No main() function, no script execution capability
    - No config.yaml dependency in this library

Expected Column Semantics (convention, not requirement):
    - Subject identifier: e.g., subjects, participant_id
    - Measurement unit: e.g., muscle, sensor, channel
    - Task/condition: e.g., tasks, movement, activity
    - Trial/repetition: e.g., trials, trial_num, rep

Note: Actual column names are flexible and passed as function parameters.

Variable Elements (passed as parameters):
    - condition_col: condition column name (e.g., cart_categories, load_level, speed)
    - dependent_var: dependent variable column name (e.g., rvc_norm_rms, emg_amplitude, force)

Usage Example:
    from anova_statistical_analysis import (
        compute_cell_means,
        find_valid_subjects,
        compute_descriptives,
        run_rm_anova,
        build_analysis_metadata,
        export_to_excel
    )
    
    # AI/user determines these at analysis time
    condition_col = "cart_categories"
    dependent_var = "rvc_norm_rms"
    alpha = 0.05
    
    # Load data
    df = pd.read_csv("data.csv")
    
    # Step 1: Compute cell means
    cell_means = compute_cell_means(
        df=df,
        dependent_var=dependent_var,
        condition_col=condition_col
    )
    
    # Step 2: Find valid subjects (complete design)
    valid_subjects, counts_pivot = find_valid_subjects(
        cell_means=cell_means,
        condition_col=condition_col,
        dependent_var=dependent_var
    )
    
    # Step 3: Filter to valid subjects
    cell_means_valid = cell_means[cell_means['subjects'].isin(valid_subjects)]
    
    # Step 4: Compute descriptive statistics
    desc_df = compute_descriptives(
        cell_means_valid=cell_means_valid,
        dependent_var=dependent_var,
        condition_col=condition_col
    )
    
    # Step 5: Run ANOVA
    anova_df = run_rm_anova(
        cell_means_valid=cell_means_valid,
        dependent_var=dependent_var,
        condition_col=condition_col
    )
    
    # Step 6: Build metadata
    metadata = build_analysis_metadata(
        valid_subjects=valid_subjects,
        data_file_name="data.csv",
        dependent_var=dependent_var,
        dependent_var_label="RVC Normalized RMS",
        condition_col=condition_col,
        n_conditions=2,
        n_tasks=3,
        alpha=alpha
    )
    
    # Step 7: Export to Excel
    export_to_excel(
        output_dir="./OUTPUT",
        excel_name="anova_results.xlsx",
        analysis_metadata=metadata,
        desc_df=desc_df,
        anova_df=anova_df,
        cell_means_valid=cell_means_valid,
        dependent_var=dependent_var,
        alpha=alpha
    )
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import polars as pl
from statsmodels.stats.anova import AnovaRM

from excel_utils import (
    create_formats,
    write_methods_sheet,
    write_descriptives_sheet,
    write_statistical_tests_sheet
)


# =============================================================================
# DATA AGGREGATION FUNCTIONS
# =============================================================================

def compute_cell_means(
    df: pd.DataFrame,
    dependent_var: str,
    condition_col: str
) -> pd.DataFrame:
    """
    Aggregate raw trial-level data to subjects × condition × tasks × muscle level.
    
    This function computes the mean of the dependent variable for each unique 
    combination of subjects, condition, tasks, and muscle.
    
    Args:
        df: Input DataFrame with trial-level data
            Required columns: ['subjects', condition_col, 'tasks', 'muscle', dependent_var]
        dependent_var: Name of the dependent variable column
        condition_col: Name of the condition column (within-subject factor)
    
    Returns:
        DataFrame with one row per subjects × condition × tasks × muscle combination,
        containing the mean of the dependent variable
    
    Raises:
        ValueError: If required columns are missing from the input DataFrame
    
    Example:
        >>> cell_means = compute_cell_means(
        ...     df=raw_data,
        ...     dependent_var="rvc_norm_rms",
        ...     condition_col="cart_categories"
        ... )
    """
    required_cols = ['subjects', condition_col, 'tasks', 'muscle', dependent_var]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input DataFrame: {missing}")

    cell_means = (
        df
        .groupby(['subjects', condition_col, 'tasks', 'muscle'], as_index=False)[dependent_var]
        .mean()
    )

    return cell_means


def find_valid_subjects(
    cell_means: pd.DataFrame,
    condition_col: str,
    dependent_var: str
) -> Tuple[pd.Index, pd.DataFrame]:
    """
    Find subjects with complete N×M design for all muscles.
    
    Identifies subjects who have valid data for all combinations of conditions
    and tasks for every muscle. This ensures the repeated-measures ANOVA
    can be computed without missing cells.
    
    Args:
        cell_means: DataFrame with cell-level means (output of compute_cell_means)
        condition_col: Name of the condition column
        dependent_var: Name of the dependent variable column
    
    Returns:
        Tuple of:
            - valid_subjects: Index of subject IDs with complete design
            - counts_pivot: subjects × muscles cell count table (for debugging)
    
    Example:
        >>> valid_subjects, counts = find_valid_subjects(
        ...     cell_means=cell_means,
        ...     condition_col="cart_categories",
        ...     dependent_var="rvc_norm_rms"
        ... )
        >>> print(f"Valid subjects: {len(valid_subjects)}")
    """
    # Calculate expected number of cells per subject per muscle
    n_conditions = cell_means[condition_col].nunique()
    n_tasks = cell_means['tasks'].nunique()
    expected_cells = n_conditions * n_tasks
    
    # Count cells per subject per muscle
    counts = (
        cell_means
        .groupby(['muscle', 'subjects'])[dependent_var]
        .count()
        .reset_index()
    )
    counts_pivot = counts.pivot(index='subjects', columns='muscle', values=dependent_var)
    
    # Select subjects where minimum cell count equals expected (complete for all muscles)
    valid_subjects = counts_pivot.index[counts_pivot.min(axis=1) == expected_cells]
    
    return valid_subjects, counts_pivot


# =============================================================================
# DESCRIPTIVE STATISTICS FUNCTIONS
# =============================================================================

def compute_descriptives(
    cell_means_valid: pd.DataFrame,
    dependent_var: str,
    condition_col: str
) -> pd.DataFrame:
    """
    Compute descriptive statistics at muscle × condition × tasks level.
    
    Calculates mean, standard deviation, standard error of the mean, and
    sample size for each combination of muscle, condition, and task.
    
    Args:
        cell_means_valid: DataFrame with cell means for valid subjects only
        dependent_var: Name of the dependent variable column
        condition_col: Name of the condition column
    
    Returns:
        DataFrame with columns: ['muscle', condition_col, 'tasks', 'mean', 'std', 'count', 'sem']
    
    Example:
        >>> desc_stats = compute_descriptives(
        ...     cell_means_valid=cell_means_valid,
        ...     dependent_var="rvc_norm_rms",
        ...     condition_col="cart_categories"
        ... )
    """
    desc = (
        cell_means_valid
        .groupby(['muscle', condition_col, 'tasks'])[dependent_var]
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    desc['sem'] = desc['std'] / np.sqrt(desc['count'])
    
    return desc


# =============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# =============================================================================

def run_rm_anova(
    cell_means_valid: pd.DataFrame,
    dependent_var: str,
    condition_col: str
) -> pd.DataFrame:
    """
    Perform N×M repeated-measures ANOVA for each muscle using statsmodels.
    
    Runs a two-way repeated-measures ANOVA with within-subject factors:
        - condition_col (e.g., cart_categories)
        - tasks
    
    The ANOVA is computed separately for each muscle.
    
    Args:
        cell_means_valid: DataFrame with cell means for valid subjects only
        dependent_var: Name of the dependent variable column
        condition_col: Name of the condition column
    
    Returns:
        DataFrame with columns:
            - muscle: Muscle name
            - effect: Effect name (condition_col, 'tasks', or 'condition:tasks')
            - num_df: Numerator degrees of freedom
            - den_df: Denominator degrees of freedom
            - F_value: F statistic
            - p_value: p-value
            - n_subjects: Number of subjects in analysis
    
    Example:
        >>> anova_results = run_rm_anova(
        ...     cell_means_valid=cell_means_valid,
        ...     dependent_var="rvc_norm_rms",
        ...     condition_col="cart_categories"
        ... )
    """
    anova_rows = []
    
    for muscle in sorted(cell_means_valid['muscle'].unique()):
        df_m = cell_means_valid[cell_means_valid['muscle'] == muscle].copy()
        df_m = df_m.dropna(subset=[dependent_var])
        
        if df_m.empty:
            continue

        # Run AnovaRM with long format data
        aov = AnovaRM(
            df_m,
            depvar=dependent_var,
            subject='subjects',
            within=[condition_col, 'tasks']
        ).fit()

        tbl = aov.anova_table

        # Extract results for each effect
        effects = [condition_col, 'tasks', f'{condition_col}:tasks']
        
        for effect in effects:
            if effect not in tbl.index:
                continue
            row = tbl.loc[effect]
            anova_rows.append({
                'muscle': muscle,
                'effect': effect,
                'num_df': float(row['Num DF']),
                'den_df': float(row['Den DF']),
                'F_value': float(row['F Value']),
                'p_value': float(row['Pr > F']),
                'n_subjects': int(df_m['subjects'].nunique())
            })

    anova_df = pd.DataFrame(anova_rows)
    return anova_df


# =============================================================================
# METADATA FUNCTIONS
# =============================================================================

def build_analysis_metadata(
    valid_subjects: pd.Index,
    data_file_name: str,
    dependent_var: str,
    dependent_var_label: str,
    condition_col: str,
    n_conditions: int,
    n_tasks: int,
    alpha: float
) -> Dict:
    """
    Build analysis metadata dictionary for the methods sheet.
    
    Creates a structured metadata dictionary that describes the analysis
    parameters and can be used to generate the methods sheet in the Excel output.
    
    Args:
        valid_subjects: Index of valid subject IDs
        data_file_name: Name of the input data file
        dependent_var: Name of the dependent variable column
        dependent_var_label: Display label for the dependent variable
        condition_col: Name of the condition column
        n_conditions: Number of condition levels
        n_tasks: Number of task levels
        alpha: Significance level
    
    Returns:
        Dictionary containing analysis metadata with keys:
            - data_file, dependent_var, analysis_type, factors, design,
            - row_unit, preprocessing, sample_criteria, alpha, n_subjects,
            - within_factors, statistical_package
    
    Example:
        >>> metadata = build_analysis_metadata(
        ...     valid_subjects=valid_subjects,
        ...     data_file_name="data.csv",
        ...     dependent_var="rvc_norm_rms",
        ...     dependent_var_label="RVC Normalized RMS",
        ...     condition_col="cart_categories",
        ...     n_conditions=2,
        ...     n_tasks=3,
        ...     alpha=0.05
        ... )
    """
    return {
        'data_file': data_file_name,
        'dependent_var': dependent_var_label,
        'analysis_type': 'rm_anova',
        'factors': [condition_col, 'tasks'],
        'design': f'{n_conditions}×{n_tasks} repeated-measures',
        'row_unit': f"subjects × {condition_col} × tasks × trials × muscle",
        'preprocessing': f'Averaged across conditions and trials for each subject×{condition_col}×task×muscle cell',
        'sample_criteria': f'Complete {n_conditions}×{n_tasks} design for all muscles',
        'alpha': alpha,
        'n_subjects': len(valid_subjects),
        'within_factors': [condition_col, 'tasks'],
        'statistical_package': 'statsmodels.AnovaRM'
    }


# =============================================================================
# EXCEL EXPORT FUNCTIONS
# =============================================================================

def export_to_excel(
    output_dir: str,
    excel_name: str,
    analysis_metadata: Dict,
    desc_df: pd.DataFrame,
    anova_df: pd.DataFrame,
    cell_means_valid: pd.DataFrame,
    dependent_var: str,
    alpha: float
) -> str:
    """
    Export analysis results to a formatted Excel file.
    
    Creates an Excel workbook with the following sheets:
        - methods: Analysis method summary (dynamically generated)
        - descriptives: Muscle × condition × task descriptive statistics
        - anova_results: Repeated-measures ANOVA results by muscle
        - cell_means: Subject-level cell means
    
    Args:
        output_dir: Directory path for output file
        excel_name: Name of the output Excel file
        analysis_metadata: Dictionary with analysis metadata (from build_analysis_metadata)
        desc_df: Descriptive statistics DataFrame (from compute_descriptives)
        anova_df: ANOVA results DataFrame (from run_rm_anova)
        cell_means_valid: Cell means DataFrame for valid subjects
        dependent_var: Name of the dependent variable column
        alpha: Significance level for highlighting
    
    Returns:
        Full path to the created Excel file
    
    Example:
        >>> excel_path = export_to_excel(
        ...     output_dir="./OUTPUT",
        ...     excel_name="anova_results.xlsx",
        ...     analysis_metadata=metadata,
        ...     desc_df=desc_df,
        ...     anova_df=anova_df,
        ...     cell_means_valid=cell_means_valid,
        ...     dependent_var="rvc_norm_rms",
        ...     alpha=0.05
        ... )
    """
    import xlsxwriter

    os.makedirs(output_dir, exist_ok=True)
    excel_path = os.path.join(output_dir, excel_name)

    workbook = xlsxwriter.Workbook(str(excel_path))

    # Create formats
    formats = create_formats(workbook)

    # 1. Methods sheet (dynamically generated)
    write_methods_sheet(workbook, analysis_metadata, formats)

    # 2. Descriptives sheet
    desc_pl = pl.from_pandas(desc_df)
    write_descriptives_sheet(
        workbook,
        desc_pl,
        formats,
        sheet_name="descriptives",
        title=f"Muscle × Condition × Task Descriptive Statistics ({analysis_metadata['dependent_var']})"
    )

    # 3. ANOVA Results sheet
    anova_pl = pl.from_pandas(anova_df)
    
    # Add significance column (p < alpha)
    anova_pl = anova_pl.with_columns(
        pl.when(pl.col("p_value") < alpha)
        .then(pl.lit("Yes"))
        .otherwise(pl.lit("No"))
        .alias("significant")
    )

    write_statistical_tests_sheet(
        workbook,
        anova_pl,
        formats,
        sheet_name="anova_results",
        title="Repeated-Measures ANOVA Results by Muscle",
        sig_col_name="significant"
    )

    # 4. Cell Means sheet
    cell_means_pl = pl.from_pandas(cell_means_valid)

    ws_cm = workbook.add_worksheet("cell_means")
    ws_cm.merge_range(
        0, 0, 0, len(cell_means_pl.columns) - 1,
        "Subject-Level Cell Means",
        formats['header']
    )

    # Column headers
    for col_idx, col_name in enumerate(cell_means_pl.columns):
        ws_cm.write(1, col_idx, col_name, formats['header'])

    # Data rows
    for row_idx, row in enumerate(cell_means_pl.iter_rows(), start=2):
        for col_idx, val in enumerate(row):
            ws_cm.write(row_idx, col_idx, val, formats['cell'])

    ws_cm.set_column(0, len(cell_means_pl.columns) - 1, 15)

    workbook.close()
    print(f"\n결과 저장 완료: {excel_path}")

    return excel_path


def save_intermediate_csvs(
    output_dir: str,
    cell_means_valid: pd.DataFrame,
    desc_df: pd.DataFrame,
    anova_df: pd.DataFrame,
    prefix: str = ""
) -> Dict[str, str]:
    """
    Save intermediate analysis results as CSV files.
    
    Useful for debugging or for importing results into other tools.
    
    Args:
        output_dir: Directory path for output files
        cell_means_valid: Cell means DataFrame for valid subjects
        desc_df: Descriptive statistics DataFrame
        anova_df: ANOVA results DataFrame
        prefix: Optional prefix for output filenames
    
    Returns:
        Dictionary with file type keys and path values
    
    Example:
        >>> csv_paths = save_intermediate_csvs(
        ...     output_dir="./OUTPUT",
        ...     cell_means_valid=cell_means_valid,
        ...     desc_df=desc_df,
        ...     anova_df=anova_df,
        ...     prefix="analysis1_"
        ... )
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cell_means_csv = os.path.join(output_dir, f"{prefix}cell_means_valid.csv")
    desc_csv = os.path.join(output_dir, f"{prefix}descriptives.csv")
    anova_csv = os.path.join(output_dir, f"{prefix}anova_results.csv")
    
    cell_means_valid.to_csv(cell_means_csv, index=False)
    desc_df.to_csv(desc_csv, index=False)
    anova_df.to_csv(anova_csv, index=False)
    
    return {
        "cell_means_csv": cell_means_csv,
        "descriptives_csv": desc_csv,
        "anova_csv": anova_csv
    }
