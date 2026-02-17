---
name: statistical-analysis
description: Perform paired t-tests and repeated-measures ANOVA for biomechanical data analysis with formatted Excel output. Use when analyzing EMG, force, kinematics data with statistical comparisons, creating auto-updating Excel reports, or applying Excel formula automation. Always outputs publication-ready Excel files.
---

# Statistical Analysis Skill

> Biomechanical data statistical analysis with Excel output

## Overview

This skill provides a complete workflow for biomechanical research:
- **Paired t-test**: Condition and task comparisons with Bonferroni correction
- **Repeated-measures ANOVA**: N×M within-subject design analysis
- **Excel output**: Formatted reports with standard color scheme and templates
- **Excel formula automation**: Auto-updating analysis with hybrid Python + formula approach

## When to Use

- Analyzing EMG, force, or kinematic data
- Comparing conditions (e.g., new vs old cart)
- Comparing tasks (e.g., lift, pull, push)
- Running repeated-measures experiments
- Generating publication-ready statistical reports in Excel
- Creating auto-updating Excel analysis sheets

## Files

| File | Purpose |
|------|---------|
| `ttest_statistical_analysis.py` | Paired t-test function library |
| `anova_statistical_analysis.py` | Repeated-measures ANOVA function library |
| `excel_utils.py` | Common Excel formatting utilities |
| `excel-format.md` | Standard formatting templates (color scheme, sheet structure) |
| `excel-formula-automation.md` | Guide for Excel formula automation workflow |

## Key Principle

**⚠️ These are FUNCTION LIBRARIES, not standalone scripts.**

The AI must:
1. Inspect the data to determine column structure
2. Identify dependent variable and condition factors
3. Call functions with explicit parameters
4. Always output results to Excel

## Usage Example

### T-test Analysis

```python
from ttest_statistical_analysis import (
    load_and_preprocess_data,
    aggregate_trials,
    paired_ttest_condition,
    export_to_excel
)

# AI determines these by inspecting data
dependent_var = "rvc_norm_rms"
condition_col = "cart_categories"
condition_values = ["new", "old"]

# Run analysis
df = load_and_preprocess_data(data_path, dependent_var, condition_col)
df_agg = aggregate_trials(df, dependent_var, condition_col)
results = paired_ttest_condition(df_agg, condition_col, condition_values)
```

### ANOVA Analysis

```python
from anova_statistical_analysis import (
    compute_cell_means,
    find_valid_subjects,
    run_rm_anova,
    export_to_excel
)

# Run analysis
cell_means = compute_cell_means(df, dependent_var, condition_col)
valid_subjects, _ = find_valid_subjects(cell_means, condition_col, dependent_var)
anova_results = run_rm_anova(cell_means_valid, dependent_var, condition_col)
```

## Excel Output Format

All analyses produce Excel files with standard sheets:
1. **methods**: Analysis methodology (dynamically generated)
2. **descriptives**: Mean, SD, SEM, N per condition
3. **statistical_tests**: Test results with significance highlighting
4. **cell_means**: Subject-level data (optional)

### Standard Color Scheme

| Element | Color Code |
|---------|------------|
| Header Background | #4472C4 (Blue) |
| Header Font | White |
| Significant Cell BG | #C6EFCE (Light Green) |
| Significant Cell Font | #006100 (Dark Green) |

See `excel-format.md` for detailed formatting specifications.
See `excel-formula-automation.md` for formula automation workflow.
