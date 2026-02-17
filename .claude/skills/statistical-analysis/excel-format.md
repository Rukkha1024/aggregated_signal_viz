# Excel Format Template Skill

This skill defines the standard Excel formatting for statistical analysis outputs in **biomechanical analysis**.

## Purpose

Provides a unified Excel format for t-test and ANOVA statistical analysis results, ensuring consistency across all biomechanical analyses.

---

## Data Structure Assumption (데이터 구조 가정)

### Fixed Columns (고정 컬럼)
All biomechanical data files should contain these core columns:
- `subjects`: Subject/participant ID
- `muscle`: Muscle name
- `tasks`: Task/movement type
- `trials` or `trial_num`: Trial number

### Variable Elements (가변 요소)
- **Condition factors**: e.g., `cart_categories`, `condition`, `load_level`, `speed`
- **Dependent variable**: e.g., `rvc_norm_rms`, `emg_amplitude`, `force`, `angle`
- **Data file name**: Changes based on analysis target

---

## ⚠️ CRITICAL DESIGN PRINCIPLE (핵심 설계 원칙)

### Column/Variable Detection is AI's Responsibility (컬럼/변수 감지는 AI의 책임)

**The Python scripts in `.claude/skills/` are FUNCTION LIBRARIES, NOT standalone programs.**

- **DO NOT** hard-code column names (e.g., `DEPENDENT_VAR = "rvc_norm_rms"`) in scripts
- **DO NOT** auto-detect meta columns via Python logic
- **AI MUST** analyze the data structure and determine:
  - Which column is the dependent variable
  - Which columns are condition factors
  - Which columns are fixed (subjects, muscle, tasks, trials)
- **AI MUST** pass these as function arguments when calling analysis functions

### Workflow

```
1. User requests statistical analysis
2. AI reads and inspects the data file (CSV/Parquet)
3. AI determines: dependent_var, condition_col, factors, etc.
4. AI calls analysis functions with explicit parameters
5. AI generates Excel output using excel_utils.py
```

### Example: How AI Should Call Functions

```python
# AI determines these values by inspecting the data:
dependent_var = "rvc_norm_rms"      # AI decision based on data inspection
condition_col = "cart_categories"   # AI decision based on data inspection
condition_values = ["new", "old"]   # AI decision based on unique values

# AI calls functions with explicit parameters
from ttest_statistical_analysis import (
    load_and_preprocess_data,
    aggregate_trials,
    paired_ttest_condition,
    build_analysis_metadata
)

df = load_and_preprocess_data(data_path, dependent_var, condition_col)
df_agg = aggregate_trials(df, dependent_var, condition_col)
results = paired_ttest_condition(df_agg, condition_col, condition_values)
```

---

## ⚠️ MANDATORY RULES (필수 규칙)

### 1. Skills Usage Requirement (스킬 사용 의무)
When performing **t-test** or **ANOVA** statistical analysis, you **MUST**:

- **For t-test**: Use or reference `.claude/skills/ttest_statistical_analysis.py`
- **For ANOVA**: Use or reference `.claude/skills/anova_statistical_analysis.py`  
- **For Excel output**: Use `.claude/skills/excel_utils.py` functions
- **For formatting**: Follow this document's specifications

### 2. Excel Output Requirement (Excel 출력 필수)
Statistical analysis **MUST always produce an Excel file** as output:

- Every t-test analysis → `OUTPUT/ttest_stats_results.xlsx` (or similar named file)
- Every ANOVA analysis → `OUTPUT/anova_stats_results.xlsx` (or similar named file)
- Excel file must contain all 4 standard sheets: `methods`, `descriptives`, `statistical_tests`, `cell_means`

### 3. Workflow
```
1. Read data (CSV/Parquet)
2. Run statistical analysis using skills scripts
3. Generate Excel output using excel_utils.py
4. Save to OUTPUT/ folder
```

---

## Format Configuration

### Color Scheme
- **Header Background**: #4472C4 (Blue)
- **Header Font**: White
- **Significant Cell Background**: #C6EFCE (Light Green)
- **Significant Cell Font**: #006100 (Dark Green)

### Cell Formatting
- **All cells**: Border, Center alignment, Vertical center
- **Headers**: Bold + Blue background + White text
- **Significant results**: Green highlight

## Sheet Structure

All statistical analysis Excel files should contain:

1. **methods**: Analysis methodology description
   - Columns: item, description
   - Content: data file info, dependent variable, preprocessing, sample criteria, analysis method

2. **descriptives**: Descriptive statistics
   - Columns: `{condition_factors}`, `tasks`, `muscle`, `mean`, `std`, `sem`, `n`, `mean_sd`
   - Content: Condition-wise descriptive statistics per muscle

3. **statistical_tests**: Statistical test results (t-test, ANOVA, etc.)
   - For t-test: `{factor}_comparison`, `task_comparison`, `muscle_group_analysis` (optional)
   - For ANOVA: `anova_results` with F-value, p-value, df

4. **cell_means** (optional): Subject-level data
   - Content: `subjects × {condition} × tasks × muscle` level data

## Sheet Naming Convention

- **Language**: English
- **Format**: lowercase_with_underscores
- **Examples**:
  - `methods`
  - `descriptives`
  - `{condition}_comparison` (e.g., `cart_comparison`, `load_comparison`)
  - `task_comparison`
  - `anova_results`
  - `muscle_group_analysis` (optional, for grouped muscle analysis)

## Implementation

When creating Excel files, use xlsxwriter with these format dictionaries:

```python
# Header format
header_format = workbook.add_format({
    'bold': True,
    'bg_color': '#4472C4',
    'font_color': 'white',
    'border': 1,
    'align': 'center',
    'valign': 'vcenter'
})

# Regular cell format
cell_format = workbook.add_format({
    'border': 1,
    'align': 'center',
    'valign': 'vcenter'
})

# Significant result format
significant_format = workbook.add_format({
    'border': 1,
    'align': 'center',
    'valign': 'vcenter',
    'bg_color': '#C6EFCE',
    'font_color': '#006100'
})
```

## Methods Sheet Dynamic Generation

The `methods` sheet should be dynamically generated based on analysis metadata, NOT hard-coded.

### Required Metadata

```python
analysis_metadata = {
    # Required fields
    'data_file': str,                    # e.g., "emg_summary.csv", "force_data.parquet"
    'dependent_var': str,                # e.g., "rvc_norm_rms (%RVC-RMS)", "peak_force (N)"
    'analysis_type': str,                # e.g., "paired_t_test" or "rm_anova"
    'factors': list,                     # e.g., ['condition', 'tasks'] - condition factors to analyze
    'preprocessing': str,                # Description of preprocessing steps
    'sample_criteria': str,              # Subject selection criteria
    'alpha': float,                      # Significance level (typically 0.05)
    'n_subjects': int,                   # Number of subjects analyzed
    
    # Row unit description (동적 생성용)
    'row_unit': str,                     # e.g., "subjects × condition × tasks × trials × muscle"
    
    # Optional fields depending on analysis type
    'bonferroni_alpha': float,           # For Bonferroni correction
    'design': str,                       # e.g., "2×3 repeated-measures"
    'within_factors': list,              # For ANOVA: list of within-subject factors
    'between_factors': list,             # For mixed ANOVA: list of between-subject factors
    'statistical_package': str,          # e.g., "scipy.stats", "statsmodels.AnovaRM"
    'muscle_groups': list,               # Optional: for grouped muscle analysis
}
```

### Methods Sheet Content

The methods sheet should translate the metadata into human-readable descriptions:

| Item | Description |
|------|-------------|
| data_file | `{data_file}` (row unit: `{row_unit}`) |
| dependent_variable | `{dependent_var}` |
| preprocessing | `{preprocessing}` |
| sample | `{sample_criteria}`, N=`{n_subjects}` |
| analysis | Auto-generated based on `{analysis_type}` |
| descriptives | Descriptive statistics (mean, std, sem, N) computed per condition |

## Usage Example

```python
from excel_utils import create_formats, write_methods_sheet

# Create workbook
workbook = xlsxwriter.Workbook('output.xlsx')

# Create formats
formats = create_formats(workbook)

# Define analysis metadata (example for EMG analysis)
metadata = {
    'data_file': 'emg_summary.csv',
    'dependent_var': 'normalized_emg (%MVC)',
    'analysis_type': 'paired_t_test',
    'factors': ['condition', 'tasks'],
    'row_unit': 'subjects × condition × tasks × trials × muscle',
    'preprocessing': 'Averaged across trials for each subject×condition×task×muscle combination',
    'sample_criteria': 'Complete 2×3 design (2 conditions × 3 tasks) for all muscles',
    'alpha': 0.05,
    'bonferroni_alpha': 0.017,
    'n_subjects': 15
}

# Write methods sheet (dynamically generated)
write_methods_sheet(workbook, metadata, formats)

workbook.close()
```

### Additional Examples

**Example 2: Force Analysis**
```python
metadata = {
    'data_file': 'force_data.parquet',
    'dependent_var': 'peak_force (N)',
    'analysis_type': 'rm_anova',
    'factors': ['load_level', 'speed'],
    'row_unit': 'subjects × load_level × speed × trials × muscle',
    'design': '3×2 repeated-measures',
    'within_factors': ['load_level (low, medium, high)', 'speed (slow, fast)'],
    'preprocessing': 'Peak force extracted from each trial',
    'sample_criteria': 'Complete design for all conditions',
    'alpha': 0.05,
    'n_subjects': 20,
    'statistical_package': 'statsmodels.AnovaRM'
}
```

**Example 3: Joint Angle Analysis**
```python
metadata = {
    'data_file': 'kinematics.csv',
    'dependent_var': 'peak_angle (degrees)',
    'analysis_type': 'paired_t_test',
    'factors': ['posture'],
    'row_unit': 'subjects × posture × tasks × trials × joint',
    'preprocessing': 'Peak angle during movement phase',
    'sample_criteria': 'All subjects completed both posture conditions',
    'alpha': 0.05,
    'n_subjects': 12
}
```

## Best Practices

1. **Never hard-code methods descriptions** - always generate from metadata
2. **Use consistent color scheme** across all analyses
3. **Apply significance highlighting** to p-value results
4. **Maintain English sheet names** for international compatibility
5. **Include all four core sheets** (methods, descriptives, statistical_tests, cell_means)
6. **Set appropriate column widths** for readability (typically 12-15 characters)
7. **Merge cells for section titles** to improve visual organization
