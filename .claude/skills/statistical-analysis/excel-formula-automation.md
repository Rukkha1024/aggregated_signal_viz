---
name: excel-formula-automation
description: Automates repetitive Excel data analysis tasks using Python + Excel formula hybrid approach. Use when creating auto-updating analysis reports, maintaining data integrity with formulas, or combining complex Python calculations with Excel's dynamic updates. Preserves original data and enables real-time recalculation.
allowed-tools: Read, Write, Bash, Glob, Grep, mcp__excel__*
---

# Excel Formula Automation Skill

> Guidelines for automating Excel data analysis with Python + Excel formula hybrid approach

---

## 📌 Overview

This Skill defines the **Work Process** and **Approach Guidelines** for automating repetitive Excel analysis tasks.

### Core Principles
1. **Preserve Original Data** - Always create analysis results in separate sheets
2. **Prioritize Auto-Update** - Use Excel formulas when possible (auto-reflects data changes)
3. **Hybrid Approach** - Pre-calculate complex logic in Python, insert results only
4. **Incremental Validation** - Verify results at each step

---

## 🔄 Work Process

### Phase 1: Data Discovery

```python
# Step 1: Understand Excel structure
import pandas as pd

# Check sheet list
xl = pd.ExcelFile('filename.xlsx')
print(xl.sheet_names)

# Check data structure
df = pd.read_excel('filename.xlsx', sheet_name='sheet_name')
print(df.columns.tolist())  # Column names
print(df.dtypes)             # Data types
print(df.head(10))           # Sample data
print(df.shape)              # Row/column count
```

**Checklist:**
- [ ] Sheet structure (data sheets vs result sheets)
- [ ] Column names and data types
- [ ] Unique values (`df['column'].unique()`)
- [ ] Missing values (`df.isnull().sum()`)

### Phase 2: Requirements Analysis

**Question Checklist:**
1. What **aggregations/calculations** are needed? (count, sum, average, ratio)
2. What **conditions** for filtering? (specific values, ranges, combinations)
3. Should results **auto-update**?
4. Is **grouped analysis** needed? (by subject, by condition)
5. Final **output format**? (summary table, detailed list, pivot)

### Phase 3: Approach Selection

| Situation | Recommended Approach | Reason |
|-----------|---------------------|--------|
| Simple conditional count | Excel formula | Auto-updates on data change |
| Ratio/average calculation | Excel formula + error handling | Needs division by zero prevention |
| Complex string aggregation | Python pre-calculation | Difficult to express in formulas |
| Dynamic list generation | Python pre-calculation | TEXTJOIN limitations |
| Large data (100k+ rows) | Python preprocessing | Formula performance issues |
| Real-time update needed | Excel formula | No manual re-execution needed |

### Phase 4: Implementation

```
[Implementation Order]
1. Header/layout setup
2. Summary statistics section (total count, ratios)
3. Detailed analysis section (grouped analysis)
4. Apply formatting (bold, borders, colors)
5. Validation
```

### Phase 5: Validation

```python
# Result verification code
from openpyxl import load_workbook

wb = load_workbook('filename.xlsx', data_only=True)  # Load formula results
ws = wb['analysis_auto']

# Check specific cell values
for row in ws.iter_rows(min_row=1, max_row=20, values_only=True):
    print(row)
```

---

## 📐 Excel Formula Patterns

### Pattern 1: Conditional Count (COUNTIFS)

```excel
=COUNTIFS(condition_range1, condition_value1, condition_range2, condition_value2)
```

**Example - Count specific condition for specific subject:**
```excel
=COUNTIFS(in!$A:$A, A5, in!$B:$B, 1)
```
- `in!$A:$A` = Subject column (full column reference handles data expansion)
- `A5` = Subject name in current row
- `in!$B:$B` = Condition column
- `1` = Condition value

**Python generation code:**
```python
f'=COUNTIFS(in!$A:$A, A{row}, in!$B:$B, 1)'
```

### Pattern 2: Multiple Condition Count

```excel
=COUNTIFS(range1, condition1, range2, condition2, range3, condition3)
```

**Example - Three conditions simultaneously:**
```python
f'=COUNTIFS(in!$A:$A, A{row}, in!$B:$B, 1, in!$C:$C, ">0")'
```

### Pattern 3: Ratio Calculation (Division by Zero Prevention)

**Basic pattern:**
```excel
=IF(denominator=0, 0, ROUND(numerator/denominator*100, 1))
```

**Example - Mixed ratio:**
```python
# Numerator: Mixed=1 count, Denominator: total count
f'=IF(C{row}=0, 0, ROUND(D{row}/C{row}*100, 1))'
```

### Pattern 4: Summary Statistics (SUMPRODUCT)

**Total sum:**
```excel
=SUMPRODUCT((in!$B:$B=1)*1)
```

**Conditional sum:**
```excel
=SUMPRODUCT((in!$A:$A<>"")*1)  -- Count non-empty cells
```

**Ratio (error prevention + rounding error handling):**
```python
f'=IF(B3=0, 0, ROUND(ROUND(B4/B3, 4)*100, 1))'
```
- Double ROUND prevents floating point errors (e.g., 49.99999... → 50.0)

### Pattern 5: Dynamic Range Reference

**Full column reference (handles data addition):**
```excel
=COUNTIFS(in!$A:$A, "condition")  -- Entire column A
```

**Explicit range (performance optimization):**
```excel
=COUNTIFS(in!$A$2:$A$1000, "condition")  -- Max 999 rows
```

---

## 🐍 Python Pre-calculation Patterns

### Pattern 1: Grouped String Aggregation

**Situation:** Concatenate values of specific condition by subject with commas
```python
# Example: List of velocity values for Mixed trials by subject
mixed_values = df[df['mixed'] == 1].groupby('Subject')['velocity'].apply(
    lambda x: ', '.join(map(str, sorted(x.unique())))
).to_dict()

# Result: {'John': '15, 35', 'Jane': '25, 45, 55'}
```

### Pattern 2: Conditional Aggregation with Static Insertion

```python
# Calculate in Python
result = df.groupby('Subject').agg({
    'trial': 'count',
    'mixed': 'sum'
}).to_dict()

# Insert as static values in Excel
ws.cell(row=5, column=9, value=mixed_values.get(subject, '-'))
```

### Pattern 3: Complex Condition Filtering

```python
# Multiple condition filter
filtered = df[
    (df['condition_A'] == 1) &
    (df['condition_B'] > 0) &
    (df['condition_C'].isin(['X', 'Y']))
]

# Aggregate results
summary = filtered.groupby('Subject').size().to_dict()
```

---

## 🎨 Formatting Patterns

### Header Styles
```python
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment

# Section header
section_font = Font(bold=True, size=11)
section_fill = PatternFill(start_color='E7E6E6', end_color='E7E6E6', fill_type='solid')

# Table header
header_font = Font(bold=True)
header_fill = PatternFill(start_color='D9E1F2', end_color='D9E1F2', fill_type='solid')

# Border
thin_border = Border(
    left=Side(style='thin'),
    right=Side(style='thin'),
    top=Side(style='thin'),
    bottom=Side(style='thin')
)
```

### Conditional Formatting (Applied via Python)
```python
from openpyxl.formatting.rule import CellIsRule

# Change color if value exceeds specific threshold
red_fill = PatternFill(start_color='FFC7CE', end_color='FFC7CE', fill_type='solid')
ws.conditional_formatting.add(
    'E5:E100',
    CellIsRule(operator='greaterThan', formula=['50'], fill=red_fill)
)
```

---

## ⚠️ Error Handling

### 1. Division by Zero
```excel
=IF(denominator=0, 0, numerator/denominator)
=IFERROR(numerator/denominator, 0)
```

### 2. Empty Data
```python
# Default value handling in Python
value = data_dict.get(key, '-')  # Show '-' if missing
value = data_dict.get(key, 0)    # Use 0 if missing
```

### 3. Floating Point Error
```excel
=ROUND(ROUND(value, 4)*100, 1)  -- Double ROUND prevents 49.9999...
```

### 4. File Open State
```python
try:
    wb.save('filename.xlsx')
except PermissionError:
    print("⚠️ File is open in Excel. Close it and run again.")
```

---

## 📁 Standard Code Template

```python
"""
Excel Auto-Analysis Script Template
"""
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
from openpyxl.utils.dataframe import dataframe_to_rows

# ============================================================
# 1. Configuration
# ============================================================
INPUT_FILE = 'data.xlsx'
INPUT_SHEET = 'raw_data'
OUTPUT_SHEET = 'analysis_auto'

# ============================================================
# 2. Data Loading and Pre-calculation
# ============================================================
df = pd.read_excel(INPUT_FILE, sheet_name=INPUT_SHEET)
subjects = df['Subject'].unique().tolist()

# Pre-calculate complex aggregations in Python
precomputed_data = df.groupby('Subject').agg({...}).to_dict()

# ============================================================
# 3. Open Excel Workbook
# ============================================================
wb = load_workbook(INPUT_FILE)

# Delete existing sheet and recreate
if OUTPUT_SHEET in wb.sheetnames:
    del wb[OUTPUT_SHEET]
ws = wb.create_sheet(OUTPUT_SHEET)

# ============================================================
# 4. Header and Summary Section
# ============================================================
current_row = 1

# Summary statistics
ws.cell(row=current_row, column=1, value='📊 SUMMARY')
current_row += 1
# ... write summary formulas

# ============================================================
# 5. Detailed Analysis Section
# ============================================================
current_row += 2
ws.cell(row=current_row, column=1, value='📋 DETAIL ANALYSIS')
current_row += 1

# Header row
headers = ['Subject', 'Total', 'Count', 'Ratio', ...]
for col, header in enumerate(headers, 1):
    cell = ws.cell(row=current_row, column=col, value=header)
    cell.font = Font(bold=True)
current_row += 1

# Data rows (formulas + static values)
for subject in subjects:
    ws.cell(row=current_row, column=1, value=subject)
    ws.cell(row=current_row, column=2, value=f'=COUNTIFS({INPUT_SHEET}!$A:$A, A{current_row})')
    # ... additional formulas
    ws.cell(row=current_row, column=9, value=precomputed_data.get(subject, '-'))
    current_row += 1

# ============================================================
# 6. Apply Formatting
# ============================================================
# Auto-adjust column widths (simple version)
for col in ws.columns:
    ws.column_dimensions[col[0].column_letter].width = 12

# ============================================================
# 7. Save
# ============================================================
try:
    wb.save(INPUT_FILE)
    print(f"✅ '{OUTPUT_SHEET}' sheet has been created.")
except PermissionError:
    print("⚠️ Close the file and run again.")
```

---

## ✅ Checklist

### Before Implementation
- [ ] Original data structure understood
- [ ] Required calculations/aggregations defined
- [ ] Auto-update requirement confirmed
- [ ] Output layout designed

### During Implementation
- [ ] No modification to original sheet
- [ ] Full column references for data expansion
- [ ] Division by zero handling added
- [ ] Complex aggregations pre-calculated in Python

### After Implementation
- [ ] Formula result values verified
- [ ] Auto-update tested after data addition
- [ ] Empty data cases tested
- [ ] File save successful

---

## 📚 Reference

### openpyxl Key Functions
- `load_workbook(filename, data_only=False)` - Load Excel file
- `wb.create_sheet(title)` - Create new sheet
- `ws.cell(row, column, value)` - Input value/formula to cell
- `ws.merge_cells('A1:D1')` - Merge cells
- `wb.save(filename)` - Save

### pandas Key Functions
- `pd.read_excel(file, sheet_name)` - Read Excel
- `df.groupby('col').agg({...})` - Grouped aggregation
- `df['col'].unique()` - Unique values
- `df[condition]` - Filtering
