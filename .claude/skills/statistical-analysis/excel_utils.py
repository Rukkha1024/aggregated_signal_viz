"""
Common Excel Generation Utilities
==================================
Reusable for all statistical analyses (t-test, ANOVA, etc.)
Primarily for biomechanics analysis.

Format templates defined in .claude/skills/excel-format.md
"""

import xlsxwriter
import polars as pl
from typing import Dict, Optional, List


def create_formats(workbook: xlsxwriter.Workbook) -> Dict[str, xlsxwriter.format.Format]:
    """
    공통 서식 생성 (header, cell, significant)

    서식 정의는 .claude/skills/excel-format.md 참조

    Args:
        workbook: xlsxwriter Workbook 객체

    Returns:
        dict: {'header': header_format, 'cell': cell_format, 'significant': significant_format}
    """
    header_format = workbook.add_format({
        'bold': True,
        'bg_color': '#4472C4',
        'font_color': 'white',
        'border': 1,
        'align': 'center',
        'valign': 'vcenter'
    })

    cell_format = workbook.add_format({
        'border': 1,
        'align': 'center',
        'valign': 'vcenter'
    })

    significant_format = workbook.add_format({
        'border': 1,
        'align': 'center',
        'valign': 'vcenter',
        'bg_color': '#C6EFCE',
        'font_color': '#006100'
    })

    return {
        'header': header_format,
        'cell': cell_format,
        'significant': significant_format
    }


def generate_methods_description(analysis_metadata: dict) -> List[dict]:
    """
    분석 메타데이터를 기반으로 methods 시트 내용을 동적 생성
    
    생체역학 분석을 위한 범용 메타데이터 처리

    Args:
        analysis_metadata: dict
            {
                # Required fields
                'data_file': str,                    # e.g., "emg_summary.csv", "force_data.parquet"
                'dependent_var': str,                # e.g., "rvc_norm_rms (%RVC-RMS)", "peak_force (N)"
                'analysis_type': str,                # e.g., 'paired_t_test', 'rm_anova'
                'factors': list,                     # e.g., ['condition', 'tasks']
                'preprocessing': str,                # Description of preprocessing steps
                'sample_criteria': str,              # Subject selection criteria
                'alpha': float,                      # Significance level
                'n_subjects': int,                   # Number of subjects analyzed
                
                # Row unit description (동적 생성용)
                'row_unit': str,                     # e.g., "subjects × condition × tasks × trials × muscle"
                                                     # If not provided, will be auto-generated from factors
                
                # Optional fields
                'bonferroni_alpha': float,           # For Bonferroni correction
                'design': str,                       # e.g., "2×3 repeated-measures"
                'within_factors': list,              # For ANOVA
                'between_factors': list,             # For mixed ANOVA
                'statistical_package': str,          # e.g., "scipy.stats", "statsmodels.AnovaRM"
                'muscle_groups': list,               # Optional: for grouped muscle analysis
            }

    Returns:
        list of dict: methods 시트에 들어갈 item-description 쌍
    """
    methods_data = []

    # 1. Data file - row_unit을 동적으로 생성
    row_unit = analysis_metadata.get('row_unit')
    if not row_unit:
        # factors가 있으면 자동 생성: subjects × {factors} × tasks × trials × muscle
        factors = analysis_metadata.get('factors', [])
        factor_str = ' × '.join(factors) if factors else 'condition'
        row_unit = f"subjects × {factor_str} × trials × muscle"
    
    methods_data.append({
        'item': 'data_file',
        'description': f"{analysis_metadata['data_file']} (row unit: {row_unit})"
    })

    # 2. Dependent variable
    methods_data.append({
        'item': 'dependent_variable',
        'description': analysis_metadata['dependent_var']
    })

    # 3. Preprocessing
    methods_data.append({
        'item': 'preprocessing',
        'description': analysis_metadata['preprocessing']
    })

    # 4. Sample
    methods_data.append({
        'item': 'sample',
        'description': f"{analysis_metadata['sample_criteria']}, N={analysis_metadata['n_subjects']}"
    })

    # 5. Analysis (동적 생성 - analysis_type에 따라)
    analysis_desc = _generate_analysis_description(analysis_metadata)
    methods_data.append({
        'item': 'analysis',
        'description': analysis_desc
    })

    # 6. Descriptives
    methods_data.append({
        'item': 'descriptives',
        'description': 'Descriptive statistics (mean, standard deviation, standard error, N) were computed per condition across subjects.'
    })

    return methods_data


def _generate_analysis_description(metadata: dict) -> str:
    """
    분석 유형에 따라 적절한 분석 설명 생성

    Args:
        metadata: analysis_metadata dict

    Returns:
        str: 분석 방법 설명
    """
    analysis_type = metadata['analysis_type']

    if analysis_type == 'paired_t_test':
        desc = (
            f"Paired t-tests were performed to compare {' × '.join(metadata['factors'])}. "
            f"Significance level: α={metadata['alpha']}"
        )

        if 'bonferroni_alpha' in metadata:
            desc += f" (Bonferroni-corrected: α={metadata['bonferroni_alpha']})"

        if 'statistical_package' in metadata:
            desc += f" using {metadata['statistical_package']}."
        else:
            desc += " using scipy.stats.ttest_rel."

    elif analysis_type == 'rm_anova':
        design = metadata.get('design', 'repeated-measures')
        within_factors = metadata.get('within_factors', metadata['factors'])

        desc = (
            f"A {design} ANOVA was performed for each muscle with within-subject factors: "
            f"{', '.join(within_factors)}. "
            f"Significance level: α={metadata['alpha']}"
        )

        if 'statistical_package' in metadata:
            desc += f" using {metadata['statistical_package']}."
        else:
            desc += "."

    else:
        # 기본 설명
        desc = f"Statistical analysis with significance level α={metadata['alpha']}."

    return desc


def write_methods_sheet(workbook: xlsxwriter.Workbook,
                        analysis_metadata: dict,
                        formats: dict,
                        sheet_name: str = 'methods'):
    """
    분석 메타데이터를 받아 methods 시트를 동적으로 작성
    하드코딩된 템플릿이 아닌, 실제 수행된 분석을 기반으로 생성

    Args:
        workbook: xlsxwriter Workbook 객체
        analysis_metadata: 분석 메타데이터 dict
        formats: create_formats()에서 반환된 서식 dict
        sheet_name: 시트 이름 (기본값: 'methods')
    """
    ws = workbook.add_worksheet(sheet_name)

    # 제목
    ws.merge_range(0, 0, 0, 1, "Analysis Methods", formats['header'])

    # 컬럼 헤더
    ws.write(1, 0, 'Item', formats['header'])
    ws.write(1, 1, 'Description', formats['header'])

    # Methods 내용 생성
    methods_data = generate_methods_description(analysis_metadata)

    # 데이터 작성
    for row_idx, row_data in enumerate(methods_data, start=2):
        ws.write(row_idx, 0, row_data['item'], formats['cell'])
        ws.write(row_idx, 1, row_data['description'], formats['cell'])

    # 열 너비 조정
    ws.set_column(0, 0, 20)  # item 컬럼
    ws.set_column(1, 1, 80)  # description 컬럼


def write_descriptives_sheet(workbook: xlsxwriter.Workbook,
                             desc_df: pl.DataFrame,
                             formats: dict,
                             sheet_name: str = "descriptives",
                             title: str = "Descriptive Statistics"):
    """
    Descriptives 시트 작성

    Args:
        workbook: xlsxwriter Workbook 객체
        desc_df: polars DataFrame (기술통계량)
        formats: create_formats()에서 반환된 서식 dict
        sheet_name: 시트 이름
        title: 시트 제목
    """
    ws = workbook.add_worksheet(sheet_name)

    # 제목
    ws.merge_range(0, 0, 0, len(desc_df.columns) - 1, title, formats['header'])

    # 컬럼 헤더
    for col_idx, col_name in enumerate(desc_df.columns):
        ws.write(1, col_idx, col_name, formats['header'])

    # 데이터 작성
    for row_idx, row in enumerate(desc_df.iter_rows(), start=2):
        for col_idx, val in enumerate(row):
            ws.write(row_idx, col_idx, val, formats['cell'])

    # 열 너비 조정
    ws.set_column(0, len(desc_df.columns) - 1, 15)


def write_statistical_tests_sheet(workbook: xlsxwriter.Workbook,
                                  test_df: pl.DataFrame,
                                  formats: dict,
                                  sheet_name: str,
                                  title: str,
                                  sig_col_name: Optional[str] = None):
    """
    통계 검정 결과 시트 작성 (유의성 강조 포함)

    Args:
        workbook: xlsxwriter Workbook 객체
        test_df: polars DataFrame (통계 검정 결과)
        formats: create_formats()에서 반환된 서식 dict
        sheet_name: 시트 이름
        title: 시트 제목
        sig_col_name: 유의성 컬럼명 (예: "significant" 또는 "bonferroni_sig")
                     이 컬럼의 "Yes" 값을 녹색으로 강조
    """
    ws = workbook.add_worksheet(sheet_name)

    # 제목
    ws.merge_range(0, 0, 0, len(test_df.columns) - 1, title, formats['header'])

    # 컬럼 헤더
    for col_idx, col_name in enumerate(test_df.columns):
        ws.write(1, col_idx, col_name, formats['header'])

    # 유의성 컬럼 인덱스 찾기
    sig_col_idx = None
    if sig_col_name and sig_col_name in test_df.columns:
        sig_col_idx = test_df.columns.index(sig_col_name)

    # 데이터 작성
    for row_idx, row in enumerate(test_df.iter_rows(), start=2):
        for col_idx, val in enumerate(row):
            # 유의성 컬럼이고 값이 "Yes"인 경우 강조
            if sig_col_idx is not None and col_idx == sig_col_idx and val == "Yes":
                ws.write(row_idx, col_idx, val, formats['significant'])
            else:
                ws.write(row_idx, col_idx, val, formats['cell'])

    # 열 너비 조정
    ws.set_column(0, len(test_df.columns) - 1, 12)
