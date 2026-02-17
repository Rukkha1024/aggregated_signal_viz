"""
Paired T-test Function Library
===============================
Primarily for biomechanics analysis (repeated-measures experimental data).

⚠️ Important: This is a FUNCTION LIBRARY. Do not execute directly.
- Meta columns (dependent_var, condition_col, etc.) are determined by AI/user and passed as function arguments
- No hardcoded settings - all configurations explicitly passed at function call

Analysis Features:
1. Condition comparison: Paired t-test per task (by measurement unit)
2. Task comparison: Multiple paired t-tests + Bonferroni correction
3. Descriptive statistics (Mean ± SD)
4. Optional grouped analysis (e.g., muscle groups)

Expected Column Semantics (names are flexible, passed as parameters):
- Subject identifier (e.g., subjects, participant_id)
- Measurement unit (e.g., muscle, sensor, channel)
- Task/condition (e.g., tasks, movement, activity)
- Trial/repetition (e.g., trials, trial_num, rep)

사용 예시:
```python
from ttest_statistical_analysis import (
    load_and_preprocess_data,
    aggregate_trials,
    calculate_descriptive_stats,
    paired_ttest_condition,
    paired_ttest_tasks,
    build_analysis_metadata,
    export_to_excel
)

# AI가 데이터를 분석하여 결정한 값들
dependent_var = "rvc_norm_rms"
condition_col = "cart_categories"
condition_values = ["new", "old"]
alpha = 0.05

# 함수 호출
df = load_and_preprocess_data(data_path, dependent_var, condition_col)
df_agg = aggregate_trials(df, dependent_var, condition_col)
results = paired_ttest_condition(df_agg, condition_col, condition_values, alpha)
```
"""

import polars as pl
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Optional, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# .claude/skills/ 폴더 내 상대 임포트
from excel_utils import create_formats, write_methods_sheet, write_descriptives_sheet, write_statistical_tests_sheet


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================
def load_and_preprocess_data(
    data_path: Path,
    dependent_var: str,
    condition_col: str,
    task_mapping: Optional[Dict[str, str]] = None
) -> pl.DataFrame:
    """
    데이터 로드 및 전처리
    
    Args:
        data_path: 데이터 파일 경로 (CSV 또는 Parquet)
        dependent_var: 종속변수 컬럼명
        condition_col: 조건 컬럼명
        task_mapping: 동작명 매핑 딕셔너리 (예: {"pull_walk": "pull"})
    
    Returns:
        전처리된 DataFrame
    """
    print(f"Loading data from: {data_path}")

    # 파일 확장자에 따라 로드 방식 결정
    if str(data_path).endswith('.parquet'):
        df = pl.read_parquet(data_path)
    else:
        df = pl.read_csv(data_path)
        
    print(f"Raw data shape: {df.shape}")
    print(f"Columns: {df.columns}")

    # 기본 정보 출력
    print(f"\nUnique subjects: {df['subjects'].n_unique()}")
    print(f"Unique {condition_col}: {df[condition_col].unique().to_list()}")
    print(f"Unique tasks: {df['tasks'].unique().to_list()}")
    print(f"Unique muscles: {df['muscle'].unique().to_list()}")

    # 종속변수 결측치 제거
    df_clean = df.filter(pl.col(dependent_var).is_not_null())
    print(f"\nAfter removing NaN in {dependent_var}: {df_clean.shape}")

    # 동작명 표준화 (매핑이 있는 경우만)
    if task_mapping:
        df_clean = df_clean.with_columns(
            pl.col("tasks").replace(task_mapping).alias("task_short")
        )
    else:
        df_clean = df_clean.with_columns(
            pl.col("tasks").alias("task_short")
        )

    return df_clean


def aggregate_trials(
    df: pl.DataFrame,
    dependent_var: str,
    condition_col: str
) -> pl.DataFrame:
    """
    각 피험자 × 조건 × 동작 × 근육 조합에서 시도들의 평균값 계산
    
    Args:
        df: 전처리된 DataFrame
        dependent_var: 종속변수 컬럼명
        condition_col: 조건 컬럼명
    
    Returns:
        집계된 DataFrame
    """
    df_agg = (
        df.group_by(["subjects", condition_col, "task_short", "muscle"])
        .agg(
            pl.col(dependent_var).mean().alias("mean_value"),
            pl.col(dependent_var).std().alias("std_value"),
            pl.col(dependent_var).count().alias("n_trials")
        )
        .sort(["subjects", condition_col, "task_short", "muscle"])
    )

    print(f"\nAggregated data shape: {df_agg.shape}")
    return df_agg


# =============================================================================
# Descriptive Statistics
# =============================================================================
def calculate_descriptive_stats(
    df_agg: pl.DataFrame,
    condition_col: str
) -> pl.DataFrame:
    """
    조건 × 동작 × 근육별 기술통계량 (Mean ± SD)
    
    Args:
        df_agg: 집계된 DataFrame
        condition_col: 조건 컬럼명
    
    Returns:
        기술통계량 DataFrame
    """
    desc_stats = (
        df_agg.group_by([condition_col, "task_short", "muscle"])
        .agg(
            pl.col("mean_value").mean().alias("grand_mean"),
            pl.col("mean_value").std().alias("grand_sd"),
            pl.col("mean_value").count().alias("n_subjects")
        )
        .sort(["muscle", "task_short", condition_col])
    )

    # SEM 계산
    desc_stats = desc_stats.with_columns(
        (pl.col("grand_sd") / pl.col("n_subjects").sqrt()).alias("sem")
    )

    # Mean ± SD 형태의 문자열 생성
    desc_stats = desc_stats.with_columns(
        (pl.col("grand_mean").round(3).cast(pl.Utf8) + " ± " + pl.col("grand_sd").round(3).cast(pl.Utf8)).alias("mean_sd")
    )

    return desc_stats


# =============================================================================
# Muscle Group Analysis Functions (Optional)
# =============================================================================
def calculate_muscle_group_mean(
    df_agg: pl.DataFrame,
    muscle_list: List[str],
    condition_col: str
) -> pl.DataFrame:
    """
    특정 근육군(muscle_list)의 평균값 계산
    
    Args:
        df_agg: 집계된 DataFrame
        muscle_list: 근육군에 포함될 근육 리스트
        condition_col: 조건 컬럼명
    
    Returns:
        근육군 평균 DataFrame
    """
    mg_df = df_agg.filter(pl.col("muscle").is_in(muscle_list))

    mg_mean = (
        mg_df.group_by(["subjects", condition_col, "task_short"])
        .agg(
            pl.col("mean_value").mean().alias("mg_mean_value")
        )
        .sort(["subjects", condition_col, "task_short"])
    )

    return mg_mean


def paired_ttest_muscle_group_by_task(
    df_agg: pl.DataFrame,
    muscle_group_name: str,
    muscle_list: List[str],
    condition_col: str,
    task_pairs: Optional[List[tuple]] = None,
    alpha: float = 0.05,
    bonferroni_comparisons: int = 3
) -> pl.DataFrame:
    """
    특정 근육군 대상 동작 간 paired t-test
    
    Args:
        df_agg: 집계된 DataFrame
        muscle_group_name: 근육군 이름
        muscle_list: 근육군에 포함될 근육 리스트
        condition_col: 조건 컬럼명
        task_pairs: 비교할 동작 쌍 리스트 (기본: lift/pull/push 조합)
        alpha: 유의수준
        bonferroni_comparisons: Bonferroni 보정 비교 횟수
    
    Returns:
        t-test 결과 DataFrame
    """
    if task_pairs is None:
        task_pairs = [("lift", "pull"), ("lift", "push"), ("pull", "push")]
    
    bonferroni_alpha = alpha / bonferroni_comparisons
    
    mg_mean = calculate_muscle_group_mean(df_agg, muscle_list, condition_col)

    # 조건 평균으로 집계
    mg_by_task = (
        mg_mean.group_by(["subjects", "task_short"])
        .agg(pl.col("mg_mean_value").mean().alias("mg_value"))
    )

    results = []

    for task1, task2 in task_pairs:
        task1_data = mg_by_task.filter(pl.col("task_short") == task1)
        task2_data = mg_by_task.filter(pl.col("task_short") == task2)

        # 공통 피험자
        common_subjects = sorted(
            set(task1_data["subjects"].to_list()) &
            set(task2_data["subjects"].to_list())
        )

        if len(common_subjects) < 2:
            continue

        task1_values = []
        task2_values = []

        for subj in common_subjects:
            val1 = task1_data.filter(pl.col("subjects") == subj)["mg_value"].to_list()
            val2 = task2_data.filter(pl.col("subjects") == subj)["mg_value"].to_list()

            if val1 and val2:
                task1_values.append(val1[0])
                task2_values.append(val2[0])

        if len(task1_values) < 2:
            continue

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(task1_values, task2_values)

        # 기술통계
        task1_mean = np.mean(task1_values)
        task2_mean = np.mean(task2_values)

        # 차이 비율 (%)
        diff_pct = ((task1_mean - task2_mean) / task2_mean * 100) if task2_mean != 0 else 0

        results.append({
            "comparison": f"{task1} vs {task2}",
            "muscle_group": muscle_group_name,
            "n_pairs": len(task1_values),
            f"{task1}_mean": round(task1_mean, 4),
            f"{task2}_mean": round(task2_mean, 4),
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_value, 6),
            "bonferroni_sig": "Yes" if p_value < bonferroni_alpha else "No",
            "diff_pct": round(diff_pct, 2),
        })

    return pl.DataFrame(results)


def paired_ttest_muscle_group_by_condition(
    df_agg: pl.DataFrame,
    muscle_group_name: str,
    muscle_list: List[str],
    condition_col: str,
    condition_values: List[str],
    alpha: float = 0.05
) -> pl.DataFrame:
    """
    특정 근육군 대상 조건 간 paired t-test (동작별)
    
    Args:
        df_agg: 집계된 DataFrame
        muscle_group_name: 근육군 이름
        muscle_list: 근육군에 포함될 근육 리스트
        condition_col: 조건 컬럼명
        condition_values: 비교할 조건값 리스트 [조건1, 조건2]
        alpha: 유의수준
    
    Returns:
        t-test 결과 DataFrame
    """
    mg_mean = calculate_muscle_group_mean(df_agg, muscle_list, condition_col)

    results = []
    tasks = mg_mean["task_short"].unique().sort().to_list()

    for task in tasks:
        subset = mg_mean.filter(pl.col("task_short") == task)

        cond1_data = subset.filter(pl.col(condition_col) == condition_values[0])
        cond2_data = subset.filter(pl.col(condition_col) == condition_values[1])

        common_subjects = sorted(
            set(cond1_data["subjects"].to_list()) &
            set(cond2_data["subjects"].to_list())
        )

        if len(common_subjects) < 2:
            continue

        cond1_values = []
        cond2_values = []

        for subj in common_subjects:
            val1 = cond1_data.filter(pl.col("subjects") == subj)["mg_mean_value"].to_list()
            val2 = cond2_data.filter(pl.col("subjects") == subj)["mg_mean_value"].to_list()

            if val1 and val2:
                cond1_values.append(val1[0])
                cond2_values.append(val2[0])

        if len(cond1_values) < 2:
            continue

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(cond1_values, cond2_values)

        # 기술통계
        cond1_mean = np.mean(cond1_values)
        cond2_mean = np.mean(cond2_values)

        # 차이 비율 (%)
        diff_pct = ((cond2_mean - cond1_mean) / cond2_mean * 100) if cond2_mean != 0 else 0

        results.append({
            "task": task,
            "muscle_group": muscle_group_name,
            "n_pairs": len(cond1_values),
            f"{condition_values[0]}_mean": round(cond1_mean, 4),
            f"{condition_values[1]}_mean": round(cond2_mean, 4),
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_value, 6),
            "significant": "Yes" if p_value < alpha else "No",
            "diff_pct": round(diff_pct, 2),
        })

    return pl.DataFrame(results)


# =============================================================================
# Statistical Tests
# =============================================================================
def paired_ttest_condition(
    df_agg: pl.DataFrame,
    condition_col: str,
    condition_values: List[str],
    alpha: float = 0.05
) -> pl.DataFrame:
    """
    조건 간 paired t-test (각 동작 × 근육 조합에서 수행)
    
    Args:
        df_agg: 집계된 DataFrame
        condition_col: 조건 컬럼명
        condition_values: 비교할 조건값 리스트 [조건1, 조건2]
        alpha: 유의수준
    
    Returns:
        t-test 결과 DataFrame
    """
    results = []

    tasks = df_agg["task_short"].unique().sort().to_list()
    muscles = df_agg["muscle"].unique().sort().to_list()

    for task in tasks:
        for muscle in muscles:
            # 해당 동작/근육 데이터 필터링
            subset = df_agg.filter(
                (pl.col("task_short") == task) &
                (pl.col("muscle") == muscle)
            )

            # 피험자별로 조건 데이터 매칭
            cond1_data = subset.filter(pl.col(condition_col) == condition_values[0])
            cond2_data = subset.filter(pl.col(condition_col) == condition_values[1])

            # 공통 피험자 찾기
            cond1_subjects = set(cond1_data["subjects"].to_list())
            cond2_subjects = set(cond2_data["subjects"].to_list())
            common_subjects = sorted(cond1_subjects & cond2_subjects)

            if len(common_subjects) < 2:
                continue

            # 공통 피험자에 대한 paired data 생성
            cond1_values = []
            cond2_values = []

            for subj in common_subjects:
                val1 = cond1_data.filter(pl.col("subjects") == subj)["mean_value"].to_list()
                val2 = cond2_data.filter(pl.col("subjects") == subj)["mean_value"].to_list()

                if val1 and val2:
                    cond1_values.append(val1[0])
                    cond2_values.append(val2[0])

            if len(cond1_values) < 2:
                continue

            # Paired t-test
            t_stat, p_value = stats.ttest_rel(cond1_values, cond2_values)

            # 기술통계
            cond1_mean = np.mean(cond1_values)
            cond2_mean = np.mean(cond2_values)
            cond1_sd = np.std(cond1_values, ddof=1)
            cond2_sd = np.std(cond2_values, ddof=1)

            # 차이 비율 (%)
            diff_pct = ((cond2_mean - cond1_mean) / cond2_mean * 100) if cond2_mean != 0 else 0

            results.append({
                "task": task,
                "muscle": muscle,
                "n_pairs": len(cond1_values),
                f"{condition_values[0]}_mean": round(cond1_mean, 4),
                f"{condition_values[0]}_sd": round(cond1_sd, 4),
                f"{condition_values[1]}_mean": round(cond2_mean, 4),
                f"{condition_values[1]}_sd": round(cond2_sd, 4),
                "t_statistic": round(t_stat, 4),
                "p_value": round(p_value, 6),
                "significant": "Yes" if p_value < alpha else "No",
                "diff_pct": round(diff_pct, 2),
            })

    return pl.DataFrame(results)


def paired_ttest_tasks(
    df_agg: pl.DataFrame,
    task_pairs: Optional[List[tuple]] = None,
    alpha: float = 0.05,
    bonferroni_comparisons: int = 3
) -> pl.DataFrame:
    """
    동작 간 paired t-test (Bonferroni 보정)
    
    Args:
        df_agg: 집계된 DataFrame
        task_pairs: 비교할 동작 쌍 리스트 (기본: lift/pull/push 조합)
        alpha: 유의수준
        bonferroni_comparisons: Bonferroni 보정 비교 횟수
    
    Returns:
        t-test 결과 DataFrame
    """
    if task_pairs is None:
        task_pairs = [("lift", "pull"), ("lift", "push"), ("pull", "push")]
    
    bonferroni_alpha = alpha / bonferroni_comparisons
    
    results = []
    muscles = df_agg["muscle"].unique().sort().to_list()

    for task1, task2 in task_pairs:
        for muscle in muscles:
            # 두 동작에 대한 데이터 필터링 (조건 무관하게 평균)
            task1_data = (
                df_agg.filter(
                    (pl.col("task_short") == task1) &
                    (pl.col("muscle") == muscle)
                )
                .group_by("subjects")
                .agg(pl.col("mean_value").mean().alias("value"))
            )

            task2_data = (
                df_agg.filter(
                    (pl.col("task_short") == task2) &
                    (pl.col("muscle") == muscle)
                )
                .group_by("subjects")
                .agg(pl.col("mean_value").mean().alias("value"))
            )

            # 공통 피험자 찾기
            task1_subjects = set(task1_data["subjects"].to_list())
            task2_subjects = set(task2_data["subjects"].to_list())
            common_subjects = sorted(task1_subjects & task2_subjects)

            if len(common_subjects) < 2:
                continue

            # Paired data 생성
            task1_values = []
            task2_values = []

            for subj in common_subjects:
                val1 = task1_data.filter(pl.col("subjects") == subj)["value"].to_list()
                val2 = task2_data.filter(pl.col("subjects") == subj)["value"].to_list()

                if val1 and val2:
                    task1_values.append(val1[0])
                    task2_values.append(val2[0])

            if len(task1_values) < 2:
                continue

            # Paired t-test
            t_stat, p_value = stats.ttest_rel(task1_values, task2_values)

            # 기술통계
            task1_mean = np.mean(task1_values)
            task2_mean = np.mean(task2_values)
            task1_sd = np.std(task1_values, ddof=1)
            task2_sd = np.std(task2_values, ddof=1)

            # 차이 비율 (%)
            diff_pct = ((task1_mean - task2_mean) / task2_mean * 100) if task2_mean != 0 else 0

            results.append({
                "comparison": f"{task1} vs {task2}",
                "muscle": muscle,
                "n_pairs": len(task1_values),
                f"{task1}_mean": round(task1_mean, 4),
                f"{task1}_sd": round(task1_sd, 4),
                f"{task2}_mean": round(task2_mean, 4),
                f"{task2}_sd": round(task2_sd, 4),
                "t_statistic": round(t_stat, 4),
                "p_value": round(p_value, 6),
                "bonferroni_sig": "Yes" if p_value < bonferroni_alpha else "No",
                "diff_pct": round(diff_pct, 2),
            })

    return pl.DataFrame(results)


# =============================================================================
# Build Analysis Metadata
# =============================================================================
def build_analysis_metadata(
    n_subjects: int,
    data_file: str,
    dependent_var: str,
    dependent_var_label: str,
    condition_col: str,
    alpha: float = 0.05,
    bonferroni_comparisons: int = 3,
    factors: Optional[List[str]] = None
) -> dict:
    """
    분석 메타데이터 생성
    
    Args:
        n_subjects: 피험자 수
        data_file: 데이터 파일명
        dependent_var: 종속변수 컬럼명
        dependent_var_label: 종속변수 표시 라벨
        condition_col: 조건 컬럼명
        alpha: 유의수준
        bonferroni_comparisons: Bonferroni 보정 비교 횟수
        factors: 분석 요인 리스트
    
    Returns:
        분석 메타데이터 딕셔너리
    """
    if factors is None:
        factors = [condition_col, 'tasks']
    
    bonferroni_alpha = alpha / bonferroni_comparisons
        
    return {
        'data_file': data_file,
        'dependent_var': dependent_var_label,
        'analysis_type': 'paired_t_test',
        'factors': factors,
        'row_unit': f"subjects × {condition_col} × tasks × trials × muscle",
        'preprocessing': f'Averaged across trials for each subject×{condition_col}×task×muscle combination',
        'sample_criteria': 'Complete design for all conditions and muscles',
        'alpha': alpha,
        'bonferroni_alpha': bonferroni_alpha,
        'n_subjects': n_subjects,
        'statistical_package': 'scipy.stats.ttest_rel',
        'comparisons': [f'{condition_col} comparison', 'tasks comparison (Bonferroni corrected)']
    }


# =============================================================================
# Export to Excel
# =============================================================================
def export_to_excel(
    desc_stats: pl.DataFrame,
    condition_ttest_results: pl.DataFrame,
    task_ttest_results: pl.DataFrame,
    analysis_metadata: dict,
    output_path: Path,
    condition_col: str,
    muscle_group_task_results: Optional[List[pl.DataFrame]] = None,
    muscle_group_condition_results: Optional[List[pl.DataFrame]] = None,
    alpha: float = 0.05,
    bonferroni_alpha: float = 0.017
):
    """
    분석 결과를 Excel 파일로 저장
    
    Args:
        desc_stats: 기술통계량 DataFrame
        condition_ttest_results: 조건 비교 t-test 결과
        task_ttest_results: 동작 비교 t-test 결과
        analysis_metadata: 분석 메타데이터
        output_path: 출력 파일 경로
        condition_col: 조건 컬럼명 (시트명에 사용)
        muscle_group_task_results: 근육군 동작 비교 결과 리스트 (선택)
        muscle_group_condition_results: 근육군 조건 비교 결과 리스트 (선택)
        alpha: 유의수준
        bonferroni_alpha: Bonferroni 보정 유의수준
    """
    import xlsxwriter

    # 출력 디렉토리 생성
    output_path.parent.mkdir(parents=True, exist_ok=True)

    workbook = xlsxwriter.Workbook(str(output_path))

    # 서식 생성
    formats = create_formats(workbook)

    # 1. Methods 시트 (동적 생성)
    write_methods_sheet(workbook, analysis_metadata, formats)

    # 2. Descriptives 시트
    write_descriptives_sheet(
        workbook,
        desc_stats,
        formats,
        sheet_name="descriptives",
        title=f"Condition × Task × Muscle Descriptive Statistics ({analysis_metadata['dependent_var']})"
    )

    # 3. Condition Comparison 시트
    write_statistical_tests_sheet(
        workbook,
        condition_ttest_results,
        formats,
        sheet_name=f"{condition_col}_comparison",
        title=f"{condition_col.replace('_', ' ').title()} Paired t-test Results by Task and Muscle",
        sig_col_name="significant"
    )

    # 4. Task Comparison 시트
    write_statistical_tests_sheet(
        workbook,
        task_ttest_results,
        formats,
        sheet_name="task_comparison",
        title=f"Task Comparison Paired t-test Results (Bonferroni α={bonferroni_alpha:.3f})",
        sig_col_name="bonferroni_sig"
    )

    # 5. Muscle Group Analysis 시트 (선택적)
    if muscle_group_task_results or muscle_group_condition_results:
        ws_mg = workbook.add_worksheet("muscle_group_analysis")

        current_row = 0
        
        # 섹션 1: 동작 간 비교
        if muscle_group_task_results and len(muscle_group_task_results) > 0 and muscle_group_task_results[0].height > 0:
            ws_mg.write(current_row, 0, f"Muscle Group Task Comparison (Bonferroni α={bonferroni_alpha:.3f})", formats['header'])
            ws_mg.merge_range(current_row, 0, current_row, 8, f"Muscle Group Task Comparison (Bonferroni α={bonferroni_alpha:.3f})", formats['header'])
            current_row += 1

            # 헤더
            mg_task_headers = list(muscle_group_task_results[0].columns)
            for col, header in enumerate(mg_task_headers):
                ws_mg.write(current_row, col, header, formats['header'])
            current_row += 1

            # 각 근육군 결과 작성
            sig_col_idx = mg_task_headers.index("bonferroni_sig") if "bonferroni_sig" in mg_task_headers else None

            for mg_result in muscle_group_task_results:
                for row in mg_result.iter_rows():
                    for col_idx, val in enumerate(row):
                        fmt = formats['significant'] if sig_col_idx == col_idx and val == "Yes" else formats['cell']
                        ws_mg.write(current_row, col_idx, val, fmt)
                    current_row += 1

            current_row += 4

        # 섹션 2: 조건 비교
        if muscle_group_condition_results and len(muscle_group_condition_results) > 0 and muscle_group_condition_results[0].height > 0:
            ws_mg.write(current_row, 0, f"Muscle Group {condition_col.replace('_', ' ').title()} Comparison (α={alpha})", formats['header'])
            ws_mg.merge_range(current_row, 0, current_row, 8, f"Muscle Group {condition_col.replace('_', ' ').title()} Comparison (α={alpha})", formats['header'])
            current_row += 1

            # 헤더
            mg_cond_headers = list(muscle_group_condition_results[0].columns)
            for col, header in enumerate(mg_cond_headers):
                ws_mg.write(current_row, col, header, formats['header'])
            current_row += 1

            # 각 근육군 결과 작성
            sig_col_idx = mg_cond_headers.index("significant") if "significant" in mg_cond_headers else None

            for mg_result in muscle_group_condition_results:
                for row in mg_result.iter_rows():
                    for col_idx, val in enumerate(row):
                        fmt = formats['significant'] if sig_col_idx == col_idx and val == "Yes" else formats['cell']
                        ws_mg.write(current_row, col_idx, val, fmt)
                    current_row += 1

        ws_mg.set_column(0, 9, 15)

    workbook.close()
    print(f"\n결과 저장 완료: {output_path}")
