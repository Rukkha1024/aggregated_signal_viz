# %% [markdown]
# # meta_data를  data\normalized_data.csv 에 추가하기

# %% [markdown]
# ## meta_data 읽기 

# %%
import polars as pl

df = pl.read_excel(
    "/mnt/c/Users/Alice/OneDrive - 청주대학교/연구실 자료/연구실_노인균형프로토콜/데이터 정리-설문, 신체계측, 기타 데이터/perturb_inform.xlsm",
    sheet_name="meta"
)

# 첫 번째 열(subject)의 값들을 컬럼명으로 사용하여 전치
items = df["subject"].to_list()  # 이것이 새 컬럼명이 됨
subjects = df.columns[1:]         # 이것이 새 행 인덱스가 됨

transpose = (
    df
    .select(subjects)
    .transpose(include_header=True, header_name="subject", column_names=items)
)

print(transpose)

# %%
# meta_subject_row: 행 = subject, 열 = item 구조
# "나이" 열 기준으로 young/old 그룹 생성

meta = transpose.with_columns(
    pl.when(pl.col("나이").cast(pl.Int32) < 30)
      .then(pl.lit("young"))
      .otherwise(pl.lit("old"))
      .alias("age_group")
)

print(meta)
print(meta.columns)

# %% [markdown]
# ## `meta` df 내 변수를 "data\normalized_data.parquet"에 추가하기
# 
# 연령변수(young/old) 생성 및 추가
# 주손 or 주발 변수 추가

# %%
normalized = pl.read_parquet("/mnt/c/Users/Alice/OneDrive - 청주대학교/근전도 분석 코드/shared_files/output/03_post_processed/normalized_data.parquet")

meta_select = meta.select(["subject", "age_group", "주손 or 주발"])
merged = normalized.join(meta_select, on="subject", how="left")

print(merged.head(5))
print(merged.columns)

# parquet 파일로 저장
merged.write_parquet("/mnt/c/Users/Alice/OneDrive - 청주대학교/근전도 분석 코드/aggregated_signal_viz/data/merged.parquet", compression="zstd", compression_level=10)
print("✓ 저장 완료")

# %%
print(merged["subject"].unique().to_list())  # subject 고유값 전체 출력


