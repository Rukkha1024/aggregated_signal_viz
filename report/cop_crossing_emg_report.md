# CoP(Cx\_zero, Cy\_zero) step/nonstep **역전(교차) 기반 EMG 윈도우** 분석 보고서 (young, mixed=1)

## 초록(Abstract)
본 분석은 `merged.parquet`에서 `mixed==1 & age_group==young` 조건만 사용하여, CoP(`Cx_zero`, `Cy_zero`)의 **step vs nonstep 평균 곡선이 교차(역전)하는 시점/구간**을 데이터로부터 직접 추정하고, 그 구간에서의 EMG(16채널) 차이를 **새로 계산**하였다(기존 feature/meta 파일 사용 없음). 분석 단위는 프로젝트 규칙에 따라 `subject-velocity-trial`이며, 통계 비교는 `subject-velocity` pair 수준에서 paired 방식으로 수행했다.  
결과적으로 (1) 다수의 `subject-velocity`에서 onset(0ms) 근처는 nonstep>step이지만 이후 교차가 발생했고(특히 `Cx_zero`), (2) **CoP가 nonstep>step인 구간에서도 EMG는 전반적으로 step>nonstep** 경향이 강했으며(`ST/SOL/MG/ESL/EST/RF/GM`), (3) 교차점 ±50ms(역전 주변)에서는 FDR 기준으로 유의한 EMG 차이가 나타나지 않았다. 이는 “초반 nonstep CoP 우세가 특정 근육 activation 증가로 설명될 것”이라는 가설을 지지하지 않으며, CoP 차이는 근활성 크기보다는 전략/기계적 제약(예: stepping/footlift)에 의해 더 크게 좌우될 가능성을 시사한다.

---

## 1. 연구 질문 및 가설
### 1.1 관찰(현상)
- `Cx_zero`, `Cy_zero`에서 nonstep과 step의 최대값 및 시간경과 패턴이 다르며, 초반에는 nonstep이 더 높고 이후 step이 역전(교차)하는 양상이 관찰됨.

### 1.2 연구 질문
- **nonstep이 더 높은 CoP 구간(또는 교차 주변 구간)**에서 EMG(근활성) 패턴이 step과 어떻게 다른가?

### 1.3 사전 가설
- H1: nonstep 조건에서 특정 근육의 activation이 step보다 더 높을 것이다.

---

## 2. 데이터 및 전처리(Methods)
### 2.1 데이터/필터
- 입력: `data/merged.parquet`
- 필터: `mixed == 1 & age_group == young`
- 분석 시간창: `-100~800ms` (onset=0ms 기준)
- EMG 채널(순서 유지): `config.yaml: signal_groups.emg.columns` (16채널)

### 2.2 시간 정렬(플랫폼 onset 기준)
각 `subject-velocity-trial`에 대해 `platform_onset`을 0ms로 두고 `DeviceFrame`을 정렬하여 `time_ms`를 구성했다(1000Hz → 1 frame ≈ 1ms).

### 2.3 CoP 교차(역전) 추정
축(axis) = `Cx_zero`, `Cy_zero` 각각에 대해:
1) `subject-velocity`별로 `time_ms`마다 CoP 평균 곡선을 생성: `mean(COP | step_TF, time_ms)`  
2) 차이곡선 정의: `diff(time) = nonstep - step`  
3) 교차 안정화를 위해 `diff`에 **이동평균 smoothing 11ms** 적용  
4) `diff`의 부호 기반으로 **A-구간(부호 구간)**을 정의(최소 길이 10ms):
   - `A_nonstep_gt_step`: `diff > 0` (CoP nonstep>step)
   - `A_step_gt_nonstep`: `diff < 0` (CoP step>nonstep)
5) **Primary crossing**: onset 이후(`>=0ms`) 첫 번째 0-crossing을 primary로 선택  
6) 교차 주변 **B-구간(교차 윈도우)** 정의(요청사항 반영):
   - `B_pre_cross`: `[t_cross-50, t_cross-1]` (50ms)
   - `B_post_cross`: `[t_cross, t_cross+49]` (50ms)

### 2.4 EMG 지표 산출(기존 feature 재사용 없음)
각 `axis × phase × subject-velocity-trial × step_TF`에서 해당 phase에 속하는 `time_ms` 샘플만 모아 EMG 지표를 계산:
- `mean`: 평균 활성(정규화된 0~1 스케일의 평균)
- `iemg`: `sum(EMG) * dt_ms` (dt=1ms, 단위는 a.u.*ms)
- `peak`: 최대 활성(정규화 스케일의 최대값)

이후 프로젝트 규칙의 최소 단위 준수를 위해:
- trial-level 지표 → 같은 `subject-velocity-step_TF` 내 trial 평균을 내어 **subject-velocity 단일값**으로 축약  
- 최종 비교는 `subject-velocity` pair에서 `nonstep vs step` paired로 수행

### 2.5 통계
- 각 `axis × phase × metric`에서 16개 근육별로 `diff = nonstep - step`을 0과 비교(단일표본 t-test; paired 차이의 평균이 0인지 검정)  
- 다중비교 보정: BH-FDR(`q<=0.05`)를 **axis×phase×metric 내부(근육 16개)**에서 적용  

---

## 3. 결과(Results)
분석 산출물은 `output/cop_crossing_emg/`에 저장되어 있다.

### 3.1 표본 크기 및 교차 특성
- 전체 trial 수: 188 (`subject-velocity-trial` 고유 조합 기준)
- `subject-velocity` pair 수: 24

#### Primary crossing 존재 여부
- `Cx_zero`: 22/24 pair에서 primary crossing(>=0ms) 탐지
- `Cy_zero`: 24/24 pair에서 primary crossing 탐지

#### Primary crossing 시간 분포(>=0ms, ms)
- `Cx_zero`: median 224.8ms (IQR 145.5–329.6), min 1.09ms, max 737.4ms  
- `Cy_zero`: median 121.2ms (IQR 63.5–165.4), min 8.97ms, max 568.4ms  
참고 그림: `output/cop_crossing_emg/crossing_hist_Cx_zero.png`, `output/cop_crossing_emg/crossing_hist_Cy_zero.png`

#### onset(0ms)에서 diff 부호(=nonstep-step)
`time_ms==0`에서:
- `Cx_zero`: nonstep>step 15/24, step>nonstep 9/24
- `Cy_zero`: nonstep>step 15/24, step>nonstep 9/24  
즉 “초반 nonstep 우세”는 다수 pair에서 관찰되지만, **모든 pair에 보편적인 패턴은 아님**.

---

### 3.2 A-구간(부호 구간): CoP가 nonstep>step인 구간에서도 EMG는 주로 step>nonstep
핵심은 **CoP의 부호 구간(A-phase)이 무엇이든**, EMG 차이의 방향이 대부분 `step>nonstep`(=mean_diff<0)로 나타났다는 점이다.

#### 3.2.1 `Cx_zero` 기준
**(1) A_nonstep_gt_step (CoP nonstep>step)**  
유의(q<=0.05) 근육들은 모두 `step>nonstep` 방향.
- mean: `ESL, EST, RF, GM, SOL` 등
- iemg: `ESL, RF, EST, GM, SOL` 등
- peak: `GM, EST, RF, SOL, ESL` 등

**(2) A_step_gt_nonstep (CoP step>nonstep)**  
가장 강한 효과는 `ST`, 그 다음으로 `SOL/MG` 계열.
- iemg: `ST` (dz=-1.36, q=1.37e-05), `SOL`, `MG` …
- mean: `ST` (dz=-1.26, q=4.25e-05), `MG`, `SOL` …
- peak: `ST`, `SOL`, `MG` …

참고 그림(효과크기 dz):  
- `output/cop_crossing_emg/effect_heatmap_Cx_zero_mean.png`  
- `output/cop_crossing_emg/effect_heatmap_Cx_zero_iemg.png`  
- `output/cop_crossing_emg/effect_heatmap_Cx_zero_peak.png`

#### 3.2.2 `Cy_zero` 기준
**(1) A_nonstep_gt_step (CoP nonstep>step)**  
유의(q<=0.05) 근육들은 모두 `step>nonstep` 방향.
- mean/iemg/peak에서 공통적으로 `RF/SOL/GM/MG/ST/EST/IO` 등이 반복

**(2) A_step_gt_nonstep (CoP step>nonstep)**  
`ST/SOL/MG` 중심의 step 우세가 나타남.
- 예외적으로 `VL`에서만 nonstep>step 방향의 유의 결과가 일부 관찰됨(대부분 지표/근육은 step 우세).

참고 그림:  
- `output/cop_crossing_emg/effect_heatmap_Cy_zero_mean.png`  
- `output/cop_crossing_emg/effect_heatmap_Cy_zero_iemg.png`  
- `output/cop_crossing_emg/effect_heatmap_Cy_zero_peak.png`

#### 3.2.3 요약(가장 빈번한 유의 근육)
유의 결과가 가장 자주 반복된 근육(축/구간/지표 통합 빈도 상위):
- `SOL`, `MG`, `ESL`, `ST`, `EST` (다수 조합에서 step>nonstep)

---

### 3.3 B-구간(교차 ±50ms): 유의한 EMG 차이 없음(FDR 기준)
요청사항에 따라 교차점 주변 ±50ms를 따로 비교했으나:
- `B_pre_cross`, `B_post_cross`에서 **FDR(q<=0.05) 유의 결과는 0건**이었다.
- 일부 근육에서 dz가 -0.5~-0.6 수준으로 관찰되었으나, 짧은 구간(50ms)과 개인차/변동성으로 인해 다중보정 후 유의성을 확보하지 못했다.

---

## 4. 해석(Discussion)
### 4.1 가설(H1) 평가
- “nonstep이 더 높은 CoP 구간에서 nonstep EMG가 더 높다”는 가설은 **지지되지 않았다**.
- 오히려 CoP가 nonstep>step인 A-구간에서도 EMG는 다수 근육에서 **step>nonstep**이었다.

### 4.2 의미(왜 이런 결과가 나왔을까?)
본 결과는 “CoP 크기 차이”가 단순히 “근활성 크기 증가”로 설명되지 않음을 시사한다. 가능한 해석은:
- step 전략은 자세 회복/발 내딛기 과정에서 **후반 근동원(posterior chain: SOL/MG/ST/ESL/EST/GM 등)**이 더 크게 필요  
- nonstep(특히 footlift 포함 가능)은 CoP 궤적의 기계적 변화(접촉조건 변화/하중 분배 변화)로 인해 CoP가 커질 수 있으나, EMG 크기가 반드시 증가하지는 않음  

### 4.3 교차 주변(B-구간)에서 유의성이 약한 이유(가설)
- ±50ms는 매우 짧은 창으로, iEMG/mean의 신호대잡음비가 낮고 개인차가 커질 수 있음  
- “교차 시점”은 CoP 기하학적 특성이라, 근활성 변화의 피크/지연과 정확히 동기화되지 않을 수 있음  

---

## 5. 한계(Limitations)
- `step_TF`는 실제 행동/상태(`state`)와 결합되어 있을 가능성이 높으며(예: step_L/step_R vs footlift), 이로 인해 CoP/EMG가 혼합된 전략을 반영할 수 있음  
- 교차는 `subject-velocity` 평균곡선 기반이며, trial별 교차 변동성은 반영하지 않음  
- 좌/우(step_L/step_R) 분리, 발 지배측(`주손 or 주발`)에 따른 정렬 등은 추가 분석이 필요

---

## 6. 결론(Conclusion)
1) CoP(`Cx_zero`, `Cy_zero`)가 **초반 nonstep 우세 후 역전**되는 패턴은 다수 `subject-velocity`에서 관찰되며, primary crossing의 중앙값은 `Cx_zero≈225ms`, `Cy_zero≈121ms` 수준이다.  
2) 그러나 **CoP nonstep>step 구간에서도 EMG는 전반적으로 step>nonstep**(특히 `ST/SOL/MG/ESL/EST/RF/GM`)이며, nonstep 활성 증가로 CoP 우세를 설명하기 어렵다.  
3) 교차점 ±50ms에서는 유의한 EMG 차이가 확인되지 않았다(FDR 기준).  

---

## 7. 재현 방법(Reproducibility)
분석 실행:
```bash
conda run -n module python script/analyze_cop_crossing_emg.py --config config.yaml
```

주요 산출물:
- 통계표: `output/cop_crossing_emg/emg_stats.csv`
- 교차 요약: `output/cop_crossing_emg/crossing_summary.csv`
- trial-level: `output/cop_crossing_emg/trial_metrics.parquet`
- subject-velocity-level: `output/cop_crossing_emg/sv_metrics.parquet`
