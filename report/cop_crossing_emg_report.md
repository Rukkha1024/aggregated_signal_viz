# CoP(Cx\_zero, Cy\_zero) step/nonstep **역전(교차) 기반 EMG 윈도우** 분석 보고서 (young, mixed=1)

## 초록(Abstract)
본 분석은 `data/merged.parquet`에서 `mixed==1 & age_group==young` 조건만 사용하여, CoP(`Cx_zero`, `Cy_zero`)의 **step vs nonstep 평균 곡선이 교차(역전)하는 시점/구간**을 데이터로부터 직접 추정하고, 그 구간에서의 EMG(16채널) 차이를 **새로 계산**하였다(기존 feature/meta 파일 재사용 없음). CoP 교차(역전)는 두 조건의 비교가 필요하므로 `subject-velocity` 평균 CoP 곡선에서 추정하되, EMG 지표는 프로젝트 규칙에 따라 최소 단위인 `subject-velocity-trial`에서 산출하였다. 또한 **EMG가 CoP보다 지연될 수 있음**을 고려하여, 모든 근육에 공통으로 **lag(0–150ms, 10ms step)**을 스캔하며 “CoP 기준 윈도우(t)에서 EMG는 (t+lag)로 관측된다”는 가정을 적용하였다.

주요 결과는 다음과 같다. (1) 다수의 `subject-velocity`에서 onset(0ms) 초반에는 nonstep>step 경향이 있으나 이후 교차가 발생했고(특히 `Cx_zero`), 교차 시간 중앙값은 `Cx_zero≈225ms`, `Cy_zero≈121ms`였다. (2) **CoP가 nonstep>step인 A-구간에서도 EMG는 대체로 step>nonstep(음의 효과)**이었으며(예: `ST/RF/GM/MG/EST`), “초반 nonstep CoP 우세가 특정 근육 activation 증가로 설명될 것”이라는 가설(H1)은 지지되지 않았다. (3) 교차점 주변 ±50ms(B-구간)에서는 mean/iEMG는 유의 차이가 없었고, peak에서만 제한적인 lag-의존 효과가 관찰되었다: `Cx_zero`의 `B_post_cross`에서 **lag=110ms에 `EHL/PL/RF`가 step>nonstep**, `Cy_zero`의 B-구간에서는 **`VL`이 nonstep>step(peak)**이 반복적으로 관찰되었다.

---

## 1. 연구 질문 및 가설
### 1.1 관찰(현상)
- `Cx_zero`, `Cy_zero`에서 nonstep과 step의 최대값 및 시간경과 패턴이 다르며, 초반에는 nonstep이 더 높고 이후 step이 역전(교차)하는 양상이 관찰됨.

### 1.2 연구 질문
- **nonstep이 더 높은 CoP 구간(또는 교차 주변 구간)**에서 EMG(근활성) 패턴이 step과 어떻게 다른가?

### 1.3 사전 가설
- H1: nonstep condition에서 특정 근육의 activation이 step condition보다 더 높을 것이다.

---

## 2. 데이터 및 방법(Methods)
### 2.1 데이터/필터
- 입력: `data/merged.parquet`
- 필터: `mixed == 1 & age_group == young`
- 분석 기준 이벤트: `platform_onset`(0ms)
- 기본 분석 시간창: `-100~800ms`
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

### 2.4 EMG 지표 산출 + 지연(lag) 스캔
각 `axis × phase × subject-velocity-trial × step_TF`에서 해당 phase에 속하는 `time_ms=t` 샘플만 모아 EMG 지표를 계산하되, **lag 스캔**을 위해 EMG는 `t+lag`에서 샘플링하였다(모든 근육 공통 lag).
- `mean`: 평균 활성(정규화된 0~1 스케일의 평균)
- `iemg`: `sum(EMG) * dt_ms` (dt=1ms, 단위 a.u.*ms)
- `peak`: 최대 활성(정규화 스케일의 최대값)

lag 스캔 설정(요청사항 반영):
- lag 범위: `0–150ms`
- step: `10ms`

### 2.5 통계(최소 단위 준수)
- 최소 단위(`subject-velocity-trial`)에서 산출한 trial-level EMG 지표를 사용해, `y ~ 1 + x`(x=nonstep=1, step=0) 회귀로 조건 차이(=mean_nonstep - mean_step)를 추정했다.
- 반복측정(동일 `subject-velocity` 내 여러 trial)을 고려하기 위해, 표준오차는 `subject-velocity`를 cluster로 하는 **cluster-robust SE**로 계산했다(df=clusters-1).
- 다중비교 보정: BH-FDR(`q<=0.05`)를 **axis×phase×metric×lag 내부(근육 16개)**에서 적용했다.
- 효과크기: Cohen’s `d`(pooled SD). 부호는 `nonstep - step` 기준.

---

## 3. 결과(Results)
### 3.1 표본 크기 및 교차 특성
- 전체 trial 수: 188 (`subject-velocity-trial` 고유 조합 기준)
- `subject-velocity` pair 수: 24

#### Primary crossing 존재 여부
- `Cx_zero`: 22/24 pair에서 primary crossing(>=0ms) 탐지
- `Cy_zero`: 24/24 pair에서 primary crossing 탐지

#### Primary crossing 시간 분포(>=0ms, ms)
- `Cx_zero`: median 224.84ms (IQR 145.45–329.64), min 1.09ms, max 737.42ms  
- `Cy_zero`: median 121.16ms (IQR 63.52–165.39), min 8.97ms, max 568.39ms  
참고 그림: `output/cop_crossing_emg/crossing_hist_Cx_zero.png`, `output/cop_crossing_emg/crossing_hist_Cy_zero.png`

---

### 3.2 A-구간(부호 구간): CoP가 nonstep>step이어도 EMG는 대체로 step>nonstep
핵심은 **CoP 부호 구간(A-phase)이 무엇이든**, EMG 차이의 방향이 대부분 `step>nonstep`(=mean_diff<0)로 나타났다는 점이다.

#### 3.2.1 `Cx_zero` 기준(예시)
- `A_nonstep_gt_step`: `mean`에서 `EST/RF/ST`가 step>nonstep(음의 효과)로 반복 관찰.
- `A_step_gt_nonstep`: `ST`가 가장 강하고, `SOL/MG` 계열이 뒤따르는 step 우세가 관찰됨(여러 lag에서 반복).

#### 3.2.2 `Cy_zero` 기준(예시)
- `A_nonstep_gt_step`: `iemg`에서 `GM/EST/MG/ST/RF`가 step>nonstep(음의 효과)로 반복 관찰.
- `A_step_gt_nonstep`: `ST/SOL/MG` 중심의 step 우세가 관찰됨.

참고 그림(효과크기 d):  
- `output/cop_crossing_emg/effect_heatmap_Cx_zero_mean_lag0.png`  
- `output/cop_crossing_emg/effect_heatmap_Cx_zero_iemg_lag0.png`  
- `output/cop_crossing_emg/effect_heatmap_Cx_zero_peak_lag110.png`  
- `output/cop_crossing_emg/effect_heatmap_Cy_zero_mean_lag0.png`  
- `output/cop_crossing_emg/effect_heatmap_Cy_zero_iemg_lag0.png`  
- `output/cop_crossing_emg/effect_heatmap_Cy_zero_peak_lag10.png`  

---

### 3.3 B-구간(교차 ±50ms): mean/iEMG는 무유의, peak에서 제한적 lag-의존 효과
요청사항(±50ms)과 EMG 지연(lag) 가능성을 반영하여 0–150ms lag를 스캔한 결과:
- **mean, iEMG**: `B_pre_cross`, `B_post_cross`에서 모든 lag에서 유의 결과 0건(FDR 기준).
- **peak**: 일부 lag에서만 유의 결과가 나타남.

#### 3.3.1 `Cx_zero` (ML 방향)
- `B_post_cross`에서 **lag=110ms**일 때 `EHL/PL/RF`가 **step>nonstep** (q<=0.05).
  - `EHL`: d=-0.434, q=0.0277
  - `PL`: d=-0.494, q=0.0277
  - `RF`: d=-0.468, q=0.0277
- `B_pre_cross`에서는 모든 lag에서 유의 결과 0건.

#### 3.3.2 `Cy_zero` (AP 방향)
- `B_post_cross`에서 **lag=100ms**일 때 `VL`이 **nonstep>step** (d=0.375, q=0.0186).
- `B_pre_cross`에서 **lag=10ms**일 때 `ESC`가 **step>nonstep** (d=-0.229, q=0.0354).
- `B_pre_cross`에서 **lag=150ms**일 때 `VL`이 **nonstep>step** (d=0.363, q=0.0211).

---

## 4. 논의(Discussion)
### 4.1 가설(H1) 평가
- “nonstep이 더 높은 CoP 구간에서 nonstep EMG가 더 높다”는 가설은 **지지되지 않았다**.
- 오히려 CoP가 nonstep>step인 A-구간에서도 EMG는 다수 근육에서 **step>nonstep**이었다.
- lag 스캔을 통해서도 nonstep>step의 유의 결과는 `VL`(Cy, peak, B-구간)에서만 제한적으로 관찰되었다.

### 4.2 해석(가능한 기전)
본 결과는 “CoP 크기 차이”가 단순히 “근활성 크기 증가”로 설명되지 않음을 시사한다. 가능한 해석은:
- step 전략은 자세 회복/발 내딛기 과정에서 **근동원 크기(특히 peak/iemg)**가 더 크게 요구될 수 있음(예: `ST`, 일부 하지 근육에서 반복 관찰).
- nonstep(또는 footlift 포함 가능)은 접촉/하중 분배 변화 등 **기계적 요인**으로 인해 CoP 곡선이 달라질 수 있으나, EMG 크기가 반드시 증가하지는 않을 수 있음.

### 4.3 EMG 지연(lag) 관점의 시사점
- `Cx_zero`의 `B_post_cross`에서 peak 차이가 **lag≈110ms**에서만 관찰된 것은, 근활성 차이가 CoP 교차(기하학적 사건)보다 늦게 드러나거나(신경-기계 지연), 또는 교차 직후 전략 전환의 “근동원 피크”가 지연되어 나타날 수 있음을 시사한다.
- 다만 유의 근육이 `EHL/PL/RF`로 제한되고(그리고 step 우세), “초반 nonstep CoP 우세 = nonstep 근활성 증가”로 연결되지는 않았다.
- `Cy_zero`에서 `VL`이 nonstep>step으로 관찰된 점은, nonstep에서 AP 방향 제어를 위해 무릎-엉덩이 전략(예: knee extensor involvement)이 상대적으로 커질 가능성을 시사하나, 효과는 중간 크기(d≈0.36–0.38)이며 추가 검증이 필요하다.

---

## 5. 한계(Limitations)
- `step_TF`는 실제 행동/상태(`state`)와 결합되어 있을 가능성이 높으며(예: step_L/step_R vs footlift), 이로 인해 CoP/EMG가 혼합된 전략을 반영할 수 있음.
- 교차는 `subject-velocity` 평균곡선 기반이며(각 trial은 step 또는 nonstep 단일 조건), trial별 교차 변동성은 직접 추정되지 않음.
- lag는 모든 근육에 공통으로 스캔하였다(근육별 electromechanical delay 차이를 반영하지 못함).
- 좌/우(step_L/step_R) 분리, 발 지배측 정렬, 속도/개인차를 반영한 계층모형 등은 후속 분석이 필요.

---

## 6. 결론(Conclusion)
1) CoP(`Cx_zero`, `Cy_zero`)가 **초반 nonstep 우세 후 역전**되는 패턴은 다수 `subject-velocity`에서 관찰되며, primary crossing의 중앙값은 `Cx_zero≈225ms`, `Cy_zero≈121ms` 수준이다.  
2) 그러나 **CoP nonstep>step 구간에서도 EMG는 전반적으로 step>nonstep**이며, nonstep 활성 증가로 CoP 우세를 설명하기 어렵다.  
3) 교차점 ±50ms(B-구간)에서는 mean/iEMG 차이가 없었고, peak에서만 제한적인 lag-의존 효과가 관찰되었다(`Cx_zero`: lag=110ms에서 `EHL/PL/RF` step 우세; `Cy_zero`: `VL` nonstep 우세).  

---

## 7. 재현 방법(Reproducibility)
분석 실행:
```bash
conda run -n module python script/analyze_cop_crossing_emg.py --config config.yaml
```

주요 산출물:
- 통계표: `output/cop_crossing_emg/emg_stats.csv`
- lag 요약(유의 근육 수): `output/cop_crossing_emg/lag_summary.csv`
- 교차 요약: `output/cop_crossing_emg/crossing_summary.csv`
- trial-level 지표: `output/cop_crossing_emg/trial_metrics.parquet`
