# `config.yaml` 옵션 가이드 (aggregated_signal_viz)

이 문서는 `aggregated_signal_viz` 프로젝트에서 `python main.py` 파이프라인을 실행할 때 사용하는 `config.yaml`의 주요 옵션을 **“어떻게 설정하면 무엇이 바뀌는지”** 관점에서 정리한 기술 문서입니다.

> 실행 환경(WSL2 기준)
> - Python 실행: `conda run -n module python ...`
> - 상대 경로 해석: `config.yaml`이 있는 폴더 기준으로 해석됩니다.

---

## 0) 빠른 시작 (가장 많이 쓰는 설정 3개)

### A. 기본 실행 (현재 `config.yaml` 사용)

```bash
conda run -n module python main.py
```

### B. 특정 config로 실행

```bash
conda run -n module python main.py --config config.yaml
```

### C. `--sample` / `--modes` / `--groups` (빠른 디버그용)

CLI 옵션 요약:

- `--sample`
  - “데이터 전체”가 아니라 **단 1개의 샘플(subject-velocity-trial 묶음)** 만 골라서 실행합니다.
  - 단, `aggregation_modes.<mode>.filter` 조건은 그대로 적용되므로,
    - 필터 조건이 너무 강하면 `[run] generated files: none`이 나올 수 있습니다.
- `--modes <mode...>`
  - 실행할 aggregation mode를 지정합니다(여러 개 가능).
  - 미지정 시: `enabled: true`인 mode 전체(그리고 일부 분석 블록)가 실행됩니다.
- `--groups <group...>`
  - 실행할 signal group을 지정합니다(예: `emg`, `forceplate` 등).

예시(EMG만, 특정 mode만, sample로 빠르게 확인):

```bash
conda run -n module python main.py --sample --modes diff_step_TF_subject --groups emg
```

### D. Plotly HTML 같이 저장하기 (PNG 옆에 `.html` 생성)

`config.yaml`에서 아래만 켜면 됩니다:

```yaml
output:
  plotly_html: true
```

### E. EMG “trial-grid by channel” Plotly HTML 추가 생성 (신규 기능)

아래를 켜면, EMG에 대해 **subject×emg_channel 단위**의 “trial grid” HTML을 추가로 생성합니다.

```yaml
output:
  emg_trial_grid_by_channel: true
```

중요: 이 옵션은 **trial-level mode(= `groupby`가 subject/velocity/trial을 포함)** 에서만 허용됩니다. (미포함이면 실행이 중단됩니다.)

---

## 1) 경로/파일 해석 규칙 (제일 자주 헷갈리는 부분)

### 1.1 상대 경로는 “config 파일 위치 기준”

예를 들어 `config.yaml`에 아래가 있으면,

```yaml
data:
  input_file: "data/merged.parquet"
```

실제 경로는 **`config.yaml`이 있는 폴더** 기준의 `data/merged.parquet` 입니다.

- 따라서 `config.yaml`을 `/tmp/config.yaml` 같은 곳으로 복사하면,
  - `data/merged.parquet`가 `/tmp/data/merged.parquet`로 해석되어 실패할 수 있습니다.

---

## 2) `data:` (입력/메타/샘플링 설정)

### 2.1 입력 파일

```yaml
data:
  input_file: "data/merged.parquet"
  features_file: "/abs/path/to/meta_merged.csv"
```

- `input_file`
  - 주 입력 데이터(merged parquet)
  - `signal_groups.*.columns`에 적힌 신호 컬럼들이 이 parquet에 있어야 합니다.
- `features_file`
  - 이벤트(특히 EMG 채널별 feature) timing을 제공하는 외부 CSV
  - 예: `TKEO_AGLR_emg_onset_timing` 같은 feature-based 이벤트가 여기서 들어옵니다.

### 2.2 컬럼명 매핑(`id_columns`)

```yaml
data:
  id_columns:
    subject: "subject"
    velocity: "velocity"
    trial: "trial_num"
    frame: "DeviceFrame"
    mocap_frame: "MocapFrame"
    onset: "platform_onset"
    offset: "platform_offset"
    task: "task"
```

이 매핑은 “데이터셋에서 어떤 컬럼이 subject/velocity/trial인지”를 알려줍니다.  
데이터 컬럼명이 바뀌었으면 여기만 수정하는 것이 가장 안전합니다(코드 수정 최소화).

### 2.3 task 필터

```yaml
data:
  task_filter: "perturb"
```

- `task_filter`가 설정되어 있으면, `id_columns.task` 컬럼이 `perturb` 인 데이터만 사용합니다.

### 2.4 샘플링 레이트

```yaml
data:
  device_sample_rate: 1000
  mocap_sample_rate: 100
```

- Mocap↔Device 프레임 변환 비율(100 Hz ↔ 1000 Hz 등)에 사용됩니다.
- 프레임/도메인 혼선을 방지하려면 여기를 “단일 소스”로 두고, 코드에 하드코딩하지 않습니다.

---

## 3) `output:` (출력 폴더 + Plotly HTML 옵션)

```yaml
output:
  base_dir: "output"
  plotly_html: false
  emg_trial_grid_by_channel: false
```

### 3.1 `base_dir`

- 모든 결과물은 기본적으로 `base_dir` 아래에 저장됩니다.
- 모드별로 `aggregation_modes.<mode>.output_dir` 하위 폴더가 생성됩니다.

### 3.2 `plotly_html`

- `true`이면, 기존 PNG 저장과 별개로 **동일한 stem의 `.html`** 을 함께 저장합니다.
  - 예: `.../step_TF_정혜진_v20.0_2_emg.png` 생성 시 `.../step_TF_정혜진_v20.0_2_emg.html`도 생성

### 3.3 `emg_trial_grid_by_channel` (신규)

- `true`이면 EMG에 대해 다음 HTML을 추가 생성합니다:
  - 위치: `<mode output_dir>/trial_grid_by_channel/`
  - 파일: `diff_step_TF_subject_<subject>_<emg_channel>_trial_grid_emg.html`
  - 단위: **subject × emg_channel**
  - 내용: 한 채널(예: TA)만 고정하고, subplot을 “trial”로 배치

#### (중요) Trial-level 모드만 허용

이 옵션은 “trial-grid”의 정의상 **trial 단위가 보장되어야** 하므로, 아래 컬럼들이 `groupby`에 포함되어야 합니다.

- `data.id_columns.subject`
- `data.id_columns.velocity`
- `data.id_columns.trial`

미포함 시 실행이 중단됩니다(안전장치).

---

## 4) `signal_groups:` (어떤 신호를 그릴지)

예:

```yaml
signal_groups:
  emg:
    columns: [TA, EHL, ...]
    grid_layout: [4, 4]
  forceplate:
    columns: [Fx_zero, Fy_zero, ...]
    grid_layout: [2, 3]
  cop:
    columns: [Cx_zero, Cy_zero]
    grid_layout: [1, 3]
  com:
    columns: [COMx_zero, COMy_zero, COMz_zero]
    grid_layout: [1, 4]
    time_base: mocap
    optional: true
```

- `columns`: 데이터에서 뽑아올 컬럼 리스트
- `grid_layout`: subplot grid (rows, cols)
- `com.optional: true`: com 데이터가 없으면 스킵(실패하지 않음)
- `com.time_base: mocap`: com은 mocap time base 사용(필요 시)

---

## 5) `interpolation:` (시간 구간/리샘플링)

```yaml
interpolation:
  target_length: 500
  start_ms: -100
  end_ms: 500
```

- `target_length`
  - 각 trial을 동일 길이로 리샘플링하는 포인트 수
- `start_ms`, `end_ms`
  - 분석/시각화에 사용하는 시간 구간
  - 이벤트(vline)도 이 범위 밖이면 자동으로 표시되지 않습니다(“왜 vline이 안 보이지?”의 주요 원인).

---

## 6) `aggregation_modes:` (필터 → 그룹 → overlay → 파일 저장 규칙)

`aggregation_modes`는 “어떤 단위로 묶어서(집계/평균/overlay) 어떤 파일명으로 저장할지”를 정의합니다.

### 6.1 기본 구조

```yaml
aggregation_modes:
  diff_step_TF_subject:
    enabled: true
    filter:
      mixed: 1
      age_group: "young"
    groupby: ["step_TF", "subject", "velocity", "trial_num"]
    color_by: ["step_TF"]
    overlay: true
    overlay_within: ["step_TF"]
    output_dir: "step_TF_subject"
    filename_pattern: "step_TF_{subject}_v{velocity}_{trial_num}_{signal_group}.png"
```

### 6.2 `filter`

- dict의 모든 조건을 AND로 적용합니다.
- 흔한 실수: 타입 mismatch
  - 예: `velocity`가 float 컬럼인데 YAML에서 `60`(int)로 적으면 의도와 다르게 필터될 수 있습니다.
  - 의심되면 `60.0`처럼 데이터 타입과 맞추는 것이 안전합니다.

### 6.3 `groupby`

- “그림/파일 생성의 기본 단위”를 정의합니다.
- 이 프로젝트의 핵심 단위는 `subject-velocity-trial` 입니다.
- `output.emg_trial_grid_by_channel: true`를 켰다면 `groupby`에
  - subject/velocity/trial이 반드시 포함되어야 합니다(안전장치로 강제).

### 6.4 `overlay` + `overlay_within` (중요)

- `overlay: true`면 여러 그룹을 한 파일에 겹쳐 그릴 수 있습니다.
  - 어떤 기준으로 “파일을 나누고”, 어떤 기준으로 “한 파일 안에서 overlay”할지는 `overlay_within`에 달려 있습니다.

개념적으로:
- `overlay_within`에 있는 필드는 **한 파일 안에서 변화(overlay)** 하고
- 그 외 `groupby - overlay_within`는 **파일을 나누는 기준(file_fields)** 이 됩니다.

### 6.5 `filename_pattern`

- `str.format()` 기반 패턴입니다. 예:
  - `"step_TF_{subject}_v{velocity}_{trial_num}_{signal_group}.png"`
- overlay를 쓰는 모드에서는 “파일을 나누는 기준”에 없는 placeholder를 쓰면 KeyError가 날 수 있습니다.
- 가장 안전한 접근:
  - overlay 사용 시에도 파일로 분리되는 field(= `groupby - overlay_within`)에 있는 키만 filename에 사용

---

## 7) `windows:` (기간(span) 정의)

```yaml
windows:
  reference_event: "platform_onset"
  definitions:
    p1:
      start_ms: "TKEO_AGLR_emg_onset_timing"
      end_ms: "TKEO_AGLR_emg_onset_timing + 125"
    p2:
      start_ms: "TKEO_AGLR_emg_onset_timing + 125"
      end_ms: "step_onset"
    p3:
      start_ms: "step_onset"
      end_ms: "600"
```

### 7.1 경계 값 형식

`start_ms` / `end_ms`는 아래 형식을 지원합니다.

- 숫자(또는 숫자 문자열): ms offset
- 문자열(이벤트 컬럼명): 이벤트 시점(ms)
- 문자열(표현식): `<event> +/- <ms>` (예: `"TKEO_AGLR_emg_onset_timing + 30"`)

### 7.2 `reference_event`

- 숫자형 경계(start/end)가 “어떤 0점을 기준으로 해석될지”를 정의합니다.
- 보통 `platform_onset`(onset=0) 기준으로 쓰면 가장 덜 헷갈립니다.

---

## 8) `x_axis_zeroing:` (x축 0점 재정렬)

```yaml
x_axis_zeroing:
  enabled: true
  reference_event: "TKEO_AGLR_emg_onset_timing"
```

- `enabled: true`이면, 각 플롯/파일에서 포함된 trial들의 `reference_event` 평균을 계산해 x축을 0으로 맞춥니다.
- EMG 채널별 이벤트(예: `TKEO_AGLR_emg_onset_timing`)는 채널별로 값이 달라질 수 있으므로,
  - trial-grid 출력에서는 채널별 평균을 기준으로 정렬되는 동작이 포함됩니다.

---

## 9) `plot_style:` (표시 on/off + 일부 스타일 파라미터)

> NOTE: 이 블록은 이미 repo에 존재하는 설정을 문서화한 것입니다.  
> 새로운 “순수 디자인(색/폰트 등)” 파라미터를 무분별하게 늘리기보다는, 공용 로직으로 관리하는 것이 유지보수에 유리합니다.

### 9.1 `plot_style.common`

- `show_*`: 요소 표시 여부
  - `show_event_vlines`, `show_windows`, `show_legend` 등
- `grid_alpha`: grid 투명도 (Plotly도 동일 개념으로 사용)
- `use_group_colors`
  - `false`: overlay에서 색 대신 라인스타일로 구분(기본은 “회색 + dash 변화”)
  - `true`: overlay 그룹별 색상을 사용
- `window_colors`: windows 기간(p1,p2,...) 배경색
- `group_linestyles`: overlay 그룹에 할당되는 라인스타일 목록

### 9.2 `plot_style.emg/forceplate/cop/com`

- EMG: `line_color`, `line_alpha`, `window_span_alpha` 등
- Forceplate: 채널별 `line_colors`(Fx/Fy/Fz 등) 지원
- COP/COM: scatter/배경/축 라벨 등

---

## 10) `event_vlines:` (이벤트 수직선)

```yaml
event_vlines:
  columns: ["platform_onset", "step_onset", "TKEO_AGLR_emg_onset_timing", "platform_offset"]
  event_labels:
    TKEO_AGLR_emg_onset_timing: "onset"
  style:
    linestyle: "--"
    linewidth: 1.5
    alpha: 0.9
  overlay_group:
    enabled: true
    mode: "linestyle"
    columns: ["TKEO_AGLR_emg_onset_timing"]
    linestyles: ["-", "--", ":", "-."]
```

### 10.1 `columns`

- 표시할 이벤트 컬럼 목록입니다.
- 이벤트 값은 다음 중 하나에서 옵니다:
  1) `data.input_file`에 컬럼이 있으면 그 값을 사용
  2) input에 없고 `data.features_file`에만 있으면 features 값을 사용

### 10.2 `event_labels`

- legend에 표시될 이름을 짧게 바꿀 때 사용합니다.

### 10.3 `palette` / `colors` (선택)

- 기본 팔레트는 matplotlib의 `C0..C9` 스타일 토큰을 사용할 수 있습니다.
- 특정 이벤트만 색을 고정하려면 `colors`로 override 합니다.

> 참고: legend는 이제 `C0` 같은 토큰도 정상적으로 컬러 표시됩니다(HTML 색상 정규화 적용).

### 10.4 `overlay_group`

- overlay plot(예: step vs nonstep)에서 특정 이벤트를 “그룹별로 한 번씩” 그리게 하는 옵션입니다.
- `enabled: false`로 끄면 그룹별 이벤트 표시/legend 중복을 줄일 수 있습니다.

---

## 11) `cop_crossing_emg_analysis:` (CoP crossing 기반 EMG 분석)

```yaml
cop_crossing_emg_analysis:
  enabled: true
  filter:
    mixed: 1
    age_group: "young"
  cop_columns: ["Cx_zero", "Cy_zero"]
  time_window_ms:
    start_ms: null
    end_ms: null
  crossing:
    smoothing_ms: 11
    diff_epsilon: 0.0
    min_segment_ms: 10
    primary_crossing_policy: "first_after_onset"
  crossing_window_ms: 50
  emg_lag_scan_ms:
    start_ms: 0
    end_ms: 150
    step_ms: 10
  metrics: ["mean", "iemg", "peak"]
  output_subdir: "cop_crossing_emg"
  report_path: "report/cop_crossing_emg_report.md"
```

- `enabled: true`면 CoP step/nonstep crossing을 계산하고, 그 기반으로 EMG 윈도우 분석을 수행합니다.
- `time_window_ms.start_ms/end_ms`가 `null`이면 `interpolation.start_ms/end_ms`를 fallback으로 사용합니다.

---

## 12) 트러블슈팅 체크리스트

### 12.1 “Plotly HTML이 안 생겨요”

- `output.plotly_html: true`인지 확인
- 실행 커맨드가 `conda run -n module python main.py ...` 인지 확인

### 12.2 “EMG trial-grid HTML이 안 생겨요”

- `output.emg_trial_grid_by_channel: true`인지 확인
- 선택한 mode가 trial-level인지 확인:
  - `aggregation_modes.<mode>.groupby`에 subject/velocity/trial 포함 필수

### 12.3 “필터가 먹지 않는 것 같아요”

- 컬럼이 실제 데이터에 존재하는지 확인(없으면 조건이 무시되거나 결과가 0이 될 수 있음)
- 타입 mismatch 주의:
  - `velocity: 60.0` vs `velocity: 60`

### 12.4 “이벤트 vline이 안 보여요”

- 이벤트 timing이 `interpolation.start_ms~end_ms` 범위 밖이면 표시되지 않습니다.
- `event_vlines.columns`가 실제로 존재/생성되는 이벤트인지 확인

### 12.5 “overlay에서 filename_pattern KeyError”

- overlay 사용 시 filename에 넣은 placeholder가 “파일 분리 기준(field)”에 없는 경우가 흔한 원인입니다.
- `overlay_within`과 `filename_pattern`을 같이 점검하세요.

---

## 13) 참고 구현(코드 위치)

- 파이프라인 엔트리: `main.py`
- 핵심 구현(aggregation/overlay/windows/vlines): `src/core/visualizer.py`
- Plotly HTML export: `src/plotting/plotly/html_export.py`
- EMG trial-grid Plotly export: `src/plotting/plotly/emg_trial_grid_by_channel.py`
- 레거시 Plotly 참고(디자인/규칙): `scripts/plotting/plotly/plotly_emg_sample.py`
