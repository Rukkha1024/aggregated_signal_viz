# aggregation_modes + event_vlines 규칙 스펙 (Korean)

이 문서는 `aggregated_signal_viz`에서 `config.yaml`의 `aggregation_modes`와 `event_vlines`가 `script/visualizer.py`에서 어떻게 해석되는지(코드 기준 동작) 규칙을 정리한다.

목표:
- “왜 파일이 한 장/여러 장으로 나뉘는지”를 재현 가능하게 설명한다.
- `filename_pattern` KeyError, filter 미적용, event_vlines(세로선) 미표시 같은 실전 문제를 빠르게 진단한다.

---

## 1) 기본 전제(처리 단위)

### 1.1 최소 처리 단위
- 내부적으로 trial은 `subject-velocity-trial` 단위로 만들어지고, 그 단위에서 리샘플링 텐서가 생성된다.
- 이는 `data.id_columns.subject`, `velocity`, `trial` 조합으로 결정된다.

### 1.2 시간축(리샘플링)
- `interpolation.start_ms ~ interpolation.end_ms` 구간을 잘라
- `interpolation.target_length` 길이로 보간/리샘플링하여
- x축은 0..1 (`x_norm`)로 통일된다.

---

## 2) aggregation_modes 스키마

```yaml
aggregation_modes:
  <mode_name>:
    enabled: true|false
    filter: {col1: value1, col2: value2, ...}
    groupby: [colA, colB, ...]
    overlay: true|false
    overlay_within: [colX, colY, ...]
    color_by: [colK, ...]
    output_dir: "..."
    filename_pattern: "....png"
```

### 2.1 enabled
- 기본값: `true` 취급
- `false`면 실행 대상에서 제외됨.

### 2.2 filter (AND 필터)
- 형식: dict만 사용. 예: `filter: { mixed: 1, age_group: "young" }`
- 의미: trial 메타에서 `col == value`를 모든 조건 AND로 결합.
- 동작 특징:
  - 필터 컬럼이 없으면 경고 출력 후 해당 조건만 스킵(전체 모드 실패 아님).
  - 비교는 `==`이므로 타입(예: `10` vs `10.0`) 불일치에 유의.

### 2.3 groupby (집계 키)
- `groupby: []` 또는 비어있으면, 단일 그룹 `("all",)`로 취급.
- `groupby`에 포함된 컬럼은 trial 단위에서 상수여야 안정적이다(같은 trial 내부에서 값이 변하면 메타 상수성 체크에서 예외 가능).

### 2.4 overlay
- `overlay: false`(기본)
  - 그룹(key)마다 파일 1개 생성(“파일 단위 = groupby key”).
- `overlay: true`
  - 한 파일 안에서 여러 그룹을 라인으로 겹쳐 그릴 수 있다.
  - 이때 파일이 1개인지/여러 개인지는 `overlay_within` 유무로 결정된다(2.5 참고).

### 2.5 overlay_within (overlay일 때 파일 분할)
`overlay=true`인 경우만 의미가 있다.

#### 2.5.1 OLD 동작(레거시 유지)
- `overlay_within`이 미설정/빈 값이면:
  - 모든 그룹 키를 “한 파일”에 전부 오버레이한다.
  - 파일 key는 사실상 `("all",)`처럼 동작.

#### 2.5.2 NEW 동작(현재 지원)
- `overlay_within`이 지정되어 있으면:
  - `file_fields = groupby - overlay_within`
  - `file_fields`로 파일을 나누고,
  - 각 파일 안에서는 `overlay_within`에 해당하는 차원들이 라인(오버레이)로 표현된다.

예:
- `groupby: ["subject", "step_TF"]`
- `overlay: true`
- `overlay_within: ["step_TF"]`
→ `file_fields = ["subject"]`
→ subject별 파일 생성, 각 파일에 step/nonstep 오버레이.

권장 규칙:
- 논리적으로 `overlay_within ⊆ groupby`로 유지한다.

### 2.6 color_by (오버레이 색상 그룹 기준)
- `emg`, `cop`, `com`에서만 적용됨.
- 색상 그룹 키는 “그룹 키(groupby 결과)”에서 값을 뽑아 만든다.
  - 실전적으로 `color_by ⊆ groupby`가 안전하다.
  - `color_by`에 groupby에 없는 필드가 있으면 값이 `None`이 되어 색상 분리가 무의미해질 수 있다(에러는 아닐 수 있음).

### 2.7 output_dir (출력 폴더)
- 모드별 출력 폴더.
- `output.base_dir`가 설정되어 있으면 그 아래로 resolve됨.
- 레거시로 `output/xxx`를 직접 넣어도 중복 `output/output`가 생기지 않도록 방어 로직이 있다.

### 2.8 filename_pattern (파일명 포맷)
- `str.format()` 템플릿.
- 항상 사용 가능한 키:
  - `{signal_group}`
- overlay=false일 때:
  - 보통 `{groupby에 포함된 필드}`를 사용 가능.
- overlay=true + overlay_within 지정(NEW)일 때:
  - `file_fields`에 포함된 필드만 안전하게 포맷 가능.
  - 패턴에 없는 키를 쓰면 `KeyError`가 발생할 수 있다.
- overlay=true + overlay_within 미지정(OLD)일 때:
  - 안전하게는 `{signal_group}` 중심으로 간단한 패턴을 권장.

---

## 3) event_vlines 스키마

권장 형태(현재 config.yaml 형태):

```yaml
event_vlines:
  columns: ["platform_onset", "step_onset", "TKEO_AGLR_emg_onset_timing"]
  event_labels:
    TKEO_AGLR_emg_onset_timing: "TKEO"
  # Optional:
  # colors:
  #   platform_onset: "black"
  # Optional:
  # palette: ["C0","C1",...,"C9"]
  style:
    linestyle: "--"
    linewidth: 1.5
    alpha: 0.9
  overlay_group:
    enabled: true
    mode: "linestyle"
    columns: ["TKEO_AGLR_emg_onset_timing"]
    linestyles: ["-","--",":","-."]
```

간단형도 지원:

```yaml
event_vlines: ["platform_onset", "step_onset"]
```

### 3.1 columns
- 세로선으로 렌더링할 “이벤트 컬럼명” 리스트.
- 중복/빈 문자열은 제거된다.

### 3.2 이벤트 시간 도메인 규칙(매우 중요)
내부적으로 이벤트는 `__event_<event>_ms` 형태의 메타 컬럼으로 변환되어 사용된다.

1) 입력 parquet에 이벤트 컬럼이 존재하면:
- 그 값은 `platform_onset`과 같은 도메인(= mocap frame)으로 해석한다.
- trial별로 onset 기준 상대 ms로 변환된다.

2) 입력 parquet에 없고 `features_file`에만 있으면:
- features_file의 이벤트 값은 “이미 onset 기준 ms”로 해석하여 사용한다.
- join 후 `__event_<event>_ms`에 채워 넣는다(기존 값이 있으면 우선).

추가:
- input/features 어디에도 없으면 경고 후 스킵된다.
- 이벤트 평균이 `interpolation.start_ms/end_ms` 범위 밖이면 세로선이 그려지지 않는다.

### 3.3 event_labels
- legend에 표시할 라벨 매핑.
- 없으면 이벤트 컬럼명이 그대로 사용된다.

### 3.4 palette / colors
- 기본 색은 팔레트를 순환하며 할당.
- `colors`에 지정된 이벤트는 색이 고정되며 팔레트보다 우선한다.

### 3.5 style
- matplotlib `axvline` 기본 kwargs.
- 개별 vline 항목이 color/linestyle을 override 할 수 있다.

### 3.6 overlay_group (오버레이 그룹별 이벤트선)
오버레이 플롯에서 “특정 이벤트를 그룹별로 따로 그려” 차이를 시각화하는 옵션.

- `enabled`: on/off
- `mode`: 현재 지원은 `"linestyle"`만 (그 외는 무시)
- `columns`: 그룹별로 다시 그릴 이벤트 목록
- `linestyles`: 그룹 순서대로 순환 적용

중복 방지 동작:
- `overlay_group.columns`에 있는 이벤트는 pooled vline에서 제거되어 한 번만 나타나도록 설계되어 있다.

---

## 4) aggregation_modes × event_vlines 결합 규칙(실전 핵심)

### 4.1 overlay=false(그룹별 파일)
- 각 그룹(key)마다 해당 그룹의 trial들에 대해 이벤트 평균을 계산해 세로선을 렌더링한다.

### 4.2 overlay=true(오버레이 플롯)
- 파일 단위로 “pooled_event_vlines”(그 파일에 포함된 trial 전체 평균 이벤트)를 계산해 먼저 그릴 수 있다.
- `event_vlines.overlay_group`가 활성화되어 있고, 특정 이벤트가 `overlay_group.columns`에 있으면:
  - 그 이벤트는 pooled에서 제외되고
  - 각 그룹별로 linestyle을 달리한 이벤트선이 그려진다.

### 4.3 EMG 채널별 이벤트(가능)
- features_file에 `emg_channel`이 있고 이벤트 값이 채널에 따라 달라지면,
  - 해당 이벤트는 EMG에서 채널별 이벤트로 취급될 수 있다.
- 이 경우 EMG subplot마다 이벤트선 x 위치가 달라질 수 있다.

---

## 5) 트러블슈팅 체크리스트

### 5.1 “파일이 한 장으로 합쳐짐”
- `overlay=true`이고 `overlay_within`이 비어있으면(미설정 포함) OLD 동작으로 한 파일에 전부 오버레이됨.
- 해결: `overlay_within`을 명시하고, 파일로 나누고 싶은 필드가 `file_fields = groupby - overlay_within`에 남도록 구성.

### 5.2 “KeyError: ... (filename_pattern)”
- 원인: `filename_pattern`의 `{placeholder}`가 그 파일의 format mapping에 없음.
- 해결:
  - overlay NEW: filename에 file_fields만 사용(또는 `{signal_group}` 위주로 단순화)
  - overlay OLD: `{signal_group}` 중심 패턴 권장

### 5.3 “filter가 안 먹음”
- 컬럼 누락: 경고 후 그 조건이 스킵될 수 있음.
- 타입 불일치: `10` vs `10.0` 같은 경우 매칭 실패 가능.

### 5.4 “event_vlines가 안 보임”
- 이벤트 컬럼이 input/features에 없는 경우: 경고 후 스킵
- 이벤트 평균이 interpolation 범위 밖인 경우: 시각화에서 제외

