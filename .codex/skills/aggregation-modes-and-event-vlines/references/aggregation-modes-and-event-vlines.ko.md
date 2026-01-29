# aggregation_modes + event_vlines 규칙 스펙 (Korean)

이 문서는 “config 기반 시계열 시각화 파이프라인”에서 `aggregation_modes`(필터/그룹/오버레이/파일 분할/파일명)와 `event_vlines`(이벤트 세로선/범례/오버레이 그룹)가 어떻게 해석되어야 안전한지 규칙을 정리한다.

이 repo(`aggregated_signal_viz`)는 구체 구현 중 하나이며, **범용 규칙**과 **프로젝트 특화 규칙(구현 디테일)**을 분리해서 설명한다.

목표:
- “왜 파일이 한 장/여러 장으로 나뉘는지”를 재현 가능하게 설명한다.
- `filename_pattern` KeyError, filter 미적용, event_vlines(세로선) 미표시 같은 실전 문제를 빠르게 진단한다.

---

## 1) 범용 개념(프로젝트/코드가 달라도 유지되는 규칙)

### 1.1 처리 단위(Processing Unit)
- 집계/캐싱/파일 분할의 최소 단위가 무엇인지 먼저 고정해야 한다.
  - 예: `subject-velocity-trial`, 또는 `session-trial`, 또는 `recording-id`
- 이후 `aggregation_modes`의 설계는 “이 단위를 기반으로 어떤 평균을 낼지, 파일을 어떻게 나눌지”의 문제로 환원된다.

### 1.2 시간축(Time Axis) 정책
- 이벤트/윈도우/세로선이 올바르게 보이려면 “시간축의 도메인”이 명확해야 한다.
  - absolute(절대시간/프레임)인지
  - onset-locked(특정 이벤트를 0으로 둔 상대시간)인지
- 리샘플링을 한다면, 시각화 x축은 보통 0..1(정규화) 같은 “파생 축”이 된다.
  - 이 경우 이벤트 위치도 같은 도메인으로 변환되어야 한다(예: ms → norm).

---

## 2) aggregation_modes 스키마(범용)

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
  - 설계 원칙: “모드를 정의할 때, 필터는 데이터 정제/부분집합 선택을 담당”하고 “groupby는 그 안에서 평균 단위를 결정”한다.
- 동작/운영 팁:
  - 컬럼이 없을 때의 정책을 명확히 한다: (A) 스킵+경고, (B) 즉시 실패, (C) 무시(로그 없음)
  - 비교는 `==`인 경우가 많아 타입(예: `10` vs `10.0`) 불일치에 유의.

### 2.3 groupby (집계 키)
- `groupby: []` 또는 비어있으면, 단일 그룹 `("all",)`로 취급.
- `groupby`에 포함된 컬럼은 “집계 단위에서 상수”여야 안전하다.
  - 같은 처리 단위 내부에서 값이 변하면, 집계/표시가 애매해지고 구현에 따라 예외가 날 수 있다.

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
- base_dir 정책이 있으면(예: `output.base_dir`) 그 아래로 resolve하는 것이 안전하다.
- 레거시 경로를 허용하더라도 “중복 prefix” 같은 흔한 실수를 자동 교정하는 로직이 유용하다.

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

## 3) event_vlines 스키마(범용)

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
- 이벤트 값이 어디에서 오고(input vs features), 어떤 도메인인지(frame vs ms), 어떤 기준점인지(onset=0인지)이 명확해야 한다.
- 일반적으로 안전한 설계는 다음 2가지 중 하나다.
  1) 모든 이벤트를 “파이프라인의 선택된 시간축 도메인”으로 변환한 후 저장(예: `event_ms_from_onset`)
  2) 이벤트 원본 도메인은 그대로 두고, 시각화 직전에 변환(단, provenance/원본 포인터 유지)
- 이벤트 세로선은 보통 “표시 구간(리샘플링 윈도우)” 밖이면 그려지지 않는다.

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

## 4) aggregation_modes × event_vlines 결합 규칙(실전 핵심, 범용)

### 4.1 overlay=false(그룹별 파일)
- 각 그룹(key)마다 해당 그룹의 trial들에 대해 이벤트 평균을 계산해 세로선을 렌더링한다.

### 4.2 overlay=true(오버레이 플롯)
- 파일 단위로 “pooled_event_vlines”(그 파일에 포함된 trial 전체 평균 이벤트)를 계산해 먼저 그릴 수 있다.
- `event_vlines.overlay_group`가 활성화되어 있고, 특정 이벤트가 `overlay_group.columns`에 있으면:
  - 그 이벤트는 pooled에서 제외되고
  - 각 그룹별로 linestyle을 달리한 이벤트선이 그려진다.

### 4.3 (선택) 채널/센서별 이벤트
- EMG 같은 다채널 신호에서는 “이벤트가 채널마다 다르게 정의”될 수 있다.
- 이 경우 이벤트는 (A) 전역 평균 1개, (B) 채널별 평균 여러 개 중 어떤 방식으로 표시할지 정책이 필요하다.

---

## 5) 트러블슈팅 체크리스트(범용)

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
- 이벤트 컬럼이 input/features에 없는 경우: 경고 후 스킵(또는 실패 정책이면 즉시 실패)
- 이벤트 평균이 표시 구간(interpolation 범위) 밖인 경우: 시각화에서 제외

---

## 6) 프로젝트 특화: aggregated_signal_viz 구현 메모(이 repo)

이 섹션은 이 repo의 현재 구현(`config.yaml`, `script/visualizer.py`)을 기준으로 한 “구체 동작”이다.

### 6.1 최소 처리 단위
- 내부적으로 trial 텐서 생성 단위는 `subject-velocity-trial`.
- 이는 `data.id_columns.subject`, `velocity`, `trial`로 결정된다.

### 6.2 시간축(리샘플링)
- `interpolation.start_ms ~ interpolation.end_ms` 구간을 잘라
- `interpolation.target_length` 길이로 보간/리샘플링하여
- x축은 0..1 (`x_norm`)로 통일된다.

### 6.3 event_vlines 이벤트 도메인(입력 parquet vs features_file)
- 내부적으로 이벤트는 `__event_<event>_ms` 형태의 메타 컬럼으로 변환되어 사용된다.
  1) 입력 parquet에 이벤트 컬럼이 존재하면:
     - 그 값은 `platform_onset`과 같은 도메인(= mocap frame)으로 해석한다.
     - trial별로 onset 기준 상대 ms로 변환된다.
  2) 입력 parquet에 없고 `features_file`에만 있으면:
     - features_file의 이벤트 값은 “이미 onset 기준 ms”로 해석하여 사용한다.
     - join 후 `__event_<event>_ms`에 채워 넣는다(기존 값이 있으면 우선).
- 이벤트 평균이 `interpolation.start_ms/end_ms` 범위 밖이면 세로선이 그려지지 않는다.

### 6.4 EMG 채널별 이벤트(구현)
- features_file에 `emg_channel`이 있고 이벤트 값이 채널에 따라 달라지면,
  - 해당 이벤트는 EMG에서 채널별 이벤트로 취급될 수 있다.
- 이 경우 EMG subplot마다 이벤트선 x 위치가 달라질 수 있다.

---

## 7) 이식/확장 체크리스트(다른 프로젝트로 옮길 때)

아래 질문에 “예/아니오”로 답하면서 규칙을 옮긴다.

1) 처리 단위가 무엇인가? (예: subject-velocity-trial)
2) 시간축은 absolute인가, onset-locked인가?
3) 이벤트 값은 어떤 도메인(frame/ms)으로 들어오는가? input과 features가 섞이는가?
4) overlay를 켰을 때 파일 분할 규칙은 무엇인가? (`overlay_within` 같은 개념이 있는가?)
5) filename_pattern에 들어가는 placeholder는 정확히 무엇인가? (file_fields vs overlay_fields)
6) filter에서 컬럼 누락 시 정책은 무엇인가? (스킵/실패)
7) 채널별 이벤트/윈도우가 필요한가? 있다면 표시 정책은 무엇인가?
