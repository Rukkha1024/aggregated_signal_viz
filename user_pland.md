
## 0. 큰 방향 요약

* 내부 기준은 계속 **onset 기준 시간(ms)** 로 잡되,
* 최종 플롯의 x축은 **정규화된 축(0~1 또는 0~N-1)** 만 사용.
* `config.yaml`의 `windows`는 계속 ms 단위로 유지.
* 새로 도입하는 **time axis(ms 범위)** 가 우선권을 갖고,
  그 범위를 벗어나는 window 구간은 **잘려도 신경 쓰지 않는 설계**로 간다.
* `unit` 파라미터는 추가하지 않고, ms만 쓴다.

---

## 1. config.yaml 설계 변경 (ms 기준 time axis + 정규화 x축)

지금 `config.yaml`의 핵심 부분은 아래와 같습니다.

```yaml
interpolation:
  enabled: true
  method: "linear"
  target_length: 1000

windows:
  reference_event: "platform_onset"
  definitions:
    p1:
      start_ms: 0
      end_ms: 150
    ...
```

여기에 **x축 시간 범위(ms)** 정보를 추가합니다. `unit`은 두지 않고 ms로 고정합니다.

```yaml
interpolation:
  enabled: true
  method: "linear"
  target_length: 1000

  # 새로 추가
  start_ms: -1000    # onset 기준, 플롯에 쓸 전체 time window 시작
  end_ms: 800        # onset 기준, 플롯에 쓸 전체 time window 끝
```

* `start_ms`, `end_ms`는 **onset 기준 ms**.
* 이 구간 `[start_ms, end_ms]`만 남겨서 보간하고,
  플롯 x축을 이 구간을 0~1 (또는 0~target_length-1)로 정규화해서 사용합니다.
* 만약 둘 중 하나를 생략하면, 지금처럼 데이터의 min/max에서 자동으로 채우는 fallback도 가능하게 할 수 있습니다. (하지만 논문 figure 만들 땐 직접 명시하는 게 좋습니다.)

`windows` 섹션은 그대로 유지합니다. (p1: 0~150, p2: 150~300, …)

---

## 2. 내부 time axis 파이프라인 (정규화까지)

현재 코드는:

1. `aligned_frame` 계산 (onset = 0 frame 기준)
2. 전체 데이터의 `aligned_frame` min~max → `target_axis` (frame 단위) 생성
3. 각 trial을 `aligned_frame` 기반으로 `target_axis`에 보간

으로 되어 있습니다.

정규화 + `start_ms`/`end_ms` 반영을 위해 다음과 같이 바꿉니다.

### 2-1. ms 범위 결정

* config에서 읽은 값:

```python
start_ms = cfg["interpolation"].get("start_ms")   # 예: -1000
end_ms   = cfg["interpolation"].get("end_ms")     # 예: 800
```

* 이 값을 그대로 **canonical time axis(ms)** 로 사용합니다.
* 만약 None이면, 기존처럼 데이터 기반으로 채우되, 일관성 위해 한 번 ms로 변환 후 사용해도 됩니다.

### 2-2. frame 범위로 변환 + 데이터 crop

device sampling rate: `device_sample_rate` (지금 1000 Hz)

```python
device_rate = self.device_rate  # 1000 Hz 가정

start_frame = start_ms * device_rate / 1000.0
end_frame   = end_ms   * device_rate / 1000.0
```

* `aligned_frame`가 onset 기준 frame이므로,

```python
df = df.filter(
    (pl.col("aligned_frame") >= start_frame) &
    (pl.col("aligned_frame") <= end_frame)
)
```

으로 **이 구간만 남기고 잘라냅니다.**

### 2-3. 보간용 target_axis (frame 단위) 생성

보간 함수는 여전히 frame 단위를 쓰는 게 최소 변경이라서, target_axis는 frame 기준으로 둡니다.

```python
self.target_axis = np.linspace(start_frame, end_frame, self.target_length)
```

* `_interpolate_group`는 그대로 `x = aligned_frame`, `f(self.target_axis)`를 호출.
* 즉, y 데이터 자체는 지금과 완전히 동일한 형태의 array(길이 `target_length`)로 남습니다.

### 2-4. 플롯 x축은 “정규화 축”으로만 사용

보간 결과는 “frame 기반 time axis” 에 맞춰져 있지만,
플롯을 그릴 때는 **항상 정규화된 0~1 축**만 씁니다.

```python
x_norm = np.linspace(0.0, 1.0, self.target_length)
```

* EMG, forceplate, COP 모두 `_plot_*` 함수에서 x축을 `self.target_axis` 대신 `x_norm`으로 교체.
* `config.yaml`의 `x_label`도 나중에 `"Normalized time (0–1)"` 같은 식으로 바꾸면 됩니다. (지금은 `"Frame (normalized)"`)

---

## 3. windows(start_ms, end_ms)를 정규화 축으로 옮기는 방법 + 충돌 처리

### 3-1. canonical axis: 항상 ms

* `windows.definitions.*.start_ms`, `end_ms`는 계속 ms.
* `interpolation.start_ms`, `end_ms`도 ms.
* feature timings (`*_onset_timing`, `*_max_amp_timing` 등)도 ms로 해석.

이렇게 하면 “시간 관련 정보는 전부 ms”로 통일됩니다.

### 3-2. 정규화 변환 함수 개념

helper를 하나 둔다고 생각하면 편합니다.

```python
time_start_ms = start_ms     # interpolation.start_ms
time_end_ms   = end_ms       # interpolation.end_ms

def ms_to_norm(ms: float) -> float:
    return (ms - time_start_ms) / (time_end_ms - time_start_ms)
```

이제 window, onset, max, force onset 전부 같은 공식을 사용합니다.

### 3-3. windows ↔ normalized axis 매핑 + 잘림(clipping)

윈도우 정의:

* `cfg["windows"]["definitions"]["p1"]` → `start_ms=0`, `end_ms=150` 같은 구조.

정규화 좌표 계산:

1. 먼저 **time axis와의 충돌을 처리**합니다.
   (즉, “time axis 값이 우선, windows는 잘려도 OK”라는 규칙 반영)

```python
raw_start = window_start_ms    # 예: 0
raw_end   = window_end_ms      # 예: 150

# 1단계: time axis 범위로 클램핑
clamped_start = max(raw_start, time_start_ms)
clamped_end   = min(raw_end, time_end_ms)

# 완전히 밖에 있으면 스킵
if clamped_start >= clamped_end:
    # 이 window는 표시하지 않음
    continue
```

2. 그 다음 정규화:

```python
win_start_norm = ms_to_norm(clamped_start)
win_end_norm   = ms_to_norm(clamped_end)
```

3. 플롯에서 사용:

```python
ax.axvspan(win_start_norm, win_end_norm, alpha=..., color=...)
```

이렇게 하면:

* `p1: 0~150ms`인데 time axis가 `[-1000, 800]`이면:

  * clamped_start = 0, clamped_end = 150 → 그대로.
* time axis가 `[-50, 100]`인데 p1이 `0~150ms`라면:

  * clamped_start = 0, clamped_end = 100 → window의 뒷부분(100~150ms)은 잘려 나감.
* time axis가 `[-1000, -100]`인데 p1이 `0~150ms`이면:

  * clamped_start = 0, clamped_end = -100 → start ≥ end → 이 window는 완전히 안 보이는 상태.

=> 요구한 대로 **“충돌 나면 time axis 기준으로 자르고, 필요한 부분만 그린다”**가 정확히 구현됩니다.

---

## 4. onset / max marker도 동일한 방식으로 처리

EMG marker 수집 코드 구조를 보면, onset, max 타이밍은 ms로 읽어오게 되어 있습니다.

```python
marker_info["onset"] = onset_val          # ms
marker_info["max"]   = max_val            # ms
```

지금은 `_plot_emg`에서 이 값을 frame 기준으로 생각하고 바로 `ax.axvline(onset_time)`을 때리지만,
정규화 축으로 갈 때는 마찬가지로:

1. time axis 범위로 클램핑:

```python
if onset_val is not None:
    if time_start_ms <= onset_val <= time_end_ms:
        onset_norm = ms_to_norm(onset_val)
        ax.axvline(onset_norm, **style["onset_marker"])
```

2. 범위 밖이면 그냥 표시하지 않음.

max도 동일:

```python
if max_val is not None:
    if time_start_ms <= max_val <= time_end_ms:
        max_norm = ms_to_norm(max_val)
        ax.axvline(max_norm, **style["max_marker"])
```

COP의 max marker도 같은 규칙으로 ms→norm 변환 후 index 계산에 사용할 수 있습니다.

---

## 5. 정리: 실제 구현 시 수정 포인트 (코드 말고 “어디를 손댈지” 관점으로)

실제 코드 수정 시에는 대략 아래 네 축으로 정리할 수 있습니다.

1. **config 읽기:**

   * `interpolation.start_ms`, `interpolation.end_ms`를 읽어서,
     `self.time_start_ms`, `self.time_end_ms` 같은 필드를 만든다.

2. **time axis + interpolation:**

   * `_build_target_axis`에서 frame_min/max 대신 `start_ms`/`end_ms` → frame으로 변환하여 사용.
   * 그 frame 구간으로 `df`를 먼저 filter해서 사용.
   * `self.target_axis`는 계속 frame 단위(보간용)로 두고,
   * 플로팅 시에는 `x_norm = np.linspace(0, 1, target_length)`를 사용.

3. **windows 변환:**

   * `_compute_window_frames`는 더 이상 “frame”을 리턴하지 않고,
   * ms 기준 정의를 읽어서 → time axis 범위로 클램핑 → `ms_to_norm()`으로 변환 → normalized 구간을 리턴.
   * 플롯에서 `ax.axvspan`은 `self.window_norm_ranges`(예: `{p1: (0.55, 0.63)}`)를 사용.

4. **markers 변환:**

   * `_collect_emg_markers`, `_collect_forceplate_markers`는 지금처럼 ms를 읽고,
   * `_plot_*` 단계에서 `ms_to_norm()`으로 변환해서 `ax.axvline()`에 넘긴다.
   * time axis 밖이면 표시하지 않는다.

