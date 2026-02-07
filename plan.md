## 수정 계획

`plotly_emg_sample.py`를 수정하여 `interpolation.start_ms` / `end_ms`가 null일 때 데이터의 실제 범위를 자동 계산하도록 변경하겠습니다.

### 변경 내용:

1. **null 체크 로직 추가** (line 680-683 부근)
   - `start_ms`가 null이면 → 데이터에서 onset 대비 최소 시간을 계산
   - `end_ms`가 null이면 → 데이터에서 onset 대비 최대 시간을 계산

2. **구현 방식**:
   - `interp_cfg.get("start_ms")`가 `None`인지 체크
   - `None`이면 LazyFrame에서 `(original_DeviceFrame - onset_device)`의 min/max를 계산하여 ms로 변환
   - 그렇지 않으면 config 값 사용
