# check_onset (샘플)

`matplotlib` 대신 `plotly`로 EMG overlay(그리드 + window span + event vline + legend 정보)를 샘플로 재현합니다.

## 실행

```bash
cd "/mnt/c/Users/Alice/OneDrive - 청주대학교/근전도 분석 코드/aggregated_signal_viz/check_onset"

# (환경 확인)
conda run -n module python -c "import sys; print(sys.executable)"

# 샘플 실행 (HTML/PNG 생성)
conda run -n module python plotly_emg_sample.py --mode step_TF_mean
```

결과물:
- `check_onset/output/emg_plotly_sample.html`
- `check_onset/output/emg_plotly_sample.png` (kaleido 설치 시)

