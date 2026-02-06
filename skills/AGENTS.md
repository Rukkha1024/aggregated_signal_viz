# Environment Issues Log

## 2026-02-06
- `conda run -n module python script/onset/vis_onset.py --config config.yaml` 실행 시 `_tkinter.TclError: not enough free memory for image buffer` 발생.
- 원인: Tk 기반 matplotlib backend가 현재 WSL/headless 실행 환경에서 메모리 부족으로 실패.
- 우회: `MPLBACKEND=Agg conda run -n module python script/onset/vis_onset.py --config config.yaml`로 정상 실행 확인.
