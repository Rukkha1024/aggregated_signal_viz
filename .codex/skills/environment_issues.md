# Environment Issues Log

## 2026-02-07
- Context: Running `script/onset/vis_onset.py` in WSL2 with the default Matplotlib backend.
- Symptom: `_tkinter.TclError: not enough free memory for image buffer`.
- Workaround: run with a non-interactive backend.
  - `MPLBACKEND=Agg conda run -n module python script/onset/vis_onset.py`
