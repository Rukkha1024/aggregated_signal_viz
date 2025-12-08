# Aggregated Signal Visualization - Standalone Repository Plan

## 목적
시계열 시그널 데이터의 집계(aggregated) 플롯을 생성하는 **독립 레포지토리**.

---

## 핵심 제약사항
- **완전 독립 레포지토리**: 외부 모듈 의존 없음
- **외부 의존성**: polars, numpy, matplotlib, scipy만 사용
- **Config 기반**: 모든 설정을 `config.yaml`에서 관리

---

## Legacy Code 분석 - Plot Type별 적용 요소

| 요소 | EMG | Forceplate | COP |
|------|-----|------------|-----|
| **Plot Type** | Line (`'b-'`) | Line (채널별 색상) | Scatter |
| **Window Span** | O (`axvspan`) | O (`axvspan`) | X |
| **Window Color (scatter)** | X | X | O (점 색상) |
| **Onset Marker** | O (`axvline`) | O (`axvline`) | X |
| **Max Marker** | O (`axvline`) | X | O (`scatter '*'`) |
| **Y축 반전** | X | X | O (Cy → -Cy) |

### 공통 스타일
```
- DPI: 300
- Grid: True, alpha=0.3
- Tick labelsize: 7
- Title fontsize: 30, fontweight='bold', pad=5
- Label fontsize: 8
- Legend: loc='best', framealpha=0.8
- tight_layout: rect=[0, 0, 1, 0.99]
- savefig: bbox_inches='tight', facecolor='white'
```

### EMG 전용
```
- Subplot size: 12x6
- Line: 'b-', linewidth=0.8, alpha=0.8
- Window span: axvspan, alpha=0.15
- Onset marker: axvline, color='red', linestyle='--', linewidth=1.5
- Max amp marker: axvline, color='orange', linestyle='--', linewidth=1.5
- Legend fontsize: 6
- X label: 'Frame (onset=0)'
- Y label: {channel}
```

### Forceplate 전용
```
- Subplot size: 12x6
- Line colors: Fx='purple', Fy='brown', Fz='green'
- Line: linewidth=0.8, alpha=0.8
- Window span: axvspan, alpha=0.15
- Onset marker: axvline, color='red', linestyle='--', linewidth=1.5
- Legend fontsize: 6
- X label: 'Frame (onset=0)'
- Y label: '{channel} Value'
```

### COP 전용
```
- Subplot size: 8x8 (정사각형)
- Scatter: s=8, alpha=0.7
- Window별 점 색상: p1='#4E79A7', p2='#F28E2B', p3='#E15759', p4='#59A14F'
- Background 점: color='lightgray', alpha=0.3, s=6
- Max marker: s=80, marker='*', color='#ED1C24', edgecolor='white', linewidth=1, zorder=10
- Legend fontsize: 5
- X label: 'Cx (R+/L-)'
- Y label: 'Cy (A+)'
- Y축 반전: y = -Cy
```

---

## 디렉토리 구조
```
aggregated_signal_viz/
├── config.yaml
├── main.py
├── visualizer.py
├── data/
│   └── (input.csv)
├── output/
│   ├── subject_mean/
│   ├── grand_mean/
│   └── filtered_mean/
└── requirements.txt
```

---

## config.yaml 설계

```yaml
# === 데이터 설정 ===
data:
  input_file: "data/processed_emg_data.csv"
  id_columns:
    subject: "subject"
    group_var: "velocity"
    trial: "trial_num"
    time_axis: "DeviceFrame"

# === 시그널 그룹 정의 ===
signal_groups:
  emg:
    columns: [TA, EHL, MG, SOL, PL, RF, VL, ST, RA, EO, IO, SCM, GM, ESC, EST, ESL]
    grid_layout: [4, 4]
  forceplate:
    columns: [Fx, Fy, Fz]
    grid_layout: [1, 3]
  cop:
    columns: [Cx, Cy]

# === 보간 설정 ===
interpolation:
  enabled: true
  method: "linear"
  target_length: 1000

# === 집계 모드 ===
aggregation_modes:
  subject_mean:
    enabled: true
    groupby: [subject]
    output_dir: "output/subject_mean"
    filename_pattern: "{subject}_mean_{signal_group}.png"
  grand_mean:
    enabled: true
    groupby: []
    output_dir: "output/grand_mean"
    filename_pattern: "grand_mean_{signal_group}.png"
  filtered_mean:
    enabled: true
    filter:
      column: "velocity"
      value: 10.0
    groupby: [subject]
    output_dir: "output/filtered_mean"
    filename_pattern: "{subject}_vel10_mean_{signal_group}.png"

# === 플롯 스타일 ===
plot_style:
  # 공통
  common:
    dpi: 300
    grid_alpha: 0.3
    tick_labelsize: 7
    title_fontsize: 20
    title_fontweight: "bold"
    title_pad: 5
    label_fontsize: 8
    legend_loc: "best"
    legend_framealpha: 0.8
    tight_layout_rect: [0, 0, 1, 0.99]
    savefig_bbox_inches: "tight"
    savefig_facecolor: "white"
    font_family: "Malgun Gothic"
  
  # EMG 전용
  emg:
    subplot_size: [12, 6]
    line_color: "blue"
    line_width: 0.8
    line_alpha: 0.8
    window_span_alpha: 0.15
    onset_marker_color: "red"
    onset_marker_linestyle: "--"
    onset_marker_linewidth: 1.5
    max_marker_color: "orange"
    max_marker_linestyle: "--"
    max_marker_linewidth: 1.5
    legend_fontsize: 6
    x_label: "Frame (normalized)"
    y_label: "{channel}"
  
  # Forceplate 전용
  forceplate:
    subplot_size: [12, 6]
    line_colors:
      Fx: "purple"
      Fy: "brown"
      Fz: "green"
    line_width: 0.8
    line_alpha: 0.8
    window_span_alpha: 0.15
    onset_marker_color: "red"
    onset_marker_linestyle: "--"
    onset_marker_linewidth: 1.5
    legend_fontsize: 6
    x_label: "Frame (normalized)"
    y_label: "{channel} Value"
  
  # COP 전용
  cop:
    subplot_size: [8, 8]
    scatter_size: 8
    scatter_alpha: 0.7
    background_color: "lightgray"
    background_alpha: 0.3
    background_size: 6
    window_colors:
      p1: "#4E79A7"
      p2: "#F28E2B"
      p3: "#E15759"
      p4: "#59A14F"
    max_marker_color: "#ED1C24"
    max_marker_size: 80
    max_marker_symbol: "*"
    max_marker_edgecolor: "white"
    max_marker_linewidth: 1
    max_marker_zorder: 10
    legend_fontsize: 5
    x_label: "Cx (R+/L-)"
    y_label: "Cy (A+)"
    y_invert: true

# === 출력 설정 ===
output:
  base_dir: "output"
  format: "png"
```

---

## 생성할 시각화 (3가지)

### 1. Subject-Level Mean
- **Groupby**: subject + DeviceFrame (보간 후)
- **출력**: `{subject}_mean_emg.png`, `{subject}_mean_forceplate.png`, `{subject}_mean_cop.png`

### 2. Grand Mean
- **Groupby**: DeviceFrame (보간 후)
- **출력**: `grand_mean_emg.png`, `grand_mean_forceplate.png`, `grand_mean_cop.png`

### 3. Velocity 10 + Subject별
- **Filter**: velocity == 10.0
- **Groupby**: subject + DeviceFrame (보간 후)
- **출력**: `{subject}_vel10_mean_emg.png`, `{subject}_vel10_mean_forceplate.png`, `{subject}_vel10_mean_cop.png`

---

## 사용 컬럼
```python
ID_COLS = ['subject', 'velocity', 'trial_num', 'DeviceFrame']
EMG_COLS = ['TA', 'EHL', 'MG', 'SOL', 'PL', 'RF', 'VL', 'ST', 'RA', 'EO', 'IO', 'SCM', 'GM', 'ESC', 'EST', 'ESL']
FP_COLS = ['Fx', 'Fy', 'Fz']
COP_COLS = ['Cx', 'Cy']
```

---

## 다음 단계
`proceed` 시 코드 구현
