# tight_layout 경고 해결 및 subplot_size 자동 조정 기능 추가

## 문제 분석

### 현재 상황
- `config.yaml`에서 EMG `grid_layout`을 `[4, 4]`에서 `[16, 1]`로 변경
- Forceplate `grid_layout`을 `[1, 3]`에서 `[3, 1]`로 변경
- 프로그램 실행 시 다음 경고 발생:
  ```
  UserWarning: Tight layout not applied. tight_layout cannot make Axes height small enough
  to accommodate all Axes decorations.
  ```

### 근본 원인
- **EMG subplot_size**: `[18, 9]` (grid_layout `[4, 4]`에 최적화)
- **변경된 grid_layout**: `[16, 1]` (4행 → 16행으로 4배 증가)
- **subplot_size 미조정**: 여전히 `[18, 9]`로 고정
- **결과**:
  1. 9인치 높이로는 16개 서브플롯을 배치하기에 공간 부족 → tight_layout 실패
  2. subplot 간격이 좁아져서 subplot title들이 서로 겹침
  3. figure suptitle과 맨 위 subplot의 title이 겹칠 가능성

### 사용자 요구사항
- grid_layout 변경 시 subplot_size를 수동으로 조정하는 불편함 해소
- **subplot당 크기 일정 유지**: subplot 1개당 크기를 설정하면 grid_layout에 따라 전체 크기 자동 계산
- subplot title 겹침 문제 해결

## 해결 방안

### 핵심 아이디어
1. **per_subplot_size 도입**: subplot 1개당 크기 설정 (인치)
2. **subplot_size 자동 계산**: `[per_subplot_size[0] × cols, per_subplot_size[1] × rows]`
3. **hspace 추가**: subplot 간 수직 간격 설정으로 title 겹침 방지
4. **tight_layout_rect 조정**: suptitle 공간 확보

### 1. 즉각적 해결 (수동 조정)
당장 경고를 해결하기 위해 config.yaml의 subplot_size를 수동으로 조정합니다.

#### config.yaml Line 107: EMG subplot_size
```yaml
# 변경 전
subplot_size: [18, 9]

# 변경 후
subplot_size: [18, 36]
```

**계산 설명:**
- grid_layout이 [4, 4] → [16, 1]로 변경 (4행 → 16행)
- subplot_size의 세로도 비례적으로 4배 증가: 9 × (16/4) = 36

### 2. 자동 조정 기능 추가 (근본적 해결)
subplot 1개당 크기를 설정하면 grid_layout에 따라 전체 크기가 자동 계산되도록 합니다.

#### 2-1. config.yaml 수정

**plot_style.emg 섹션 (Line ~107)** - per_subplot_size 추가:
```yaml
emg:
  per_subplot_size: [4.5, 2.25]  # subplot 1개당 크기 [가로, 세로] (인치)
  hspace: 0.5                     # subplot 간 수직 간격
  # subplot_size는 제거 (자동 계산됨)
  line_color: "blue"
  # ... 나머지 설정 ...
```

**계산 예시:**
- `grid_layout: [4, 4]` → `subplot_size: [18, 9]` (4.5×4, 2.25×4)
- `grid_layout: [16, 1]` → `subplot_size: [4.5, 36]` (4.5×1, 2.25×16)

**plot_style.forceplate 섹션 (Line ~124)** - per_subplot_size 추가:
```yaml
forceplate:
  per_subplot_size: [4.0, 2.0]  # subplot 1개당 크기 [가로, 세로] (인치)
  hspace: 0.4                    # subplot 간 수직 간격
  # subplot_size는 제거
  line_colors:
    # ... 나머지 설정 ...
```

**plot_style.common 섹션 (Line ~100)** - suptitle 공간 확보:
```yaml
common:
  # ... 기존 설정 ...
  tight_layout_rect: [0, 0, 1, 0.97]  # [left, bottom, right, top]
  # ... 기존 설정 ...
```

#### 2-2. visualizer.py 수정

**파일**: `/mnt/c/Users/Alice/OneDrive - 청주대학교/근전도 분석 코드/aggregated_signal_viz/visualizer.py`

**_build_emg_style 메서드 수정 (Line 97~126)**:
```python
def _build_emg_style(self, cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    defaults = {
        "per_subplot_size": [4.5, 2.25],  # subplot 1개당 크기 [가로, 세로]
        "hspace": 0.5,  # subplot 간 수직 간격
        # ... 기존 defaults ...
    }
    style = self._merge_style(defaults, cfg)

    # Auto-calculate subplot_size based on grid_layout and per_subplot_size
    grid = self.config["signal_groups"]["emg"]["grid_layout"]  # [rows, cols]
    per_size = style.get("per_subplot_size", [4.5, 2.25])

    # subplot_size = [가로, 세로] = [per_subplot_size[0] × cols, per_subplot_size[1] × rows]
    subplot_size = (per_size[0] * grid[1], per_size[1] * grid[0])
    style["subplot_size"] = subplot_size

    # ... 나머지 코드 (onset_marker, max_marker 설정 등) ...
    return style
```

**_build_forceplate_style 메서드 수정 (Line 128~149)**:
동일한 방식으로 자동 계산 로직 추가:
```python
def _build_forceplate_style(self, cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    defaults = {
        "per_subplot_size": [4.0, 2.0],  # subplot 1개당 크기 [가로, 세로]
        "hspace": 0.4,  # subplot 간 수직 간격
        # ... 기존 defaults ...
    }
    style = self._merge_style(defaults, cfg)

    # Auto-calculate subplot_size based on grid_layout and per_subplot_size
    grid = self.config["signal_groups"]["forceplate"]["grid_layout"]  # [rows, cols]
    per_size = style.get("per_subplot_size", [4.0, 2.0])

    # subplot_size = [가로, 세로] = [per_subplot_size[0] × cols, per_subplot_size[1] × rows]
    subplot_size = (per_size[0] * grid[1], per_size[1] * grid[0])
    style["subplot_size"] = subplot_size

    # ... 나머지 코드 (onset_marker 설정 등) ...
    return style
```

**_plot_emg 메서드 수정 (Line 495~573)** - hspace 적용:
```python
def _plot_emg(...) -> None:
    rows, cols = self.config["signal_groups"]["emg"]["grid_layout"]
    hspace = self.emg_style.get("hspace", 0.5)

    fig, axes = plt.subplots(
        rows, cols,
        figsize=self.emg_style["subplot_size"],
        dpi=self.common_style["dpi"],
        gridspec_kw={'hspace': hspace}  # subplot 간격 설정
    )
    # ... 기존 코드 ...

    # tight_layout 호출 (기존 Line 567)
    fig.tight_layout(rect=self.common_style["tight_layout_rect"])
```

**_plot_forceplate 메서드 수정 (Line 575~641)** - hspace 적용:
```python
def _plot_forceplate(...) -> None:
    rows, cols = self.config["signal_groups"]["forceplate"]["grid_layout"]
    hspace = self.forceplate_style.get("hspace", 0.4)

    fig, axes = plt.subplots(
        rows, cols,
        figsize=self.forceplate_style["subplot_size"],
        dpi=self.common_style["dpi"],
        gridspec_kw={'hspace': hspace}  # subplot 간격 설정
    )
    # ... 기존 코드 ...

    # tight_layout 호출 (기존 Line 635)
    fig.tight_layout(rect=self.common_style["tight_layout_rect"])
```

## 구현 순서

### 단계 1: 즉각적 해결 (수동 조정)
1. `config.yaml` Line 100: `tight_layout_rect`를 `[0, 0, 1, 0.97]`로 변경
2. `config.yaml` Line 107: `subplot_size`를 `[18, 36]`으로 변경
3. 프로그램 실행하여 경고 해결 및 title 겹침 확인

### 단계 2: 자동 조정 기능 추가
1. `config.yaml` 수정
   - EMG: `subplot_size` 제거, `per_subplot_size: [4.5, 2.25]`, `hspace: 0.5` 추가
   - Forceplate: `subplot_size` 제거, `per_subplot_size: [4.0, 2.0]`, `hspace: 0.4` 추가
2. `visualizer.py`의 `_build_emg_style` 메서드에 자동 계산 로직 추가
3. `visualizer.py`의 `_build_forceplate_style` 메서드에 자동 계산 로직 추가
4. `visualizer.py`의 `_plot_emg` 메서드에 `hspace` 적용
5. `visualizer.py`의 `_plot_forceplate` 메서드에 `hspace` 적용
6. 프로그램 실행하여 자동 계산 작동 확인

## 수정 후 검증

### 1단계 검증 (수동 조정)
```bash
conda run -n <env_name> python main.py
```
**확인 사항:**
- [ ] tight_layout 경고가 사라짐
- [ ] `output/filtered_mean/old_mean_emg.png`, `young_mean_emg.png` 생성
- [ ] subplot title들이 서로 겹치지 않음
- [ ] figure suptitle과 subplot title이 겹치지 않음
- [ ] Legend가 명확하게 표시됨

### 2단계 검증 (자동 계산)
**테스트 1**: `grid_layout: [4, 4]`, `per_subplot_size: [4.5, 2.25]`
- [ ] subplot_size가 자동으로 `(18.0, 9.0)`으로 계산됨
- [ ] hspace 0.5가 올바르게 적용됨

**테스트 2**: `grid_layout`을 `[8, 2]`로 변경
- [ ] subplot_size가 자동으로 `(9.0, 18.0)`으로 계산됨 (4.5×2, 2.25×8)
- [ ] title 겹침 없이 정상 표시

**테스트 3**: `grid_layout`을 `[16, 1]`로 변경
- [ ] subplot_size가 자동으로 `(4.5, 36.0)`으로 계산됨 (4.5×1, 2.25×16)
- [ ] hspace 적용으로 subplot title 겹침 없음

**테스트 4**: `per_subplot_size` 값 조정 테스트
- [ ] `[3.0, 2.0]` → 더 작은 subplot
- [ ] `[6.0, 3.0]` → 더 큰 subplot

**테스트 5**: hspace 값 조정 테스트
- [ ] hspace 0.3: 간격이 좁음
- [ ] hspace 0.5: 적절한 간격
- [ ] hspace 0.8: 간격이 넓음

## 핵심 파일

### config.yaml
**파일**: `/mnt/c/Users/Alice/OneDrive - 청주대학교/근전도 분석 코드/aggregated_signal_viz/config.yaml`
- Line 100: `tight_layout_rect` 수정 (1단계)
- Line 107 (1단계): EMG `subplot_size` → `[18, 36]` 수정
- Line 107~110 (2단계): EMG `subplot_size` 제거, `per_subplot_size: [4.5, 2.25]`, `hspace: 0.5` 추가
- Line 124~127 (2단계): Forceplate `per_subplot_size: [4.0, 2.0]`, `hspace: 0.4` 추가

### visualizer.py
**파일**: `/mnt/c/Users/Alice/OneDrive - 청주대학교/근전도 분석 코드/aggregated_signal_viz/visualizer.py`
- Line 97~126: `_build_emg_style` - per_subplot_size 기반 자동 계산, hspace 추가
- Line 128~149: `_build_forceplate_style` - per_subplot_size 기반 자동 계산, hspace 추가
- Line 505~507: `_plot_emg` - plt.subplots()에 gridspec_kw={'hspace': ...} 추가
- Line 584~586: `_plot_forceplate` - plt.subplots()에 gridspec_kw={'hspace': ...} 추가

## 참고 사항

### Matplotlib 레이아웃 관련
- **tight_layout()**: matplotlib의 자동 레이아웃 조정 기능
- **tight_layout_rect**: `[left, bottom, right, top]` 형식, suptitle 등을 위한 공간 확보
- **hspace**: subplot 간 수직 간격 (높이 대비 비율, 기본값 0.2)
- **gridspec_kw**: plt.subplots()에 전달하여 subplot 간격 등을 설정

### 설정 값 의미
- **per_subplot_size**: subplot 1개당 크기 (인치 단위), `[가로, 세로]`
- **subplot_size**: 전체 figure의 크기 (인치 단위), `[가로, 세로]` - **자동 계산됨**
- **grid_layout**: subplot 배치, `[rows, cols]` 형식 (행, 열)

### 자동 계산 로직
```python
grid = [rows, cols]  # signal_groups.emg.grid_layout
per_size = [width_per_subplot, height_per_subplot]  # plot_style.emg.per_subplot_size

# 자동 계산
subplot_size = (per_size[0] × grid[1], per_size[1] × grid[0])
             = (가로_per_subplot × 열_수, 세로_per_subplot × 행_수)
```

**예시:**
- `grid_layout: [4, 4]`, `per_subplot_size: [4.5, 2.25]`
  - → `subplot_size: (18.0, 9.0)` = (4.5×4, 2.25×4)
- `grid_layout: [16, 1]`, `per_subplot_size: [4.5, 2.25]`
  - → `subplot_size: (4.5, 36.0)` = (4.5×1, 2.25×16)

### hspace 권장 값
- **0.2~0.3**: 간격이 좁음 (subplot이 많을 때 title 겹칠 수 있음)
- **0.4~0.5**: 적절한 간격 (권장)
- **0.6 이상**: 간격이 넓음 (subplot 수가 적을 때)
