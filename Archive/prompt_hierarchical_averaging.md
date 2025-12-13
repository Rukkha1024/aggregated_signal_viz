**목표:** `aggregated_signal_viz/visualizer.py`의 시각화 품질 개선을 위해 집계 방식(Aggregation Logic)을 변경합니다.

**배경:**
현재는 모든 트라이얼을 한 번에 평균(Pooled Mean) 내고 있어, 피험자 간 타이밍 편차(Jitter)로 인해 반응이 빠른 근육(TA 등)의 Onset이 실제보다 느리고 완만하게 보이는 왜곡이 발생합니다. 이를 해결하기 위해 **계층적 평균(Hierarchical Averaging)**을 도입하고자 합니다.

**지시 사항:**

`aggregated_signal_viz/visualizer.py` 파일의 `AggregatedSignalVisualizer` 클래스 내 `_aggregate_group` 메서드를 다음 로직으로 전면 수정해주세요.

1.  **동적 집계 방식 결정:**
    *   메서드에 입력된 `records` 리스트를 검사하여 포함된 고유 피험자(`subject`)의 수를 확인합니다.
    
2.  **Case 1: 다중 피험자인 경우 (Unique Subjects > 1)**
    *   `grand_mean`, `age_group_mean` 등이 여기에 해당합니다.
    *   **Step 1:** `records`를 `subject`별로 그룹화합니다.
    *   **Step 2:** 각 피험자 그룹에 대해 채널별 평균(Subject Mean)을 먼저 계산합니다. (`np.nanmean` 사용)
    *   **Step 3:** 계산된 '피험자별 평균 파형'들을 다시 모아서, 최종 평균(Mean of Subject Means)을 계산합니다.
    *   이렇게 함으로써 트라이얼 수가 많은 피험자 쪽으로 결과가 편향되는 것을 막고, 신호의 대표성을 높입니다.

3.  **Case 2: 단일 피험자인 경우 (Unique Subjects == 1)**
    *   `subject_mean` 혹은 필터링 결과 피험자가 한 명뿐인 경우가 해당합니다.
    *   기존과 동일하게 모든 `records`를 `vstack` 하여 단순 평균(Simple Mean)을 계산합니다.

4.  **코드 구조 제안:**
    *   `_aggregate_group` 메서드 내에서 분기 처리를 하거나, 실제 평균 계산을 수행하는 헬퍼 메서드(예: `_compute_mean_of_signals`)를 만들어 재사용성을 높이세요.

**수정 대상 파일:**
*   `aggregated_signal_viz/visualizer.py`