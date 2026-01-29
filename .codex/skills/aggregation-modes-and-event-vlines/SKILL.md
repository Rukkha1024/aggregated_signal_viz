---
name: aggregation-modes-and-event-vlines
description: aggregated_signal_viz에서 config.yaml의 aggregation_modes 및 event_vlines 규칙(overlay/overlay_within, filename_pattern, filter, event_vlines.overlay_group 등)을 정확히 해석하고 안전한 설정안을 제안한다. config.yaml을 수정하거나 새 aggregation mode/event vline을 추가할 때, 그림이 왜 “한 파일로 합쳐지거나” “여러 파일로 쪼개지는지” 원인을 찾을 때, KeyError(파일명 포맷), 필터 미적용(타입 불일치/컬럼 누락), 이벤트 세로선/범례(overlay_group) 동작을 점검할 때 사용한다.
---

# Aggregation Modes And Event Vlines

## Overview

이 skill은 `aggregated_signal_viz`에서 `config.yaml`의 `aggregation_modes`(집계/오버레이/출력)와 `event_vlines`(이벤트 세로선/오버레이 그룹) 설정이 실제 코드(`script/visualizer.py`)에서 어떻게 해석되는지 규칙을 정리하고, 안전한 설정 패턴/디버깅 체크리스트를 제공한다.

## Quick Start (작업 절차)

1) `config.yaml`에서 `aggregation_modes` / `event_vlines` 블록을 확인한다.
2) 코드 기준 동작을 확정해야 하면 `script/visualizer.py`에서 아래 키워드를 찾아 실제 분기(OLD/NEW)를 확인한다.
   - `overlay_within`, `filename_pattern`, `_apply_filter_indices`, `_collect_event_vlines`, `overlay_group`
3) 설정을 바꾸기 전, 아래 “안전 규칙”에 맞게 `groupby`/`overlay_within`/`filename_pattern` 정합성을 먼저 맞춘다.
4) 상세 규칙/예시는 `references/aggregation-modes-and-event-vlines.ko.md`를 연다.

## 핵심 안전 규칙(요약)

- `aggregation_modes.<mode>.filter`는 dict만 사용하고, 조건은 AND로 결합된다. (컬럼이 없으면 경고 후 해당 조건만 스킵)
- `overlay=true`일 때 파일 분할은 `overlay_within` 유무로 갈린다.
  - `overlay_within`이 비어있으면(미설정 포함) **OLD 동작**: 모든 그룹을 한 파일에 오버레이
  - `overlay_within`이 있으면 **NEW 동작**: `file_fields = groupby - overlay_within` 기준으로 파일을 쪼개고, 각 파일 안에서 오버레이
- `filename_pattern`은 `str.format()`이므로, 오버레이 NEW에서는 **`file_fields`에 없는 키를 쓰면 KeyError**가 날 수 있다.
- `color_by`는 “그룹 키(groupby 결과)”에서 값을 뽑아 색상 그룹을 만든다. 보통 `color_by ⊆ groupby`가 안전하다.
- `event_vlines.overlay_group`는 오버레이 플롯에서 특정 이벤트를 “그룹별 linestyle”로 다시 그리기 위한 옵션이며, `overlay_group.columns`에 지정된 이벤트는 pooled vline에서 제외되어 중복을 피한다.

## 자주 나는 문제(증상 → 원인 → 해결)

- “원래 여러 파일로 나뉘어야 하는데 한 파일로만 나온다”
  - 원인: `overlay=true` + `overlay_within` 미설정(OLD 동작) → 전부 한 파일로 합쳐짐
  - 해결: `overlay_within`을 명시하고, 원하는 파일 분할 기준이 `file_fields = groupby - overlay_within`에 남도록 조정
- “png 생성 중 KeyError 발생”
  - 원인: `filename_pattern`에 들어간 `{placeholder}`가 그 파일의 format mapping에 없음 (특히 overlay NEW에서 흔함)
  - 해결: overlay NEW에서는 filename에 **file_fields만** 사용하거나, 패턴을 `{signal_group}` 중심으로 단순화
- “filter를 줬는데 안 먹는 것 같다”
  - 원인: (1) 컬럼이 meta에 없음(경고 후 스킵), (2) 타입 불일치(예: 10 vs 10.0)
  - 해결: 입력 데이터 컬럼 존재/타입을 확인하고, config 값을 데이터 타입에 맞춤
- “event vline이 안 보인다”
  - 원인: (1) 이벤트 컬럼이 input/features에 없음(경고 후 스킵), (2) 이벤트 평균이 interpolation 시간축 밖
  - 해결: `event_vlines.columns`와 이벤트 도메인(입력 parquet vs features_file)을 점검하고, `interpolation.start_ms/end_ms` 범위도 확인

## Reference

- 자세한 규칙/스키마/예시 YAML: `references/aggregation-modes-and-event-vlines.ko.md`
