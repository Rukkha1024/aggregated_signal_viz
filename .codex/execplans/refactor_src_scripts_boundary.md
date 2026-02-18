# ExecPlan: src/scripts 경계 리팩토링과 matplotlib 모듈 실체화

이 ExecPlan은 living document입니다. 구현 중 `Progress`, `Surprises & Discoveries`, `Decision Log`, `Outcomes & Retrospective`를 지속적으로 업데이트합니다. 이 문서는 `.codex/skills/PLANS.md`를 따릅니다.

## Purpose / Big Picture

`Archive/src&script_rule.md` 기준에 맞춰 `src`를 재사용 로직 중심으로 정리하고, 비어 있던 `src/plotting/matplotlib` 모듈을 실제 역할 모듈로 전환합니다. 동시에 `scripts` 내부의 공용 가능한 로직 일부를 `src/core`로 추출하여 경계를 명확히 합니다. 완료 후에는 `main.py --sample` 결과가 기존과 동등함을 MD5 비교와 육안 확인으로 검증합니다.

## Progress

- [x] (2026-02-18 11:40Z) 요구사항/의사결정 확정.
- [x] (2026-02-18 11:52Z) baseline 실행 및 before MD5 생성.
- [x] (2026-02-18 12:02Z) matplotlib 모듈 리팩토링 적용(`line/scatter/task` 실체화, `shared.py` 추가).
- [x] (2026-02-18 12:05Z) scripts 공용 로직 src 이관(`aggregation_mode_utils.py` 추가, `vis_onset.py` 연결).
- [x] (2026-02-18 12:08Z) 실행/MD5/육안 검증 완료(`NO_MD5_DIFF`, 대표 플롯 3개 확인).
- [x] (2026-02-18 12:10Z) 이슈/해결 기록 완료(`.codex/issue.md`, 글로벌 troubleshooting 기록).
- [ ] 한국어 3줄 이상 커밋 완료.

## Surprises & Discoveries

- Observation: 출력 파일명에 공백과 한글이 많아 일반 `xargs` 기반 md5 계산이 실패함.
  Evidence: `md5sum ... No such file or directory` 다수 발생.
- Observation: `find -print0 | xargs -0`로 안정적으로 해결됨.
  Evidence: before 파일/해시 라인 수 1562/1562 일치.

## Decision Log

- Decision: 리팩토링 범위는 `src + scripts` 경계 재정리까지 포함.
  Rationale: 사용자 명시 선택.
  Date/Author: 2026-02-18 / Codex

- Decision: 동작 불변을 원칙으로 하며 시각 스타일 튜닝은 제외.
  Rationale: 회귀 리스크 최소화.
  Date/Author: 2026-02-18 / Codex

- Decision: Matplotlib는 `common` 집중 구조를 완화하고 `line/scatter/task` 실체화를 우선.
  Rationale: empty/얇은 모듈 문제를 직접 해결.
  Date/Author: 2026-02-18 / Codex

## Outcomes & Retrospective

1. 구조 개선 결과:
   - `src/plotting/matplotlib/line.py`, `scatter.py`, `task.py`가 단순 재export를 벗어나 타입 정규화/디스패치 책임을 갖는 모듈로 전환됨.
   - `src/plotting/matplotlib/shared.py`를 추가해 공통 변환 유틸을 중앙화함.
   - `scripts/plotting/errorbar/vis_onset.py` 내부 aggregation mode 보조 함수를 `src/core/aggregation_mode_utils.py`로 이관함.
2. 검증 결과:
   - `conda run -n module python main.py --sample` 성공.
   - `conda run -n module python scripts/diagnostics/check_zero_tick.py` 성공(`OK: tick label 0 is present`).
   - before/after md5 비교 결과 `NO_MD5_DIFF`.
   - 대표 플롯 3개(EMG/forceplate/COP) 육안 확인 완료.
3. 잔여 기술부채:
   - `src/plotting/matplotlib/common.py`는 여전히 대형 파일이며, 향후 단계적 분할 여지가 있음.

## Context and Orientation

현재 렌더링 핵심은 `src/plotting/matplotlib/common.py`에 집중되어 있고 `line.py`, `scatter.py`, `task.py`는 얇은 위임 모듈입니다. 실행 흐름은 `main.py` -> `scripts/pipeline/run_visualizer.py` -> `src/core/visualizer.py`이며, `visualizer`가 `task.plot_task`를 process-pool에서 실행합니다. `scripts/plotting/errorbar/vis_onset.py`에는 `aggregation_modes` 관련 공용 함수가 스크립트 내부에 있습니다.

## Plan of Work

우선 baseline 결과를 고정한 뒤, `src/plotting/matplotlib/shared.py`를 추가하고 `line.py`, `scatter.py`, `task.py`를 각각 라인/스캐터/작업 디스패치 책임을 갖는 모듈로 변경합니다. 이때 기존 동작을 보존하기 위해 레거시 구현은 `common.py`를 재사용하되, 엔트리포인트의 타입/입력 정규화 책임을 분리합니다. 다음으로 `scripts/plotting/errorbar/vis_onset.py`의 `_as_str_list`, `_get_mode_cfg`, `_coerce_bool`, `_apply_filters` 성격의 공용 함수를 `src/core/aggregation_mode_utils.py`로 옮기고 스크립트에서 import하도록 변경합니다. 이후 샘플 실행, diagnostics, md5 비교, 육안 확인, 기록/커밋 순으로 마무리합니다.

## Concrete Steps

작업 디렉터리: repo root

1. `conda run -n module python main.py --sample`
2. `find output -type f \( -name '*.png' -o -name '*.html' \) -print0 | sort -z > .codex/tmp/refactor_before_files.zlist`
3. `xargs -0 -a .codex/tmp/refactor_before_files.zlist md5sum > .codex/tmp/refactor_before_md5.txt`
4. 코드 리팩토링 적용
5. `conda run -n module python main.py --sample`
6. `conda run -n module python scripts/diagnostics/check_zero_tick.py`
7. after md5 생성 후 `diff -u` 비교

## Validation and Acceptance

- `main.py --sample` 실행이 성공해야 합니다.
- 산출물 생성이 정상이어야 합니다.
- before/after MD5가 동일해야 하며, 차이가 있으면 사유를 문서화합니다.
- 최소 3개 대표 플롯을 육안으로 비교합니다.

## Idempotence and Recovery

- 모든 검증 명령은 반복 실행 가능합니다.
- 실패 시 마지막 정상 상태에서 함수 단위로 재적용합니다.
- 사용자 지시 없는 파괴적 git 명령은 사용하지 않습니다.

## Artifacts and Notes

- before hash: `.codex/tmp/refactor_before_md5.txt`
- before file list: `.codex/tmp/refactor_before_files.txt`

## Interfaces and Dependencies

- 신규 파일: `src/core/aggregation_mode_utils.py`
- 신규 파일: `src/plotting/matplotlib/shared.py`
- 수정 파일: `src/plotting/matplotlib/line.py`, `src/plotting/matplotlib/scatter.py`, `src/plotting/matplotlib/task.py`
- 수정 파일: `scripts/plotting/errorbar/vis_onset.py`
- 런타임: `conda run -n module python`

---

Revision note (2026-02-18): 초기 실행계획을 대화 블록에서 저장형 문서로 전환하고 baseline 발견사항(print0/xargs-0)을 반영함.
Revision note (2026-02-18): 구현 결과(모듈 실체화, src 이관, 검증 결과)를 Progress/Outcomes에 반영함.
