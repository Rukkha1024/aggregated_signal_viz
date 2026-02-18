# ExecPlan: Refactor `src/core/visualizer.py` into Focused Core Modules / `src/core/visualizer.py` 책임 분해 리팩터링

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` are maintained as work progresses.

EN: This document is maintained in accordance with `.codex/PLANS.md`.
KR: 이 문서는 `.codex/PLANS.md` 규칙에 따라 유지됩니다.

## Purpose / Big Picture

EN: The previous `src/core/visualizer.py` contained data loading, interpolation, event/window logic, task construction, style building, and marker extraction in a single file (2,383 lines). This refactor separates these concerns into dedicated `src/core` modules while preserving pipeline behavior. After this change, maintainers can change one concern without re-reading the entire monolith.

KR: 기존 `src/core/visualizer.py`는 데이터 로딩, 보간, 이벤트/윈도우 계산, 태스크 생성, 스타일 구성, 마커 추출이 한 파일(2,383줄)에 결합되어 있었습니다. 이번 리팩터링은 동작을 유지한 채 책임을 `src/core` 하위 모듈로 분리합니다. 변경 후에는 특정 관심사만 독립적으로 수정할 수 있습니다.

## Progress

- [x] (2026-02-18 02:41Z) EN: Captured baseline outputs and MD5 signatures before refactor (`.codex/tmp/visualizer_before_md5.txt`).
  KR: 리팩터링 전 기준선 출력과 MD5를 고정했습니다(`.codex/tmp/visualizer_before_md5.txt`).
- [x] (2026-02-18 03:03Z) EN: Added modular files: `visualizer_types/data/events/tasks/style/markers` under `src/core`.
  KR: `src/core`에 `visualizer_types/data/events/tasks/style/markers` 모듈을 추가했습니다.
- [x] (2026-02-18 03:08Z) EN: Rewrote `src/core/visualizer.py` as orchestration-focused class using mixins.
  KR: `src/core/visualizer.py`를 믹스인 기반 오케스트레이터 구조로 재작성했습니다.
- [x] (2026-02-18 03:10Z) EN: Verified syntax by compiling all new/updated core modules.
  KR: 신규/수정된 core 모듈 전체 문법 컴파일 검증을 완료했습니다.
- [x] (2026-02-18 03:14Z) EN: Ran full pipeline and diagnostics, then captured after MD5.
  KR: 전체 파이프라인과 진단 스크립트를 실행하고 after MD5를 수집했습니다.
- [x] (2026-02-18 03:15Z) EN: Compared before/after MD5 (`MD5_NO_DIFF`).
  KR: before/after MD5 비교 결과(`MD5_NO_DIFF`)를 확인했습니다.
- [ ] (2026-02-18 03:16Z) EN: Final repository records and commit.
  KR: 저장소 기록 정리 및 커밋.

## Surprises & Discoveries

- Observation: EN: Even with major structural split, generated outputs remained byte-identical in this environment.
  KR: 구조를 크게 분해했음에도 이 환경에서는 출력이 바이트 단위로 동일했습니다.
  Evidence: `MD5_NO_DIFF` with 1,658 output files.

- Observation: EN: The longest practical risk area is overlay/event/window coupling, not interpolation.
  KR: 실제 회귀 위험은 보간보다 overlay/event/window 결합부에 집중되어 있었습니다.
  Evidence: Focused module boundaries preserved original method logic and call order.

## Decision Log

- Decision: EN: Keep public API stable (`AggregatedSignalVisualizer`, `ensure_output_dirs`).
  KR: 퍼블릭 API(`AggregatedSignalVisualizer`, `ensure_output_dirs`)는 유지했습니다.
  Rationale: Prevent caller changes in `main.py` and `scripts/pipeline/run_visualizer.py`.
  Date/Author: 2026-02-18 / Codex

- Decision: EN: Use mixin-based extraction in `src/core` rather than introducing new runtime entrypoints.
  KR: 새로운 런타임 엔트리포인트를 만들지 않고 `src/core` 믹스인 분해를 선택했습니다.
  Rationale: Low-risk structural refactor with behavior preservation.
  Date/Author: 2026-02-18 / Codex

- Decision: EN: Keep full-mode verification and MD5 comparison even though acceptance allows visual equivalence.
  KR: 완료 기준이 시각 동등이어도 전체 모드 실행+MD5 비교를 유지했습니다.
  Rationale: Stronger evidence for non-regression.
  Date/Author: 2026-02-18 / Codex

## Outcomes & Retrospective

EN: `src/core/visualizer.py` was reduced from 2,383 lines to 287 lines, with responsibilities moved into focused modules. The pipeline run and diagnostic checks succeeded, and before/after MD5 matched exactly. The refactor improves maintainability while keeping behavior stable.

KR: `src/core/visualizer.py`는 2,383줄에서 287줄로 축소되었고, 책임은 전용 모듈로 분리되었습니다. 파이프라인 실행과 진단 검증이 성공했고 before/after MD5도 완전히 동일했습니다. 동작 안정성을 유지하면서 유지보수성을 개선했습니다.

## Context and Orientation

EN: The runtime entry is `main.py`, which delegates to `scripts/pipeline/run_visualizer.py`. That script imports `AggregatedSignalVisualizer` from `src/core/visualizer.py`. Plot execution uses `src/plotting/matplotlib/task.py` in process pools. This refactor only changes internal organization inside `src/core`.

KR: 실행 진입점은 `main.py`이며, `scripts/pipeline/run_visualizer.py`를 통해 `src/core/visualizer.py`의 `AggregatedSignalVisualizer`를 호출합니다. 플로팅 실행은 `src/plotting/matplotlib/task.py`를 프로세스 풀에서 사용합니다. 이번 변경은 `src/core` 내부 구조만 재구성합니다.

## Plan of Work

EN: Move methods by concern into dedicated modules (`data`, `events`, `tasks`, `style`, `markers`, `types`), then slim `visualizer.py` into orchestration plus initialization. Preserve method bodies and call order to avoid behavior changes.

KR: 관심사별로 메서드를 전용 모듈(`data`, `events`, `tasks`, `style`, `markers`, `types`)로 이동하고, `visualizer.py`는 초기화+오케스트레이션만 담당하도록 축소합니다. 동작 변경을 피하기 위해 기존 메서드 본문과 호출 순서를 유지합니다.

## Concrete Steps

Working directory: repository root.

1. `conda run -n module python main.py`
2. `find output -type f \( -name '*.png' -o -name '*.html' \) -print0 | sort -z > .codex/tmp/visualizer_before_files.zlist`
3. `xargs -0 -a .codex/tmp/visualizer_before_files.zlist md5sum > .codex/tmp/visualizer_before_md5.txt`
4. Create/modify core modules and slim `src/core/visualizer.py`.
5. `conda run -n module python -m py_compile src/core/visualizer.py src/core/visualizer_data.py src/core/visualizer_events.py src/core/visualizer_tasks.py src/core/visualizer_style.py src/core/visualizer_markers.py src/core/visualizer_types.py`
6. `conda run -n module python main.py`
7. `conda run -n module python scripts/diagnostics/check_zero_tick.py`
8. Capture after MD5 and diff:
   `find output -type f \( -name '*.png' -o -name '*.html' \) -print0 | sort -z > .codex/tmp/visualizer_after_files.zlist`
   `xargs -0 -a .codex/tmp/visualizer_after_files.zlist md5sum > .codex/tmp/visualizer_after_md5.txt`
   `diff -u .codex/tmp/visualizer_before_md5.txt .codex/tmp/visualizer_after_md5.txt > .codex/tmp/visualizer_md5.diff || true`

## Validation and Acceptance

EN: Acceptance requires successful full pipeline execution with enabled modes/groups and successful diagnostics. Visual equivalence is the minimum requirement; MD5 equality is recorded as stronger evidence.

KR: 완료 기준은 enabled 모드/그룹 전체 실행 성공과 진단 스크립트 성공입니다. 최소 기준은 시각 동등이며, MD5 동일성은 더 강한 근거로 기록합니다.

Observed results:

- `main.py` run: success.
- `scripts/diagnostics/check_zero_tick.py`: `OK: tick label 0 is present`.
- MD5 diff: `MD5_NO_DIFF` (1,658 files).

## Idempotence and Recovery

EN: Commands above are repeatable. If any step fails, rerun the same step after fixing the immediate error. No destructive git commands are required.

KR: 위 명령은 반복 실행 가능합니다. 실패 시 원인 수정 후 동일 단계를 재실행하면 됩니다. 파괴적 git 명령은 필요하지 않습니다.

## Artifacts and Notes

- `.codex/tmp/visualizer_before_files.zlist`
- `.codex/tmp/visualizer_before_md5.txt`
- `.codex/tmp/visualizer_after_files.zlist`
- `.codex/tmp/visualizer_after_md5.txt`
- `.codex/tmp/visualizer_md5.diff`

## Interfaces and Dependencies

EN: Public interface remains unchanged in `src/core/visualizer.py`. New internal modules are:

- `src/core/visualizer_types.py`
- `src/core/visualizer_data.py`
- `src/core/visualizer_events.py`
- `src/core/visualizer_tasks.py`
- `src/core/visualizer_style.py`
- `src/core/visualizer_markers.py`

KR: `src/core/visualizer.py`의 퍼블릭 인터페이스는 유지되었습니다. 신규 내부 모듈은 다음과 같습니다.

- `src/core/visualizer_types.py`
- `src/core/visualizer_data.py`
- `src/core/visualizer_events.py`
- `src/core/visualizer_tasks.py`
- `src/core/visualizer_style.py`
- `src/core/visualizer_markers.py`

---

Revision note (2026-02-18): EN/KR synchronized plan created after implementation and validation evidence was embedded for restartability.
Revision note (2026-02-18): 구현/검증 완료 상태를 반영해 재시작 가능한 EN/KR 동기화 문서로 작성함.
