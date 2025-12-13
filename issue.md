## Legend / None 표시 이슈 정리 및 해결

### 1) `diff_step_TF_age_group_mean_forceplate.png`에서 왜 `age_group=None`?

원인:
- `groupby`가 실패한 게 아니라, **입력 데이터에 `age_group` 컬럼이 없거나 값이 null(None)** 이라서, 그룹 키가 `(step_TF, None)` 형태로 만들어진 경우입니다.
- 이 프로젝트의 `script/visualizer.py`는 모드에서 요구하는 meta 컬럼이 데이터에 없으면 **해당 컬럼을 `None`으로 채워서**(missing meta cols → `pl.lit(None)`) 계속 진행합니다. 그래서 legend에도 `None`이 그대로 노출됩니다.

해결:
- `config.yaml`의 `data.input_file`이 **`age_group`이 포함된 parquet**(예: `data/merged.parquet`)를 가리키도록 해야 합니다.
- 만약 `data/merged.parquet`를 쓰는데도 `None`이 보이면, join 결과에 `age_group` null이 남아있는 케이스입니다(메타 엑셀에 없는 subject 등). 이 경우 upstream에서 메타를 보완하거나, 누락 subject를 제외/보정해야 합니다.


### 2) `diff_age_group_subject_mean_forceplate.png`에서 subject가 전부 labeling 되는 문제

의도(커밋 `b7209fd9e8d49cbc5dbd3262c6e155fa5a51f338`):
- groupby 컬럼의 unique 값이 `legend_label_threshold`(기본 6) 이상이면 해당 컬럼은 legend 라벨 생성에서 제외.

문제 원인(기존 동작):
- `groupby=["age_group","subject"]`에서 subject unique 값이 6 이상이면 라벨은 `age_group`만 남는 게 맞는데,
  같은 `age_group` 라벨(young/old)이 **subject 개수만큼 반복해서 plot**되며,
  matplotlib이 이를 **중복 제거 없이 legend에 전부 쌓아버리는 구조**였습니다.

해결(코드 반영):
- overlay plot에서 **(1) 라벨이 `None`이면 legend에 넣지 않도록** 처리하고,
- **(2) 같은 라벨은 축(axis)당 1번만 legend에 표시**하도록 중복 라벨을 `_nolegend_`로 바꿔서 막았습니다.
  - 결과적으로 `groupby=["age_group","subject"]`여도 legend에는 `young`, `old`만 남습니다.


### 3) `diff_age_group_forceplate.png`에서 `None` 값이 나오는 이유

원인:
- (1) `age_group` 컬럼 자체가 없거나, (2) `age_group`에 null이 섞여 있거나,
- 또는 (3) legend 숨김 처리에서 `label=None`을 matplotlib에 넘겨서 legend 항목이 문자열 `"None"`으로 생성되는 경우가 있었습니다.

해결(코드 반영):
- overlay plot에서 `label=None`을 그대로 넘기지 않고, 라벨이 없을 때는 `_nolegend_`로 처리하도록 수정했습니다.
- 동일 라벨 중복도 `_nolegend_`로 처리해서 legend가 깔끔하게 유지됩니다.


## 적용된 변경점
- `script/visualizer.py`:
  - overlay(`_plot_forceplate_overlay`, `_plot_emg_overlay`)에서
    - `label is None` → `_nolegend_`
    - 동일 label 반복 → `_nolegend_`


## 재생성/확인 방법
- (WSL2, conda env: `module`)
  - `conda run -n module python main.py --groups forceplate --modes diff_step_TF_age_group_mean diff_age_group_subject_mean diff_age_group_mean`
