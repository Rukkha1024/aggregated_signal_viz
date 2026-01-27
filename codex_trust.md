# Codex `.codex/` 폴더가 비활성화되는 경우 (Trusted 설정)

Codex CLI가 아래와 같은 경고를 출력하며 프로젝트의 `.codex/` 폴더(스킬/규칙/프로젝트 설정)를 로드하지 못할 때가 있습니다.

```
⚠ The following config folders are disabled:
  1. <project>/.codex
     Add <project> as a trusted project in ~/.codex/config.toml.
```

## 해결 방법

`~/.codex/config.toml`에 프로젝트 경로를 `trusted`로 등록합니다.

예시:

```toml
[projects."/mnt/c/Users/Alice/OneDrive - 청주대학교/근전도 분석 코드/aggregated_signal_viz"]
trust_level = "trusted"
```

등록 후 `codex`를 다시 실행하면 `.codex/` 폴더가 활성화됩니다.

