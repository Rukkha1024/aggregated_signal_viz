## Reusable prompt for minor skill updates (Write in English)

Copy/paste this prompt to the AI when you want a small, repeatable update added to the existing skill.

```text
Write all new/edited content in ENGLISH (including SKILL.md frontmatter description and body). Do NOT write Korean in the skill file.

Task: Add a small, repeatable troubleshooting note to the existing Codex skill for this repo.
Target file: `.codex/skills/aggregated-signal-figure/SKILL.md`

Workflow constraints (must follow):
1) Propose a brief plan first (what/where you’ll edit and why), then STOP and wait for my explicit approval: I will type "proceed".
2) After "proceed", apply the edit (keep it minimal; do not create a new skill).
3) Commit with a Korean git commit message.
4) Run a lightweight verification command using conda env `module`:
   - `conda run -n module python -c "import sys; print(sys.executable)"`

What to add to the skill (English, concise):
- Update the YAML frontmatter `description` with trigger keywords relevant to this issue (so the skill triggers on similar requests).
- Add a short section describing:
  - Symptom
  - Root cause (why it happens)
  - Fix pattern (where it’s controlled in code/config)
  - “Reference implementation in this repo” with file path(s) to search

Scope: Only edit the SKILL.md file unless I explicitly ask for code changes.
```
