# Project Requirements Brief

This template must be completed before any ExecPlan is authored. The agent should use this template to guide the requirements discovery session with the user. Fields marked (AGENT) should be filled by the agent through conversation. Fields marked (USER) require direct user input.

---

## One-Line Summary (AGENT)

<!-- Format: "This tool/system [does X] so that [user] can [achieve Y]." -->


## Problem Statement (AGENT)

<!-- Why does this project need to exist? What pain point or gap does it address? Write 2-3 sentences based on user input. -->


## Success Criteria (USER)

<!-- What does "done" look like? List concrete, observable outcomes. Example: "Running `python main.py --input data.csv` produces a figure saved to `output/fig1.png` showing muscle synergy patterns." -->

1. 
2. 
3. 

## Inputs

### Data Sources (USER)

- Format(s): <!-- e.g., CSV, MAT, C3D, JSON -->
- Location / example path: <!-- e.g., `data/raw/emg_subject01.csv` -->
- Schema or structure: <!-- column names, sampling rate, units, etc. -->
- Volume: <!-- e.g., 12 subjects × 3 conditions × 8 channels -->

### External Dependencies (AGENT)

- Required libraries: <!-- e.g., numpy, scipy, matplotlib -->
- Required services: <!-- e.g., Google Sheets API, database -->
- Required hardware/environment: <!-- e.g., GPU, specific OS -->

## Outputs

### Primary Deliverables (USER)

- Format(s): <!-- e.g., PNG figures, PDF report, new CSV, Python package -->
- Destination: <!-- e.g., `output/` directory, Google Drive, terminal stdout -->
- Naming convention: <!-- e.g., `{subject}_{condition}_synergy.png` -->

### Secondary Artifacts (AGENT)

- Logs: <!-- where and what level -->
- Intermediate files: <!-- cached results, temp files -->
- Documentation: <!-- README, docstrings, usage examples -->

## Constraints

### Technical (AGENT)

- Language: <!-- e.g., Python 3.10+ -->
- Framework: <!-- e.g., none, Django, React -->
- Must use: <!-- libraries or tools that are non-negotiable -->
- Must avoid: <!-- e.g., no MATLAB, no paid APIs -->

### Operational (AGENT)

- Runtime budget: <!-- e.g., must complete in under 5 minutes -->
- Platform: <!-- e.g., Windows 11, Ubuntu 24, macOS -->
- Deployment: <!-- one-off script, CLI tool, installable package, web app -->

## Scope

### In Scope (AGENT)

<!-- Numbered list of what this project WILL do, derived from conversation. -->

1. 
2. 
3. 

### Out of Scope (USER)

<!-- Explicitly state what this project will NOT do to prevent scope creep. -->

1. 
2. 
3. 

## Prior Art and References (USER)

<!-- Existing code, papers, repos, or examples the agent should study before planning. -->

- Reference implementation: <!-- e.g., `scripts/old_analysis.py` in this repo -->
- Paper / method: <!-- e.g., Chvatal et al. (2011), NMF decomposition -->
- Similar project: <!-- e.g., GitHub link, internal tool -->

## User Expertise Level (AGENT)

<!-- How technical is the user? This determines the level of detail in the ExecPlan. -->

- Domain expertise: <!-- e.g., expert in biomechanics, novice in web dev -->
- Coding proficiency: <!-- e.g., intermediate Python, no JS experience -->
- Will the user modify the code after delivery? <!-- yes / no -->

## Open Questions (AGENT)

<!-- Unresolved items that surfaced during discovery. Must be answered before ExecPlan authoring begins. -->

1. 
2. 

---

## Requirements Confirmed

- [ ] User has reviewed and approved all sections above.
- [ ] All Open Questions have been resolved.
- [ ] Agent may now proceed to author the ExecPlan per `.codex/PLANS.md`.