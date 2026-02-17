---
description: Execute statistical analysis pipeline using plan file with orchestrator subagent
---

**IMPORTANT**: This is a multi-step statistical analysis workflow requiring preprocessing, statistical analysis, visualization, and reporting.

## Instructions

Read the plan file and use the orchestrator subagent to coordinate this entire analysis pipeline.

### Plan File Location

The plan file will be searched in the following order:
1. **User-specified**: `${1}` (e.g., `/analyze my_custom_plan.md`)
2. **Current directory**: `./plan.md` (default if no argument provided)
3. **Claude config directory**: `./.claude/plan.md` (fallback location)

### Orchestrator Selection

The orchestrator to use will be determined by:
1. **Plan file specification**: If the plan file contains an `orchestrator` field, use that subagent
2. **Default**: Use `main-orchestrator` if not specified in the plan file

The plan file should contain:
- **Orchestrator** (optional): Name of the orchestrator subagent to use (defaults to `main-orchestrator`)
- **Dataset path**: Location of the data file(s) to analyze
- **Methodology**: Statistical methods, alpha levels, correction methods
- **Hypotheses**: What needs to be tested or verified
- **Data schema**: Column definitions and meanings
- **Expected output format**: Reference files or desired output structure

## Task

Execute the complete analysis workflow using the orchestrator subagent (default: `main-orchestrator`):
1. Data context understanding and preprocessing
2. Statistical analysis (descriptive and inferential)
3. Results reporting and table generation
4. Visualization and figure creation
5. Quality verification and validation

**Do NOT modify the plan file itself** - only read it and execute the analysis pipeline based on its contents.
