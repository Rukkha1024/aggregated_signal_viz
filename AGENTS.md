## Work Procedure
Always follow this procedure when performing tasks:
1. **Plan the changes**: Before making any code modifications, create a detailed plan outlining what will be changed and why
2. **Get user confirmation**: Present the plan to the user and wait for explicit confirmation before proceeding
3. **Modify code**: Make the necessary code changes according to the confirmed plan
4. **Git commit in Korean**: Commit your changes with a Korean commit message
5. **Run the modified code**: Execute the modified code to verify your work


---
## Environment rules
- Use the existing conda env: `module` (WSL2).
- Always run Python/pip as: `conda run -n module python` / `conda run -n module pip`.
- **Do not** create or activate any `venv` or `.venv` or run `uv venv`.
- If a package is missing, prefer:
  1) `mamba/conda install -n module <pkg>` (if available)
  2) otherwise `conda run -n module pip install <pkg>`
- Before running Python, verify the interpreter path with:
  `conda run -n module python -c "import sys; print(sys.executable)"`

---
## **Codebase Rule: Configuration Management**

절대 사용자의 컨펌이 있기 전에는 코드 실행을 진행하지 마라. 사용자가 정확하게 `proceed`라고 하지 않는 이상 절대로 코드 수정을 하지 않는다. 

### **Core Principle: Centralized Control**
The primary goal is to centralize shared values across multiple scripts. This ensures consistency and minimizes code modifications when parameters change.

### **Items to Include in Config Files:**
1.  **Paths and Directories:** Define paths to data, logs, and outputs (e.g., `RAW_DATA_DIR`, `OUTPUT_DIR`).
2.  **File Identification Patterns:** Store regex or fixed strings for parsing filenames (e.g., `VELOCITY_PATTERN`, `TRIAL_PATTERNS`).
3.  **Data Structure Definitions:** List column names for data extraction or processing (e.g., `FORCEPLATE_COLUMNS`, `METADATA_COLS`).
4.  **Fixed Processing Constants:** Define constants derived from the experimental setup (e.g., `FRAME_RATIO`, `FORCEPLATE_DATA_START`).
5.  **Tunable Analysis Parameters:** Specify parameters that researchers might adjust (e.g., filter cutoffs, normalization methods).
6.  **Shared Texts:** Centralize common log messages or report headers (e.g., `STAGE03_SUMMARY_HEADER`).

### **Exclusion Rule:**
- **Visualization Settings:** Do not include settings related to the visual appearance of plots (e.g., colors, fonts, line styles). These should be managed within the visualization code itself.

---

## **Codebase Rule: Perturbation Task Data Processing**

### **Scope: This document defines the rules governing data structure, time frames, and processing units specifically for the `perturb` task within this codebase.**

## **1. Core Principles**

### **1.1. Unit of Analysis**
The fundamental, indivisible unit for all data processing, analysis, and file management is the unique combination of **`subject-velocity-trial`**.

*   **Subject:** The unique identifier for a participant (e.g., '김연옥').
*   **Velocity:** The perturbation speed condition (e.g., '5.0').
*   **Trial:** The specific trial number for a given subject and velocity (e.g., '1').
*   **Implementation:** All intermediate and final data files are named and grouped using this unit. Data merging and aggregation operations are performed at this level. 

### **1.2. Data Source Authority**
*   The `platform on-offset.xlsx` file is the definitive source for event timing (`platform_onset`, `platform_offset`). These timings are recorded in the **MocapFrame** unit.
*   The `config.yaml` file is the definitive source for signal processing parameters (e.g., filter cutoffs, processing order) and muscle channel names.

## **2. Time Frame Definitions and Rules**

This codebase utilizes four distinct time frame systems. Adherence to these definitions is critical for data integrity.

### **2.1. `MocapFrame`**
*   **Description:** The **absolute** time frame system based on the motion capture system's sampling rate.
*   **Sampling Rate:** **100Hz**.
*   **Role:**
    *   Acts as the master clock for synchronizing all heterogeneous data sources (EMG, Forceplate, COM).
    *   Serves as the reference unit for all event timings defined in `platform on-offset.xlsx`.
*   **Rule:** This value must remain unchanged throughout the pipeline and represents the original, global timestamp from the raw data.

### **2.2. `original_DeviceFrame`**
*   **Description:** The **absolute** time frame system based on the EMG/Forceplate sensor's sampling rate.
*   **Sampling Rate:** **1000Hz**.
*   **Role:** A high-resolution version of the absolute time frame, preserved to maintain linkage to the original raw sensor data after segmentation.
*   **Rule:** It is calculated as `MocapFrame * 10`. This column is created during Stage 03 as a backup of the initial `DeviceFrame` before it is localized.

### **2.3. `DeviceFrame`**
*   **Description:** A **relative** (local) time frame system.
*   **Sampling Rate:** **1000Hz**.
*   **Role:** Represents the time elapsed *within* a single, segmented `subject-velocity-trial` unit.
*   **Rule:** For each trial segment, this frame counter **must** start at `0`. It is generated in Stage 03 by resetting the frame index for each extracted segment.


## **3. Frame Conversion and Relationship**

*   **`FRAME_RATIO`:** The constant conversion factor between Mocap and Device frames is **10**.
*   **Conversion Formulas:**
    *   To Device Frame: `DeviceFrame = MocapFrame * 10`
    *   To Mocap Frame: `MocapFrame = DeviceFrame // 10` (integer division)
*   **Hierarchy:** `MocapFrame` and `original_DeviceFrame` are global/absolute. `DeviceFrame` is local/relative to a trial. `resampled_frame` is normalized and abstract.