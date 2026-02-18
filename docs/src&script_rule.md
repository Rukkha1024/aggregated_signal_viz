# Project Directory Standard: src vs. scripts

## 1. Overview
This document defines the architectural separation between the `src` and `scripts` directories. Adherence to this standard ensures the separation of concerns between **business logic definitions** and **execution workflows**.

## 2. Directory Specifications

### 2.1. `src` (Source)
**Role:** Repository for reusable logic, library code, and core application definitions.

*   **Content:**
    *   Function and class definitions.
    *   Constant declarations and configuration schemas.
    *   Data structures and models.
    *   Internal modules and packages.
*   **Characteristics:**
    *   **Importable:** Designed to be imported by other modules or scripts.
    *   **Passive:** Code must not execute operations immediately upon import (no side effects at module level).
    *   **Stateless:** Focuses on logic and transformation rules, not the state of a specific execution run.
*   **Prohibition:**
    *   Must not contain top-level execution blocks (e.g., `if __name__ == "__main__":`).
    *   Must not contain hard-coded execution parameters specific to a single run.

### 2.2. `scripts` (Scripts)
**Role:** Repository for executable entry points, automation tasks, and workflows.

*   **Content:**
    *   CLI (Command Line Interface) tools.
    *   Build, deployment, and testing automation files.
    *   Data migration and pipeline orchestration workflows.
    *   One-off utility tasks.
*   **Characteristics:**
    *   **Executable:** Designed to be run directly via the terminal or job scheduler.
    *   **Imperative:** executes a linear sequence of steps to achieve a specific outcome.
    *   **Orchestration:** Connects inputs (arguments, environment variables) with logic defined in `src`.
*   **Prohibition:**
    *   Must not contain core business logic that might be needed elsewhere.
    *   Should not be imported by code within the `src` directory.

## 3. Interaction Protocol

1.  **Dependency Direction:**
    *   `scripts` **depends on** `src`.
    *   `src` **must never depend on** `scripts`.
2.  **Logic Abstraction:**
    *   If a logic block inside a script becomes complex or reusable, it must be refactored and moved to `src`.
3.  **Execution Flow:**
    *   The `scripts` directory acts as the **Driver**, invoking the functionality defined in the `src` **Library**.