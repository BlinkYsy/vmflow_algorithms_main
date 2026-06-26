# Backend Smoke Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal `Test_demo` smoke-test entry that verifies the requested `gpuID` is reflected by the actual TorchScript / ONNX backend logs and provider selection.

**Architecture:** Extend `Test_demo/test_main.cpp` with one self-contained backend-smoke namespace plus a new CLI selection. The smoke test will build init XML dynamically, capture console output during model initialization, assert the expected backend/provider text for `gpuID=0/1`, and reuse the existing DLL init/clean interfaces to avoid changing inference libraries.

**Tech Stack:** C++17, existing `Test_demo` executable, DeepLearning AI/OCR DLL init APIs, Win32 console redirection helpers, MSBuild.

---

### Task 1: Add smoke-test scaffolding

**Files:**
- Modify: `E:\vmflow_algorithms_main\Test_demo\test_main.cpp`

- [ ] **Step 1: Add shared smoke-test helpers for XML building and console capture**

Add small helpers near the existing `test_AI_common` utilities for:
- building init XML from runtime parameters
- temporarily redirecting `stdout`/`stderr`
- searching captured logs for expected substrings

- [ ] **Step 2: Run a build to verify the new helper declarations fail until implemented completely**

Run: `MSBuild.exe E:\vmflow_algorithms_main\Test_demo\Test_demo.vcxproj /t:Build /p:Configuration=Release /p:Platform=x64 /m`
Expected: compile failure if declarations and definitions are incomplete

- [ ] **Step 3: Implement the helper bodies with minimal scope**

Implement only the helpers needed by the smoke entry:
- `buildBackendSmokeInitXml(...)`
- `captureConsoleOutput(...)`
- `containsAllExpectedMarkers(...)`

- [ ] **Step 4: Rebuild to verify helper integration succeeds**

Run: `MSBuild.exe E:\vmflow_algorithms_main\Test_demo\Test_demo.vcxproj /t:Build /p:Configuration=Release /p:Platform=x64 /m`
Expected: build succeeds or exposes the next concrete integration error

### Task 2: Add the backend smoke entry point

**Files:**
- Modify: `E:\vmflow_algorithms_main\Test_demo\test_main.cpp`

- [ ] **Step 1: Add a `test_backend_smoke` namespace with one init/clean verification flow**

Implement a namespace that:
- accepts CLI args: runtime kind, module kind, model path, `gpuID`, optional `deviceType`
- chooses the matching `DeepAI_init_*` / `DeepAI_clean_*` pair
- captures logs during init
- verifies expected backend markers before cleanup

- [ ] **Step 2: Add a failing CLI route for the new selection before wiring the full flow**

Add a new menu/selection branch such as `"7"` / `test_backend_smoke` that currently returns a clear “missing arguments” error when invoked without enough parameters.

- [ ] **Step 3: Run the executable to verify the new route is reachable**

Run: `E:\vmflow_algorithms_main\_Release\Test_demo.exe 7`
Expected: non-zero exit plus a usage message mentioning required args

- [ ] **Step 4: Finish the route wiring and argument parsing**

Update `runSelectedCase(...)` and the interactive menu so `selection == "7"` dispatches to the new smoke namespace and forwards remaining CLI args.

### Task 3: Validate gpuID-specific backend behavior

**Files:**
- Modify: `E:\vmflow_algorithms_main\Test_demo\test_main.cpp`

- [ ] **Step 1: Define exact log expectations for TorchScript and ONNX**

TorchScript expectations:
- `Current inference backend: TorchScript / GPU (cuda:0)`
- `Current inference backend: TorchScript / GPU (cuda:1)`

ONNX expectations:
- success path: `Current inference backend: ONNXRuntime / GPU (cuda:0|1)`
- fallback path: `requested GPU, actual provider: CPUExecutionProvider`

- [ ] **Step 2: Build Release after the final smoke-test logic is in place**

Run: `MSBuild.exe E:\vmflow_algorithms_main\Test_demo\Test_demo.vcxproj /t:Build /p:Configuration=Release /p:Platform=x64 /m`
Expected: build succeeds

- [ ] **Step 3: Run one TorchScript smoke case for `gpuID=0` and one for `gpuID=1`**

Run:
- `E:\vmflow_algorithms_main\_Release\Test_demo.exe 7 torchscript detect <MODEL_PATH> 0 <IMAGE_PATH>`
- `E:\vmflow_algorithms_main\_Release\Test_demo.exe 7 torchscript detect <MODEL_PATH> 1 <IMAGE_PATH>`

Expected:
- exit code `0` if matching backend marker is found
- log contains `TorchScript / GPU (cuda:0)` and `TorchScript / GPU (cuda:1)` respectively

- [ ] **Step 4: Run one ONNX smoke case for `gpuID=0` and one for `gpuID=1`**

Run:
- `E:\vmflow_algorithms_main\_Release\Test_demo.exe 7 onnx detect <MODEL_PATH> 0 <IMAGE_PATH>`
- `E:\vmflow_algorithms_main\_Release\Test_demo.exe 7 onnx detect <MODEL_PATH> 1 <IMAGE_PATH>`

Expected:
- exit code `0` if either GPU provider log matches requested card or CPU fallback is explicitly reported
- log contains either `ONNXRuntime / GPU (cuda:0|1)` or the fallback provider text

- [ ] **Step 5: Print a concise PASS/FAIL summary per case**

Each smoke run should print:
- requested runtime
- requested module kind
- requested `gpuID`
- matched backend/provider marker
- final verdict
