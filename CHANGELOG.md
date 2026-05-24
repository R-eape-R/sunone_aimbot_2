# Implemented Changelog

This changelog records the implemented source changes in the current working
tree. It is an implementation report rather than a short release note, so it
names the systems and files that changed and explains why the changes exist.

Line numbers are intentionally not used here because they shift as the source is
edited.

## Unreleased - Current Source

### Circle Mask Change

The old `circle_mask` setting was a pixel-level frame mask. When it was enabled,
capture produced or reused a circular mask image and copied only the pixels
inside that circle into a new CPU frame. The detection frame itself was
physically altered before inference.

That had an important side effect in the CUDA/TensorRT build. In the capture
loop, `circle_mask=true` made the fast-path decision choose `prefer_gpu=false`.
Instead of this path:

```text
DXGI GPU capture -> CUDA interop -> TensorRT
```

the app fell back toward this path:

```text
CPU capture -> CPU pixel mask -> CPU preprocess -> copy to CUDA -> TensorRT
```

That CPU path is what correlated with the GPU spike behavior changing when the
GUI or overlay state changed.

The new Circle FOV system is separate from the legacy pixel mask. The app now
keeps the captured frame rectangular and unmodified. Circular behavior is
applied after detection by checking whether each detection center is inside the
configured circle.

Implemented pieces:

- Added `sunone_aimbot_2/capture/circle_fov.h` with center, radius, and
  point-in-circle helpers.
- Added config fields:
  - `circle_fov_enabled`
  - `circle_fov_radius_percent`
  - `circle_fov_show_preview`
  - `game_overlay_draw_circle_fov`
- Existing configs with only `circle_mask=true` migrate to Circle FOV. The
  legacy pixel mask is disabled unless newer Circle FOV keys already exist and
  the user explicitly enables the legacy mask.
- The legacy pixel mask remains available as `circle_mask`, but it is treated as
  an explicit CPU pixel-mask option rather than the recommended circular FOV
  behavior.
- Added Circle FOV filtering to both DML and TensorRT detection output.
- Added GUI controls for enabling Circle FOV, adjusting radius from `1..100`,
  previewing the circle, and explicitly enabling the legacy pixel mask.
- Added game-overlay drawing for Circle FOV without modifying capture pixels.

Why this matters:

- Circle FOV now limits targets without forcing a CPU-side frame rewrite.
- CUDA/TensorRT can keep the direct GPU capture path when other settings allow
  it.
- The overlay/preview circle is only a visual guide; it is much cheaper than the
  old per-frame pixel mask.

### TensorRT GPU Fast Path

During diagnosis, TensorRT CPU-frame preprocessing was added so CPU capture
frames can be converted, packed, and copied into TensorRT input memory. That CPU
path is still available for CPU-submitted frames.

A review caught that `processFrameGpu` had also been routed through the CPU
preprocess path by downloading every `GpuMat` before preprocessing. That removed
the point of the GPU capture fast path.

Current behavior:

- `TrtDetector::preProcess(const cv::Mat&)` uses CPU color conversion, resize,
  RGB conversion, float conversion, HWC-to-CHW tensor packing, and
  `cudaMemcpyAsync` into TensorRT input memory.
- `TrtDetector::preProcess(const cv::cuda::GpuMat&)` now keeps GPU frames on
  device.
- GPU frames use CUDA color conversion, CUDA resize, CUDA float conversion, and
  `launch_hwc_to_chw_norm`.
- `CMakeLists.txt` compiles `sunone_aimbot_2/detector/cuda_preprocess.cu` for
  CUDA builds so the CUDA HWC-to-CHW kernel is linked into `ai.exe`.
- Regression checks now protect the intended split:
  - CPU frames may use CPU preprocessing.
  - GPU frames must not force a device-to-host download before preprocessing.

Why this matters:

- The GPU fast path remains:

```text
DXGI GPU capture -> CUDA interop -> CUDA preprocess -> TensorRT
```

- CPU-submitted frames remain supported for compatibility and diagnostics.

### Config Persistence

Fixed config save/load issues that could silently corrupt behavior after reload:

- `config.cpp` now loads `show_fps` before saving can preserve it.
- `config.cpp` restores normal floating formatting before writing `[Games]`.

Why this matters:

- `std::fixed << std::setprecision(...)` could leak into later game-profile
  saves.
- Small yaw/pitch values such as `0.022` could be written as `0.0`.
- A bad saved profile could make mouse conversion produce no movement after
  reload.

### Shutdown And Lifetime Safety

Implemented safer shutdown ordering:

- Added `TrtDetector::requestStop()`.
- Main shutdown now requests TensorRT detector shutdown before joining the TRT
  inference thread.
- Game overlay shutdown is signaled before deleting the DML detector.

Why this matters:

- Prevents possible CUDA/TensorRT shutdown hangs.
- Prevents the game overlay thread from reading a detector object after it has
  been freed.

### Stale Detection Clearing

TensorRT pause paths now fully publish cleared detections:

- Clear boxes.
- Clear classes.
- Clear confidences.
- Increment the detection buffer version.
- Notify waiting consumers.

Why this matters:

- Consumers react to detection-buffer version changes.
- Without a version update, stale detections could remain visible or active
  after pause.

### Capture Initialization And Recovery

Capture backends now expose initialization state and fail cleanly:

- Duplication API capture exposes `isInitialized()`.
- UDP capture tracks and exposes initialization state.
- Failed Duplication API or UDP capture setup returns `nullptr`.

Why this matters:

- The retry logic retries null capture objects.
- Previously, a failed capture object could be retained forever, leaving capture
  permanently unavailable until restart.

### Capture Diagnostics

Added `[CaptureDiag]` logging for diagnosing capture path selection and GPU
spikes.

The diagnostic line reports:

- selected backend and capture method
- capture FPS
- CUDA use and preview/window state
- `prefer_gpu`
- `need_cpu_copy`
- GPU capture attempts and success/failure counters
- GPU timeout, not-ready, lost, acquire-failed, missing-texture counters
- CUDA map, array, and copy failure counters
- TensorRT GPU/CPU submission counters
- CPU fallback and CPU path frame counters
- DML submission counters

Why this matters:

- It shows whether the runtime is using direct GPU capture, CPU capture, CPU
  fallback, TensorRT GPU submit, TensorRT CPU submit, or DML submit.
- It made the Circle Mask and preview-window behavior visible instead of
  speculative.

### kmboxNet Reliability

Hardened the network kmbox path:

- Added UDP receive timeouts.
- Added a command mutex around shared packet state.
- Made monitor state atomic.
- Fixed keyboard queue copies to avoid overreads by using safe overlapping-copy
  behavior.

Why this matters:

- Dropped UDP responses could previously hang calls indefinitely.
- Concurrent monitor/input operations could interleave shared packets.
- Keyboard queue movement had unsafe copy lengths.

### Arduino Input Thread Safety

Improved Arduino input lifetime and flag safety:

- Arduino button-state flags are atomic.
- Keyboard listener reads use atomic loads.
- The listener thread joins before serial close.

Why this matters:

- The serial listener writes these flags while keyboard logic reads them.
- Closing serial while the listener may still use it was a shutdown race.

### Overlay Config Save Reliability

Added immediate overlay config saving:

- Added `OverlayConfig_SaveNow()`.
- The overlay force-saves before hiding.
- The overlay force-saves during shutdown.

Why this matters:

- Debounced config saves could be skipped if the overlay was closed quickly
  after changing settings.

### Game Overlay Monitor Placement

Fixed monitor placement for the game overlay:

- The game overlay uses the configured monitor index.
- It uses the selected monitor's actual left/top bounds.

Why this matters:

- Secondary monitor overlays could be sized correctly but positioned at the
  primary monitor origin.

### Image And Model Math

Fixed image/model coordinate issues:

- Depth letterbox resize now copies the resized image into the computed offset.
- DML fixed-input coordinate scaling now uses separate X and Y scales.

Why this matters:

- Depth preprocessing was effectively stretching instead of letterboxing.
- Non-square fixed DML model inputs could map boxes incorrectly.

### TensorRT Export Cleanup

Cleaned up TensorRT export/build resource ownership:

- Serialized model memory is released on success and failure paths.
- Related TensorRT/CUDA resources are cleaned up during repeated export/build
  attempts.

Why this matters:

- Repeated model export attempts could leak resources.

### Build Entry Points

Added and reworked build entry points:

- Added `tools/build_dml.ps1`.
- Added `tools/build_cuda.ps1`.
- Added `tools/build_common.ps1`.
- Reworked `build_dml.bat` and `build_cuda.bat` into thin wrappers.
- Added root launchers:
  - `BUILDER.ps1`
  - `BUILDER.bat`
  - `RUN_BUILDER.bat`
- Added no-download local rebuild launchers:
  - `build_no-options.ps1`
  - `build_no-options.bat`

Why this matters:

- The old batch build path was tied to Visual Studio/MSBuild behavior and was
  difficult to extend.
- PowerShell gives prompts, dependency checks, download manifests, version
  detection, dry-run support, and better Windows path handling while preserving
  double-click and command-line workflows.

### Build Prompt Flow

The full build scripts now ask:

- OpenCV already built?
- Download or update needed files?

They also support scripted overrides:

```powershell
-OpenCvAlreadyBuilt true
-DownloadOrUpdateNeeded true
-NonInteractive
-DryRun
```

Why this matters:

- First-run users get guided setup.
- Repeatable scripted builds do not need interactive prompts.

### Ninja Build Automation

The automated CMake build now uses Ninja Multi-Config by default:

- `CMakeLists.txt` accepts Ninja generators instead of requiring Visual Studio
  project generation.
- Build scripts call `cmake --build ... --config Release --parallel`.

Why this matters:

- Ninja is faster and easier for automated builds.
- Ninja Multi-Config keeps the familiar Release/Debug configuration model.

### Visual Studio Environment Detection

`tools/build_common.ps1` initializes the compiler environment:

- Uses `vswhere.exe`.
- Imports `VsDevCmd.bat`.
- Finds Ninja from PATH, Visual Studio's bundled Ninja, or `tools/.bin`.

Why this matters:

- CUDA and Ninja on Windows still need the MSVC compiler environment.
- Users no longer need to manually open a Developer Command Prompt.

### NuGet Automation

Build scripts restore NuGet packages into `packages/`:

- Default restore uses pinned versions from `sunone_aimbot_2/packages.config`.
- `-UseLatestPackages` installs latest package versions without rewriting the
  Visual Studio project.

Why this matters:

- CMake/Ninja can use the newest valid package folders.
- The old `.vcxproj` remains stable instead of being mutated by package restore.

### Guided NVIDIA Downloads

CUDA setup can write:

- `build/dependency-downloads.json`
- `build/dependency-downloads.md`

The CUDA build script can open browser pages with:

```powershell
-OpenBrowserForDownloads
```

After the user downloads NVIDIA files, the script scans Downloads and caches
matches under:

```text
sunone_aimbot_2/modules/_downloads
```

Why this matters:

- NVIDIA downloads can require login or license acceptance.
- The script avoids credential handling while still automating everything after
  the user has downloaded the files.

### Best-Compatible CUDA Selection

Added `Get-BestCompatibleCudaDependencySet` in `tools/build_common.ps1`.

It detects:

- GPU information
- installed CUDA
- TensorRT folders
- optional cuDNN
- CUDA architecture

It records selected dependency information in:

```text
build/dependency-resolution.json
```

Why this matters:

- The goal is "latest compatible," not blindly newest.
- The selected dependency set is recorded for debugging and repeatability.

### OpenCV Folder Layout

DML OpenCV now targets:

```text
sunone_aimbot_2/modules/opencv/build/dml
```

CUDA OpenCV now builds under:

```text
sunone_aimbot_2/modules/opencv/build/cuda
```

CUDA OpenCV installs to:

```text
sunone_aimbot_2/modules/opencv/build/cuda/install
```

Why this matters:

- DML and CUDA builds no longer accidentally consume each other's OpenCV
  runtime.

### Dynamic OpenCV Version Detection

Removed the hard dependency on `opencv_world4130`.

Current behavior:

- CMake and scripts detect `opencv_world*.lib`.
- The matching same-stem `.dll` is selected.
- Release builds avoid debug-suffixed OpenCV libraries such as
  `opencv_world4130d.lib`.

Why this matters:

- Newer OpenCV versions can be dropped or built into the expected folder and
  picked up automatically.

### cuDNN Optional Path

CUDA OpenCV automation now disables cuDNN for OpenCV DNN:

- `WITH_CUDNN=OFF`
- `OPENCV_DNN_CUDA=OFF`

Why this matters:

- The app uses TensorRT for inference, not OpenCV DNN.
- Disabling OpenCV DNN cuDNN support removes a fragile dependency without losing
  the app's TensorRT inference or OpenCV CUDA preprocessing behavior.

### Source Module Automation

Build automation prepares missing core source modules:

- `sunone_aimbot_2/modules/SimpleIni.h`
- `sunone_aimbot_2/modules/serial`

Why this matters:

- Full automation should prepare all required build inputs rather than stopping
  at OpenCV, TensorRT, and NuGet.

### Runtime Packaging

CMake copies runtime DLLs beside `ai.exe` when available:

- OpenCV runtime DLL
- ONNX Runtime and DirectML DLLs
- CUDA/TensorRT related runtime DLLs for CUDA builds
- GHUB mouse DLL
- Razer `chroma_lighting.dll`

CMake also deploys selected training assets:

- root training scripts
- PID governor helper scripts
- neural tracker helper scripts
- base ONNX models
- model metadata JSON files

It intentionally does not deploy training data, tests, caches, or pycache
folders.

### Ignore Rules

Updated `.gitignore` for generated/downloaded build artifacts:

- `tools/.bin/`
- `sunone_aimbot_2/modules/_downloads`
- `sunone_aimbot_2/modules/TensorRT-*`

Why this matters:

- Tool caches, downloaded archives, and extracted NVIDIA SDKs should not pollute
  git status.

### Razer Controls

Added the optional Razer controls module:

- `sunone_aimbot_2/modules/razer-controls`

Runtime DLL name:

```text
chroma_lighting.dll
```

Added runtime wrapper:

- `sunone_aimbot_2/mouse/rzctl.h`
- `sunone_aimbot_2/mouse/rzctl.cpp`

Implemented behavior:

- Dynamically loads `chroma_lighting.dll` with `LoadLibraryW`.
- Searches beside `ai.exe`, under `controls`, and under the module release
  folder relative to both executable and current working directory.
- Resolves and validates required exports:
  - `init`
  - `mouse_move`
  - `mouse_click`
  - `keyboard_input`
  - `mouse_move_status`
  - `mouse_click_status`
- Prefers status-returning movement/click exports when available.
- Routes movement and click down/up through the Razer wrapper when
  `input_method = RAZER`.
- Adds startup, reassignment, GUI status, CMake source inclusion, CMake runtime
  copying, and shutdown handling.

No fallback input method is used when `RAZER` is selected. If the DLL is missing,
exports are missing, or initialization fails, the Razer path reports not
connected instead of silently moving through another backend.

### Teensy 4.1 RawHID Controls

Added the `TEENSY41_HID` input method.

Added implementation:

- `sunone_aimbot_2/mouse/Teensy41RawHid.h`
- `sunone_aimbot_2/mouse/Teensy41RawHid.cpp`

Implemented behavior:

- Fixed 64-byte host packet and 64-byte device packet contract.
- Host reports are sent with a report ID prefix, giving a 65-byte HID write.
- Packet fields include magic, protocol version, command, host button mask,
  dx/dy, wheel, horizontal wheel, and sequence number.
- Device reports parse button events, status events, and sequence
  acknowledgements.
- RawHID discovery is configurable through serial, VID, PID, usage page, usage
  ID, open index, packet timeout, and reconnect interval.
- Background reader thread reconnects on disconnect and validates magic/version
  before applying device events.
- Device button events update local and global state:
  - button 1 -> shooting / left mouse
  - button 2 -> zooming / right mouse
  - button 5 -> aiming / X2 mouse
- Movement and click routing support `TEENSY41_HID` in `MouseThread`.
- Keyboard/button listener can consume Teensy physical side-button state instead
  of falling back to Win32 mouse state.
- Added startup, reassignment, GUI controls, save/reconnect, CMake source
  inclusion, and shutdown handling.

No fallback input method is used when `TEENSY41_HID` is selected. If no device is
connected or writes fail, movement is not silently rerouted to another backend.

### Other Mouse And Control Robustness

Expanded `input_method` support and documentation:

- `WIN32`
- `GHUB`
- `RAZER`
- `ARDUINO`
- `RP2350`
- `TEENSY41_HID`
- `KMBOX_NET`
- `KMBOX_A`
- `MAKCU`

Other control updates:

- Added input-method change signaling from the GUI so device changes can trigger
  reconnect or reassignment.
- Added `KMBOX_A` one-field `PIDVID` config in `PPPPVVVV` format.
- Added `KmboxAConnection::parsePidVid`.

### Neural Tracker Association

Added optional neural tracker association support:

- `sunone_aimbot_2/neural/NeuralTracker.h`
- `sunone_aimbot_2/neural/NeuralTracker.cpp`

Implemented behavior:

- 16-feature model contract through `NeuralTrackerFeatureCount`.
- `INeuralTracker` runtime scorer interface.
- CPU ONNX Runtime inference.
- CUDA/TensorRT inference support in CUDA builds.
- Model path resolution relative to configured path, working directory, and
  executable directory.
- ONNX input feature-count validation.
- Score normalization.
- Integration into `AimbotTarget` association scoring.
- Optional CSV logging for feature values, scores, acceptance, and selection.

Added config fields:

- `neural_tracker_enabled`
- `neural_tracker_runtime`
- `neural_tracker_model_path`
- `neural_tracker_blend`
- `neural_tracker_log_enabled`
- `neural_tracker_debug_enabled`
- `neural_tracker_log_path`

Added overlay support:

- Neural sidebar tab.
- CPU/CUDA runtime selection.
- association model discovery and path edit.
- association blend slider.
- debug/log controls.

If disabled, missing, or unavailable, the tracker uses the classical association
path.

### PID Governor Support

Added PID governor config/UI/training support:

- `pid_governor_enabled`
- `pid_governor_speed`
- `pid_governor_blend`

Implemented behavior:

- Load/save support.
- Speed clamped to `1..100`.
- Blend clamped to `1..100`.
- Neural tab controls for enable, speed, and blend.
- Offline training modules under `sunone_aimbot_2/modules/training`.
- Scripts for synthetic dataset generation, training, ONNX export, and
  evaluation.
- Base model artifacts and metadata.

Current boundary:

- The runtime source exposes and persists the governor controls and packages
  model/training assets.
- A dedicated PID governor inference class is not yet wired into the live mouse
  controller output.

### Training Modules

Moved project training assets into:

```text
sunone_aimbot_2/modules/training
```

Added base model artifacts:

- `models/pid_governor.onnx`
- `models/neural_tracker.onnx`

Added scripts:

- `generate_pid_dataset.py`
- `train_pid_governor.py`
- `export_pid_governor_onnx.py`
- `evaluate_pid_governor.py`
- `generate_neural_tracker_dataset.py`
- `train_neural_tracker.py`
- `export_neural_tracker_onnx.py`

Build automation bootstraps missing deployable ONNX models before CMake
packaging when possible.

### Overlay And GUI

Added or updated GUI surfaces:

- Capture tab Circle FOV controls.
- Debug tab Circle FOV preview and neural tracker debug controls.
- Mouse tab Razer and Teensy status/config controls.
- Neural tab for association model/runtime/blend and PID governor controls.
- Game overlay Circle FOV draw option.
- Config dirty/reconnect signaling for runtime and control-device changes.

### Runtime Config

Added defaults, persistence, and validation for:

- Circle FOV and legacy pixel mask.
- game overlay Circle FOV drawing.
- Razer input method selection.
- Teensy 4.1 RawHID selection and filters.
- `KMBOX_A` PIDVID.
- neural tracker association.
- PID governor controls.

Added clamping for:

- Circle FOV radius.
- neural tracker runtime and blend.
- PID governor speed/blend.
- Teensy RawHID usage/filter timing values.

### Documentation

Updated documentation for the current source:

- Rewrote `README.md` as a high-level project overview.
- Rewrote `docs/build.md` for the new DML, CUDA, and no-options build flows.
- Rewrote `docs/config.md` to match current config keys and defaults.
- Rewrote `docs/guides.md` with practical setup and troubleshooting guidance.
- Documented Circle FOV vs legacy `circle_mask`.
- Documented Razer and Teensy setup, including no-fallback behavior.
- Documented capture diagnostics and DML/CUDA model selection.

### Regression Contracts

Added and maintained `tools/regression_checks.ps1`.

Regression checks cover:

- config persistence and game profile formatting.
- Circle FOV migration and detector filtering.
- TensorRT GPU-frame preprocessing staying on device.
- CMake compiling `cuda_preprocess.cu` for CUDA builds.
- capture diagnostics counters.
- capture backend initialization failures returning null.
- Razer wrapper and `chroma_lighting.dll` naming.
- Teensy RawHID packet behavior and input routing.
- kmboxNet and Arduino safety changes.
- overlay force-save behavior.
- neural tracker interface, runtime selection, and overlay controls.
- PID governor config/UI ranges.
- training model presence and build bootstrapping.
- builder dispatch and no-options builder behavior.
- OpenCV build-script guardrails.

### Current Boundaries And Notes

- `build_no-options` assumes the selected build tree already exists. Use the full
  builder first when dependencies, OpenCV, packages, or CMake configure state are
  missing.
- Razer and Teensy inputs intentionally do not fall back to another control
  method when selected.
- Circle FOV is the recommended circular constraint. Legacy Pixel Circle Mask is
  still present but should only be used when the actual pixels outside the circle
  must be blacked out.
- CPU TensorRT preprocessing remains available for CPU-submitted frames.
- GPU TensorRT preprocessing now stays on device for GPU-submitted frames.
- Neural tracker association is optional. If disabled, missing, or unavailable,
  target tracking uses the classical association path.
- PID governor has config/UI/training/model packaging in the current source.
  Runtime PID governor inference still needs a C++ controller integration step.
