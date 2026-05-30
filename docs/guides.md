# Usage and Troubleshooting Guides

This page is for practical setup and diagnosis. It avoids source-code detail unless it helps you solve a real problem.

## First Launch Checklist

1. Pick the right build:
   - Use **DML** if you want easiest setup or non-NVIDIA compatibility.
   - Use **CUDA + TensorRT** if you have the NVIDIA CUDA/TensorRT stack ready.
2. Put the detector model in the `models` folder beside `ai.exe`:
   - DML uses `.onnx`.
   - TensorRT normally uses `.engine`; a CUDA build can also build an `.engine` from a selected `.onnx`.
3. Start the app once so `config.ini` is generated.
4. Open the GUI or overlay and save settings from there when possible.
5. Confirm the selected `input_method` matches the device/control path you actually want.

## Choosing DML vs CUDA

### Use DML When

- You want the least complicated Windows GPU path.
- You are testing compatibility.
- You do not want to install CUDA and TensorRT.
- You are using an ONNX model.

### Use CUDA + TensorRT When

- You have an NVIDIA GPU.
- CUDA Toolkit and TensorRT are installed.
- You want the highest-performance backend.
- You are using a TensorRT `.engine` model, or you have an `.onnx` ready for first-time engine generation.

## DML Runs But Does Not Detect Anything

Check these first:

```ini
backend = DML
ai_model = your_model.onnx
confidence_threshold = 0.10
class_player = 0
class_head = 1
```

Common causes:

- The selected model is a TensorRT `.engine` instead of `.onnx`.
- `confidence_threshold` is too high for the model.
- The model class IDs do not match `class_player` and `class_head`.
- `dml_device_id` points to the wrong GPU.
- The capture source is not actually showing the target content.

## CUDA Runs But GPU Usage Spikes

Start by checking which features force CPU-readable frames:

- Debug/preview window: `show_window = true`.
- Data collection.
- Screenshots.
- Any feature that needs pixels on the CPU for display or saving.

For the current CUDA path, the recommended FOV limiter is:

```ini
circle_fov_enabled = true
```

If the spike disappears when the GUI or overlay is open, compare the capture diagnostics in both states. The GUI/preview can change whether the app requests CPU copies, which can make the runtime path different from the closed-GUI path.

## Reading Capture Diagnostics

When diagnostic logging prints a line like this:

```text
[CaptureDiag] backend=TRT method=duplication_api capture_fps=60 use_cuda=true show_window=true prefer_gpu=false need_cpu_copy=true ...
```

Use these fields:

| Field | Meaning |
|---|---|
| `backend` | Active inference backend, such as `TRT` or `DML`. |
| `method` | Capture method, usually `duplication_api`. |
| `use_cuda` | Whether CUDA capture is allowed by config/build. |
| `show_window` | Whether preview/debug window is open. |
| `prefer_gpu` | Whether the capture path is trying to stay GPU-side. |
| `need_cpu_copy` | Whether some active feature needs CPU-readable pixels. |
| `gpu_attempts` | Number of GPU capture attempts. If this is `0`, the GPU path is not being attempted. |
| `gpu_ok` | Successful GPU capture frames. |
| `gpu_timeout`, `gpu_not_ready`, `gpu_lost` | GPU capture failure categories. |
| `cuda_map_failed`, `cuda_array_failed`, `cuda_copy_failed` | CUDA interop/copy failure categories. |
| `cpu_path_frames` | Frames captured through the CPU path. |
| `trt_cpu_submitted` | Frames sent to TensorRT from CPU-prepared input. |
| `trt_gpu_submitted` | Frames submitted to TensorRT from GPU path. |
| `dml_submitted` | Frames sent to DirectML. |

For example, this pattern means TensorRT is running, but capture/preprocess is going through CPU frames:

```text
prefer_gpu=false need_cpu_copy=true gpu_attempts=0 cpu_path_frames=6000 trt_cpu_submitted=6000
```

That is not automatically wrong. It means some current setting or feature is selecting a CPU-readable path.

## Circle FOV Guide

### Recommended Setup

```ini
circle_fov_enabled = true
circle_fov_radius_percent = 100
circle_fov_show_preview = true
game_overlay_draw_circle_fov = true
```

Use the GUI or overlay to visualize the circle. Lower `circle_fov_radius_percent` to make the active area smaller.

### Does Drawing the Circle Add Much Overhead?

Drawing the circle in the GUI preview or game overlay is small compared with capture and inference. The overlay depiction only matters when the GUI or overlay is actually open and configured to show it.

## Control Method Guide

Set the control method with:

```ini
input_method = WIN32
```

Valid values:

```text
WIN32, GHUB, RAZER, ARDUINO, RP2350, TEENSY41_HID, KMBOX_NET, KMBOX_A, MAKCU
```

Hardware methods are explicit. If you select `RAZER` or `TEENSY41_HID` and the matching runtime is not available, the app does not switch to another method for you.

## Razer Setup

Use:

```ini
input_method = RAZER
```

The runtime wrapper loads:

```text
rzctl.dll
```

Expected project path:

```text
sunone_aimbot_2\rzctl.dll
```

The DLL can also be placed beside `ai.exe`. If the DLL is missing or exports are wrong, Razer movement will not work and no fallback control method is used.

Quick checks:

- Confirm the DLL filename is exactly `rzctl.dll`.
- Confirm it is the same architecture as the app, usually x64.
- Confirm `input_method = RAZER`.
- Watch console logs for `[Razer]` messages.

## Teensy 4.1 RawHID Setup

Use:

```ini
input_method = TEENSY41_HID
```

Default HID filters:

```ini
teensy_hid_serial = AUTO
teensy_hid_vid_filter = AUTO
teensy_hid_pid_filter = AUTO
teensy_hid_usage_page = 65451
teensy_hid_usage_id = 512
teensy_hid_open_index = 0
```

The firmware must match the expected RawHID-style packet interface. The current path sends report-ID-prefixed 64-byte packets.

Button mapping used by the control path:

| Button ID | Meaning |
|---:|---|
| `1` | Shoot / left mouse. |
| `2` | Zoom / right mouse. |
| `5` | Aiming / side button state. |

Quick checks:

- Confirm the Teensy is visible as a HID device.
- Start with all filters set to `AUTO`.
- If multiple matching devices exist, adjust `teensy_hid_open_index`.
- Watch console logs for `[Mouse] Using TEENSY41_HID input.`

## Overlay and GUI Behavior

There are two separate display concepts:

- The GUI/debug preview window, controlled mostly by `show_window`.
- The game overlay, controlled by `game_overlay_enabled` and related overlay keys.

The GUI preview can require CPU copies because it has to show captured pixels. The game overlay can draw boxes, future positions, frame outlines, Circle FOV, and optional icons.

Useful overlay keys:

```ini
overlay_exclude_from_capture = true
game_overlay_enabled = false
game_overlay_compensate_latency = true
game_overlay_draw_circle_fov = true
```

If you are measuring performance, test with the preview closed and then open so you can see how CPU-copy needs change.

### Latency Compensation

The app compensates for the delay between frame capture, inference, and display.
This keeps aim and overlay drawing closer to what is currently on screen instead
of where the target was when the frame was captured.

Use this key only if the game overlay boxes appear over-corrected or under-corrected:

```ini
game_overlay_compensate_latency = false
```

This setting affects the visual overlay correction. It does not switch input
methods and does not create a fallback control path.

## Data Collection Guide

Data collection can be useful for model work but can affect performance:

```ini
collect_data_while_playing = false
collect_only_when_targets_present = true
collect_save_every_n_frames = 15
collect_jpeg_quality = 95
auto_label_data = true
```

If capture diagnostics show `need_cpu_copy=true`, data collection may be one reason.

## Build Workflow for Everyday Changes

Use the full builder when setting up a backend or changing dependencies:

```powershell
.\BUILDER.bat
```

Use the no-options builder after the backend already builds and you only changed app code:

```powershell
.\build_no-options.bat
```

The no-options builder only asks DML or CUDA and then builds the existing CMake target. It does not download, restore, update, or rebuild OpenCV.

## Common Recipes

### Lowest Setup Burden

```ini
backend = DML
ai_model = your_model.onnx
input_method = WIN32
circle_fov_enabled = true
```

### CUDA/TensorRT Performance Test

```ini
backend = TRT
ai_model = your_model.engine
capture_method = duplication_api
capture_use_cuda = true
show_window = false
collect_data_while_playing = false
```

Then check `[CaptureDiag]` output.

### Provider Benchmark

Use this when you want a repeatable ONNX Runtime provider comparison without starting capture, overlay, input devices, or file logging:

```powershell
.\ai.exe --benchmark-providers
.\ai.exe --benchmark-providers cpu,dml-gpu --bench-runs 200 --bench-warmup 20
.\ai.exe --benchmark-providers dml-gpu,dml-cpu --bench-model models\your_model.onnx
.\ai.exe --benchmark-providers cuda --bench-model models\your_model.onnx --bench-cuda-model models\your_model.engine
```

The benchmark prints one final CSV-style summary in seconds. Providers are `cpu`, `cuda`, `dml-gpu`, and `dml-cpu`; `cuda` uses the TensorRT engine path in CUDA builds, and `dml-cpu` uses DirectML through the WARP software adapter. If no model is passed, the benchmark skips ONNX candidates that cannot initialize on the requested available providers.

Each benchmark run appends rows to `benchmark_results\provider_benchmark.csv` under the repository root with the commit id, dirty flag, `model_family`, `onnx_model`, `cuda_engine_model`, `provider_model`, provider, and timing data. For `cuda`, `provider_model` is the `.engine` actually used by TensorRT. Use `--bench-no-save` for a disposable run.

### Razer Control Test

```ini
input_method = RAZER
```

Make sure `rzctl.dll` is beside `ai.exe` or in:

```text
sunone_aimbot_2
```

### Teensy Control Test

```ini
input_method = TEENSY41_HID
teensy_hid_serial = AUTO
teensy_hid_vid_filter = AUTO
teensy_hid_pid_filter = AUTO
```

Start broad with `AUTO`, then narrow filters after the device is confirmed.

## When Something Feels Wrong

Use this order:

1. Confirm build family: DML or CUDA.
2. Confirm model type: `.onnx` for DML, `.engine` for TensorRT, or `.onnx` when you want CUDA to generate a new engine.
3. Confirm capture is showing the expected content.
4. Confirm selected `input_method` has its required runtime/device.
5. Turn on useful logs or diagnostics.
6. Rebuild through the matching wrapper and run a real DML or CUDA smoke test after source changes.
