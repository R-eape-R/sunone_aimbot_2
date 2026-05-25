# Build Guide

This guide explains the current build system in plain terms first, then adds the technical details needed for local development.

## Which Builder Should I Use?

| Tool | Best for | What it does |
|---|---|---|
| `BUILDER.bat` | Most people building from source. | Opens an interactive choice for DML or CUDA, prepares dependencies, configures CMake, and builds. |
| `BUILDER.ps1` | Same as above, but easier to script. | Lets you pass `-Backend DML` or `-Backend CUDA` directly. |
| `build_no-options.bat` | Fast local rebuilds after the full build already worked once. | Rebuilds `ai` from an existing build tree. No downloads, updates, NuGet restore, OpenCV setup, or dependency prompts. |
| `tools/build_dml.ps1` | Direct DML backend build. | Restores DML/ONNX dependencies and prepares the DML build tree. |
| `tools/build_cuda.ps1` | Direct CUDA/TensorRT backend build. | Resolves CUDA/TensorRT/OpenCV CUDA dependencies and prepares the CUDA build tree. |

If you are unsure, run `BUILDER.bat`.

## Requirements

### Required for Both Builds

- Windows 10 or Windows 11.
- Visual Studio 2022 or Build Tools with the **Desktop development with C++** workload.
- Windows SDK with C++/WinRT headers.
- CMake.
- PowerShell.

The builder can download or restore some project dependencies when you allow it. It cannot install Visual Studio for you.

### DML Build Requirements

- A Windows GPU/driver that supports DirectML.
- NuGet packages restored by the build scripts:
  - `Microsoft.ML.OnnxRuntime.DirectML`
  - `Microsoft.AI.DirectML`
- OpenCV DML layout under:

```text
sunone_aimbot_2\modules\opencv\build\dml
```

The DML builder can prepare the OpenCV layout when downloads are enabled.

### CUDA Build Requirements

- NVIDIA GPU.
- CUDA Toolkit.
- TensorRT 10 Windows binary SDK.
- OpenCV built with CUDA under:

```text
sunone_aimbot_2\modules\opencv\build\cuda\install
```

The CUDA builder can guide dependency downloads and build OpenCV CUDA when downloads are enabled. OpenCV CUDA builds are large and can take a while.

## Full Interactive Build

From the repository root:

```powershell
.\BUILDER.bat
```

Choose:

- `DML` for the DirectML build.
- `CUDA` for the CUDA + TensorRT build.

The batch file keeps the console open after a double-click run so you can read errors.

## Full Scripted Build

DML:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\BUILDER.ps1 -Backend DML
```

CUDA:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\BUILDER.ps1 -Backend CUDA
```

Useful options:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\BUILDER.ps1 -Backend CUDA -OpenCvAlreadyBuilt true
powershell -NoProfile -ExecutionPolicy Bypass -File .\BUILDER.ps1 -Backend CUDA -DownloadOrUpdateNeeded false
powershell -NoProfile -ExecutionPolicy Bypass -File .\BUILDER.ps1 -Backend DML -DryRun
```

`BUILDER.ps1` forwards backend-specific arguments to `tools/build_dml.ps1` or `tools/build_cuda.ps1`.

## No-Options Rebuild

Use this after the full builder has already prepared the selected backend at least once:

```powershell
.\build_no-options.bat
```

or:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\build_no-options.ps1 -Backend DML
powershell -NoProfile -ExecutionPolicy Bypass -File .\build_no-options.ps1 -Backend CUDA
```

This script intentionally does only this:

```text
cmake --build build\<backend> --config Release --target ai --parallel
```

It is useful when you changed C++ or UI code and want a quick rebuild. It is not a dependency setup tool. If the build tree is missing, stale, or configured for the wrong dependency paths, run `BUILDER.bat` again.

## Direct Backend Scripts

DML:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\build_dml.ps1
```

CUDA:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\build_cuda.ps1
```

Examples:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\build_cuda.ps1 -CudaArchBin 8.6
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\build_cuda.ps1 -CudaArchBin all
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\build_cuda.ps1 -SkipOpenCvBuild
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\build_dml.ps1 -UseLatestPackages
```

`-CudaArchBin all` expands to:

```text
7.5;8.0;8.6;8.7;8.8;8.9;9.0;10.0;10.3;11.0;12.0;12.1
```

## Output Folders

Default outputs:

```text
build\dml\Release\ai.exe
build\cuda\Release\ai.exe
```

The output folder also receives runtime DLLs and selected assets that the executable needs at runtime.

## Runtime Files Copied by CMake

When `AIMBOT_COPY_RUNTIME_DLLS` is enabled, CMake copies available runtime DLLs next to `ai.exe`, including:

- OpenCV runtime DLL.
- `ghub_mouse.dll`, when present.
- `chroma_lighting.dll`, when present.
- ONNX Runtime DLLs.
- DirectML DLL for DML support.
- TensorRT and CUDA provider DLLs for CUDA builds.
- cuDNN DLL if a compatible cuDNN layout is found.

The Razer control DLL is expected at:

```text
sunone_aimbot_2\modules\razer-controls\x64\Release\chroma_lighting.dll
```

You can override its CMake cache path with `AIMBOT_RAZER_CONTROL_DLL`.

## Important CMake Options

| Option | Meaning |
|---|---|
| `AIMBOT_USE_CUDA` | `ON` for CUDA + TensorRT, `OFF` for DML. |
| `AIMBOT_COPY_RUNTIME_DLLS` | Copies runtime DLLs beside `ai.exe`. |
| `AIMBOT_RAZER_CONTROL_DLL` | Source path for `chroma_lighting.dll`. |
| `AIMBOT_OPENCV_DML_ROOT` | DML OpenCV layout root. |
| `AIMBOT_OPENCV_CUDA_ROOT` | CUDA OpenCV layout root. |
| `AIMBOT_TENSORRT_ROOT` | TensorRT SDK root. |
| `AIMBOT_CUDNN_ROOT` | Optional cuDNN root. |

## CUDA Build Notes

The CUDA build uses TensorRT for the main TRT backend and still links ONNX Runtime/DirectML libraries because DML-related runtime paths and tooling remain available in the shared project.

For best performance:

- Use a TensorRT `.engine` model with the CUDA build.
- Keep the legacy `circle_mask` off unless you specifically need it.
- Use Circle FOV instead of the old pixel mask for normal FOV limiting.
- Turn off preview/debug windows when measuring capture performance, because preview requires CPU-readable pixels.

The current CUDA preprocess path can intentionally run CPU preprocessing and then copy the tensor to CUDA input. That is useful for diagnosis and compatibility, but pure GPU paths are faster when available.

## DML Build Notes

The DML build expects ONNX models. A TensorRT `.engine` model will not run through the DML backend.

If DML launches but does not detect targets:

- Check that the selected AI model is `.onnx`.
- Lower `confidence_threshold` temporarily.
- Confirm the configured class IDs match the model.
- Confirm `backend = DML`.
- Check `dml_device_id` if the system has more than one GPU.

## Validation

Run the regression checks after build-system or integration changes:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\regression_checks.ps1
```

This checks important source contracts such as:

- Razer and Teensy runtime wiring.
- no-options builder behavior.
- training asset packaging.
- Circle FOV and legacy mask behavior.
- frame-age latency compensation while preserving detection confidences.
- config and UI integration contracts.

## Troubleshooting

| Problem | What to check |
|---|---|
| `Build tree not found` from `build_no-options` | Run `BUILDER.bat` first for that backend. |
| CUDA dependency missing | Install CUDA Toolkit and TensorRT Windows binary SDK, then rerun the CUDA builder. |
| OpenCV CUDA compile errors | Check CUDA, Visual Studio compiler, OpenCV, and contrib compatibility. If you only need CPU preprocess for diagnosis, keep legacy GPU-heavy capture features off and rebuild the main app after the dependency tree is healthy. |
| DML build cannot find OpenCV | Run the DML builder with downloads enabled or set OpenCV CMake paths manually. |
| Missing `chroma_lighting.dll` | Build or copy the Razer controls DLL to `sunone_aimbot_2\modules\razer-controls\x64\Release\chroma_lighting.dll` or next to `ai.exe`. |
| Training bootstrap fails | Install training requirements or place the expected `.onnx` files in `sunone_aimbot_2\modules\training\models`. |
| App runs but Razer or Teensy control does nothing | The selected control method does not fall back. Check the selected `input_method`, the Razer DLL, or the Teensy RawHID endpoint. |
