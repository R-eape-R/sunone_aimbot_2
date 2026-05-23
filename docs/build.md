# Build From Source

The supported automated build path is CMake with `Ninja Multi-Config`.
The batch files remain as compatibility wrappers, but the real entry points are:

```powershell
powershell -ExecutionPolicy Bypass -File tools/build_dml.ps1
powershell -ExecutionPolicy Bypass -File tools/build_cuda.ps1
```

Both scripts ask:

1. `OpenCV already built?`
2. `Download or update needed files?`

Use `-NonInteractive`, `-OpenCvAlreadyBuilt $true`, and
`-DownloadOrUpdateNeeded $true` for unattended runs.

## Toolchain

Required:

* Windows 10 or 11 x64
* Visual Studio with C++ build tools
* CMake

The scripts use `vswhere.exe` and `VsDevCmd.bat` to initialize the MSVC
environment. Ninja is resolved from PATH, Visual Studio's bundled copy, or a
repo-local cache under `tools/.bin`.

## Dependency Setup

NuGet packages are restored into `packages/`. In default mode the pinned
versions from `sunone_aimbot_2/packages.config` are restored. With
`-UseLatestPackages`, the newest package versions are installed into `packages/`
without rewriting Visual Studio project files.

Required source modules are also prepared automatically when download/update is
enabled:

* `sunone_aimbot_2/modules/SimpleIni.h`
* `sunone_aimbot_2/modules/serial`

The build writes:

```text
build/dependency-resolution.json
build/dependency-downloads.json
build/dependency-downloads.md
```

The resolution file records which versions and paths were selected. The
download files are temporary checklists for files that need user-assisted
downloads, such as NVIDIA packages behind login or license prompts.

## NVIDIA Downloads

CUDA Toolkit may need administrator installation. TensorRT archives are copied
into `sunone_aimbot_2/modules/_downloads` and extracted under
`sunone_aimbot_2/modules`.

When browser-assisted downloads are needed, run:

```powershell
powershell -ExecutionPolicy Bypass -File tools/build_cuda.ps1 -OpenBrowserForDownloads
```

Download the listed NVIDIA files to your Downloads folder, then return to the
terminal and continue. The script scans Downloads, copies matching files into
`sunone_aimbot_2/modules/_downloads`, and installs or extracts them.

## OpenCV Layout

OpenCV builds are backend-specific:

```text
sunone_aimbot_2/modules/opencv/
  build/
    dml/
      include/opencv2/opencv.hpp
      x64/vc*/lib/opencv_world*.lib
      x64/vc*/bin/opencv_world*.dll
    cuda/
      install/
        include/opencv2/opencv.hpp
        x64/vc*/lib/opencv_world*.lib
        x64/vc*/bin/opencv_world*.dll
```

CMake detects the available `opencv_world*.lib` and matching DLL dynamically.
The library name is no longer pinned to one OpenCV version.

## DML Build

```powershell
powershell -ExecutionPolicy Bypass -File tools/build_dml.ps1
```

The DML build uses the prebuilt OpenCV Windows package and installs its build
layout into `sunone_aimbot_2/modules/opencv/build/dml`.

## CUDA Build

```powershell
powershell -ExecutionPolicy Bypass -File tools/build_cuda.ps1 -OpenBrowserForDownloads
```

The CUDA build requires CUDA Toolkit, TensorRT, ONNX Runtime, and a CUDA-enabled
OpenCV build. The helper builds OpenCV with CUDA under
`sunone_aimbot_2/modules/opencv/build/cuda` and installs it under
`sunone_aimbot_2/modules/opencv/build/cuda/install`.

cuDNN is optional for this project. The program uses TensorRT and CUDA/OpenCV
preprocessing, not OpenCV DNN inference. When cuDNN is absent, the OpenCV CUDA
helper disables both `WITH_CUDNN` and `OPENCV_DNN_CUDA`.

For CUDA 13.2 and newer CCCL layouts, the OpenCV helper also patches the cloned
OpenCV contrib `cudev` header before configuring. This keeps OpenCV 4.13.0's
CUDA modules building when NVIDIA's libcu++ namespace helper macros use their
new CCCL names.

Useful options:

```powershell
powershell -ExecutionPolicy Bypass -File tools/build_cuda.ps1 -CudaArchBin 8.6
powershell -ExecutionPolicy Bypass -File tools/build_cuda.ps1 -CudaArchBin all
powershell -ExecutionPolicy Bypass -File tools/build_cuda.ps1 -UseLatestPackages
```

Run the produced executable from:

```text
build/dml/Release/ai.exe
build/cuda/Release/ai.exe
```
