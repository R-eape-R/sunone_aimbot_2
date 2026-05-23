param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
)

$ErrorActionPreference = 'Stop'

function Read-Source([string]$RelativePath) {
    Get-Content -Raw -LiteralPath (Join-Path $RepoRoot $RelativePath)
}

function Assert-Contains([string]$Text, [string]$Pattern, [string]$Message) {
    if ($Text -notmatch $Pattern) {
        throw $Message
    }
}

function Assert-NotContains([string]$Text, [string]$Pattern, [string]$Message) {
    if ($Text -match $Pattern) {
        throw $Message
    }
}

function Assert-FileExists([string]$Path, [string]$Message) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw $Message
    }
}

function Assert-Order([string]$Text, [string]$First, [string]$Second, [string]$Message) {
    $firstIndex = $Text.IndexOf($First)
    $secondIndex = $Text.IndexOf($Second)
    if ($firstIndex -lt 0 -or $secondIndex -lt 0 -or $firstIndex -gt $secondIndex) {
        throw $Message
    }
}

$config = Read-Source 'sunone_aimbot_2/config/config.cpp'
Assert-Order $config 'std::defaultfloat << std::setprecision(6)' 'file << "[Games]\n";' `
    'Config saves must restore floating precision before writing game profiles.'
Assert-Contains $config 'show_fps\s*=\s*get_bool\("show_fps",\s*false\)' `
    'Config load must read show_fps before save can preserve it.'
Assert-Contains $config 'circle_mask\s*=\s*hasCircleFovSetting\s*\?\s*legacyCircleMask\s*:\s*false' `
    'Legacy pixel circle masks must migrate to the lightweight circle FOV instead of forcing CPU capture.'
Assert-Contains $config 'circle_fov_radius_percent\s*=\s*get_long\("circle_fov_radius_percent",\s*100\)' `
    'Circle FOV radius must be configurable and default to the old full mask size.'

$main = Read-Source 'sunone_aimbot_2/sunone_aimbot_2.cpp'
Assert-Contains $main 'trt_detector\.requestStop\(\);\s*trt_detThread\.join\(\);' `
    'CUDA shutdown must signal the TRT inference thread before joining it.'
Assert-Order $main 'gameOverlayShouldExit.store(true);' 'delete dml_detector;' `
    'Game overlay must stop before dml_detector is deleted.'

$trtHeader = Read-Source 'sunone_aimbot_2/detector/trt_detector.h'
Assert-Contains $trtHeader 'void\s+requestStop\(\);' `
    'TrtDetector must expose a stop request method for clean shutdown.'

$trt = Read-Source 'sunone_aimbot_2/detector/trt_detector.cpp'
Assert-Contains $trt 'void\s+TrtDetector::requestStop\(\)' `
    'TrtDetector requestStop implementation is required.'
Assert-Contains $trt 'detectionBuffer\.version\+\+;\s*detectionBuffer\.cv\.notify_all\(\);' `
    'TRT pause path must publish cleared detections to consumers.'
Assert-Contains $trt 'void\s+TrtDetector::copyCpuTensorToDevice' `
    'TRT preprocessing must prepare the input tensor on CPU before copying it to CUDA.'
Assert-Contains $trt 'frame\.download\(cpuDownloadedFrame\);' `
    'TRT GPU frames must download to CPU before preprocessing.'
Assert-NotContains $trt 'launch_hwc_to_chw_norm' `
    'TRT preprocessing must not use the CUDA HWC-to-CHW kernel.'
Assert-Contains $trt 'filterDetectionsByCircleFov\(detections\)' `
    'TRT detections must support lightweight circle FOV filtering without pixel masking.'

$dml = Read-Source 'sunone_aimbot_2/detector/dml_detector.cpp'
Assert-Contains $dml 'filterDetectionsByCircleFov\(filteredDetections\)' `
    'DML detections must support lightweight circle FOV filtering without pixel masking.'

$ddaHeader = Read-Source 'sunone_aimbot_2/capture/duplication_api_capture.h'
Assert-Contains $ddaHeader 'bool\s+isInitialized\(\)\s+const' `
    'Duplication API capture must expose initialization status.'
Assert-Contains $ddaHeader 'enum class GpuCaptureStatus' `
    'CUDA capture diagnostics must preserve GPU capture failure reasons.'

$udpHeader = Read-Source 'sunone_aimbot_2/capture/udp_capture.h'
Assert-Contains $udpHeader 'bool\s+isInitialized\(\)\s+const' `
    'UDP capture must expose initialization status.'

$capture = Read-Source 'sunone_aimbot_2/capture/capture.cpp'
Assert-Contains $capture 'capture->isInitialized\(\)' `
    'createCapturer must reject failed capture backend objects.'
Assert-Contains $capture '\[CaptureDiag\]' `
    'Capture diagnostics must log CUDA direct-capture counters.'
Assert-Contains $capture 'GetNextFrameGpu\(screenshotGpu,\s*&gpuStatus\)' `
    'Capture diagnostics must record GPU capture status.'

$circleFov = Read-Source 'sunone_aimbot_2/capture/circle_fov.h'
Assert-Contains $circleFov 'pointInsideCircleFov' `
    'Circle FOV helper must use point math instead of per-frame pixel masking.'

$kmbox = Read-Source 'sunone_aimbot_2/mouse/kmbox_net/kmboxNet.cpp'
Assert-Contains $kmbox 'SO_RCVTIMEO' `
    'kmboxNet sockets must use receive timeouts.'
Assert-Contains $kmbox 'std::mutex\s+g_kmNetCommandMutex' `
    'kmboxNet command state must be serialized.'
Assert-Contains $kmbox 'std::memmove\(&softkeyboard\.button\[0\],\s*&softkeyboard\.button\[1\],\s*9\)' `
    'kmboxNet key queue full shift must not read beyond button[9].'
Assert-Contains $kmbox 'std::memmove\(&softkeyboard\.button\[i\],\s*&softkeyboard\.button\[i \+ 1\],\s*9 - i\)' `
    'kmboxNet key release shift must not read beyond button[9].'
Assert-Contains $kmbox 'if \(ret >= static_cast<int>\(sizeof\(hw_mouse\) \+ sizeof\(hw_keyboard\)\)\)' `
    'kmboxNet monitor must validate datagram size before copying reports.'

$arduinoHeader = Read-Source 'sunone_aimbot_2/mouse/Arduino.h'
Assert-Contains $arduinoHeader 'std::atomic<bool>\s+aiming_active' `
    'Arduino active button state must be atomic across listener and keyboard threads.'

$arduino = Read-Source 'sunone_aimbot_2/mouse/Arduino.cpp'
Assert-Order $arduino 'listening_thread_.join();' 'serial_.close();' `
    'Arduino listener thread must stop before closing the serial object.'

$overlayDirty = Read-Source 'sunone_aimbot_2/overlay/config_dirty.h'
Assert-Contains $overlayDirty 'OverlayConfig_SaveNow' `
    'Overlay config dirty helper must support forced save on hide/shutdown.'

$overlay = Read-Source 'sunone_aimbot_2/overlay/overlay.cpp'
Assert-Contains $overlay 'OverlayConfig_SaveNow\(\);' `
    'Overlay must flush pending config changes when it is hidden.'

$gameOverlay = Read-Source 'sunone_aimbot_2/runtime/game_overlay_loop.cpp'
Assert-Contains $gameOverlay 'GetMonitorHandleByIndex\(overlayMonitorIndex\)' `
    'Game overlay must use the configured capture monitor instead of always choosing primary.'
Assert-Contains $gameOverlay 'SetWindowBounds\(pr\.left,\s*pr\.top,\s*pw,\s*ph\)' `
    'Game overlay window bounds must honor monitor offsets.'

$depth = Read-Source 'sunone_aimbot_2/depth/depth_utils.cpp'
Assert-Contains $depth 'rgb\.copyTo\(out\(cv::Rect\(xOffset,\s*yOffset,\s*rgb\.cols,\s*rgb\.rows\)\)\)' `
    'Depth resize must letterbox resized content instead of stretching it.'

$repoRoot = Split-Path -Parent $PSScriptRoot
$buildCommonPath = Join-Path $repoRoot 'tools/build_common.ps1'
$buildCudaPsPath = Join-Path $repoRoot 'tools/build_cuda.ps1'
$buildDmlPsPath = Join-Path $repoRoot 'tools/build_dml.ps1'
$builderPath = Join-Path $repoRoot 'BUILDER.ps1'
$builderBatPath = Join-Path $repoRoot 'BUILDER.bat'
$buildOpenCvCudaPath = Join-Path $repoRoot 'tools/build_opencv_cuda.ps1'
$setupOpenCvDmlPath = Join-Path $repoRoot 'tools/setup_opencv_dml.ps1'
$cmakePath = Join-Path $repoRoot 'CMakeLists.txt'

Assert-FileExists $buildCommonPath 'Shared build automation helper must exist.'
Assert-FileExists $buildCudaPsPath 'CUDA PowerShell build entry point must exist.'
Assert-FileExists $buildDmlPsPath 'DML PowerShell build entry point must exist.'
Assert-FileExists $builderPath 'Root BUILDER launcher must exist.'
Assert-FileExists $builderBatPath 'Root double-click BUILDER batch launcher must exist.'

$buildCommon = Get-Content -LiteralPath $buildCommonPath -Raw
$buildCudaPs = Get-Content -LiteralPath $buildCudaPsPath -Raw
$buildDmlPs = Get-Content -LiteralPath $buildDmlPsPath -Raw
$builder = Get-Content -LiteralPath $builderPath -Raw
$builderBat = Get-Content -LiteralPath $builderBatPath -Raw
$buildOpenCvCuda = Get-Content -LiteralPath $buildOpenCvCudaPath -Raw
$setupOpenCvDml = Get-Content -LiteralPath $setupOpenCvDmlPath -Raw
$cmake = Get-Content -LiteralPath $cmakePath -Raw

Assert-Contains $buildCommon 'dependency-downloads\.json' `
    'Dependency automation must write a temporary download manifest.'
Assert-Contains $buildCommon '\[void\]\(Read-Host "Download the listed files' `
    'Guided dependency downloads must not leak Read-Host input into pipeline output.'
Assert-Contains $buildCommon 'VsDevCmd\.bat' `
    'Build automation must bootstrap the Visual Studio developer environment.'
Assert-Contains $buildCommon 'Get-BestCompatibleCudaDependencySet' `
    'Build automation must choose a best-compatible CUDA dependency set.'
Assert-Contains $buildCommon 'sunone_aimbot_2\\modules\\_downloads' `
    'Downloaded third-party archives must be cached under modules.'
Assert-Contains $buildCommon 'tools\\\.bin' `
    'Build automation must have a repo-local tool cache for helper tools.'
Assert-Contains $buildCommon 'Ensure-CoreSourceModules' `
    'Build automation must prepare required source modules automatically.'
Assert-Contains $buildCommon 'brofield/simpleini/master/SimpleIni\.h' `
    'Build automation must download SimpleIni.h when it is missing.'
Assert-Contains $buildCommon 'github\.com/wjwwood/serial\.git' `
    'Build automation must clone the serial module when it is missing.'
Assert-Contains $buildCommon '\$Configuration -eq ''Debug''' `
    'OpenCV PowerShell detection must avoid Release builds selecting debug OpenCV libraries.'

Assert-Contains $buildCudaPs 'Ninja Multi-Config' `
    'CUDA project build must use Ninja Multi-Config by default.'
Assert-Contains $buildCudaPs 'Resolve-OptionalBoolean .*OpenCV already built' `
    'CUDA build must prompt whether OpenCV is already built.'
Assert-Contains $buildCudaPs 'Resolve-OptionalBoolean .*Download or update needed files' `
    'CUDA build must prompt whether dependency downloads or updates are needed.'
Assert-Contains $buildCudaPs 'modules\\opencv\\build\\cuda\\install' `
    'CUDA OpenCV install root must live under modules/opencv/build/cuda/install.'
Assert-Contains $buildCudaPs 'TensorRT-10\*\.Windows\*\.zip' `
    'CUDA dependency guidance must prefer the TensorRT Windows binary SDK archive.'
Assert-Contains $buildCudaPs 'TensorRT-10\*\.win10\*\.zip' `
    'CUDA dependency guidance must accept older TensorRT Windows SDK archive naming.'
Assert-NotContains $buildCudaPs '"TensorRT-10\*\.zip"' `
    'CUDA dependency guidance must not accept TensorRT source-only zip archives.'
Assert-Contains $buildCudaPs 'valid Windows SDK layout' `
    'CUDA dependency setup must explain invalid TensorRT source archive layouts.'

Assert-Contains $buildDmlPs 'Ninja Multi-Config' `
    'DML project build must use Ninja Multi-Config by default.'
Assert-Contains $buildDmlPs 'Resolve-OptionalBoolean .*OpenCV already built' `
    'DML build must prompt whether OpenCV is already built.'
Assert-Contains $buildDmlPs 'Resolve-OptionalBoolean .*Download or update needed files' `
    'DML build must prompt whether dependency downloads or updates are needed.'
Assert-Contains $buildDmlPs 'modules\\opencv\\build\\dml' `
    'DML OpenCV root must live under modules/opencv/build/dml.'

Assert-Contains $builder 'Select build backend' `
    'BUILDER must prompt for DML or CUDA.'
Assert-Contains $builder 'tools\\build_dml\.ps1' `
    'BUILDER must dispatch DML builds to tools/build_dml.ps1.'
Assert-Contains $builder 'tools\\build_cuda\.ps1' `
    'BUILDER must dispatch CUDA builds to tools/build_cuda.ps1.'
Assert-Contains $builder 'No backend was selected' `
    'BUILDER must handle closed double-click prompts without a null dereference.'
Assert-Contains $builder 'Usage:' `
    'BUILDER must support help output.'
Assert-Contains $builder '\$forwardedArgs' `
    'BUILDER must normalize common child build flags before dispatch.'
Assert-Contains $builder 'OpenCvAlreadyBuilt' `
    'BUILDER must forward OpenCV build state to backend scripts.'
Assert-Contains $builder 'DownloadOrUpdateNeeded' `
    'BUILDER must forward dependency download choices to backend scripts.'
Assert-Contains $builder 'DryRun' `
    'BUILDER must support non-destructive backend dry runs.'
Assert-Contains $builder '@forwardedArgs @BuildArgs' `
    'BUILDER must pass normalized flags before raw extra backend arguments.'
Assert-Contains $builderBat 'Double-click deployment launcher' `
    'BUILDER.bat must identify itself as a double-click deployment launcher.'
Assert-Contains $builderBat 'pause' `
    'BUILDER.bat must keep the console open after double-click runs.'
Assert-Contains $builderBat 'Deployment complete' `
    'BUILDER.bat must show a clear completion message.'

Assert-Contains $buildOpenCvCuda 'Ninja Multi-Config' `
    'OpenCV CUDA helper must use Ninja Multi-Config by default.'
Assert-Contains $buildOpenCvCuda 'modules\\opencv\\build\\cuda' `
    'OpenCV CUDA helper must build under modules/opencv/build/cuda.'
Assert-Contains $buildOpenCvCuda 'OPENCV_DNN_CUDA=OFF' `
    'OpenCV CUDA helper must disable OpenCV DNN CUDA when cuDNN is disabled.'
Assert-Contains $buildOpenCvCuda 'Repair-OpenCvCudevZipHeader' `
    'OpenCV CUDA helper must patch cudev for CUDA 13.2 CCCL namespace macros.'
Assert-Contains $buildOpenCvCuda '_CCCL_BEGIN_NAMESPACE_CUDA_STD' `
    'OpenCV CUDA helper must support CUDA 13.2 renamed libcu++ namespace macros.'

Assert-Contains $setupOpenCvDml 'modules\\opencv\\build\\dml' `
    'DML OpenCV helper must install into modules/opencv/build/dml.'
Assert-Contains $setupOpenCvDml 'opencv_world\*\.lib' `
    'DML OpenCV helper must detect OpenCV world library version dynamically.'
Assert-Contains $setupOpenCvDml 'opencv-"\s*\+\s*\$OpenCvVersion\s*\+\s*"-extract' `
    'DML OpenCV extraction must use a scratch folder outside the target build root.'
Assert-Contains $setupOpenCvDml 'Copy-Item -Path' `
    'DML OpenCV install copy must expand the extracted build wildcard.'

Assert-Contains $cmake 'CMAKE_GENERATOR MATCHES "\^Ninja"' `
    'CMake must allow Ninja generators.'
Assert-Contains $cmake 'opencv_world\*\.lib' `
    'CMake must detect OpenCV world library version dynamically.'
Assert-Contains $cmake '_stem MATCHES "d\$"' `
    'CMake OpenCV detection must avoid Release builds selecting debug OpenCV libraries.'
Assert-Contains $cmake 'modules/opencv/build/dml' `
    'CMake DML OpenCV default must use modules/opencv/build/dml.'
Assert-Contains $cmake 'modules/opencv/build/cuda/install' `
    'CMake CUDA OpenCV default must use modules/opencv/build/cuda/install.'
Assert-NotContains $cmake 'cuda_preprocess\.cu' `
    'CUDA app builds must not compile the CUDA preprocessing kernel when preprocessing is CPU-based.'

Write-Host 'Regression checks passed.'
