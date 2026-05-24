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
$configHeader = Read-Source 'sunone_aimbot_2/config/config.h'
Assert-Order $config 'std::defaultfloat << std::setprecision(6)' 'file << "[Games]\n";' `
    'Config saves must restore floating precision before writing game profiles.'
Assert-Contains $config 'show_fps\s*=\s*get_bool\("show_fps",\s*false\)' `
    'Config load must read show_fps before save can preserve it.'
Assert-Contains $config 'circle_mask\s*=\s*hasCircleFovSetting\s*\?\s*legacyCircleMask\s*:\s*false' `
    'Legacy pixel circle masks must migrate to the lightweight circle FOV instead of forcing CPU capture.'
Assert-Contains $config 'circle_fov_radius_percent\s*=\s*get_long\("circle_fov_radius_percent",\s*100\)' `
    'Circle FOV radius must be configurable and default to the old full mask size.'
Assert-Contains $configHeader 'bool\s+game_overlay_compensate_latency' `
    'Config header must expose game overlay latency compensation.'
Assert-Contains $config 'game_overlay_compensate_latency\s*=\s*true' `
    'Game overlay latency compensation must default enabled.'
Assert-Contains $config 'get_bool\("game_overlay_compensate_latency",\s*true\)' `
    'Config load must read game overlay latency compensation.'
Assert-Contains $config 'game_overlay_compensate_latency = ' `
    'Config save must persist game overlay latency compensation.'

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
Assert-Contains $trt 'detectionBuffer\.clear\(\);' `
    'TRT pause path must publish cleared detections to consumers.'
Assert-Contains $trt 'void\s+TrtDetector::copyCpuTensorToDevice' `
    'TRT CPU-frame preprocessing must prepare the input tensor on CPU before copying it to CUDA.'
Assert-Contains $trt 'launch_hwc_to_chw_norm' `
    'TRT GPU-frame preprocessing must keep fast-path frames on device and use the CUDA HWC-to-CHW kernel.'
Assert-NotContains $trt 'frame\.download\(cpuDownloadedFrame\);' `
    'TRT GPU-frame preprocessing must not force a GPU-to-CPU download in the fast path.'
Assert-Contains $trt 'filterDetectionsByCircleFov\(detections\)' `
    'TRT detections must support lightweight circle FOV filtering without pixel masking.'
Assert-Contains (Read-Source 'CMakeLists.txt') 'detector/cuda_preprocess\.cu' `
    'CUDA CMake target must compile cuda_preprocess.cu for the TRT GPU preprocess kernel.'

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
Assert-Contains $capture 'std::chrono::steady_clock::time_point\s+frameTimestamp' `
    'Capture must timestamp frames before preprocessing or detector submission.'
Assert-Contains $capture 'trt_detector\.processFrameGpu\(screenshotGpu,\s*frameTimestamp\)' `
    'TRT GPU capture path must pass the original frame timestamp into inference.'
Assert-Contains $capture 'dml_detector->processFrame\(detectionFrame,\s*screenshotCpu,\s*frameTimestamp\)' `
    'DML capture path must preserve frame timestamps through preprocessing.'
Assert-Contains $capture 'trt_detector\.processFrame\(detectionFrame,\s*screenshotCpu,\s*frameTimestamp\)' `
    'TRT CPU capture path must preserve frame timestamps through preprocessing.'

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

$teensyRawHidHeaderPath = Join-Path $repoRoot 'sunone_aimbot_2/mouse/Teensy41RawHid.h'
$teensyRawHidSourcePath = Join-Path $repoRoot 'sunone_aimbot_2/mouse/Teensy41RawHid.cpp'
$rzctlHeaderPath = Join-Path $repoRoot 'sunone_aimbot_2/mouse/rzctl.h'
$rzctlSourcePath = Join-Path $repoRoot 'sunone_aimbot_2/mouse/rzctl.cpp'
$razerDllPath = Join-Path $repoRoot 'sunone_aimbot_2/modules/razer-controls/x64/Release/chroma_lighting.dll'
Assert-FileExists $teensyRawHidHeaderPath 'TEENSY41_HID input must provide a RawHID header.'
Assert-FileExists $teensyRawHidSourcePath 'TEENSY41_HID input must provide a RawHID implementation.'
Assert-FileExists $rzctlHeaderPath 'RAZER input must provide a runtime rzctl wrapper header.'
Assert-FileExists $rzctlSourcePath 'RAZER input must provide a runtime rzctl wrapper implementation.'
Assert-FileExists $razerDllPath 'RAZER input must keep a release chroma_lighting.dll runtime available.'

$teensyRawHid = Get-Content -LiteralPath $teensyRawHidSourcePath -Raw
Assert-Contains $teensyRawHid 'hid_enumerate' `
    'TEENSY41_HID must enumerate RawHID devices through hidapi.'
Assert-Contains $teensyRawHid 'Teensy41RawHidPacketSize \+ 1' `
    'TEENSY41_HID must send report-id-prefixed 64 byte packets.'
Assert-Contains $teensyRawHid 'aiming\.store\(pressed' `
    'TEENSY41_HID physical side-button events must update global aiming state.'

$rzctlWrapper = Get-Content -LiteralPath $rzctlSourcePath -Raw
Assert-Contains $rzctlWrapper 'LoadLibraryW' `
    'RAZER input must load chroma_lighting.dll dynamically at runtime.'
Assert-Contains $rzctlWrapper 'mouse_move_status' `
    'RAZER input must prefer status-returning movement exports.'
Assert-Contains $rzctlWrapper 'chroma_lighting\.dll' `
    'RAZER input must load the DLL under the chroma_lighting.dll runtime name.'
Assert-Contains $rzctlWrapper 'modules.*razer-controls.*RazerControlDllName' `
    'RAZER input must search the modules/razer-controls runtime DLL path.'

$mouseHeader = Read-Source 'sunone_aimbot_2/mouse/mouse.h'
Assert-Contains $mouseHeader 'RzctlMouse\* rzctl' `
    'MouseThread must retain a Razer device pointer.'
Assert-Contains $mouseHeader 'Teensy41RawHid\* teensy41RawHid' `
    'MouseThread must retain a TEENSY41_HID device pointer.'
Assert-Contains $mouseHeader 'MotionCompensationSample' `
    'MouseThread must keep a short motion trail for latency compensation.'
Assert-Contains $mouseHeader 'getMotionCompensationSince' `
    'MouseThread must expose summed movement since a capture timestamp.'
Assert-Contains $mouseHeader 'std::chrono::steady_clock::time_point observationTime' `
    'MouseThread prediction and movement APIs must accept capture observation time.'

$mouseSource = Read-Source 'sunone_aimbot_2/mouse/mouse.cpp'
Assert-Contains $mouseSource 'const std::string inputMethod = config\.input_method' `
    'MouseThread must route by the selected input_method instead of falling through by available pointer.'
Assert-Order $mouseSource 'inputMethod == "TEENSY41_HID"' 'inputMethod == "KMBOX_NET"' `
    'TEENSY41_HID movement must route before network/software fallback devices.'
Assert-Order $mouseSource 'inputMethod == "RAZER"' 'inputMethod == "GHUB"' `
    'RAZER movement must route before GHub and Win32 fallback.'
Assert-NotContains $mouseSource 'else if \(gHub\)\s*\{\s*gHub->mouse_xy' `
    'Selected control backends must not fall through to GHub movement.'
Assert-Contains $mouseSource 'if \(inputMethod == "WIN32"\)' `
    'Win32 movement must only run when WIN32 is explicitly selected.'
Assert-Contains $mouseSource 'recordMotionCompensationStep\(dx,\s*dy\)' `
    'MouseThread must record actual submitted movement for latency compensation.'
Assert-Contains $mouseSource 'MouseThread::getMotionCompensationSince' `
    'MouseThread must be able to sum movement after a frame timestamp.'
Assert-Contains $mouseSource 'MouseThread::mouseCountsToScreenPixels' `
    'Motion compensation must use the same game-profile conversion as the wind trail.'
Assert-Contains $mouseSource 'currentDetectionDelaySec\(double observationAgeSec' `
    'Prediction delay compensation must use measured frame age when available.'

$keyboardListener = Read-Source 'sunone_aimbot_2/keyboard/keyboard_listener.cpp'
Assert-Contains $keyboardListener 'config\.input_method == "TEENSY41_HID"' `
    'TEENSY41_HID mouse-button input must be selected explicitly instead of falling back to Win32 mouse state.'
Assert-Contains $keyboardListener 'teensy41RawHid->aimingActive\(\)' `
    'Keyboard listener must consume Teensy RawHID physical side-button state.'

$overlayDirty = Read-Source 'sunone_aimbot_2/overlay/config_dirty.h'
Assert-Contains $overlayDirty 'OverlayConfig_SaveNow' `
    'Overlay config dirty helper must support forced save on hide/shutdown.'

$overlay = Read-Source 'sunone_aimbot_2/overlay/overlay.cpp'
Assert-Contains $overlay 'OverlayConfig_SaveNow\(\);' `
    'Overlay must flush pending config changes when it is hidden.'

$gameOverlay = Read-Source 'sunone_aimbot_2/runtime/game_overlay_loop.cpp'
Assert-Contains $gameOverlay 'GetMonitorHandleByIndex\(overlayMonitorIndex\)' `
    'Game overlay must use the configured capture monitor instead of always choosing primary.'
Assert-Contains $gameOverlay 'resolveGameOverlayMonitorBounds' `
    'Game overlay must recompute monitor bounds inside its render loop.'
Assert-Contains $gameOverlay 'lastOverlayMonitorIndex' `
    'Game overlay must track monitor changes while running.'
Assert-Contains $gameOverlay 'overlayMonitorChanged' `
    'Game overlay must refresh its window when monitor_idx changes at runtime.'
Assert-Contains $gameOverlay 'SetWindowBounds\(pr\.left,\s*pr\.top,\s*pw,\s*ph\)' `
    'Game overlay window bounds must honor monitor offsets.'
Assert-Contains $gameOverlay 'detectionTimestamp\s*=\s*detectionBuffer\.frameTimestamp' `
    'Game overlay must read the source frame timestamp for box projection.'
Assert-Contains $gameOverlay 'projectDetectionBox' `
    'Game overlay must project boxes through the latency compensation helper.'
Assert-Contains $gameOverlay 'getMotionCompensationSince\(compensationTimestamp\)' `
    'Game overlay must offset stale detection boxes by mouse movement since capture.'
Assert-Contains $gameOverlay 'velocityX' `
    'Game overlay must use track velocity when projecting debug boxes.'

$neuralHeaderPath = Join-Path $RepoRoot 'sunone_aimbot_2/neural/NeuralTracker.h'
$neuralSourcePath = Join-Path $RepoRoot 'sunone_aimbot_2/neural/NeuralTracker.cpp'
$drawNeuralPath = Join-Path $RepoRoot 'sunone_aimbot_2/overlay/draw_neural.cpp'
Assert-FileExists $neuralHeaderPath 'Neural tracker association must provide a runtime header.'
Assert-FileExists $neuralSourcePath 'Neural tracker association must provide a runtime implementation.'
Assert-FileExists $drawNeuralPath 'Overlay must expose neural tracker association controls.'

$neuralHeader = Get-Content -LiteralPath $neuralHeaderPath -Raw
$neuralSource = Get-Content -LiteralPath $neuralSourcePath -Raw
$drawNeural = Get-Content -LiteralPath $drawNeuralPath -Raw
$aimbotTargetHeader = Read-Source 'sunone_aimbot_2/mouse/AimbotTarget.h'
$aimbotTargetSource = Read-Source 'sunone_aimbot_2/mouse/AimbotTarget.cpp'
$detectionBuffer = Read-Source 'sunone_aimbot_2/detector/detection_buffer.h'
$mouseLoop = Read-Source 'sunone_aimbot_2/runtime/mouse_thread_loop.cpp'
$drawSettings = Read-Source 'sunone_aimbot_2/overlay/draw_settings.h'
$drawDebug = Read-Source 'sunone_aimbot_2/overlay/draw_debug.cpp'
$drawGameOverlay = Read-Source 'sunone_aimbot_2/overlay/draw_game_overlay.cpp'

Assert-Contains $neuralHeader 'NeuralTrackerFeatureCount\s*=\s*16' `
    'Neural tracker model contract must stay at 16 association features.'
Assert-Contains $neuralHeader 'class\s+INeuralTracker' `
    'Neural tracker must expose an optional scorer interface.'
Assert-Contains $neuralHeader 'createNeuralTracker\(const std::string& modelPath,\s*const std::string& runtime\)' `
    'Neural tracker factory must select CPU or CUDA runtime from config.'
Assert-Contains $neuralSource 'resolveNeuralModelPath' `
    'Neural tracker must resolve model paths relative to the app and working directory.'
Assert-Contains $neuralSource 'ONNX model missing' `
    'Missing neural models must leave the classical tracker active instead of failing startup.'
Assert-Contains $neuralSource 'class\s+TrtNeuralTracker' `
    'CUDA builds must provide a TensorRT neural tracker implementation.'
Assert-Contains $neuralSource 'buildEngineFromOnnx' `
    'CUDA neural tracker must build a TensorRT engine from ONNX when needed.'
Assert-Contains $neuralSource 'enqueueV3' `
    'CUDA neural tracker must run TensorRT inference through enqueueV3.'
Assert-Contains $neuralSource 'cudaMemcpyAsync' `
    'CUDA neural tracker must move association features and scores through CUDA buffers.'

Assert-Contains $config 'neural_tracker_enabled' `
    'Config must persist neural tracker enable state.'
Assert-Contains $config 'neural_tracker_runtime\s*=\s*"CPU"' `
    'Neural tracker runtime must default to CPU/ONNX so DML builds remain usable.'
Assert-Contains $config 'get_string\("neural_tracker_runtime",\s*"CPU"\)' `
    'Config load must read neural tracker runtime.'
Assert-Contains $config 'neural_tracker_runtime != "CPU" && neural_tracker_runtime != "CUDA"' `
    'Config must reject unsupported neural tracker runtimes.'
Assert-Contains $config 'neural_tracker_blend' `
    'Config must expose a neural/classical association blend.'
Assert-Contains $config 'neural_tracker_log_enabled' `
    'Config must expose neural tracker association logging.'
Assert-Contains $config 'neural_tracker_debug_enabled' `
    'Config must expose neural tracker overlay debug labels.'

Assert-Contains $configHeader 'bool\s+pid_governor_enabled' `
    'Config header must expose PID governor enable state.'
Assert-Contains $configHeader 'int\s+pid_governor_speed' `
    'Config header must expose integer PID governor speed.'
Assert-Contains $configHeader 'int\s+pid_governor_blend' `
    'Config header must expose integer PID governor blend percentage.'
Assert-Contains $configHeader 'int\s+pid_governor_lead_percent' `
    'Config header must expose integer PID governor target lead percentage.'
Assert-Contains $config 'pid_governor_enabled\s*=\s*false' `
    'PID governor must default disabled until runtime support is explicitly enabled.'
Assert-Contains $config 'pid_governor_speed\s*=\s*5' `
    'PID governor speed must have a conservative default inside the 1-100 range.'
Assert-Contains $config 'pid_governor_blend\s*=\s*50' `
    'PID governor blend must default to a midpoint percentage.'
Assert-Contains $config 'pid_governor_lead_percent\s*=\s*10' `
    'PID governor target lead must default to the requested plus-10-percent baseline.'
Assert-Contains $config 'get_bool\("pid_governor_enabled",\s*false\)' `
    'Config load must read PID governor enable state.'
Assert-Contains $config 'get_long\("pid_governor_speed",\s*5\)' `
    'Config load must read PID governor speed as an integer.'
Assert-Contains $config 'get_long\("pid_governor_blend",\s*50\)' `
    'Config load must read PID governor blend as an integer percentage.'
Assert-Contains $config 'get_long\("pid_governor_lead_percent",\s*10\)' `
    'Config load must read PID governor target lead as an integer percentage.'
Assert-Contains $config 'pid_governor_speed < 1' `
    'Config must clamp PID governor speed to the requested lower bound.'
Assert-Contains $config 'pid_governor_speed > 100' `
    'Config must clamp PID governor speed to the requested upper bound.'
Assert-Contains $config 'pid_governor_blend < 1' `
    'Config must clamp PID governor blend to the requested lower bound.'
Assert-Contains $config 'pid_governor_blend > 100' `
    'Config must clamp PID governor blend to the requested upper bound.'
Assert-Contains $config 'pid_governor_lead_percent < 0' `
    'Config must clamp PID governor target lead to the requested lower bound.'
Assert-Contains $config 'pid_governor_lead_percent > 50' `
    'Config must clamp PID governor target lead to the requested upper bound.'
Assert-Contains $config 'pid_governor_enabled = ' `
    'Config save must persist PID governor enable state.'
Assert-Contains $config 'pid_governor_speed = ' `
    'Config save must persist PID governor speed.'
Assert-Contains $config 'pid_governor_blend = ' `
    'Config save must persist PID governor blend.'
Assert-Contains $config 'pid_governor_lead_percent = ' `
    'Config save must persist PID governor target lead percentage.'
Assert-Contains $drawNeural 'ImGui::Checkbox\("Enable PID governor",\s*&config\.pid_governor_enabled\)' `
    'Neural tab must expose PID governor enable state.'
Assert-Contains $drawNeural 'ImGui::SliderInt\("Governor speed",\s*&config\.pid_governor_speed,\s*1,\s*100\)' `
    'Neural tab must expose PID governor speed as an integer 1-100 slider.'
Assert-Contains $drawNeural 'ImGui::SliderInt\("Governor blend",\s*&config\.pid_governor_blend,\s*1,\s*100\)' `
    'Neural tab must expose PID governor blend as an integer 1-100 slider.'
Assert-Contains $drawNeural 'ImGui::SliderInt\("Target lead %",\s*&config\.pid_governor_lead_percent,\s*0,\s*50\)' `
    'Neural tab must expose PID governor target lead as an integer 0-50 percent slider.'

Assert-Contains $detectionBuffer 'std::vector<float>\s+confidences' `
    'DetectionBuffer must publish detector confidence with each box for neural association.'
Assert-Contains $detectionBuffer 'std::chrono::steady_clock::time_point\s+frameTimestamp' `
    'DetectionBuffer must preserve the source capture timestamp for latency compensation.'
Assert-Contains $detectionBuffer 'std::chrono::steady_clock::time_point\s+publishTimestamp' `
    'DetectionBuffer must publish when detections became visible to consumers.'
Assert-Contains $detectionBuffer 'set\(\s*const std::vector<cv::Rect>& newBoxes,\s*const std::vector<int>& newClasses,\s*const std::vector<float>& newConfidences,\s*std::chrono::steady_clock::time_point newFrameTimestamp = \{\}' `
    'DetectionBuffer set must preserve confidences and accept an optional frame timestamp.'
Assert-Contains $detectionBuffer 'void\s+clear\(\)' `
    'DetectionBuffer must provide a single clear path that also clears confidences and timestamps.'
Assert-Contains $detectionBuffer 'outFrameTimestamp' `
    'DetectionBuffer get overloads must expose frame timestamps to consumers.'
Assert-Contains $dml 'currentFrameTimestamp' `
    'DML detector must store the timestamp associated with the queued frame.'
Assert-Contains $dml 'detectionBuffer\.set\(boxes,\s*classes,\s*confidences,\s*frameTimestamp\)' `
    'DML detector must publish per-detection confidences with the source timestamp.'
Assert-Contains $dml 'detectionBuffer\.clear\(\)' `
    'DML pause/reload clear paths must clear detections, confidences, and timestamps.'
Assert-Contains $trt 'currentFrameTimestamp' `
    'TRT detector must store the timestamp associated with the queued CPU/GPU frame.'
Assert-Contains $trt 'detectionBuffer\.set\(boxes,\s*classes,\s*confidences,\s*frameTimestamp\)' `
    'TRT detector must publish per-detection confidences with the source timestamp.'
Assert-Contains $trt 'detectionBuffer\.clear\(\)' `
    'TRT pause/reload clear paths must clear detections, confidences, and timestamps.'
Assert-Contains $mouseLoop 'detectionBuffer\.confidences' `
    'Mouse runtime must copy confidences from DetectionBuffer.'
Assert-Contains $mouseLoop 'detectionTimestamp\s*=\s*detectionBuffer\.frameTimestamp' `
    'Mouse runtime must copy source frame timestamps from DetectionBuffer.'
Assert-Contains $mouseLoop 'targetTracker\.update\(\s*boxes,\s*classes,\s*confidences,' `
    'Mouse runtime must pass confidences into the target tracker.'
Assert-Contains $mouseLoop 'detectionTimestamp\s*\)' `
    'Mouse runtime must pass detection timestamps into tracker and movement compensation.'

Assert-Contains $aimbotTargetHeader 'float\s+confidence' `
    'Tracked targets must retain detector confidence.'
Assert-Contains $aimbotTargetHeader 'lastNeuralScore' `
    'Track debug state must expose the most recent neural association score.'
Assert-Contains $aimbotTargetHeader 'velocityX' `
    'Track debug state must expose track velocity for overlay latency compensation.'
Assert-Contains $aimbotTargetHeader 'lastUpdate' `
    'Track debug state must expose update timestamps for overlay latency compensation.'
Assert-Contains $aimbotTargetHeader 'const std::vector<float>& confidences' `
    'MultiTargetTracker must accept per-detection confidences.'
Assert-Contains $aimbotTargetHeader 'std::chrono::steady_clock::time_point observationTime' `
    'MultiTargetTracker must accept capture observation time for frame-age-aware velocity.'
Assert-Contains $aimbotTargetSource 'neural/NeuralTracker\.h' `
    'MultiTargetTracker must use the neural tracker scorer.'
Assert-Contains $aimbotTargetSource 'buildNeuralFeatures' `
    'MultiTargetTracker must build the neural association feature vector.'
Assert-Contains $aimbotTargetSource 'neuralBonus' `
    'MultiTargetTracker must blend neural association scores into classical matching.'
Assert-Contains $aimbotTargetSource 'logNeuralTrackerAssociation' `
    'MultiTargetTracker must support CSV logging for association diagnostics.'

Assert-Contains $drawSettings 'void draw_neural\(\);' `
    'Overlay draw declarations must include the Neural tab.'
Assert-Contains $overlay '"Neural"' `
    'Overlay sidebar must include a Neural tab.'
Assert-Contains $drawNeural 'Association runtime' `
    'Neural overlay must let the user choose CPU or CUDA association runtime.'
Assert-Contains $drawNeural 'Association model' `
    'Neural overlay must let the user choose an association model.'
Assert-Contains $drawNeural 'Association blend' `
    'Neural overlay must let the user adjust neural/classical association blend.'
Assert-Contains $drawDebug 'neural_tracker_log_enabled' `
    'Debug overlay must expose neural association CSV logging.'
Assert-Contains $drawDebug 'neural_tracker_debug_enabled' `
    'Debug overlay must expose neural association debug labels.'
Assert-Contains $drawGameOverlay 'ImGui::Checkbox\("Compensate Overlay Latency",\s*&config\.game_overlay_compensate_latency\)' `
    'Game overlay settings must expose latency compensation.'

$vcxproj = Read-Source 'sunone_aimbot_2/sunone_aimbot_2.vcxproj'
$vcxprojFilters = Read-Source 'sunone_aimbot_2/sunone_aimbot_2.vcxproj.filters'
Assert-Contains $vcxproj 'neural\\NeuralTracker\.cpp' `
    'Visual Studio project must compile the neural tracker runtime.'
Assert-Contains $vcxproj 'overlay\\draw_neural\.cpp' `
    'Visual Studio project must compile the Neural overlay tab.'
Assert-Contains $vcxproj 'neural\\NeuralTracker\.h' `
    'Visual Studio project must include the neural tracker header.'
Assert-Contains $vcxprojFilters 'neural\\NeuralTracker\.cpp' `
    'Visual Studio filters must show the neural tracker runtime.'
Assert-Contains $vcxprojFilters 'overlay\\draw_neural\.cpp' `
    'Visual Studio filters must show the Neural overlay tab.'
Assert-Contains $vcxprojFilters 'neural\\NeuralTracker\.h' `
    'Visual Studio filters must show the neural tracker header.'

$depth = Read-Source 'sunone_aimbot_2/depth/depth_utils.cpp'
Assert-Contains $depth 'rgb\.copyTo\(out\(cv::Rect\(xOffset,\s*yOffset,\s*rgb\.cols,\s*rgb\.rows\)\)\)' `
    'Depth resize must letterbox resized content instead of stretching it.'

$repoRoot = Split-Path -Parent $PSScriptRoot
$buildCommonPath = Join-Path $repoRoot 'tools/build_common.ps1'
$buildCudaPsPath = Join-Path $repoRoot 'tools/build_cuda.ps1'
$buildDmlPsPath = Join-Path $repoRoot 'tools/build_dml.ps1'
$builderPath = Join-Path $repoRoot 'BUILDER.ps1'
$builderBatPath = Join-Path $repoRoot 'BUILDER.bat'
$buildNoOptionsPath = Join-Path $repoRoot 'build_no-options.ps1'
$buildNoOptionsBatPath = Join-Path $repoRoot 'build_no-options.bat'
$buildOpenCvCudaPath = Join-Path $repoRoot 'tools/build_opencv_cuda.ps1'
$setupOpenCvDmlPath = Join-Path $repoRoot 'tools/setup_opencv_dml.ps1'
$cmakePath = Join-Path $repoRoot 'CMakeLists.txt'
$debugMainPath = Join-Path $repoRoot 'sunone_aimbot_2/debug_harness/debug_main.cpp'
$nanoSimRootPath = Join-Path $repoRoot 'sunone_aimbot_2/modules/nano_sim_3d'
$nanoSimIndexPath = Join-Path $nanoSimRootPath 'index.html'
$nanoSimDiagPath = Join-Path $nanoSimRootPath 'src/diagnostic_analysis.js'
$nanoSimAppPath = Join-Path $nanoSimRootPath 'src/app.js'
$trainingRootPath = Join-Path $repoRoot 'sunone_aimbot_2/modules/training'
$pidGovernorModelPath = Join-Path $trainingRootPath 'models/pid_governor.onnx'
$neuralTrackerModelPath = Join-Path $trainingRootPath 'models/neural_tracker.onnx'

Assert-FileExists $buildCommonPath 'Shared build automation helper must exist.'
Assert-FileExists $buildCudaPsPath 'CUDA PowerShell build entry point must exist.'
Assert-FileExists $buildDmlPsPath 'DML PowerShell build entry point must exist.'
Assert-FileExists $builderPath 'Root BUILDER launcher must exist.'
Assert-FileExists $builderBatPath 'Root double-click BUILDER batch launcher must exist.'
Assert-FileExists $buildNoOptionsPath 'No-options main-program build launcher must exist.'
Assert-FileExists $buildNoOptionsBatPath 'Double-click no-options main-program build launcher must exist.'
Assert-FileExists $debugMainPath 'Optional ai_debug launcher source must exist.'
Assert-FileExists $nanoSimIndexPath 'NanoSim browser UI must exist.'
Assert-FileExists $nanoSimDiagPath 'NanoSim convergence diagnostic analyzer must exist.'
Assert-FileExists $nanoSimAppPath 'NanoSim browser app must exist.'
Assert-FileExists $trainingRootPath 'Training modules must live under sunone_aimbot_2/modules/training.'
Assert-FileExists $pidGovernorModelPath 'Training modules must carry a base PID governor ONNX model when available.'
Assert-FileExists $neuralTrackerModelPath 'Training modules must carry a base neural tracker ONNX model when available.'

$buildCommon = Get-Content -LiteralPath $buildCommonPath -Raw
$buildCudaPs = Get-Content -LiteralPath $buildCudaPsPath -Raw
$buildDmlPs = Get-Content -LiteralPath $buildDmlPsPath -Raw
$builder = Get-Content -LiteralPath $builderPath -Raw
$builderBat = Get-Content -LiteralPath $builderBatPath -Raw
$buildNoOptions = Get-Content -LiteralPath $buildNoOptionsPath -Raw
$buildNoOptionsBat = Get-Content -LiteralPath $buildNoOptionsBatPath -Raw
$buildOpenCvCuda = Get-Content -LiteralPath $buildOpenCvCudaPath -Raw
$setupOpenCvDml = Get-Content -LiteralPath $setupOpenCvDmlPath -Raw
$cmake = Get-Content -LiteralPath $cmakePath -Raw
$debugMain = Get-Content -LiteralPath $debugMainPath -Raw
$nanoSimIndex = Get-Content -LiteralPath $nanoSimIndexPath -Raw
$nanoSimDiag = Get-Content -LiteralPath $nanoSimDiagPath -Raw
$nanoSimApp = Get-Content -LiteralPath $nanoSimAppPath -Raw

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
Assert-Contains $buildCommon 'function\s+Ensure-TrainingBaseModels' `
    'Build automation must create base training ONNX models before packaging if they are missing.'
Assert-Contains $buildCommon 'generate_pid_dataset\.py' `
    'Build automation must know how to bootstrap the PID governor dataset.'
Assert-Contains $buildCommon 'train_pid_governor\.py' `
    'Build automation must know how to train the PID governor base model.'
Assert-Contains $buildCommon 'export_pid_governor_onnx\.py' `
    'Build automation must know how to export the PID governor base ONNX model.'
Assert-Contains $buildCommon 'generate_neural_tracker_dataset\.py' `
    'Build automation must know how to bootstrap the neural tracker dataset.'
Assert-Contains $buildCommon 'train_neural_tracker\.py' `
    'Build automation must know how to train the neural tracker base model.'
Assert-Contains $buildCommon 'export_neural_tracker_onnx\.py' `
    'Build automation must know how to export the neural tracker base ONNX model.'
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
Assert-Contains $buildCudaPs 'Ensure-TrainingBaseModels' `
    'CUDA build automation must ensure base training models before CMake packages assets.'
Assert-Contains $buildCudaPs 'BuildDebugHarness' `
    'CUDA build automation must be able to configure the optional NanoSim debug harness.'
Assert-Contains $buildCudaPs 'AIMBOT_BUILD_DEBUG_HARNESS' `
    'CUDA build automation must pass AIMBOT_BUILD_DEBUG_HARNESS to CMake.'

Assert-Contains $buildDmlPs 'Ninja Multi-Config' `
    'DML project build must use Ninja Multi-Config by default.'
Assert-Contains $buildDmlPs 'Ensure-TrainingBaseModels' `
    'DML build automation must ensure base training models before CMake packages assets.'
Assert-Contains $buildDmlPs 'Resolve-OptionalBoolean .*OpenCV already built' `
    'DML build must prompt whether OpenCV is already built.'
Assert-Contains $buildDmlPs 'Resolve-OptionalBoolean .*Download or update needed files' `
    'DML build must prompt whether dependency downloads or updates are needed.'
Assert-Contains $buildDmlPs 'modules\\opencv\\build\\dml' `
    'DML OpenCV root must live under modules/opencv/build/dml.'
Assert-Contains $buildDmlPs 'BuildDebugHarness' `
    'DML build automation must be able to configure the optional NanoSim debug harness.'
Assert-Contains $buildDmlPs 'AIMBOT_BUILD_DEBUG_HARNESS' `
    'DML build automation must pass AIMBOT_BUILD_DEBUG_HARNESS to CMake.'

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
Assert-Contains $builder 'BuildDebugHarness' `
    'BUILDER must forward optional debug harness builds to backend scripts.'
Assert-Contains $builder '@forwardedArgs @BuildArgs' `
    'BUILDER must pass normalized flags before raw extra backend arguments.'
Assert-Contains $builderBat 'Double-click deployment launcher' `
    'BUILDER.bat must identify itself as a double-click deployment launcher.'
Assert-Contains $builderBat 'pause' `
    'BUILDER.bat must keep the console open after double-click runs.'
Assert-Contains $builderBat 'Deployment complete' `
    'BUILDER.bat must show a clear completion message.'
Assert-Contains $buildNoOptions 'Select build backend' `
    'build_no-options must ask only which backend to build.'
Assert-Contains $buildNoOptions 'build\\dml' `
    'build_no-options must build the existing DML build tree directly.'
Assert-Contains $buildNoOptions 'build\\cuda' `
    'build_no-options must build the existing CUDA build tree directly.'
Assert-Contains $buildNoOptions 'cmake"\s+@buildArgs' `
    'build_no-options must call cmake --build directly instead of running dependency setup.'
Assert-Contains $buildNoOptions '--build' `
    'build_no-options must run the main CMake build command.'
Assert-Contains $buildNoOptions '\$target\s*=\s*if \(\$DebugHarness\) \{\s*"ai_debug"\s*\} else \{\s*"ai"\s*\}' `
    'build_no-options must default to the main ai executable target.'
Assert-Contains $buildNoOptions 'DebugHarness' `
    'build_no-options must expose an explicit optional debug harness target switch.'
Assert-Contains $buildNoOptions 'ai_debug' `
    'build_no-options must be able to build only the optional ai_debug target.'
Assert-Contains $buildNoOptions 'Import-VisualStudioEnvironment' `
    'build_no-options must import the Visual Studio compiler environment without running dependency setup.'
Assert-NotContains $buildNoOptions 'tools\\build_dml|tools\\build_cuda|nuget|restore|DownloadOrUpdateNeeded|OpenCvAlreadyBuilt|Invoke-WebRequest|build_opencv|setup_opencv|UseLatestPackages|OpenBrowserForDownloads|Resolve-OptionalBoolean|Restore-NuGetPackages|Ensure-CoreSourceModules' `
    'build_no-options must not run dependency setup, package restore, OpenCV setup, downloads, or update prompts.'
Assert-Contains $buildNoOptionsBat 'No-download main program builder' `
    'build_no-options.bat must identify itself as the no-download builder.'
Assert-Contains $buildNoOptionsBat 'build_no-options\.ps1' `
    'build_no-options.bat must launch build_no-options.ps1.'
Assert-Contains $buildNoOptionsBat 'pause' `
    'build_no-options.bat must keep the console open after double-click runs.'

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
Assert-Contains $buildOpenCvCuda '7\.5;8\.0;8\.6;8\.7;8\.8;8\.9;9\.0;10\.0;10\.3;11\.0;12\.0;12\.1' `
    'OpenCV CUDA helper all-arch preset must cover Turing, Ampere/Ada, Hopper, and Blackwell variants.'

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
Assert-Contains $cmake 'detector/cuda_preprocess\.cu' `
    'CMake must compile the CUDA preprocessing kernel used by TensorRT GPU-frame preprocessing.'
Assert-Contains $cmake 'AIMBOT_BUILD_DEBUG_HARNESS' `
    'CMake must expose the optional NanoSim debug harness build switch.'
Assert-Contains $cmake 'add_executable\(ai_debug' `
    'CMake must build ai_debug as a separate executable when the debug harness is enabled.'
Assert-Contains $cmake 'aimbot_copy_nanosim_assets' `
    'CMake must copy NanoSim assets beside ai_debug.exe.'
Assert-Contains $cmake 'debug/nano_sim_3d' `
    'CMake must deploy NanoSim under the debug/nano_sim_3d runtime folder.'
Assert-Contains $cmake 'mouse/rzctl\.cpp' `
    'CMake must compile the Razer runtime wrapper.'
Assert-Contains $cmake 'mouse/Teensy41RawHid\.cpp' `
    'CMake must compile the Teensy 4.1 RawHID input backend.'
Assert-Contains $cmake 'AIMBOT_RAZER_CONTROL_DLL' `
    'CMake must copy chroma_lighting.dll when the Razer backend runtime is present.'
Assert-Contains $cmake 'neural/NeuralTracker\.cpp' `
    'CMake must compile neural tracker association runtime support.'
Assert-Contains $cmake 'overlay/draw_neural\.cpp' `
    'CMake must compile the Neural overlay settings tab.'
Assert-Contains $cmake 'AIMBOT_TRAINING_SOURCE_DIR' `
    'CMake must treat modules/training as the source for deployable training assets.'
Assert-Contains $cmake 'aimbot_copy_training_assets' `
    'CMake must copy selected training scripts and models next to ai.exe after build.'
Assert-Contains $cmake 'generate_pid_dataset\.py' `
    'CMake must copy the PID dataset generator for deployable base-model workflows.'
Assert-Contains $cmake 'train_pid_governor\.py' `
    'CMake must copy the PID trainer for deployable base-model workflows.'
Assert-Contains $cmake 'export_pid_governor_onnx\.py' `
    'CMake must copy the PID ONNX exporter for deployable base-model workflows.'
Assert-Contains $cmake 'generate_neural_tracker_dataset\.py' `
    'CMake must copy the neural tracker dataset generator for deployable base-model workflows.'
Assert-Contains $cmake 'train_neural_tracker\.py' `
    'CMake must copy the neural tracker trainer for deployable base-model workflows.'
Assert-Contains $cmake 'export_neural_tracker_onnx\.py' `
    'CMake must copy the neural tracker ONNX exporter for deployable base-model workflows.'
Assert-Contains $cmake 'models/\*\.onnx' `
    'CMake must copy available base ONNX models into the output training/models folder.'
Assert-NotContains $cmake '\$\{AIMBOT_TRAINING_SOURCE_DIR\}/\*\.py' `
    'CMake must not deploy every root training script; only essential scripts should be copied.'
Assert-NotContains $cmake 'training/data' `
    'CMake must not deploy training data into runtime build folders.'
Assert-NotContains $cmake 'yolo_workspace' `
    'CMake must not deploy YOLO workspace data into runtime build folders.'

Assert-Contains $debugMain 'Config runtimeConfig' `
    'ai_debug must load the normal project config before launching NanoSim.'
Assert-Contains $debugMain 'debugHarness=1&movement=simulation' `
    'ai_debug must launch NanoSim with NanoSim as the default simulation-only movement runtime.'
Assert-Contains $debugMain 'collectModelOptions' `
    'ai_debug must discover deployable ONNX/engine models for the NanoSim model selector.'
Assert-Contains $debugMain 'ai_model=' `
    'ai_debug must pass the selected project model into NanoSim.'
Assert-Contains $debugMain 'model_options=' `
    'ai_debug must pass available model choices into NanoSim.'
Assert-Contains $debugMain 'auto_aim=' `
    'ai_debug must pass Auto Aim state into NanoSim.'
Assert-Contains $debugMain 'button_pause=' `
    'ai_debug must pass the pause binding into NanoSim for F3-style toggles.'
Assert-Contains $debugMain 'confidence_threshold=' `
    'ai_debug must pass detection confidence into NanoSim project runtime controls.'
Assert-Contains $debugMain 'pid_governor_speed=' `
    'ai_debug must pass PID governor speed into NanoSim project runtime controls.'
Assert-Contains $debugMain 'pid_governor_lead_percent=' `
    'ai_debug must pass PID governor target lead into NanoSim project runtime controls.'
Assert-Contains $debugMain 'circle_fov_radius_percent=' `
    'ai_debug must pass Circle FOV radius into NanoSim project runtime controls.'
Assert-NotContains $debugMain 'MouseThread|SendInput|RzctlMouse|Teensy41RawHid|kmbox|KMBOX' `
    'ai_debug must not directly instantiate or call physical mouse/control backends.'

Assert-Contains $nanoSimIndex 'Project Runtime' `
    'NanoSim must show project-shaped runtime controls instead of standalone controller controls.'
Assert-Contains $nanoSimIndex 'Main GUI Mirror' `
    'NanoSim must mirror the main GUI tabs for debug-harness settings.'
Assert-Contains $nanoSimIndex 'id="mainGuiTabs"' `
    'NanoSim must expose tab controls matching the main overlay.'
Assert-Contains $nanoSimIndex 'Model Selector' `
    'NanoSim must expose a model selector sourced from the project runtime.'
Assert-Contains $nanoSimIndex 'id="autoAimToggle"' `
    'NanoSim must expose Auto Aim as a project-facing target setting.'
Assert-Contains $nanoSimIndex 'id="aimModeValue"' `
    'NanoSim must show whether simulation Auto Aim is active or paused.'
Assert-Contains $nanoSimIndex 'id="confidenceThreshold"' `
    'NanoSim must expose the project confidence threshold knob.'
Assert-Contains $nanoSimIndex 'id="nmsThreshold"' `
    'NanoSim must expose the project NMS threshold knob.'
Assert-Contains $nanoSimIndex 'id="pidGovernorSpeed"' `
    'NanoSim must expose the PID governor speed knob.'
Assert-Contains $nanoSimIndex 'id="pidGovernorBlend"' `
    'NanoSim must expose the PID governor blend knob.'
Assert-Contains $nanoSimIndex 'id="pidGovernorLead"' `
    'NanoSim must expose the PID governor target lead knob.'
Assert-Contains $nanoSimIndex 'id="circleFovRadius"' `
    'NanoSim must expose the Circle FOV radius knob.'
Assert-Contains $nanoSimIndex 'id="simDefaults"[^>]*hidden' `
    'NanoSim internal simulator defaults must stay hidden from the project-facing debug UI.'
Assert-NotContains $nanoSimIndex '>Controller Diagnostics<' `
    'NanoSim must not expose old standalone controller diagnostics as visible controls.'
Assert-NotContains $nanoSimIndex '>Trainer Monitor<' `
    'NanoSim must not expose old standalone trainer controls as visible controls.'
Assert-NotContains $nanoSimIndex '>Internal Controller Steers<' `
    'NanoSim must not expose the old internal controller steering toggle.'
Assert-NotContains $nanoSimIndex '>Simulation Movement Sink<' `
    'NanoSim must not expose the old movement sink toggle.'

Assert-Contains $nanoSimDiag 'rankConvergenceIssues' `
    'NanoSim must rank convergence issues for the debug harness.'
Assert-Contains $nanoSimDiag 'multi_module_interaction' `
    'NanoSim diagnostics must detect multiple modules contributing to convergence issues.'
Assert-Contains $nanoSimApp 'window\.nanoSimGetSnapshot' `
    'NanoSim must expose telemetry snapshots to the debug harness.'
Assert-Contains $nanoSimApp 'window\.nanoSimApplyMovement' `
    'NanoSim must expose a simulation-only movement injection API.'
Assert-Contains $nanoSimApp 'controllerSteersView\.disabled\s*=\s*state\.debugHarness' `
    'NanoSim debug mode must disable internal controller steering as a movement source.'
Assert-Contains $nanoSimApp 'handleAutoAimHotkey' `
    'NanoSim must toggle simulation Auto Aim from the configured pause binding.'
Assert-Contains $nanoSimApp 'shouldApplySimulationAim' `
    'NanoSim must gate controller movement on simulation Auto Aim state.'
Assert-Contains $nanoSimApp 'createCartoonTargetRig' `
    'NanoSim must render a dynamic procedural cartoon target without vendored model dependencies.'
Assert-Contains $nanoSimApp 'populateModelSelect' `
    'NanoSim must populate its model selector from ai_debug launch settings.'
Assert-Contains $nanoSimApp 'collectProjectKnobSnapshot' `
    'NanoSim must collect project-shaped GUI knobs into the runtime snapshot.'
Assert-Contains $nanoSimApp 'syncProjectKnobs' `
    'NanoSim must keep visible project knobs synchronized with simulator timing inputs.'
Assert-Contains $nanoSimApp 'model_options' `
    'NanoSim must parse model choices from ai_debug launch settings.'
Assert-Contains $nanoSimApp 'pid_governor_lead_percent' `
    'NanoSim must parse PID governor target lead from ai_debug launch settings.'
Assert-Contains $nanoSimApp 'velocityLeadPercent' `
    'NanoSim controller settings must receive the PID governor target lead percentage.'
Assert-Contains $nanoSimApp 'rankConvergenceIssues\(state\.diagnosticSamples\)' `
    'NanoSim app must feed recent telemetry into the convergence issue ranker.'
Assert-NotContains $drawNeural 'NANOSIM' `
    'Production overlay tabs must not add NanoSim as a physical control method.'

Write-Host 'Regression checks passed.'
