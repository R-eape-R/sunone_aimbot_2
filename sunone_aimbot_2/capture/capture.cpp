#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <iostream>
#include <atomic>
#include <thread>
#include <mutex>
#include <algorithm>
#include <chrono>
#include <timeapi.h>
#include <condition_variable>
#include <filesystem>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>
#include <immintrin.h>

#include "capture.h"
#ifdef USE_CUDA
#include "trt_detector.h"
#include "depth/depth_anything_trt.h"
#include "depth/depth_mask.h"
#include "tensorrt/nvinf.h"
#endif
#include "sunone_aimbot_2.h"
#include "keycodes.h"
#include "keyboard_listener.h"
#include "other_tools.h"
#include "duplication_api_capture.h"
#include "winrt_capture.h"
#include "virtual_camera.h"
#include "udp_capture.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "windowsapp.lib")

cv::Mat latestFrame;
std::mutex frameMutex;

int screenWidth = 0;
int screenHeight = 0;

std::atomic<int> captureFrameCount(0);
std::atomic<int> captureFps(0);
std::chrono::time_point<std::chrono::high_resolution_clock> captureFpsStartTime;

std::deque<cv::Mat> frameQueue;

#ifdef USE_CUDA
namespace
{
std::mutex g_detectionSuppressionMaskMutex;
cv::Mat g_detectionSuppressionMask;
}

static void UpdateDetectionSuppressionMask(const cv::Mat& mask)
{
    std::lock_guard<std::mutex> lock(g_detectionSuppressionMaskMutex);
    if (!mask.empty() && mask.type() == CV_8UC1)
        g_detectionSuppressionMask = mask.clone();
    else
        g_detectionSuppressionMask.release();
}

cv::Mat getCurrentDetectionSuppressionMask()
{
    std::lock_guard<std::mutex> lock(g_detectionSuppressionMaskMutex);
    return g_detectionSuppressionMask.clone();
}
#endif

namespace
{
struct CaptureThreadConfig
{
    std::string capture_method;
    int capture_fps = 0;
    int detection_resolution = 0;
    int monitor_idx = 0;
    bool circle_fov_enabled = false;
    int circle_fov_radius_percent = 100;
    bool capture_borders = true;
    bool capture_cursor = true;
    std::string capture_target;
    std::string capture_window_title;
    std::string virtual_camera_name;
    int virtual_camera_width = 0;
    int virtual_camera_heigth = 0;
    std::string udp_ip;
    int udp_port = 0;
    std::string backend;
    std::vector<std::string> screenshot_button;
    int screenshot_delay = 0;
    bool show_window = false;
    bool verbose = false;
#ifdef USE_CUDA
    bool depth_inference_enabled = false;
    bool depth_mask_enabled = false;
    std::string depth_model_path;
    int depth_mask_fps = 0;
    int depth_mask_near_percent = 0;
    int depth_mask_expand = 0;
    bool depth_mask_invert = false;
    bool capture_use_cuda = true;
#endif
};

CaptureThreadConfig SnapshotCaptureConfig()
{
    std::lock_guard<std::mutex> cfgLock(configMutex);
    CaptureThreadConfig snapshot;
    snapshot.capture_method = config.capture_method;
    snapshot.capture_fps = config.capture_fps;
    snapshot.detection_resolution = config.detection_resolution;
    snapshot.monitor_idx = config.monitor_idx;
    snapshot.circle_fov_enabled = config.circle_fov_enabled;
    snapshot.circle_fov_radius_percent = config.circle_fov_radius_percent;
    snapshot.capture_borders = config.capture_borders;
    snapshot.capture_cursor = config.capture_cursor;
    snapshot.capture_target = config.capture_target;
    snapshot.capture_window_title = config.capture_window_title;
    snapshot.virtual_camera_name = config.virtual_camera_name;
    snapshot.virtual_camera_width = config.virtual_camera_width;
    snapshot.virtual_camera_heigth = config.virtual_camera_heigth;
    snapshot.udp_ip = config.udp_ip;
    snapshot.udp_port = config.udp_port;
    snapshot.backend = config.backend;
    snapshot.screenshot_button = config.screenshot_button;
    snapshot.screenshot_delay = config.screenshot_delay;
    snapshot.show_window = config.show_window;
    snapshot.verbose = config.verbose;
#ifdef USE_CUDA
    snapshot.depth_inference_enabled = config.depth_inference_enabled;
    snapshot.depth_mask_enabled = config.depth_mask_enabled;
    snapshot.depth_model_path = config.depth_model_path;
    snapshot.depth_mask_fps = config.depth_mask_fps;
    snapshot.depth_mask_near_percent = config.depth_mask_near_percent;
    snapshot.depth_mask_expand = config.depth_mask_expand;
    snapshot.depth_mask_invert = config.depth_mask_invert;
    snapshot.capture_use_cuda = config.capture_use_cuda;
#endif
    return snapshot;
}

std::string NormalizeCaptureMethod(const std::string& method)
{
    if (method == "duplication_api" || method == "winrt" || method == "virtual_camera" || method == "udp_capture")
        return method;
    return "duplication_api";
}

class TimerResolutionGuard
{
public:
    void Enable()
    {
        if (!enabled_)
        {
            timeBeginPeriod(1);
            enabled_ = true;
        }
    }

    void Disable()
    {
        if (enabled_)
        {
            timeEndPeriod(1);
            enabled_ = false;
        }
    }

    ~TimerResolutionGuard()
    {
        Disable();
    }

private:
    bool enabled_{ false };
};

class WinrtApartmentGuard
{
public:
    void Ensure(bool required)
    {
        if (required && !initialized_)
        {
            winrt::init_apartment(winrt::apartment_type::multi_threaded);
            initialized_ = true;
        }
        else if (!required && initialized_)
        {
            winrt::uninit_apartment();
            initialized_ = false;
        }
    }

    ~WinrtApartmentGuard()
    {
        if (initialized_)
            winrt::uninit_apartment();
    }

private:
    bool initialized_{ false };
};

class ScreenshotWriter
{
public:
    ScreenshotWriter()
    {
        writerThread_ = std::thread([this]() { Run(); });
    }

    ~ScreenshotWriter()
    {
        Stop();
    }

    void Enqueue(const std::string& filename, cv::Mat frame)
    {
        if (filename.empty() || frame.empty())
            return;

        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.size() >= maxPendingFrames_)
            queue_.pop();
        queue_.emplace(filename, std::move(frame));
        cv_.notify_one();
    }

private:
    void Stop()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_one();

        if (writerThread_.joinable())
            writerThread_.join();
    }

    void Run()
    {
        std::error_code ec;
        std::filesystem::create_directories("screenshots", ec);

        while (true)
        {
            std::pair<std::string, cv::Mat> job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this]() { return stop_ || !queue_.empty(); });
                if (stop_ && queue_.empty())
                    break;

                job = std::move(queue_.front());
                queue_.pop();
            }

            try
            {
                const std::filesystem::path outputPath = std::filesystem::path("screenshots") / job.first;
                cv::imwrite(outputPath.string(), job.second);
            }
            catch (const std::exception& e)
            {
                std::cerr << "[Capture] Screenshot save failed: " << e.what() << std::endl;
            }
            catch (...)
            {
                std::cerr << "[Capture] Screenshot save failed: unknown exception." << std::endl;
            }
        }
    }

private:
    static constexpr size_t maxPendingFrames_ = 8;

    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<std::pair<std::string, cv::Mat>> queue_;
    std::thread writerThread_;
    bool stop_{ false };
};
} // namespace

std::vector<cv::Mat> getBatchFromQueue(int batch_size)
{
    std::vector<cv::Mat> batch;
    std::lock_guard<std::mutex> lk(frameMutex);
    const size_t target_size = (batch_size > 0) ? static_cast<size_t>(batch_size) : 0;
    const size_t n = std::min(frameQueue.size(), target_size);

    for (size_t i = 0; i < n; ++i)
        batch.push_back(frameQueue[frameQueue.size() - n + i]);

    while (batch.size() < target_size && !batch.empty())
        batch.push_back(batch.back().clone());
    return batch;
}

void captureThread(int CAPTURE_WIDTH, int CAPTURE_HEIGHT)
{
    try
    {
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);

        CaptureThreadConfig currentCfg = SnapshotCaptureConfig();
        if (currentCfg.verbose)
            std::cout << "[Capture] OpenCV version: " << CV_VERSION << std::endl;

        int captureWidth = std::max(1, CAPTURE_WIDTH);
        int captureHeight = std::max(1, CAPTURE_HEIGHT);
        if (currentCfg.detection_resolution > 0)
        {
            captureWidth = currentCfg.detection_resolution;
            captureHeight = currentCfg.detection_resolution;
        }

#ifdef USE_CUDA
        depth_anything::DepthAnythingTrt depthMaskFallbackModel;
        std::string depthMaskFallbackModelPath;
#endif

        WinrtApartmentGuard winrtApartment;
        auto createCapturer = [&](const CaptureThreadConfig& cfg, int width, int height) -> std::unique_ptr<IScreenCapture>
        {
            try
            {
                const std::string method = NormalizeCaptureMethod(cfg.capture_method);
                if (method != cfg.capture_method)
                    std::cout << "[Capture] Unknown capture_method '" << cfg.capture_method << "'. Falling back to duplication_api." << std::endl;

                if (method == "duplication_api")
                {
                    if (cfg.verbose)
                        std::cout << "[Capture] Using Duplication API" << std::endl;
                    return std::make_unique<DuplicationAPIScreenCapture>(width, height, cfg.monitor_idx);
                }

                if (method == "winrt")
                {
                    if (cfg.verbose)
                        std::cout << "[Capture] Using WinRT" << std::endl;

                    WinRTScreenCapture::Options options;
                    options.target = cfg.capture_target;
                    options.windowTitle = cfg.capture_window_title;
                    options.monitorIndex = cfg.monitor_idx;
                    options.captureBorders = cfg.capture_borders;
                    options.captureCursor = cfg.capture_cursor;

                    return std::make_unique<WinRTScreenCapture>(width, height, options);
                }

                if (method == "virtual_camera")
                {
                    if (cfg.verbose)
                        std::cout << "[Capture] Using Virtual Camera" << std::endl;
                    return std::make_unique<VirtualCameraCapture>(
                        cfg.virtual_camera_width,
                        cfg.virtual_camera_heigth,
                        cfg.virtual_camera_name,
                        cfg.capture_fps,
                        cfg.verbose
                    );
                }

                if (cfg.verbose)
                    std::cout << "[Capture] Using UDP capture" << std::endl;
                return std::make_unique<UDPCapture>(width, height, cfg.udp_ip, cfg.udp_port);
            }
            catch (const std::exception& e)
            {
                std::cerr << "[Capture] Failed to initialize '" << cfg.capture_method
                    << "' capture: " << e.what() << std::endl;
                return nullptr;
            }
        };

        std::string desiredCaptureMethod = NormalizeCaptureMethod(currentCfg.capture_method);
        winrtApartment.Ensure(desiredCaptureMethod == "winrt");

        std::unique_ptr<IScreenCapture> capturer = createCapturer(currentCfg, captureWidth, captureHeight);
        std::string activeCapturerMethod = capturer ? desiredCaptureMethod : std::string();
        auto lastCapturerCreateAttempt = std::chrono::steady_clock::now();

        auto clearCaptureFrames = [&]()
        {
            std::lock_guard<std::mutex> lock(frameMutex);
            latestFrame.release();
            frameQueue.clear();
        };

        auto clearDetections = [&]()
        {
            std::lock_guard<std::mutex> lock(detectionBuffer.mutex);
            detectionBuffer.boxes.clear();
            detectionBuffer.classes.clear();
            detectionBuffer.version++;
            detectionBuffer.cv.notify_all();
        };

        auto markCaptureUnavailable = [&]()
        {
            clearCaptureFrames();
            clearDetections();
            frameCV.notify_one();
        };

        bool captureUnavailable = false;
        auto setCaptureUnavailable = [&]()
        {
            if (captureUnavailable)
                return;
            captureUnavailable = true;
            markCaptureUnavailable();
        };
        auto setCaptureAvailable = [&]()
        {
            captureUnavailable = false;
        };

        setCaptureUnavailable();

        TimerResolutionGuard timerResolution;
        if (currentCfg.capture_fps > 0)
        {
            timerResolution.Enable();
        }

        captureFpsStartTime = std::chrono::high_resolution_clock::now();

        ScreenshotWriter screenshotWriter;
        auto lastSaveTime = std::chrono::steady_clock::now();
        auto lastSuccessfulFrameTime = std::chrono::steady_clock::now();
        constexpr auto staleFrameTimeout = std::chrono::milliseconds(500);

        while (!shouldExit)
        {
            auto loop_start_time = std::chrono::high_resolution_clock::now();

            try
            {
                currentCfg = SnapshotCaptureConfig();

                if (capture_fps_changed.exchange(false))
                {
                    if (currentCfg.capture_fps > 0)
                        timerResolution.Enable();
                    else
                        timerResolution.Disable();
                }

                const bool resolutionMismatchDetected = (currentCfg.detection_resolution > 0 && currentCfg.detection_resolution != captureWidth);

                if (detection_resolution_changed.exchange(false) ||
                    capture_method_changed.exchange(false) ||
                    capture_cursor_changed.exchange(false) ||
                    capture_borders_changed.exchange(false) ||
                    capture_window_changed.exchange(false) ||
                    resolutionMismatchDetected)
                {
                    setCaptureUnavailable();

                    if (currentCfg.detection_resolution > 0)
                    {
                        captureWidth = currentCfg.detection_resolution;
                        captureHeight = currentCfg.detection_resolution;
                    }

                    const std::string nextMethod = NormalizeCaptureMethod(currentCfg.capture_method);
                    desiredCaptureMethod = nextMethod;
                    const bool nextNeedsWinrt = (nextMethod == "winrt");

                    if (capturer)
                    {
                        const bool activeWasWinrt = (activeCapturerMethod == "winrt");
                        capturer.reset();
                        activeCapturerMethod.clear();
                        if (activeWasWinrt && !nextNeedsWinrt)
                            winrtApartment.Ensure(false);
                    }

                    winrtApartment.Ensure(nextNeedsWinrt);

                    if (nextMethod == "virtual_camera")
                        VirtualCameraCapture::GetAvailableVirtualCameras(true);

                    capturer = createCapturer(currentCfg, captureWidth, captureHeight);
                    if (capturer)
                        activeCapturerMethod = nextMethod;
                    else
                        activeCapturerMethod.clear();

                    lastCapturerCreateAttempt = std::chrono::steady_clock::now();
                }

                if (!capturer)
                {
                    const auto now = std::chrono::steady_clock::now();
                    if (now - lastCapturerCreateAttempt >= std::chrono::seconds(1))
                    {
                        desiredCaptureMethod = NormalizeCaptureMethod(currentCfg.capture_method);
                        winrtApartment.Ensure(desiredCaptureMethod == "winrt");

                        if (desiredCaptureMethod == "virtual_camera")
                            VirtualCameraCapture::GetAvailableVirtualCameras(true);

                        capturer = createCapturer(currentCfg, captureWidth, captureHeight);
                        lastCapturerCreateAttempt = now;

                        if (capturer)
                        {
                            activeCapturerMethod = desiredCaptureMethod;
                            lastSuccessfulFrameTime = now;
                        }
                    }

                    setCaptureUnavailable();
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }

                const bool screenshotEnabled =
                    !currentCfg.screenshot_button.empty() && currentCfg.screenshot_button[0] != "None";
                const auto screenshotNow = std::chrono::steady_clock::now();
                const auto screenshotElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                    screenshotNow - lastSaveTime
                ).count();
                const bool screenshotRequested =
                    screenshotEnabled &&
                    isAnyKeyPressed(currentCfg.screenshot_button) &&
                    screenshotElapsedMs >= currentCfg.screenshot_delay;
#ifdef USE_CUDA
                const bool needCpuCopyFromGpu = screenshotRequested || currentCfg.show_window;
#endif

                cv::Mat screenshotCpu;
                cv::Mat detectionFrame;
                bool frameSubmittedToDetector = false;

#ifdef USE_CUDA
                static bool lastDepthInferenceEnabled = true;
                if (!currentCfg.depth_inference_enabled)
                {
                    if (lastDepthInferenceEnabled)
                    {
                        auto& depthMask = depth_anything::GetDepthMaskGenerator();
                        depthMask.reset();
                        depthMaskFallbackModel.reset();
                        depthMaskFallbackModelPath.clear();
                    }
                    UpdateDetectionSuppressionMask(cv::Mat());
                    lastDepthInferenceEnabled = false;
                }
                else
                {
                    lastDepthInferenceEnabled = true;
                }

                const bool depthMaskEnabled = currentCfg.depth_inference_enabled && currentCfg.depth_mask_enabled;
                const bool preferGpuCapturePath =
                    currentCfg.backend == "TRT" &&
                    NormalizeCaptureMethod(currentCfg.capture_method) == "duplication_api" &&
                    currentCfg.capture_use_cuda &&
                    !currentCfg.circle_fov_enabled &&
                    !depthMaskEnabled;

                if (preferGpuCapturePath)
                {
                    auto* duplicationCapture = dynamic_cast<DuplicationAPIScreenCapture*>(capturer.get());
                    if (duplicationCapture)
                    {
                        cv::cuda::GpuMat screenshotGpu;
                        if (duplicationCapture->GetNextFrameGpu(screenshotGpu))
                        {
                            trt_detector.processFrameGpu(screenshotGpu);
                            frameSubmittedToDetector = true;

                            if (needCpuCopyFromGpu)
                                screenshotGpu.download(screenshotCpu);
                        }
                    }
                }
#endif

                
                if (currentCfg.backend == "DML" && dml_detector)
                {
                    const float* hardwareTensor = capturer->GetPrecomputedTensor();
                    if (hardwareTensor)
                    {
                        dml_detector->processPrecomputedTensor(hardwareTensor);
                        capturer->UnlockTensor();
                        frameSubmittedToDetector = true;

                        if (currentCfg.show_window)
                        {
                            screenshotCpu = capturer->GetNextFrameCpu();
                        }
                    }
                }

                if (!frameSubmittedToDetector)
                {
                    screenshotCpu = capturer->GetNextFrameCpu();

                    if (screenshotCpu.empty())
                    {
                        const auto now = std::chrono::steady_clock::now();
                        if (now - lastSuccessfulFrameTime >= staleFrameTimeout)
                            setCaptureUnavailable();

                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        continue;
                    }

                    if (NormalizeCaptureMethod(currentCfg.capture_method) == "virtual_camera")
                    {
                        const int targetW = std::max(1, captureWidth);
                        const int targetH = std::max(1, captureHeight);
                        const int roiW = std::min(targetW, screenshotCpu.cols);
                        const int roiH = std::min(targetH, screenshotCpu.rows);

                        if (roiW <= 0 || roiH <= 0) continue;

                        const int x = std::max(0, (screenshotCpu.cols - roiW) / 2);
                        const int y = std::max(0, (screenshotCpu.rows - roiH) / 2);
                        cv::Mat centered = screenshotCpu(cv::Rect(x, y, roiW, roiH));

                        if (roiW != targetW || roiH != targetH)
                        {
                            cv::resize(centered, screenshotCpu, cv::Size(targetW, targetH), 0, 0, cv::INTER_LINEAR);
                        }
                        else
                        {
                            screenshotCpu = centered;
                        }
                    }

                    detectionFrame = screenshotCpu;

                    if (currentCfg.backend == "DML" && dml_detector)
                    {
                        dml_detector->processFrame(detectionFrame);
                    }
#ifdef USE_CUDA
                    else if (currentCfg.backend == "TRT")
                    {
                        trt_detector.processFrame(detectionFrame);
                    }
#endif
                }

                if (frameSubmittedToDetector || !screenshotCpu.empty())
                {
                    lastSuccessfulFrameTime = std::chrono::steady_clock::now();
                    setCaptureAvailable();
                }

                if (!screenshotCpu.empty())
                {
                    std::lock_guard<std::mutex> lock(frameMutex);
                    latestFrame = screenshotCpu;
                    if (frameQueue.size() >= 1)
                        frameQueue.pop_front();
                    frameQueue.push_back(latestFrame);
                }
                frameCV.notify_one();

                if (screenshotRequested)
                {
                    cv::Mat saveMat = screenshotCpu.clone();
                    if (!saveMat.empty())
                    {
                        auto epoch_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch()
                        ).count();
                        std::string filename = std::to_string(epoch_time) + ".jpg";
                        screenshotWriter.Enqueue(filename, std::move(saveMat));
                        lastSaveTime = screenshotNow;
                    }
                }

                captureFrameCount++;
                auto currentTime = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsedTime = currentTime - captureFpsStartTime;
                if (elapsedTime.count() >= 1.0)
                {
                    captureFps = static_cast<int>(captureFrameCount / elapsedTime.count());
                    captureFrameCount = 0;
                    captureFpsStartTime = currentTime;
                }
            }
            catch (const std::exception& e)
            {
                std::cerr << "[Capture] Loop exception: " << e.what() << std::endl;
                capturer.reset();
                activeCapturerMethod.clear();
                winrtApartment.Ensure(false);
                setCaptureUnavailable();
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            catch (...)
            {
                std::cerr << "[Capture] Loop exception: unknown." << std::endl;
                capturer.reset();
                activeCapturerMethod.clear();
                winrtApartment.Ensure(false);
                setCaptureUnavailable();
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }

            
            if (currentCfg.capture_fps > 0)
            {
                double targetFrameTimeMs = 1000.0 / currentCfg.capture_fps;
                auto target_time = loop_start_time + std::chrono::duration<double, std::milli>(targetFrameTimeMs);
                
                while (true)
                {
                    auto now = std::chrono::high_resolution_clock::now();
                    if (now >= target_time)
                        break;

                    auto remaining_us = std::chrono::duration_cast<std::chrono::microseconds>(target_time - now).count();
                    
                    if (remaining_us > 1500)
                    {
                        std::this_thread::sleep_for(std::chrono::microseconds(remaining_us - 1500));
                    }
                    else if (remaining_us > 100)
                    {
                        std::this_thread::yield(); 
                    }
                    else
                    {
                        _mm_pause();
                    }
                }
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Capture] Unhandled exception: " << e.what() << std::endl;
        throw;
    }
}
