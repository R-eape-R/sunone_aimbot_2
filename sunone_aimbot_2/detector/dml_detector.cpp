#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>
#include <dml_provider_factory.h>
#include <wrl/client.h>
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <limits>
#include <algorithm>
#include <dxgi.h>
#include <cmath>

#include "dml_detector.h"
#include "sunone_aimbot_2.h"
#include "postProcess.h"
#include "capture.h"
#include "capture/circle_fov.h" 
#include "other_tools.h"
#ifdef USE_CUDA
#include "depth/depth_mask.h"
#endif

extern std::atomic<bool> detector_model_changed;
extern std::atomic<bool> detection_resolution_changed;
extern std::atomic<bool> detectionPaused;

namespace
{
bool tryInt64ToInt(int64_t value, int* out)
{
    if (!out) return false;
    if (value < static_cast<int64_t>(std::numeric_limits<int>::min()) ||
        value > static_cast<int64_t>(std::numeric_limits<int>::max()))
    {
        return false;
    }
    *out = static_cast<int>(value);
    return true;
}

#ifdef USE_CUDA
bool intersectsDepthMask(const cv::Rect& box, const cv::Mat& mask)
{
    if (box.width <= 0 || box.height <= 0 || mask.empty() || mask.type() != CV_8UC1)
        return false;
    const cv::Rect imageBounds(0, 0, mask.cols, mask.rows);
    const cv::Rect clipped = box & imageBounds;
    if (clipped.width <= 0 || clipped.height <= 0) return false;
    const int cx = clipped.x + clipped.width / 2;
    const int cy = clipped.y + clipped.height / 2;
    if (mask.at<uint8_t>(cy, cx) != 0) return true;
    const cv::Mat roi = mask(clipped);
    return cv::countNonZero(roi) > 0;
}

void filterDetectionsByDepthMask(std::vector<Detection>& detections)
{
    static cv::Mat holdTtl;
    if (detections.empty()) return;
    if (!config.depth_inference_enabled || !config.depth_mask_enabled)
    {
        holdTtl.release();
        return;
    }
    const int holdFrames = std::clamp(config.depth_mask_hold_frames, 0, 120);
    cv::Mat currentMask = getCurrentDetectionSuppressionMask();
    cv::Mat suppressionMask;
    if (holdFrames <= 0)
    {
        holdTtl.release();
        suppressionMask = currentMask;
    }
    else
    {
        if (!currentMask.empty() && currentMask.type() == CV_8UC1)
        {
            if (holdTtl.empty() || holdTtl.size() != currentMask.size())
                holdTtl = cv::Mat::zeros(currentMask.size(), CV_16UC1);
            cv::subtract(holdTtl, cv::Scalar(1), holdTtl);
            holdTtl.setTo(cv::Scalar(static_cast<uint16_t>(holdFrames)), currentMask);
        }
        else if (!holdTtl.empty())
        {
            cv::subtract(holdTtl, cv::Scalar(1), holdTtl);
        }
        if (!holdTtl.empty() && cv::countNonZero(holdTtl) > 0)
        {
            cv::compare(holdTtl, cv::Scalar(0), suppressionMask, cv::CMP_GT);
        }
        else
        {
            suppressionMask.release();
        }
    }
    if (suppressionMask.empty() || suppressionMask.type() != CV_8UC1) return;
    detections.erase(
        std::remove_if(detections.begin(), detections.end(),
            [&suppressionMask](const Detection& det) { return intersectsDepthMask(det.box, suppressionMask); }),
        detections.end());
}
#else
void filterDetectionsByDepthMask(std::vector<Detection>&) {}
#endif
}


void filterDetectionsByCircleFov(std::vector<Detection>& detections)
{
    if (detections.empty() || !config.circle_fov_enabled)
        return;

    const cv::Size detectionSize(config.detection_resolution, config.detection_resolution);
    detections.erase(
        std::remove_if(detections.begin(), detections.end(),
            [&detectionSize](const Detection& det)
            {
                const cv::Point2f center(
                    static_cast<float>(det.box.x) + static_cast<float>(det.box.width) * 0.5f,
                    static_cast<float>(det.box.y) + static_cast<float>(det.box.height) * 0.5f);
                return !pointInsideCircleFov(center, detectionSize, config.circle_fov_radius_percent);
            }),
        detections.end());
}

std::string GetDMLDeviceName(int deviceId)
{
    Microsoft::WRL::ComPtr<IDXGIFactory1> dxgiFactory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory)))) return "Unknown";
    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
    if (FAILED(dxgiFactory->EnumAdapters1(deviceId, &adapter))) return "Invalid device ID";
    DXGI_ADAPTER_DESC1 desc;
    if (FAILED(adapter->GetDesc1(&desc))) return "Failed to get description";
    std::wstring wname(desc.Description);
    return WideToUtf8(wname);
}

DirectMLDetector::DirectMLDetector(const std::string& model_path)
    : env(ORT_LOGGING_LEVEL_WARNING, "DML_Detector"),
      memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);
    session_options.EnableCpuMemArena();
    session_options.DisableMemPattern();
    session_options.DisableProfiling();
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    
    session_options.AddConfigEntry("session.intra_op.allow_spinning", "1");
    session_options.AddConfigEntry("session.inter_op.allow_spinning", "1");

    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(session_options, config.dml_device_id));
    if (config.verbose)
        std::cout << "[DirectML] Using adapter: " << GetDMLDeviceName(config.dml_device_id) << std::endl;

    initializeModel(model_path);
}

DirectMLDetector::~DirectMLDetector()
{
    shouldExit = true;
    inferenceCV.notify_all();
}

void DirectMLDetector::initializeModel(const std::string& model_path)
{
    input_tensor_pin.reset();
    output_tensor_pin.reset();
    session = Ort::Session(nullptr);

    isStaticModel = (model_path.find("_static") != std::string::npos);
    if (config.verbose)
    {
        std::cout << "[DirectML] Loading path context: " 
                  << (isStaticModel ? "Optimized Static Graph" : "Dynamic Graph") << std::endl;
    }

    std::wstring model_path_wide(model_path.begin(), model_path.end());
    session = Ort::Session(env, model_path_wide.c_str(), session_options);

    input_name = session.GetInputNameAllocated(0, allocator).get();
    output_name = session.GetOutputNameAllocated(0, allocator).get();

    Ort::TypeInfo input_type_info = session.GetInputTypeInfo(0);
    input_shape = input_type_info.GetTensorTypeAndShapeInfo().GetShape();

    if (input_shape.size() >= 4) {
        model_height = static_cast<int>(input_shape[2]);
        model_width = static_cast<int>(input_shape[3]);
    }

    if (model_height <= 0) model_height = config.detection_resolution;
    if (model_width <= 0) model_width = config.detection_resolution;

    model_height = static_cast<int>(std::round(model_height / 32.0f) * 32);
    model_width = static_cast<int>(std::round(model_width / 32.0f) * 32);
    if (model_height < 32) model_height = 32;
    if (model_width < 32) model_width = 32;

    Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(0);
    current_output_shape = output_type_info.GetTensorTypeAndShapeInfo().GetShape();

    if (current_output_shape[0] == -1) current_output_shape[0] = 1;
    if (current_output_shape.size() == 3 && current_output_shape[2] == -1) {
        current_output_shape[2] = (21 * model_width * model_height) / 1024;
    }

    fallback_tensor_values.assign(3 * model_height * model_width, 0.0f);
    current_input_shape = { 1, 3, model_height, model_width };

    input_tensor_pin = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(
        memory_info, fallback_tensor_values.data(), fallback_tensor_values.size(),
        current_input_shape.data(), current_input_shape.size()));

    if (isStaticModel)
    {
        size_t output_element_count = 1;
        for (auto dim : current_output_shape) {
            output_element_count *= (dim > 0 ? dim : 1);
        }
        output_tensor_values.assign(output_element_count, 0.0f);

        output_tensor_pin = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(
            memory_info, output_tensor_values.data(), output_tensor_values.size(),
            current_output_shape.data(), current_output_shape.size()));
    }

    bool isStatic = true;
    for (auto d : input_shape) if (d <= 0) isStatic = false;
    if (isStatic != config.fixed_input_size)
    {
        config.fixed_input_size = isStatic;
        detector_model_changed.store(true);
    }
}

std::vector<Detection> DirectMLDetector::detect(const cv::Mat& input_frame)
{
    std::vector<cv::Mat> batch = { input_frame };
    auto batchResult = detectBatch(batch);
    if (!batchResult.empty()) return batchResult[0];
    return {};
}

std::vector<std::vector<Detection>> DirectMLDetector::detectBatch(const std::vector<cv::Mat>& frames)
{
    std::vector<std::vector<Detection>> empty;
    if (frames.empty() && !isPrecomputedFrameReady) return empty;
    const size_t batch_size = 1; 

    const int target_h = model_height;
    const int target_w = model_width;

    auto t0 = std::chrono::steady_clock::now();
    
    Ort::Value* activeInputTensorPtr = input_tensor_pin.get();
    std::vector<Ort::Value> dynamicOutputTensorHolder;

    if (isPrecomputedFrameReady)
    {
        activeInputTensorPtr = const_cast<Ort::Value*>(
            new Ort::Value(Ort::Value::CreateTensor<float>(
                memory_info, const_cast<float*>(precomputedTensorValues), 3 * target_h * target_w,
                current_input_shape.data(), current_input_shape.size()))
        );
    }
    else
    {
        cv::Mat bgrFrame;
        switch (frames[0].channels())
        {
        case 4: cv::cvtColor(frames[0], bgrFrame, cv::COLOR_BGRA2BGR); break;
        case 3: bgrFrame = frames[0]; break;
        case 1: cv::cvtColor(frames[0], bgrFrame, cv::COLOR_GRAY2BGR); break;
        default: bgrFrame = cv::Mat::zeros(frames[0].size(), CV_8UC3); break;
        }

        cv::Mat resized;
        cv::resize(bgrFrame, resized, cv::Size(target_w, target_h));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f);

        float* destRawData = fallback_tensor_values.data();
        const float* src = reinterpret_cast<const float*>(resized.data);
        for (int h = 0; h < target_h; ++h)
            for (int w = 0; w < target_w; ++w)
                for (int c = 0; c < 3; ++c)
                {
                    size_t dstIdx = static_cast<size_t>(c) * target_h * target_w
                        + static_cast<size_t>(h) * target_w + static_cast<size_t>(w);
                    destRawData[dstIdx] = src[(h * target_w + w) * 3 + c];
                }
    }
    auto t1 = std::chrono::steady_clock::now();

    const char* input_names[] = { input_name.c_str() };
    const char* output_names[] = { output_name.c_str() };

    auto t2 = std::chrono::steady_clock::now();
    float* outData = nullptr;
    std::vector<int64_t> outShape;

    if (isStaticModel && output_tensor_pin)
    {
        session.Run(Ort::RunOptions{ nullptr }, input_names, activeInputTensorPtr, 1, output_names, output_tensor_pin.get(), 1);
        outData = output_tensor_values.data();
        outShape = current_output_shape;
    }
    else
    {
        dynamicOutputTensorHolder = session.Run(Ort::RunOptions{ nullptr }, input_names, activeInputTensorPtr, 1, output_names, 1);
        outData = dynamicOutputTensorHolder.front().GetTensorMutableData<float>();
        outShape = dynamicOutputTensorHolder.front().GetTensorTypeAndShapeInfo().GetShape();
    }
    auto t3 = std::chrono::steady_clock::now();

    if (isPrecomputedFrameReady) {
        delete activeInputTensorPtr;
    }

    int rows = static_cast<int>(outShape[1]);
    int cols = static_cast<int>(outShape[2]);
    const int num_classes = rows - 4;

    std::vector<std::vector<Detection>> batchDetections(batch_size);
    std::chrono::duration<double, std::milli> nmsTimeTmp{ 0 };

    auto t4 = std::chrono::steady_clock::now(); 
    std::vector<Detection> detections = postProcessYoloDML(outData, { rows, cols }, num_classes, config.confidence_threshold, config.nms_threshold, &nmsTimeTmp);

    float scale_x = 1.0f;
    float scale_y = 1.0f;
    if (isPrecomputedFrameReady)
    {
        scale_x = static_cast<float>(config.detection_resolution) / target_w;
        scale_y = static_cast<float>(config.detection_resolution) / target_h;
    }
    else
    {
        scale_x = static_cast<float>(frames[0].cols) / target_w;
        scale_y = static_cast<float>(frames[0].rows) / target_h;
    }

    for (auto& d : detections)
    {
        d.box.x = static_cast<int>(d.box.x * scale_x);
        d.box.y = static_cast<int>(d.box.y * scale_y);
        d.box.width = static_cast<int>(d.box.width * scale_x);
        d.box.height = static_cast<int>(d.box.height * scale_y);
    }

    batchDetections[0] = std::move(detections);
    auto t5 = std::chrono::steady_clock::now();

    lastPreprocessTimeDML = t1 - t0;
    lastInferenceTimeDML = t3 - t2;
    lastCopyTimeDML = t4 - t3;
    lastPostprocessTimeDML = t5 - t4;
    lastNmsTimeDML = nmsTimeTmp;

    return batchDetections;
}

int DirectMLDetector::getNumberOfClasses()
{
    Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(0);
    std::vector<int64_t> output_shape = output_type_info.GetTensorTypeAndShapeInfo().GetShape();
    if (output_shape.size() == 3)
    {
        int channels = 0;
        if (!tryInt64ToInt(output_shape[1], &channels)) return -1;
        return channels - 4;
    }
    return -1;
}

void DirectMLDetector::processPrecomputedTensor(const float* tensor_data)
{
    if (!tensor_data) return;
    std::unique_lock<std::mutex> lock(inferenceMutex);
    precomputedTensorValues = tensor_data;
    isPrecomputedFrameReady = true;
    frameReady = true;
    inferenceCV.notify_one();
}

void DirectMLDetector::processFrame(const cv::Mat& frame)
{
    if (detectionPaused)
    {
        std::lock_guard<std::mutex> lock(detectionBuffer.mutex);
        detectionBuffer.boxes.clear();
        detectionBuffer.classes.clear();
        detectionBuffer.version++;
        detectionBuffer.cv.notify_all();
        return;
    }
    std::unique_lock<std::mutex> lock(inferenceMutex);
    currentFrame = frame;
    isPrecomputedFrameReady = false;
    frameReady = true;
    inferenceCV.notify_one();
}

void DirectMLDetector::dmlInferenceThread()
{
    try
    {
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

        while (!shouldExit)
        {
            if (detector_model_changed.load())
            {
                initializeModel("models/" + config.ai_model);
                detection_resolution_changed.store(true);
                detector_model_changed.store(false);
                std::cout << "[DML] Detector reloaded: " << config.ai_model << std::endl;
            }

            if (detectionPaused)
            {
                std::lock_guard<std::mutex> lock(detectionBuffer.mutex);
                detectionBuffer.boxes.clear();
                detectionBuffer.classes.clear();
                detectionBuffer.version++;
                detectionBuffer.cv.notify_all();
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            cv::Mat frame;
            bool hasNewFrame = false;
            bool usePrecomputed = false;
            {
                std::unique_lock<std::mutex> lock(inferenceMutex);
                if (!frameReady && !shouldExit)
                    inferenceCV.wait(lock, [this] { return frameReady || shouldExit; });

                if (shouldExit) break;

                if (frameReady)
                {
                    usePrecomputed = isPrecomputedFrameReady;
                    if (!usePrecomputed) frame = std::move(currentFrame);
                    frameReady = false;
                    hasNewFrame = true;
                }
            }

            if (hasNewFrame)
            {
                std::vector<cv::Mat> batchFrames;
                if (!usePrecomputed)
                {
                    if (frame.empty()) continue;
                    batchFrames.push_back(frame);
                }

                auto detectionsBatch = detectBatch(batchFrames);

                {
                    std::unique_lock<std::mutex> lock(inferenceMutex);
                    isPrecomputedFrameReady = false;
                }

                if (detectionsBatch.empty()) continue;

                const std::vector<Detection>& detections = detectionsBatch.back();
                std::vector<Detection> filteredDetections = detections;
                
                
                filterDetectionsByDepthMask(filteredDetections);
                filterDetectionsByCircleFov(filteredDetections);

                std::lock_guard<std::mutex> lock(detectionBuffer.mutex);
                detectionBuffer.boxes.clear();
                detectionBuffer.classes.clear();
                for (const auto& d : filteredDetections) {
                    detectionBuffer.boxes.push_back(d.box);
                    detectionBuffer.classes.push_back(d.classId);
                }
                detectionBuffer.version++;
                detectionBuffer.cv.notify_all();
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "[DML] Inference thread crashed: " << e.what() << std::endl;
        shouldExit = true;
        inferenceCV.notify_all();
        detectionBuffer.cv.notify_all();
    }
    catch (...)
    {
        std::cerr << "[DML] Inference thread crashed: unknown exception." << std::endl;
        shouldExit = true;
        inferenceCV.notify_all();
        detectionBuffer.cv.notify_all();
    }
}
