#ifndef DIRECTML_DETECTOR_H
#define DIRECTML_DETECTOR_H

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <vector>
#include <memory>

#include "postProcess.h"

class DirectMLDetector
{
public:
    DirectMLDetector(const std::string& model_path);
    ~DirectMLDetector();

    std::vector<Detection> detect(const cv::Mat& input_frame);
    std::vector<std::vector<Detection>> detectBatch(const std::vector<cv::Mat>& frames);

    void dmlInferenceThread();
    void processFrame(const cv::Mat& frame);
    void processPrecomputedTensor(const float* tensor_data);

    int getNumberOfClasses();

    std::chrono::duration<double, std::milli> lastInferenceTimeDML;
    std::chrono::duration<double, std::milli> lastPreprocessTimeDML;
    std::chrono::duration<double, std::milli> lastCopyTimeDML;
    std::chrono::duration<double, std::milli> lastPostprocessTimeDML;
    std::chrono::duration<double, std::milli> lastNmsTimeDML;

    std::condition_variable inferenceCV;
    std::atomic<bool> shouldExit = false;

    int model_height = -1;
    int model_width = -1;

private:
    Ort::Env env;
    Ort::Session session{ nullptr };
    Ort::SessionOptions session_options;
    Ort::AllocatorWithDefaultOptions allocator;

    std::string input_name;
    std::string output_name;
    std::vector<int64_t> input_shape;

    
    bool isStaticModel = false;
    std::vector<int64_t> current_input_shape;
    std::vector<int64_t> current_output_shape;
    std::vector<float> fallback_tensor_values;
    std::vector<float> output_tensor_values;

    std::unique_ptr<Ort::Value> input_tensor_pin;
    std::unique_ptr<Ort::Value> output_tensor_pin;

    std::mutex inferenceMutex;
    cv::Mat currentFrame;
    const float* precomputedTensorValues = nullptr;
    bool isPrecomputedFrameReady = false;
    bool frameReady = false;

    void initializeModel(const std::string& model_path);
    Ort::MemoryInfo memory_info;
};

#endif // DIRECTML_DETECTOR_H
