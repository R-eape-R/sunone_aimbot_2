#include <algorithm>
#include <numeric>
#include <chrono>
#include <limits>
#include <cmath>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp> 

#include "postProcess.h"
#include "sunone_aimbot_2.h"
#ifdef USE_CUDA
#include "trt_detector.h"
#endif





void NMS(std::vector<Detection>& detections, float nmsThreshold, std::chrono::duration<double, std::milli>* nmsTime)
{
    if (detections.empty()) return;

    if (nmsThreshold <= 0.0f)
    {
        if (nmsTime) *nmsTime = std::chrono::duration<double, std::milli>(0);
        return;
    }

    auto t0 = std::chrono::steady_clock::now();

   
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    boxes.reserve(detections.size());
    confs.reserve(detections.size());

    for (const auto& d : detections) {
        boxes.push_back(d.box);
        confs.push_back(d.confidence);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs, 0.0f, nmsThreshold, indices);

    std::vector<Detection> result;
    result.reserve(indices.size());
    for (int idx : indices) {
        result.push_back(detections[idx]);
    }
    detections = std::move(result);

    auto t1 = std::chrono::steady_clock::now();
    if (nmsTime) {
        *nmsTime = t1 - t0;
    }
}

#ifdef USE_CUDA
std::vector<Detection> postProcessYolo(
    const float* output,
    const std::vector<int64_t>& shape,
    int numClasses,
    float confThreshold,
    float nmsThreshold,
    std::chrono::duration<double, std::milli>* nmsTime
)
{
    std::vector<Detection> detections;
    detections.reserve(256);

    if (shape.size() < 3) return detections;

    int64_t rows = shape[1];
    int64_t cols = shape[2];
    const float img_scale = trt_detector.img_scale;

    if (cols == 6)
    {
        int64_t numDetections = rows;
        for (int i = 0; i < numDetections; ++i)
        {
            const float* det = output + i * cols;
            float confidence = det[4];

            if (confidence > confThreshold)
            {
                int classId = static_cast<int>(det[5]);

                float cx = det[0];
                float cy = det[1];
                float dx = det[2];
                float dy = det[3];

                Detection detection;
                detection.box.x = static_cast<int>(cx * img_scale);
                detection.box.y = static_cast<int>(cy * img_scale);
                detection.box.width = static_cast<int>((dx - cx) * img_scale);
                detection.box.height = static_cast<int>((dy - cy) * img_scale);
                detection.confidence = confidence;
                detection.classId = classId;

                detections.push_back(detection);
            }
        }
    }
    else
    {
        for (int i = 0; i < cols; ++i)
        {
            const float* col_data = output + i;

            float cx = col_data[0 * cols];
            float cy = col_data[1 * cols];
            float ow = col_data[2 * cols];
            float oh = col_data[3 * cols];

            float maxScore = 0.0f;
            int maxClassId = 0;
            for (int c = 0; c < numClasses; ++c)
            {
                float score = col_data[(4 + c) * cols];
                if (score > maxScore)
                {
                    maxScore = score;
                    maxClassId = c;
                }
            }

            if (maxScore > confThreshold)
            {
                const float half_ow = 0.5f * ow;
                const float half_oh = 0.5f * oh;

                Detection det;
                det.box.x = static_cast<int>((cx - half_ow) * img_scale);
                det.box.y = static_cast<int>((cy - half_oh) * img_scale);
                det.box.width = static_cast<int>(ow * img_scale);
                det.box.height = static_cast<int>(oh * img_scale);
                det.confidence = maxScore;
                det.classId = maxClassId;

                detections.push_back(det);
            }
        }
    }

    NMS(detections, nmsThreshold, nmsTime);
    return detections;
}
#endif





std::vector<Detection> postProcessYoloDML(
    const float* output,
    const std::vector<int64_t>& shape,
    int numClasses,
    float confThreshold,
    float nmsThreshold,
    std::chrono::duration<double, std::milli>* nmsTime
)
{
    std::vector<Detection> detections;
    
    int64_t rows, cols;
    if (shape.size() == 2) { rows = shape[0]; cols = shape[1]; }
    else if (shape.size() == 3) { rows = shape[1]; cols = shape[2]; }
    else return detections;

    if (rows <= 0 || cols <= 0) return detections;
    const int rows_i = static_cast<int>(rows);
    const int cols_i = static_cast<int>(cols);

    
    if (cols_i == 6)
    {
        const int numDetections = rows_i;
        detections.reserve(static_cast<size_t>(numDetections));
        for (int i = 0; i < numDetections; ++i)
        {
            const float* det = output + i * cols_i;
            float confidence = det[4];
            if (confidence > confThreshold)
            {
                int classId = static_cast<int>(det[5]);
                float cx = det[0];
                float cy = det[1];
                float dx = det[2];
                float dy = det[3];

                cv::Rect box;
                box.x = static_cast<int>(cx);
                box.y = static_cast<int>(cy);
                box.width = static_cast<int>(dx - cx);
                box.height = static_cast<int>(dy - cy);
                detections.push_back(Detection{ box, confidence, classId });
            }
        }
        NMS(detections, nmsThreshold, nmsTime);
        return detections;
    }

    
    static thread_local std::vector<float> max_scores;
    static thread_local std::vector<int> max_class_ids;
    
    if (max_scores.size() < static_cast<size_t>(cols_i)) {
        max_scores.resize(cols_i);
        max_class_ids.resize(cols_i);
    }
    
    std::fill_n(max_scores.begin(), cols_i, 0.0f);
    std::fill_n(max_class_ids.begin(), cols_i, -1);

    const float* p_classes = output + 4 * cols_i;

    for (int c = 0; c < numClasses; ++c) {
        const float* class_row = p_classes + c * cols_i;
        for (int i = 0; i < cols_i; ++i) {
            if (class_row[i] > max_scores[i]) {
                max_scores[i] = class_row[i];
                max_class_ids[i] = c;
            }
        }
    }

    const float* p_cx = output;
    const float* p_cy = output + cols_i;
    const float* p_w  = output + 2 * cols_i;
    const float* p_h  = output + 3 * cols_i;

    detections.reserve(200); 

    for (int i = 0; i < cols_i; ++i) {
        if (max_scores[i] > confThreshold) {
            float ow = p_w[i];
            float oh = p_h[i];

            cv::Rect box;
            box.x = static_cast<int>(p_cx[i] - 0.5f * ow);
            box.y = static_cast<int>(p_cy[i] - 0.5f * oh);
            box.width = static_cast<int>(ow);
            box.height = static_cast<int>(oh);

            detections.push_back(Detection{ box, max_scores[i], max_class_ids[i] });
        }
    }

    if (!detections.empty())
    {
        NMS(detections, nmsThreshold, nmsTime);
    }
    return detections;
}
