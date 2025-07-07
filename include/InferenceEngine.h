// InferenceEngine.h
#pragma once

#include <iostream>
#include <vector>
#include <string>
#include "NvInfer.h"
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <memory>
#include <map>
#include "CudaKernels.h"

// Forward declaration
struct Detection;

class MyLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();

    bool BuildEngineFromOnnx(const std::string& onnx_path, bool use_fp16, int workspace_size, std::string& error_message, std::mutex& error_mutex);
    bool LoadModel(const std::string& engine_path);
    void Infer(
        float* device_input_buffer,
        int screen_w,
        int screen_h,
        std::vector<Detection>& detections,
        float confidence_threshold,
        float nms_threshold
    );
    bool IsReady() const;
    int GetInputWidth() const;
    int GetInputHeight() const;

private:
    // These are now handled by a single CUDA kernel
    // void Letterbox(const cv::Mat& image, cv::Mat& out_image);
    // void Normalize(cv::Mat& image);
    void PostProcess(const std::vector<float>& output, std::vector<Detection>& detections, float conf_threshold, float iou_threshold, int capture_w, int capture_h, int screen_w, int screen_h);
    void NMS(std::vector<Detection>& detections, float iou_threshold);
    float IoU(const Detection& a, const Detection& b);

    std::unique_ptr<nvinfer1::IRuntime> m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;
    
    // CUDA resources - separate streams for overlapping operations
    cudaStream_t m_stream = nullptr;          // Backward compatibility (points to compute stream)
    cudaStream_t m_compute_stream = nullptr;  // For inference operations
    cudaStream_t m_transfer_stream = nullptr; // For memory transfers
    std::map<std::string, void*> m_gpu_buffers_map;
    unsigned char* m_pinned_buffer = nullptr;           // Pinned memory buffer on the host for raw capture data
    unsigned char* m_raw_capture_buffer_gpu = nullptr;  // Buffer on the GPU for the raw screen capture

    int m_input_width = 0;
    int m_input_height = 0;
    size_t m_output_buffer_size = 0;

    const char* m_input_tensor_name = "images";
    const char* m_output_tensor_name = "output0";

    MyLogger m_logger;
};
