// InferenceEngine.cpp
#define NOMINMAX
#include "../include/InferenceEngine.h"
#include "../include/GPUBufferPool.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <mutex>
#include "NvOnnxParser.h"
#include "types.h"
#include <numeric> 
#include <opencv2/opencv.hpp>

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace fs = std::filesystem;

void MyLogger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cerr << msg << std::endl;
        }
    }

InferenceEngine::InferenceEngine() {
    m_runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(m_logger));
    
    // Create separate streams for compute and memory transfers (performance optimization)
    cudaStreamCreate(&m_compute_stream);
    cudaStreamCreate(&m_transfer_stream);
    m_stream = m_compute_stream; // Maintain backward compatibility

    // The capture resolution is hardcoded for now
    const int capture_width = 320;
    const int capture_height = 320;
    const size_t raw_capture_size = capture_width * capture_height * 4;

    // Use buffer pool for GPU memory management
    auto& pool = GPUBufferPool::GetInstance();
    m_raw_capture_buffer_gpu = pool.GetBuffer(raw_capture_size);
    
    // Allocate Pinned Host Memory for raw capture input
    cudaHostAlloc((void**)&m_pinned_buffer, raw_capture_size, cudaHostAllocDefault);
}

InferenceEngine::~InferenceEngine() {
    // Clean up CUDA streams
    if (m_compute_stream) cudaStreamDestroy(m_compute_stream);
    if (m_transfer_stream) cudaStreamDestroy(m_transfer_stream);
    
    if (m_pinned_buffer) cudaFreeHost(m_pinned_buffer);
    
    // Return buffers to pool instead of freeing
    auto& pool = GPUBufferPool::GetInstance();
    if (m_raw_capture_buffer_gpu) {
        const size_t raw_capture_size = 320 * 320 * 4;
        pool.ReturnBuffer(m_raw_capture_buffer_gpu, raw_capture_size);
    }
    
    // Return GPU buffers for the model to pool
    if (m_gpu_buffers_map.count(m_input_tensor_name)) {
        size_t input_size = m_input_height * m_input_width * 3 * sizeof(float);
        pool.ReturnBuffer(m_gpu_buffers_map[m_input_tensor_name], input_size);
    }
    if (m_gpu_buffers_map.count(m_output_tensor_name)) {
        pool.ReturnBuffer(m_gpu_buffers_map[m_output_tensor_name], m_output_buffer_size * sizeof(float));
    }
}

int InferenceEngine::GetInputHeight() const {
    return m_input_height;
}

bool InferenceEngine::LoadModel(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) return false;

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    m_engine.reset(m_runtime->deserializeCudaEngine(engine_data.data(), size));
    if (!m_engine) return false;

    m_context.reset(m_engine->createExecutionContext());
    if (!m_context) return false;

    // Optimize buffer reuse - only reallocate if dimensions changed
    auto& pool = GPUBufferPool::GetInstance();
    
    // Check if we can reuse existing buffers
    bool can_reuse_buffers = (m_input_height == input_dims.d[2] && m_input_width == input_dims.d[3]);
    
    if (!can_reuse_buffers) {
        // Return old buffers to pool before allocating new ones
        if (m_pinned_buffer) {
            cudaFreeHost(m_pinned_buffer);
            m_pinned_buffer = nullptr;
        }
        if (m_raw_capture_buffer_gpu) {
            const size_t old_raw_capture_size = 320 * 320 * 4;
            pool.ReturnBuffer(m_raw_capture_buffer_gpu, old_raw_capture_size);
            m_raw_capture_buffer_gpu = nullptr;
        }
        if (m_gpu_buffers_map.count(m_input_tensor_name)) {
            size_t old_input_size = m_input_height * m_input_width * 3 * sizeof(float);
            pool.ReturnBuffer(m_gpu_buffers_map[m_input_tensor_name], old_input_size);
        }
        if (m_gpu_buffers_map.count(m_output_tensor_name)) {
            pool.ReturnBuffer(m_gpu_buffers_map[m_output_tensor_name], m_output_buffer_size * sizeof(float));
        }
        m_gpu_buffers_map.clear();
    } else {
        Logger::GetInstance().Log("Reusing existing GPU buffers - dimensions unchanged");
        return true; // Skip reallocation
    }
    
    // Get input and output tensor dimensions
    auto input_dims = m_engine->getTensorShape(m_input_tensor_name);
    m_input_height = input_dims.d[2];
    m_input_width = input_dims.d[3];
    size_t input_size = 1;
    for (int j = 0; j < input_dims.nbDims; ++j) {
        input_size *= input_dims.d[j];
    }

    auto output_dims = m_engine->getTensorShape(m_output_tensor_name);
    m_output_buffer_size = 1;
    for (int j = 0; j < output_dims.nbDims; ++j) {
        m_output_buffer_size *= output_dims.d[j];
    }

    // Use buffer pool for GPU memory allocation
    m_gpu_buffers_map[m_input_tensor_name] = pool.GetBuffer(input_size * sizeof(float));
    m_gpu_buffers_map[m_output_tensor_name] = pool.GetBuffer(m_output_buffer_size * sizeof(float));

    // Allocate buffers for the raw capture and preprocessing
    // The capture resolution is hardcoded for now, matching ScreenCapture.cpp
    const int capture_width = 320;
    const int capture_height = 320;
    const size_t raw_capture_size = static_cast<size_t>(capture_width) * capture_height * 4; // BGRA

    m_raw_capture_buffer_gpu = pool.GetBuffer(raw_capture_size);
    cudaHostAlloc((void**)&m_pinned_buffer, raw_capture_size, cudaHostAllocDefault);

    return true;
}

bool InferenceEngine::BuildEngineFromOnnx(const std::string& onnx_path, bool use_fp16, int workspace_size, std::string& error_message, std::mutex& error_mutex) {
    {
        std::lock_guard<std::mutex> lock(error_mutex);
        error_message = "Creating builder...";
    }
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    if (!builder) return false;

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    
    {
        std::lock_guard<std::mutex> lock(error_mutex);
        error_message = "Creating network definition...";
    }
    auto network = builder->createNetworkV2(explicitBatch);
    if (!network) return false;

    {
        std::lock_guard<std::mutex> lock(error_mutex);
        error_message = "Creating builder config...";
    }
    auto config = builder->createBuilderConfig();
    if (!config) return false;

    {
        std::lock_guard<std::mutex> lock(error_mutex);
        error_message = "Creating ONNX parser...";
    }
    auto parser = nvonnxparser::createParser(*network, m_logger);
    if (!parser) return false;

    {
        std::lock_guard<std::mutex> lock(error_mutex);
        error_message = "Parsing ONNX file: " + fs::path(onnx_path).filename().string();
    }
    if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::lock_guard<std::mutex> lock(error_mutex);
        error_message = "Error: Failed to parse ONNX file!";
        std::cerr << "Failed to parse ONNX file" << std::endl;
        return false;
    }

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, static_cast<size_t>(workspace_size) * 1024 * 1024);
    if (use_fp16) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    
    {
        std::lock_guard<std::mutex> lock(error_mutex);
        error_message = "Building TensorRT engine... (This may take a while)";
    }
    std::cout << "Building TensorRT engine... (This may take a while)" << std::endl;
    
    nvinfer1::IHostMemory* plan = builder->buildSerializedNetwork(*network, *config);
    if (!plan) {
        std::lock_guard<std::mutex> lock(error_mutex);
        error_message = "Error: Failed to build serialized network!";
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(error_mutex);
        error_message = "Saving engine to file...";
    }
    // Save engine to file
    std::string enginePath = onnx_path.substr(0, onnx_path.find_last_of('.')) + ".engine";
    std::ofstream engineFile(enginePath, std::ios::binary);
    engineFile.write(reinterpret_cast<const char*>(plan->data()), plan->size());

    std::cout << "Engine built and saved to " << enginePath << std::endl;

    // Cleanup
    delete plan;
    delete parser;
    delete config;
    delete network;
    // builder is a unique_ptr and is destroyed automatically

    return true;
}

float InferenceEngine::IoU(const Detection& a, const Detection& b) {
    float x1 = a.xmin > b.xmin ? a.xmin : b.xmin;
    float y1 = a.ymin > b.ymin ? a.ymin : b.ymin;
    float x2 = a.xmax < b.xmax ? a.xmax : b.xmax;
    float y2 = a.ymax < b.ymax ? a.ymax : b.ymax;
    float intersection_w = x2 > x1 ? x2 - x1 : 0.0f;
    float intersection_h = y2 > y1 ? y2 - y1 : 0.0f;
    float intersection_area = intersection_w * intersection_h;

    if (intersection_area == 0.0f) return 0.0f;

    float area_a = (a.xmax - a.xmin) * (a.ymax - a.ymin);
    float area_b = (b.xmax - b.xmin) * (b.ymax - b.ymin);
    float union_area = area_a + area_b - intersection_area;

    return intersection_area / union_area;
}

void InferenceEngine::NMS(std::vector<Detection>& detections, float iou_threshold) {
    if (detections.empty()) return;

    // Sort detections by confidence in descending order
    std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
        return a.confidence > b.confidence;
    });

    std::vector<Detection> kept_detections;
    std::vector<bool> is_suppressed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); ++i) {
        if (is_suppressed[i]) continue;
        kept_detections.push_back(detections[i]);
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (is_suppressed[j]) continue;
            if (IoU(detections[i], detections[j]) > iou_threshold) {
                is_suppressed[j] = true;
            }
        }
    }
    detections = kept_detections;
}

void InferenceEngine::PostProcess(const std::vector<float>& output, std::vector<Detection>& detections, float conf_threshold, float iou_threshold, int capture_w, int capture_h, int screen_w, int screen_h) {
    detections.clear();

    const int num_detections = 2100;

    // Calculate the offset of the capture area from the top-left of the screen
    int offset_x = (screen_w - capture_w) / 2;
    int offset_y = (screen_h - capture_h) / 2;

    std::vector<Detection> candidates;
    for (int i = 0; i < num_detections; ++i) {
        float confidence = output[i + 4 * num_detections];
        if (confidence < conf_threshold) {
            continue;
    }

        // The model's output coordinates are relative to the 320x320 capture area.
        // We need to add the offset to map them to the full screen space.
        float model_xc = output[i];
        float model_yc = output[i + num_detections];
        float model_w = output[i + 2 * num_detections];
        float model_h = output[i + 3 * num_detections];

        Detection det;
        det.x_center = static_cast<int>(model_xc + offset_x);
        det.y_center = static_cast<int>(model_yc + offset_y);
        det.w = static_cast<int>(model_w);
        det.h = static_cast<int>(model_h);
        det.xmin = static_cast<int>(det.x_center - model_w / 2);
        det.ymin = static_cast<int>(det.y_center - model_h / 2);
        det.xmax = static_cast<int>(det.x_center + model_w / 2);
        det.ymax = static_cast<int>(det.y_center + model_h / 2);
        det.confidence = confidence;
        det.class_id = 0;
        candidates.push_back(det);
    }

    NMS(candidates, iou_threshold);
    detections = candidates;
}

void InferenceEngine::Infer(float* device_input_buffer, int screen_w, int screen_h, std::vector<Detection>& detections, float confidence_threshold, float nms_threshold)
{
    if (!m_context) return;

    // The input buffer is already preprocessed and on the device.
    // We can directly use it for inference.
    if (!m_context->setTensorAddress(m_input_tensor_name, device_input_buffer))
    {
        m_logger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to set input tensor address.");
        return;
    }
    if (!m_context->setTensorAddress(m_output_tensor_name, m_gpu_buffers_map[m_output_tensor_name]))
    {
        m_logger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to set output tensor address.");
        return;
    }

    // Run inference
    if (!m_context->enqueueV3(m_stream)) {
        m_logger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to enqueue inference.");
        return;
    }

    // Synchronize stream to wait for inference to complete
    cudaStreamSynchronize(m_stream);

    // Copy output from device to host
    std::vector<float> host_output_buffer(m_output_buffer_size);
    cudaMemcpyAsync(host_output_buffer.data(), m_gpu_buffers_map[m_output_tensor_name], m_output_buffer_size * sizeof(float), cudaMemcpyDeviceToHost, m_stream);
    cudaStreamSynchronize(m_stream);

    // Post-process the raw output
    PostProcess(host_output_buffer, detections, confidence_threshold, nms_threshold, m_input_width, m_input_height, screen_w, screen_h);
}

bool InferenceEngine::IsReady() const {
    return m_context != nullptr;
    }

int InferenceEngine::GetInputWidth() const {
    return m_input_width;
}
