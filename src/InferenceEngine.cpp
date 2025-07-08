// InferenceEngine.cpp
#define NOMINMAX
#include "../include/InferenceEngine.h"
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
#include <chrono>

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
    
    // Create multiple CUDA streams for better overlap
    cudaStreamCreateWithFlags(&m_inference_stream, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&m_copy_stream, cudaStreamNonBlocking);

    // The capture resolution is configurable now
    InitializeBuffers(320, 320); // Default size, can be changed later
}

InferenceEngine::~InferenceEngine() {
    CleanupResources();
}

void InferenceEngine::CleanupResources() {
    // Cleanup CUDA streams
    if (m_inference_stream) {
        cudaStreamSynchronize(m_inference_stream);
        cudaStreamDestroy(m_inference_stream);
        m_inference_stream = nullptr;
    }
    if (m_copy_stream) {
        cudaStreamSynchronize(m_copy_stream);
        cudaStreamDestroy(m_copy_stream);
        m_copy_stream = nullptr;
    }
    
    // Cleanup pinned memory
    if (m_pinned_buffer) {
        cudaFreeHost(m_pinned_buffer);
        m_pinned_buffer = nullptr;
    }
    
    // Cleanup GPU buffers
    if (m_raw_capture_buffer_gpu) {
        cudaFree(m_raw_capture_buffer_gpu);
        m_raw_capture_buffer_gpu = nullptr;
    }
    
    // Free GPU buffers for the model
    for (auto& buffer_pair : m_gpu_buffers_map) {
        if (buffer_pair.second) {
            cudaFree(buffer_pair.second);
        }
    }
    m_gpu_buffers_map.clear();
    
    // Cleanup host output buffer
    if (m_host_output_buffer) {
        delete[] m_host_output_buffer;
        m_host_output_buffer = nullptr;
    }
}

void InferenceEngine::InitializeBuffers(int capture_width, int capture_height) {
    const size_t raw_capture_size = capture_width * capture_height * 4; // BGRA

    // Allocate GPU buffer for the raw capture
    if (cudaMalloc(&m_raw_capture_buffer_gpu, raw_capture_size) != cudaSuccess) {
        std::cerr << "Failed to allocate raw capture buffer on GPU" << std::endl;
        return;
    }
    
    // Allocate Pinned Host Memory for raw capture input
    if (cudaHostAlloc((void**)&m_pinned_buffer, raw_capture_size, cudaHostAllocDefault) != cudaSuccess) {
        std::cerr << "Failed to allocate pinned host memory" << std::endl;
        return;
    }
    
    m_capture_width = capture_width;
    m_capture_height = capture_height;
}

int InferenceEngine::GetInputHeight() const {
    return m_input_height;
}

bool InferenceEngine::LoadModel(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Failed to open engine file: " << engine_path << std::endl;
        return false;
    }

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    m_engine.reset(m_runtime->deserializeCudaEngine(engine_data.data(), size));
    if (!m_engine) {
        std::cerr << "Failed to deserialize CUDA engine" << std::endl;
        return false;
    }

    m_context.reset(m_engine->createExecutionContext());
    if (!m_context) {
        std::cerr << "Failed to create execution context" << std::endl;
        return false;
    }

    // Clean up old buffers before allocating new ones
    CleanupModelBuffers();
    
    // Get input and output tensor dimensions
    auto input_dims = m_engine->getTensorShape(m_input_tensor_name);
    m_input_height = static_cast<int>(input_dims.d[2]);
    m_input_width = static_cast<int>(input_dims.d[3]);
    size_t input_size = 1;
    for (int j = 0; j < input_dims.nbDims; ++j) {
        input_size *= input_dims.d[j];
    }

    auto output_dims = m_engine->getTensorShape(m_output_tensor_name);
    m_output_buffer_size = 1;
    for (int j = 0; j < output_dims.nbDims; ++j) {
        m_output_buffer_size *= output_dims.d[j];
    }

    // Allocate GPU buffers for model input/output with error checking
    if (cudaMalloc(&m_gpu_buffers_map[m_input_tensor_name], input_size * sizeof(float)) != cudaSuccess) {
        std::cerr << "Failed to allocate input buffer" << std::endl;
        return false;
    }
    
    if (cudaMalloc(&m_gpu_buffers_map[m_output_tensor_name], m_output_buffer_size * sizeof(float)) != cudaSuccess) {
        std::cerr << "Failed to allocate output buffer" << std::endl;
        return false;
    }

    // Allocate host output buffer for async copying
    m_host_output_buffer = new(std::nothrow) float[m_output_buffer_size];
    if (!m_host_output_buffer) {
        std::cerr << "Failed to allocate host output buffer" << std::endl;
        return false;
    }

    // Re-initialize capture buffers if needed
    if (m_capture_width != 320 || m_capture_height != 320) {
        // Clean up old buffers
        if (m_pinned_buffer) cudaFreeHost(m_pinned_buffer);
        if (m_raw_capture_buffer_gpu) cudaFree(m_raw_capture_buffer_gpu);
        
        // Reinitialize with default size
        InitializeBuffers(320, 320);
    }

    return true;
}

void InferenceEngine::CleanupModelBuffers() {
    if (m_gpu_buffers_map.count(m_input_tensor_name) && m_gpu_buffers_map[m_input_tensor_name]) {
        cudaFree(m_gpu_buffers_map[m_input_tensor_name]);
        m_gpu_buffers_map.erase(m_input_tensor_name);
    }
    if (m_gpu_buffers_map.count(m_output_tensor_name) && m_gpu_buffers_map[m_output_tensor_name]) {
        cudaFree(m_gpu_buffers_map[m_output_tensor_name]);
        m_gpu_buffers_map.erase(m_output_tensor_name);
    }
    
    if (m_host_output_buffer) {
        delete[] m_host_output_buffer;
        m_host_output_buffer = nullptr;
    }
}

bool InferenceEngine::BuildEngineFromOnnx(const std::string& onnx_path, bool use_fp16, int workspace_size, std::string& error_message, std::mutex& error_mutex) {
    {
        std::lock_guard<std::mutex> lock(error_mutex);
        error_message = "Creating builder...";
    }
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    if (!builder) {
        std::lock_guard<std::mutex> lock(error_mutex);
        error_message = "Error: Failed to create builder!";
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    
    {
        std::lock_guard<std::mutex> lock(error_mutex);
        error_message = "Creating network definition...";
    }
    auto network = builder->createNetworkV2(explicitBatch);
    if (!network) {
        std::lock_guard<std::mutex> lock(error_mutex);
        error_message = "Error: Failed to create network!";
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(error_mutex);
        error_message = "Creating builder config...";
    }
    auto config = builder->createBuilderConfig();
    if (!config) {
        std::lock_guard<std::mutex> lock(error_mutex);
        error_message = "Error: Failed to create builder config!";
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(error_mutex);
        error_message = "Creating ONNX parser...";
    }
    auto parser = nvonnxparser::createParser(*network, m_logger);
    if (!parser) {
        std::lock_guard<std::mutex> lock(error_mutex);
        error_message = "Error: Failed to create ONNX parser!";
        return false;
    }

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

    // Enhanced builder configuration
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, static_cast<size_t>(workspace_size) * 1024 * 1024);
    
    if (use_fp16) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    
    // Enable optimizations
    config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
    
    {
        std::lock_guard<std::mutex> lock(error_mutex);
        error_message = "Building TensorRT engine... (This may take a while)";
    }
    std::cout << "Building TensorRT engine... (This may take a while)" << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    nvinfer1::IHostMemory* plan = builder->buildSerializedNetwork(*network, *config);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
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
    if (!engineFile) {
        std::lock_guard<std::mutex> lock(error_mutex);
        error_message = "Error: Failed to create engine file!";
        delete plan;
        return false;
    }
    
    engineFile.write(reinterpret_cast<const char*>(plan->data()), plan->size());
    engineFile.close();

    std::cout << "Engine built and saved to " << enginePath << " (took " << duration.count() << " seconds)" << std::endl;

    // Cleanup
    delete plan;
    delete parser;
    delete config;
    delete network;

    return true;
}

float InferenceEngine::IoU(const Detection& a, const Detection& b) {
    const float x1 = std::max(static_cast<float>(a.xmin), static_cast<float>(b.xmin));
    const float y1 = std::max(static_cast<float>(a.ymin), static_cast<float>(b.ymin));
    const float x2 = std::min(static_cast<float>(a.xmax), static_cast<float>(b.xmax));
    const float y2 = std::min(static_cast<float>(a.ymax), static_cast<float>(b.ymax));
    
    const float intersection_w = std::max(0.0f, x2 - x1);
    const float intersection_h = std::max(0.0f, y2 - y1);
    const float intersection_area = intersection_w * intersection_h;

    if (intersection_area == 0.0f) return 0.0f;

    const float area_a = (a.xmax - a.xmin) * (a.ymax - a.ymin);
    const float area_b = (b.xmax - b.xmin) * (b.ymax - b.ymin);
    const float union_area = area_a + area_b - intersection_area;

    return union_area > 0.0f ? intersection_area / union_area : 0.0f;
}

void InferenceEngine::NMS(std::vector<Detection>& detections, float iou_threshold) {
    if (detections.empty()) return;

    // Sort detections by confidence in descending order
    std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
        return a.confidence > b.confidence;
    });

    std::vector<Detection> kept_detections;
    kept_detections.reserve(detections.size()); // Pre-allocate for efficiency
    
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
    detections = std::move(kept_detections);
}

void InferenceEngine::PostProcess(const std::vector<float>& output, std::vector<Detection>& detections, float conf_threshold, float iou_threshold, int capture_w, int capture_h, int screen_w, int screen_h) {
    detections.clear();

    const int num_detections = 2100;

    // Calculate the offset of the capture area from the top-left of the screen
    const int offset_x = (screen_w - capture_w) / 2;
    const int offset_y = (screen_h - capture_h) / 2;

    std::vector<Detection> candidates;
    candidates.reserve(num_detections); // Pre-allocate for efficiency
    
    for (int i = 0; i < num_detections; ++i) {
        const float confidence = output[i + 4 * num_detections];
        if (confidence < conf_threshold) {
            continue;
        }

        // The model's output coordinates are relative to the capture area.
        // We need to add the offset to map them to the full screen space.
        const float model_xc = output[i];
        const float model_yc = output[i + num_detections];
        const float model_w = output[i + 2 * num_detections];
        const float model_h = output[i + 3 * num_detections];

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
    detections = std::move(candidates);
}

void InferenceEngine::Infer(
    float* device_input_buffer,
    int screen_w,
    int screen_h,
    std::vector<Detection>& detections,
    float confidence_threshold,
    float iou_threshold
) {
    if (!m_context) {
        std::cerr << "Inference context not ready" << std::endl;
        return;
    }

    // Set tensor addresses
    if (!m_context->setTensorAddress(m_input_tensor_name, device_input_buffer)) {
        m_logger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to set input tensor address.");
        return;
    }
    if (!m_context->setTensorAddress(m_output_tensor_name, m_gpu_buffers_map[m_output_tensor_name])) {
        m_logger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to set output tensor address.");
        return;
    }

    // Run inference asynchronously
    if (!m_context->enqueueV3(m_inference_stream)) {
        m_logger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to enqueue inference.");
        return;
    }

    // Copy output from device to host asynchronously
    cudaError_t err = cudaMemcpyAsync(
        m_host_output_buffer, 
        m_gpu_buffers_map[m_output_tensor_name], 
        m_output_buffer_size * sizeof(float), 
        cudaMemcpyDeviceToHost, 
        m_copy_stream
    );
    
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy output data: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Synchronize only the copy stream for minimal blocking
    cudaStreamSynchronize(m_copy_stream);

    // Post-processing on the CPU using the host buffer
    std::vector<float> output_vector(m_host_output_buffer, m_host_output_buffer + m_output_buffer_size);
    PostProcess(output_vector, detections, confidence_threshold, iou_threshold, GetInputWidth(), GetInputHeight(), screen_w, screen_h);
}

bool InferenceEngine::IsReady() const {
    return m_context != nullptr;
}

int InferenceEngine::GetInputWidth() const {
    return m_input_width;
}
