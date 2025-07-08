#include "Aimbot.h"
#include "InferenceEngine.h"
#include "Aiming.h"
#include "Logger.h"
#include "types.h"
#include "Path.h"
#include "CudaKernels.h"
#include <windows.h>
#include <filesystem>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>

namespace fs = std::filesystem;

Aimbot::Aimbot() 
    : m_config_manager(*this),
      m_aiming(std::make_unique<Aiming>(m_aiming_settings)),
      m_adaptive_frame_rate(144), // Start with 144 FPS
      m_frame_time_history(10)    // Keep last 10 frame times for smoothing
{
    m_inference_engine = std::make_unique<InferenceEngine>();
    m_preprocessed_buffer_gpu = AllocatePreprocessedBuffer(m_capture_width, m_capture_height);
    m_config_manager.Load();
    Logger::GetInstance().Log("Aimbot Initialized with adaptive performance optimization.");
}

Aimbot::~Aimbot() {
    Stop();
    CleanupResources();
    m_config_manager.Save();
}

void Aimbot::CleanupResources() {
    if (m_preprocessed_buffer_gpu) {
        cudaFree(m_preprocessed_buffer_gpu);
        m_preprocessed_buffer_gpu = nullptr;
    }
    if (m_cuda_graphics_resource) {
        RemoveFromResourceCache(reinterpret_cast<ID3D11Texture2D*>(m_cuda_graphics_resource));
        m_cuda_graphics_resource = nullptr;
    }
    CleanupGraphicsResourceCache(); // Clean up global resource cache
}

void Aimbot::Run() {
    if (m_running) return;
    m_running = true;

    m_inference_thread = std::thread(&Aimbot::InferenceThread, this);
    Logger::GetInstance().Log("Aimbot threads started with optimized performance.");
}

void Aimbot::Stop() {
    if (!m_running) return;
    m_running = false;
    if (m_inference_thread.joinable()) {
        m_inference_thread.join();
    }
    CleanupResources();
}

bool Aimbot::IsRunning() const {
    return m_running;
}

void Aimbot::LoadAndStart(const std::string& model_path, bool use_fp16, int workspace_size) {
    if (is_building_engine) return;

    Stop();
    
    m_last_loaded_model_path = model_path;

    std::string ext = fs::path(model_path).extension().string();
    if (ext == ".onnx") {
        std::thread(&Aimbot::BuildEngineThread, this, model_path, use_fp16, workspace_size).detach();
    } else if (ext == ".engine") {
        if (m_inference_engine->LoadModel(model_path)) {
            Logger::GetInstance().Log("Successfully loaded TensorRT engine: %s", model_path.c_str());
            Run();
        } else {
            Logger::GetInstance().Log("Error: Failed to load TensorRT engine: %s", model_path.c_str());
        }
    }
}

std::string Aimbot::GetLastLoadedModelPath() const {
    return m_last_loaded_model_path;
}

void Aimbot::SetLastLoadedModelPath(const std::string& path) {
    m_last_loaded_model_path = path;
}

void Aimbot::SetSelectedModel(const std::string& model_name) {
    m_last_loaded_model_path = (get_models_path() / model_name).string();
}

void Aimbot::InitiateModelBuild(const std::string& model_path) {
    if (is_building_engine) return;
    Stop();
    std::thread(&Aimbot::BuildEngineThread, this, model_path, fp16_mode, workspace_size).detach();
}

void Aimbot::LoadModel(const std::string& engine_path) {
    if (is_building_engine) return;
    Stop();
    if (m_inference_engine->LoadModel(engine_path)) {
        Logger::GetInstance().Log("Successfully loaded TensorRT engine: %s", engine_path.c_str());
        m_last_loaded_model_path = engine_path;
        Run();
    } else {
        Logger::GetInstance().Log("Error: Failed to load TensorRT engine: %s", engine_path.c_str());
    }
}

std::string Aimbot::GetBuildStatus() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(build_status_mutex));
    return build_status_message;
}

void Aimbot::UpdateAdaptiveFrameRate(std::chrono::microseconds frame_time) {
    m_frame_time_history.push_back(frame_time.count());
    
    // Calculate average frame time
    long long avg_frame_time = 0;
    for (const auto& time : m_frame_time_history) {
        avg_frame_time += time;
    }
    avg_frame_time /= m_frame_time_history.size();
    
    // Adaptive frame rate logic
    const long long target_120fps = 8333;  // 8.33ms in microseconds
    const long long target_144fps = 6944;  // 6.94ms in microseconds
    const long long target_240fps = 4167;  // 4.17ms in microseconds
    
    if (avg_frame_time < target_240fps) {
        m_adaptive_frame_rate = 240;
    } else if (avg_frame_time < target_144fps) {
        m_adaptive_frame_rate = 180;
    } else if (avg_frame_time < target_120fps) {
        m_adaptive_frame_rate = 144;
    } else {
        m_adaptive_frame_rate = 120;
    }
}

void Aimbot::InferenceThread() {
    if (!m_capture.Init()) {
        Logger::GetInstance().Log("Failed to initialize screen capture");
        return;
    }
    
    int screen_w = m_capture.GetDesktopWidth();
    int screen_h = m_capture.GetDesktopHeight();
    
    // Performance monitoring variables
    std::chrono::high_resolution_clock::time_point last_fps_update = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    float current_fps = 0.0f;
    
    Logger::GetInstance().Log("Inference thread started - Screen: %dx%d", screen_w, screen_h);

    while (m_running) {
        auto frame_start_time = std::chrono::high_resolution_clock::now();

        if (!m_inference_engine->IsReady()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        ID3D11Texture2D* captured_frame = m_capture.CaptureScreen(m_capture_width, m_capture_height);
        
        if (captured_frame) {
            // Use optimized preprocessing with resource caching
            if (PreprocessD3D11TextureOptimized(
                captured_frame, 
                m_preprocessed_buffer_gpu, 
                m_capture_width, 
                m_capture_height, 
                m_inference_engine->GetInputWidth(), 
                m_inference_engine->GetInputHeight(), 
                &m_cuda_graphics_resource))
            {
                std::vector<Detection> new_detections;
                
                // Run inference with async operations
                m_inference_engine->Infer(
                    (float*)m_preprocessed_buffer_gpu,
                    screen_w,
                    screen_h,
                    new_detections,
                    m_aiming_settings.confidence_threshold,
                    this->iou_threshold
                );

                // Process aiming if we have detections
                if (m_aiming && !new_detections.empty()) {
                    auto [aim_dx, aim_dy] = m_aiming->ProcessDetections(new_detections, screen_w, screen_h);
                    auto [recoil_dx, recoil_dy] = m_aiming->ControlRecoil(m_recoil_settings);

                    m_aiming->ApplyMouseMovement(aim_dx + recoil_dx, aim_dy + recoil_dy);
                }

                // Update shared detections with minimal locking
                {
                    std::lock_guard<std::mutex> lock(m_detections_mutex);
                    m_detections = std::move(new_detections);
                }
                
                // Update performance metrics
                frame_count++;
                auto now = std::chrono::high_resolution_clock::now();
                auto fps_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_fps_update);
                
                if (fps_elapsed.count() >= 1000) { // Update FPS every second
                    current_fps = frame_count * 1000.0f / fps_elapsed.count();
                    frame_count = 0;
                    last_fps_update = now;
                    
                    if (current_fps < 100.0f) {
                        Logger::GetInstance().Log("Performance warning: FPS dropped to %.1f", current_fps);
                    }
                }
            }
        }

        // Adaptive frame rate limiting
        auto frame_end_time = std::chrono::high_resolution_clock::now();
        auto frame_duration = std::chrono::duration_cast<std::chrono::microseconds>(frame_end_time - frame_start_time);
        
        UpdateAdaptiveFrameRate(frame_duration);
        
        auto target_frame_duration = std::chrono::microseconds(1000000 / m_adaptive_frame_rate);
        if (frame_duration < target_frame_duration) {
            auto sleep_time = target_frame_duration - frame_duration;
            std::this_thread::sleep_for(sleep_time);
        }
    }
    
    m_capture.Cleanup();
    Logger::GetInstance().Log("Inference thread terminated");
}

void Aimbot::BuildEngineThread(const std::string& onnx_path, bool use_fp16, int ws_size) {
    is_building_engine = true;
    
    try {
        if (m_inference_engine->BuildEngineFromOnnx(onnx_path, use_fp16, ws_size, build_status_message, build_status_mutex)) {
            Logger::GetInstance().Log("Engine built successfully from %s", onnx_path.c_str());
            std::string engine_path = fs::path(onnx_path).replace_extension(".engine").string();
            LoadAndStart(engine_path, use_fp16, ws_size);
        } else {
            Logger::GetInstance().Log("Error: Failed to build engine from %s", onnx_path.c_str());
        }
    } catch (const std::exception& e) {
        Logger::GetInstance().Log("Exception in BuildEngineThread: %s", e.what());
    }
    
    is_building_engine = false;
}

int Aimbot::GetAdaptiveFrameRate() const {
    return m_adaptive_frame_rate;
} 