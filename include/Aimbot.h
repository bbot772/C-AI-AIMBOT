// Aimbot.h
#pragma once

#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <memory>
#include <filesystem>
#include "ConfigManager.h"
#include "AimingSettings.h"
#include "ScreenCapture.h"
#include "types.h"
#include <cuda_runtime.h>
#include "InferenceEngine.h"
#include "PersonalizationSettings.h"

// Forward declarations to speed up compilation
class Aiming;
class InferenceEngine;
struct ID3D11Texture2D;
struct cudaGraphicsResource;

class Aimbot {
public:
    Aimbot();
    ~Aimbot();

    void Run();
    void Stop();
    bool IsRunning() const;

    void LoadAndStart(const std::string& model_path, bool use_fp16, int workspace_size);
    std::string GetLastLoadedModelPath() const;
    void SetLastLoadedModelPath(const std::string& path);
    void SetSelectedModel(const std::string& model_name);
    void InitiateModelBuild(const std::string& model_path);
    void LoadModel(const std::string& engine_path);
    
    std::string GetBuildStatus() const;
    
    // Public members for easy access from UI
    AimingSettings m_aiming_settings;
    PersonalizationSettings m_personalization_settings;
    ConfigManager m_config_manager;
    
    bool fp16_mode = true;
    int workspace_size = 4; // in GB
    std::atomic<bool> is_building_engine{false};

    // UI state variables
    std::atomic<int> killswitch_key{ 0 };
    std::atomic<int> hide_toggle_key{ 0 };

    int batch_size = 1;

    // GPU-related resources
    const int m_capture_width = 320;
    const int m_capture_height = 320;
    float* m_preprocessed_buffer_gpu = nullptr;
    cudaGraphicsResource* m_cuda_graphics_resource = nullptr;

private:
    void InferenceThread();
    void BuildEngineThread(const std::string& onnx_path, bool use_fp16, int ws_size);

    std::atomic<bool> m_running{false};
    std::thread m_inference_thread;

    // Must be declared after m_aiming_settings
    std::unique_ptr<InferenceEngine> m_inference_engine;
    std::unique_ptr<Aiming> m_aiming;
    ScreenCapture m_capture;

    std::vector<Detection> m_detections;
    std::mutex m_detections_mutex;
    
    // Model loading and building state
    std::string m_last_loaded_model_path;
    std::string build_status_message;
    mutable std::mutex build_status_mutex;
};
