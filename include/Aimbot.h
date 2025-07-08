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
#include "RecoilSettings.h"

// Forward declarations to speed up compilation
class Aiming;
class InferenceEngine;
struct ID3D11Texture2D;
struct cudaGraphicsResource;

struct PlotData {
    std::vector<float> error;
    std::vector<float> output;
    int max_size;
    int offset;
    PlotData(int size = 200) : max_size(size), offset(0) {
        error.resize(size);
        output.resize(size);
    }
    void AddPoint(float error_val, float output_val) {
        error[offset] = error_val;
        output[offset] = output_val;
        offset = (offset + 1) % max_size;
    }
};

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
    RecoilSettings m_recoil_settings;
    ConfigManager m_config_manager;
    
    bool fp16_mode = true;
    int workspace_size = 1024;
    int batch_size = 1;
    float iou_threshold = 0.5f;
    std::atomic<bool> is_building_engine{false};

    // UI state variables
    std::atomic<int> killswitch_key{ 0 };
    std::atomic<int> hide_toggle_key{ 0 };
    std::atomic<bool> m_is_recording_pid{false};

    // GPU-related resources
    const int m_capture_width = 320;
    const int m_capture_height = 320;
    float* m_preprocessed_buffer_gpu = nullptr;
    cudaGraphicsResource* m_cuda_graphics_resource = nullptr;

    // Plotting Data
    std::mutex plot_data_mutex;
    PlotData pid_plot_data;
    void ClearPlotData() {
        std::lock_guard<std::mutex> lock(plot_data_mutex);
        pid_plot_data.error.assign(pid_plot_data.max_size, 0);
        pid_plot_data.output.assign(pid_plot_data.max_size, 0);
        pid_plot_data.offset = 0;
    }

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
