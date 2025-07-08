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

namespace fs = std::filesystem;

Aimbot::Aimbot() 
    : m_config_manager(*this),
      m_aiming(std::make_unique<Aiming>(m_aiming_settings))
{
    m_inference_engine = std::make_unique<InferenceEngine>();
    m_preprocessed_buffer_gpu = AllocatePreprocessedBuffer(m_capture_width, m_capture_height);
    m_config_manager.Load();
    Logger::GetInstance().Log("Aimbot Initialized.");
}

Aimbot::~Aimbot() {
    Stop();
    if (m_preprocessed_buffer_gpu) {
        cudaFree(m_preprocessed_buffer_gpu);
    }
    if (m_cuda_graphics_resource) {
        cudaGraphicsUnregisterResource(m_cuda_graphics_resource);
    }
    m_config_manager.Save();
}

void Aimbot::Run() {
    if (m_running) return;
    m_running = true;

    m_inference_thread = std::thread(&Aimbot::InferenceThread, this);
    Logger::GetInstance().Log("Aimbot threads started.");
}

void Aimbot::Stop() {
    if (!m_running) return;
    m_running = false;
    if (m_inference_thread.joinable()) {
        m_inference_thread.join();
    }
    if (m_cuda_graphics_resource) {
        cudaGraphicsUnregisterResource(m_cuda_graphics_resource);
        m_cuda_graphics_resource = nullptr;
    }
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
    // This function can be called from the UI thread, so we must lock the mutex
    // to safely read the status message that's written by the build thread.
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(build_status_mutex));
    return build_status_message;
}

void Aimbot::InferenceThread() {
    m_capture.Init();
    
    int screen_w = m_capture.GetDesktopWidth();
    int screen_h = m_capture.GetDesktopHeight();
    
    // Define the target frame duration for 144 FPS
    constexpr auto target_frame_duration = std::chrono::microseconds(1000000 / 144);

    while (m_running) {
        auto frame_start_time = std::chrono::high_resolution_clock::now();

        if (!m_inference_engine->IsReady()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        ID3D11Texture2D* captured_frame = m_capture.CaptureScreen(m_capture_width, m_capture_height);
        
        if (captured_frame) {
            // Preprocess the frame on the GPU
            if (PreprocessD3D11Texture(captured_frame, m_preprocessed_buffer_gpu, m_capture_width, m_capture_height, m_inference_engine->GetInputWidth(), m_inference_engine->GetInputHeight(), &m_cuda_graphics_resource))
            {
                std::vector<Detection> new_detections;
                // Pass the pre-processed buffer to the inference engine
                m_inference_engine->Infer(
                    (float*)m_preprocessed_buffer_gpu,
                    screen_w,
                    screen_h,
                    new_detections,
                    m_aiming_settings.confidence_threshold,
                    this->iou_threshold // Pass the IOU threshold
                );

                // If we have a target, process it
                if (m_aiming) {
                    auto [aim_dx, aim_dy] = m_aiming->ProcessDetections(new_detections, screen_w, screen_h);
                    auto [recoil_dx, recoil_dy] = m_aiming->ControlRecoil(m_recoil_settings);

                    m_aiming->ApplyMouseMovement(aim_dx + recoil_dx, aim_dy + recoil_dy);
                    
                    // The returned data from ProcessDetections is no longer PID data, so plotting is disabled for now.
                    // A proper implementation would require ProcessDetections to return both movement and error data.
                    // if (m_is_recording_pid) {
                    //     std::lock_guard<std::mutex> lock(plot_data_mutex);
                    //     pid_plot_data.AddPoint(pid_data.first, pid_data.second);
                    // }
                }

                // Detections are processed, clear for next frame
                new_detections.clear();

                // The old logic for sharing detections is kept for potential ESP visualization
                {
                    std::lock_guard<std::mutex> lock(m_detections_mutex);
                    m_detections = new_detections;
                }
            }
        }

        // Cap the frame rate to 144 FPS to reduce GPU usage
        auto frame_end_time = std::chrono::high_resolution_clock::now();
        auto frame_duration = std::chrono::duration_cast<std::chrono::microseconds>(frame_end_time - frame_start_time);
        
        if (frame_duration < target_frame_duration) {
            std::this_thread::sleep_for(target_frame_duration - frame_duration);
        }
    }
    m_capture.Cleanup();
}

void Aimbot::BuildEngineThread(const std::string& onnx_path, bool use_fp16, int ws_size) {
    is_building_engine = true;
    
    if (m_inference_engine->BuildEngineFromOnnx(onnx_path, use_fp16, ws_size, build_status_message, build_status_mutex)) {
        Logger::GetInstance().Log("Engine built successfully from %s", onnx_path.c_str());
        std::string engine_path = fs::path(onnx_path).replace_extension(".engine").string();
        LoadAndStart(engine_path, use_fp16, ws_size);
    } else {
        Logger::GetInstance().Log("Error: Failed to build engine from %s", onnx_path.c_str());
    }
    
    is_building_engine = false;
} 