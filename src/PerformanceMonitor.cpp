// src/PerformanceMonitor.cpp
#include "PerformanceMonitor.h"
#include "Logger.h"
#include <iostream>
#include <chrono>
#include <algorithm>

PerformanceMonitor::PerformanceMonitor() 
    : m_initialized(false), 
      m_device(nullptr), 
      m_stop_worker(false),
      m_update_interval_ms(250) // Reduced from 500ms for more responsive monitoring
{
    // Constructor
}

PerformanceMonitor::~PerformanceMonitor() {
    Shutdown();
}

bool PerformanceMonitor::Init() {
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        Logger::GetInstance().Log("Failed to initialize NVML: %s", nvmlErrorString(result));
        return false;
    }

    result = nvmlDeviceGetHandleByIndex(0, &m_device);
    if (result != NVML_SUCCESS) {
        Logger::GetInstance().Log("Failed to get handle for device 0: %s", nvmlErrorString(result));
        nvmlShutdown();
        return false;
    }

    char name[NVML_DEVICE_NAME_BUFFER_SIZE];
    result = nvmlDeviceGetName(m_device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
    if (result == NVML_SUCCESS) {
        m_metrics.deviceName = name;
    }

    // Get static device information once
    unsigned int max_power;
    result = nvmlDeviceGetEnforcedPowerLimit(m_device, &max_power);
    if (result == NVML_SUCCESS) {
        m_static_power_limit = max_power;
    }

    m_initialized = true;
    Logger::GetInstance().Log("NVML Initialized Successfully. Monitoring: %s", m_metrics.deviceName.c_str());
    
    // Start the worker thread with high performance settings
    m_stop_worker = false;
    m_worker_thread = std::thread(&PerformanceMonitor::WorkerLoop, this);

    return true;
}

void PerformanceMonitor::Shutdown() {
    m_stop_worker = true;
    if (m_worker_thread.joinable()) {
        m_worker_thread.join();
    }
    
    if (m_initialized) {
        nvmlShutdown();
        m_initialized = false;
        Logger::GetInstance().Log("NVML Shutdown.");
    }
}

void PerformanceMonitor::WorkerLoop() {
    // Set thread priority for better performance monitoring accuracy
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    
    auto last_update = std::chrono::high_resolution_clock::now();
    GpuMetrics local_metrics;
    local_metrics.deviceName = m_metrics.deviceName; // Copy device name once
    
    while (!m_stop_worker) {
        if (!m_initialized) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_update);
        
        if (elapsed.count() >= m_update_interval_ms) {
            UpdateMetrics(local_metrics);
            
            // Atomic update of shared metrics
            {
                std::lock_guard<std::mutex> lock(m_metrics_mutex);
                m_metrics = local_metrics;
            }
            
            last_update = current_time;
        }

        // Adaptive sleep based on workload
        auto sleep_duration = std::min(50, static_cast<int>(m_update_interval_ms / 5));
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_duration));
    }
}

void PerformanceMonitor::UpdateMetrics(GpuMetrics& metrics) {
    nvmlReturn_t result;

    // Temperature
    result = nvmlDeviceGetTemperature(m_device, NVML_TEMPERATURE_GPU, &metrics.temperature);
    if (result != NVML_SUCCESS) {
        metrics.temperature = 0;
    }

    // Utilization (most important metric)
    nvmlUtilization_t utilization;
    result = nvmlDeviceGetUtilizationRates(m_device, &utilization);
    if (result == NVML_SUCCESS) {
        metrics.utilizationGpu = utilization.gpu;
        metrics.utilizationMemory = utilization.memory;
    } else {
        metrics.utilizationGpu = 0;
        metrics.utilizationMemory = 0;
    }

    // Memory Info
    nvmlMemory_t memory;
    result = nvmlDeviceGetMemoryInfo(m_device, &memory);
    if (result == NVML_SUCCESS) {
        metrics.memoryUsed = memory.used;
        metrics.memoryTotal = memory.total;
    } else {
        metrics.memoryUsed = 0;
        metrics.memoryTotal = 0;
    }

    // Power Usage
    result = nvmlDeviceGetPowerUsage(m_device, &metrics.powerUsage);
    if (result != NVML_SUCCESS) {
        metrics.powerUsage = 0;
    }
    
    // Use cached power limit instead of querying every time
    metrics.powerLimit = m_static_power_limit;
}

GpuMetrics PerformanceMonitor::GetMetrics() {
    std::lock_guard<std::mutex> lock(m_metrics_mutex);
    return m_metrics;
}

void PerformanceMonitor::SetUpdateInterval(int interval_ms) {
    m_update_interval_ms = std::max(100, std::min(2000, interval_ms)); // Clamp between 100ms and 2s
}

float PerformanceMonitor::GetGpuUtilizationSmoothed() {
    std::lock_guard<std::mutex> lock(m_metrics_mutex);
    
    // Add current utilization to history
    m_gpu_utilization_history.push_back(m_metrics.utilizationGpu);
    
    // Keep only last 10 samples for smoothing
    if (m_gpu_utilization_history.size() > 10) {
        m_gpu_utilization_history.pop_front();
    }
    
    // Calculate moving average
    float sum = 0.0f;
    for (unsigned int util : m_gpu_utilization_history) {
        sum += util;
    }
    
    return m_gpu_utilization_history.empty() ? 0.0f : sum / m_gpu_utilization_history.size();
}

bool PerformanceMonitor::IsGpuOverloaded() {
    return GetGpuUtilizationSmoothed() > 95.0f;
}
