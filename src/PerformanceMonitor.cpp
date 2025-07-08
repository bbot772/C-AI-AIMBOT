// src/PerformanceMonitor.cpp
#include "PerformanceMonitor.h"
#include "Logger.h"
#include <iostream>
#include <chrono>

PerformanceMonitor::PerformanceMonitor() : m_initialized(false), m_device(nullptr), m_stop_worker(false) {
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

    m_initialized = true;
    Logger::GetInstance().Log("NVML Initialized Successfully. Monitoring: %s", m_metrics.deviceName.c_str());
    
    // Start the worker thread
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
    while (!m_stop_worker) {
        if (!m_initialized) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        GpuMetrics temp_metrics;
        // Keep device name
        temp_metrics.deviceName = m_metrics.deviceName;

    nvmlReturn_t result;

    // Get Temperature
        result = nvmlDeviceGetTemperature(m_device, NVML_TEMPERATURE_GPU, &temp_metrics.temperature);
        if (result != NVML_SUCCESS) temp_metrics.temperature = 0;

    // Get Utilization
    nvmlUtilization_t utilization;
    result = nvmlDeviceGetUtilizationRates(m_device, &utilization);
    if (result == NVML_SUCCESS) {
            temp_metrics.utilizationGpu = utilization.gpu;
            temp_metrics.utilizationMemory = utilization.memory;
    } else {
            temp_metrics.utilizationGpu = 0;
            temp_metrics.utilizationMemory = 0;
    }

    // Get Memory Info
    nvmlMemory_t memory;
    result = nvmlDeviceGetMemoryInfo(m_device, &memory);
    if (result == NVML_SUCCESS) {
            temp_metrics.memoryUsed = memory.used;
            temp_metrics.memoryTotal = memory.total;
    } else {
            temp_metrics.memoryUsed = 0;
            temp_metrics.memoryTotal = 0;
    }

    // Get Power Usage
        result = nvmlDeviceGetPowerUsage(m_device, &temp_metrics.powerUsage);
        if (result != NVML_SUCCESS) temp_metrics.powerUsage = 0;

        result = nvmlDeviceGetEnforcedPowerLimit(m_device, &temp_metrics.powerLimit);
        if (result != NVML_SUCCESS) temp_metrics.powerLimit = 0;
    
        // Lock and update the shared metrics
        {
            std::lock_guard<std::mutex> lock(m_metrics_mutex);
            m_metrics = temp_metrics;
}

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

GpuMetrics PerformanceMonitor::GetMetrics() {
    std::lock_guard<std::mutex> lock(m_metrics_mutex);
    return m_metrics;
}
