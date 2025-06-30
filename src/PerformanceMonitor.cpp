// src/PerformanceMonitor.cpp
#include "PerformanceMonitor.h"
#include "Logger.h"
#include <iostream>

PerformanceMonitor::PerformanceMonitor() {
    // Constructor
}

PerformanceMonitor::~PerformanceMonitor() {
    if (m_initialized) {
        Shutdown();
    }
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
    return true;
}

void PerformanceMonitor::Shutdown() {
    nvmlShutdown();
    m_initialized = false;
    Logger::GetInstance().Log("NVML Shutdown.");
}

bool PerformanceMonitor::UpdateMetrics() {
    if (!m_initialized) return false;

    nvmlReturn_t result;

    // Get Temperature
    result = nvmlDeviceGetTemperature(m_device, NVML_TEMPERATURE_GPU, &m_metrics.temperature);
    if (result != NVML_SUCCESS) m_metrics.temperature = 0;

    // Get Utilization
    nvmlUtilization_t utilization;
    result = nvmlDeviceGetUtilizationRates(m_device, &utilization);
    if (result == NVML_SUCCESS) {
        m_metrics.utilizationGpu = utilization.gpu;
        m_metrics.utilizationMemory = utilization.memory;
    } else {
        m_metrics.utilizationGpu = 0;
        m_metrics.utilizationMemory = 0;
    }

    // Get Memory Info
    nvmlMemory_t memory;
    result = nvmlDeviceGetMemoryInfo(m_device, &memory);
    if (result == NVML_SUCCESS) {
        m_metrics.memoryUsed = memory.used;
        m_metrics.memoryTotal = memory.total;
    } else {
        m_metrics.memoryUsed = 0;
        m_metrics.memoryTotal = 0;
    }

    // Get Power Usage
    result = nvmlDeviceGetPowerUsage(m_device, &m_metrics.powerUsage);
    if (result != NVML_SUCCESS) m_metrics.powerUsage = 0;

    result = nvmlDeviceGetEnforcedPowerLimit(m_device, &m_metrics.powerLimit);
    if (result != NVML_SUCCESS) m_metrics.powerLimit = 0;
    
    return true;
}

const GpuMetrics& PerformanceMonitor::GetMetrics() const {
    return m_metrics;
}
