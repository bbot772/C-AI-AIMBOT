// include/PerformanceMonitor.h
#pragma once

#include <string>
#include <vector>
#include <nvml.h>

struct GpuMetrics {
    std::string   deviceName;
    unsigned int  temperature;
    unsigned int  utilizationGpu;
    unsigned int  utilizationMemory;
    unsigned long long memoryTotal;
    unsigned long long memoryUsed;
    unsigned int  powerUsage;
    unsigned int  powerLimit;
};

class PerformanceMonitor {
public:
    PerformanceMonitor();
    ~PerformanceMonitor();

    bool Init();
    void Shutdown();
    bool UpdateMetrics();
    const GpuMetrics& GetMetrics() const;

private:
    bool m_initialized = false;
    nvmlDevice_t m_device;
    GpuMetrics m_metrics;
};
