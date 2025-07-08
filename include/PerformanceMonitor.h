// include/PerformanceMonitor.h
#pragma once

#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
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
    GpuMetrics GetMetrics();

private:
    void WorkerLoop();

    bool m_initialized = false;
    nvmlDevice_t m_device;
    GpuMetrics m_metrics;

    std::thread m_worker_thread;
    std::mutex m_metrics_mutex;
    std::atomic<bool> m_stop_worker;
};
