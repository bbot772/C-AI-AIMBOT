// include/PerformanceMonitor.h
#pragma once

#include <nvml.h>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <deque>

struct GpuMetrics {
    std::string deviceName;
    unsigned int temperature = 0;
    unsigned int utilizationGpu = 0;
    unsigned int utilizationMemory = 0;
    unsigned long long memoryUsed = 0;
    unsigned long long memoryTotal = 0;
    unsigned int powerUsage = 0;
    unsigned int powerLimit = 0;
};

class PerformanceMonitor {
public:
    PerformanceMonitor();
    ~PerformanceMonitor();

    bool Init();
    void Shutdown();
    
    GpuMetrics GetMetrics();
    
    // Enhanced monitoring features
    void SetUpdateInterval(int interval_ms);
    float GetGpuUtilizationSmoothed();
    bool IsGpuOverloaded();

private:
    void WorkerLoop();
    void UpdateMetrics(GpuMetrics& metrics);

    bool m_initialized;
    nvmlDevice_t m_device;
    
    // Threading
    std::thread m_worker_thread;
    std::atomic<bool> m_stop_worker;
    
    // Metrics and synchronization
    GpuMetrics m_metrics;
    std::mutex m_metrics_mutex;
    
    // Performance optimization
    int m_update_interval_ms;
    unsigned int m_static_power_limit = 0; // Cached power limit
    std::deque<unsigned int> m_gpu_utilization_history;
};
