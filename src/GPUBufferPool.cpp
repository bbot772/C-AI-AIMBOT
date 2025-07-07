#include "GPUBufferPool.h"
#include "Logger.h"
#include <cuda_runtime.h>

void* GPUBufferPool::GetBuffer(size_t size) {
    size_t aligned_size = AlignSize(size);
    
    std::lock_guard<std::mutex> lock(m_mutex);
    
    // Check if we have a buffer of this size available
    auto it = m_available_buffers.find(aligned_size);
    if (it != m_available_buffers.end() && !it->second.empty()) {
        void* buffer = it->second.front();
        it->second.pop();
        
        m_stats.reuse_count++;
        m_stats.currently_used += aligned_size;
        
        return buffer;
    }
    
    // No available buffer, allocate new one
    void* buffer = nullptr;
    cudaError_t err = cudaMalloc(&buffer, aligned_size);
    
    if (err != cudaSuccess) {
        Logger::GetInstance().Log("GPU Buffer allocation failed: %s", cudaGetErrorString(err));
        return nullptr;
    }
    
    // Track buffer size for return
    m_buffer_sizes[buffer] = aligned_size;
    
    m_stats.total_allocated += aligned_size;
    m_stats.currently_used += aligned_size;
    m_stats.allocation_count++;
    
    Logger::GetInstance().Log("Allocated GPU buffer: %zu bytes (aligned from %zu)", aligned_size, size);
    
    return buffer;
}

void GPUBufferPool::ReturnBuffer(void* buffer, size_t size) {
    if (!buffer) return;
    
    std::lock_guard<std::mutex> lock(m_mutex);
    
    // Find the actual allocated size
    auto size_it = m_buffer_sizes.find(buffer);
    if (size_it == m_buffer_sizes.end()) {
        Logger::GetInstance().Log("Warning: Returning unknown buffer to pool");
        return;
    }
    
    size_t aligned_size = size_it->second;
    
    // Add to available buffers
    m_available_buffers[aligned_size].push(buffer);
    m_stats.currently_used -= aligned_size;
    m_stats.pool_size++;
}

void GPUBufferPool::Clear() {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    // Free all buffers
    for (auto& [size, queue] : m_available_buffers) {
        while (!queue.empty()) {
            void* buffer = queue.front();
            queue.pop();
            cudaFree(buffer);
        }
    }
    
    m_available_buffers.clear();
    m_buffer_sizes.clear();
    
    Logger::GetInstance().Log("GPU Buffer Pool cleared. Total freed: %zu bytes", m_stats.total_allocated);
    
    // Reset stats
    m_stats = MemoryStats{};
}

GPUBufferPool::MemoryStats GPUBufferPool::GetStats() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_stats;
}