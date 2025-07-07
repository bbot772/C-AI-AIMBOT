#pragma once

#include <unordered_map>
#include <queue>
#include <mutex>
#include <cuda_runtime.h>

// GPU Buffer Pool for optimized memory management
// Reduces allocation overhead and prevents memory fragmentation
class GPUBufferPool {
public:
    static GPUBufferPool& GetInstance() {
        static GPUBufferPool instance;
        return instance;
    }

    // Get a buffer of specified size (reuses existing if available)
    void* GetBuffer(size_t size);
    
    // Return buffer to pool for reuse
    void ReturnBuffer(void* buffer, size_t size);
    
    // Clear all cached buffers (call on shutdown)
    void Clear();
    
    // Get memory usage statistics
    struct MemoryStats {
        size_t total_allocated = 0;
        size_t currently_used = 0;
        size_t pool_size = 0;
        size_t allocation_count = 0;
        size_t reuse_count = 0;
    };
    
    MemoryStats GetStats() const;

private:
    GPUBufferPool() = default;
    ~GPUBufferPool() { Clear(); }
    
    // Delete copy constructor and assignment operator
    GPUBufferPool(const GPUBufferPool&) = delete;
    GPUBufferPool& operator=(const GPUBufferPool&) = delete;
    
    struct BufferInfo {
        void* ptr;
        size_t size;
        bool in_use;
    };
    
    mutable std::mutex m_mutex;
    std::unordered_map<size_t, std::queue<void*>> m_available_buffers;
    std::unordered_map<void*, size_t> m_buffer_sizes;
    
    // Statistics
    mutable MemoryStats m_stats;
    
    // Helper to round up size to alignment boundary
    size_t AlignSize(size_t size) const {
        const size_t alignment = 256; // GPU memory alignment
        return (size + alignment - 1) & ~(alignment - 1);
    }
};