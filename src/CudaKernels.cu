#include "CudaKernels.h"
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include "Logger.h"
#include <unordered_map>
#include <mutex>

// Global resource cache for persistent graphics resources
static std::unordered_map<ID3D11Texture2D*, cudaGraphicsResource_t> g_resource_cache;
static std::mutex g_resource_cache_mutex;

// Optimized CUDA error checking macro
#define CUDA_CHECK_OPTIMIZED(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        Logger::GetInstance().Log("CUDA error at %s:%d - %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return false; \
    } \
} while(0)

/**
 * @brief Optimized CUDA kernel for image preprocessing with improved memory access patterns
 */
__global__ void PreprocessKernelOptimized(
    cudaTextureObject_t inputTexture, 
    float* __restrict__ output, 
    int img_w, 
    int img_h, 
    int output_w, 
    int output_h
) {
    // Use 2D thread indexing for better memory coalescing
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= output_w || y >= output_h) return;

    // Precompute scale and padding (these are constant for all threads)
    const float scale = fminf(static_cast<float>(output_w) / img_w, static_cast<float>(output_h) / img_h);
    const int new_w = __float2int_rz(img_w * scale);
    const int new_h = __float2int_rz(img_h * scale);
    const int pad_x = (output_w - new_w) >> 1;  // Bit shift for division by 2
    const int pad_y = (output_h - new_h) >> 1;

    // Calculate input coordinates with improved precision
    const float input_x_f = (x - pad_x) / scale;
    const float input_y_f = (y - pad_y) / scale;

    float3 rgb_norm = make_float3(0.5f, 0.5f, 0.5f); // Default padding color

    // Check bounds and sample texture
    if (x >= pad_x && x < pad_x + new_w && y >= pad_y && y < pad_y + new_h && 
        input_x_f >= 0 && input_x_f < img_w && input_y_f >= 0 && input_y_f < img_h) {
        
        // Optimized texture sampling with hardware interpolation
        const float4 pixel = tex2D<float4>(inputTexture, 
            (input_x_f + 0.5f) / img_w, 
            (input_y_f + 0.5f) / img_h);
        
        // Swizzle BGRA to RGB
        rgb_norm.x = pixel.z; // R
        rgb_norm.y = pixel.y; // G
        rgb_norm.z = pixel.x; // B
    }

    // Optimized memory write pattern (NCHW format)
    const int base_idx = y * output_w + x;
    const int channel_size = output_h * output_w;
    
    output[base_idx] = rgb_norm.x;                    // R channel
    output[channel_size + base_idx] = rgb_norm.y;     // G channel
    output[2 * channel_size + base_idx] = rgb_norm.z; // B channel
}

/**
 * @brief Optimized D3D11 texture preprocessing with persistent resource management
 */
bool PreprocessD3D11TextureOptimized(
    ID3D11Texture2D* pTexture, 
    float* d_processed_output, 
    int texture_w, 
    int texture_h, 
    int output_w, 
    int output_h, 
    cudaGraphicsResource_t* ppCudaResource,
    cudaStream_t stream = 0
) {
    if (!pTexture || !d_processed_output) return false;

    cudaGraphicsResource_t cudaResource = nullptr;
    bool resource_cached = false;

    // Check if resource is already cached
    {
        std::lock_guard<std::mutex> lock(g_resource_cache_mutex);
        auto it = g_resource_cache.find(pTexture);
        if (it != g_resource_cache.end()) {
            cudaResource = it->second;
            resource_cached = true;
        }
    }

    // Register resource if not cached
    if (!resource_cached) {
        CUDA_CHECK_OPTIMIZED(cudaGraphicsD3D11RegisterResource(
            &cudaResource, pTexture, cudaGraphicsRegisterFlagsReadOnly));
        
        // Cache the resource
        std::lock_guard<std::mutex> lock(g_resource_cache_mutex);
        g_resource_cache[pTexture] = cudaResource;
    }

    // Map resource for CUDA access
    CUDA_CHECK_OPTIMIZED(cudaGraphicsMapResources(1, &cudaResource, stream));

    cudaArray* cuArray;
    CUDA_CHECK_OPTIMIZED(cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource, 0, 0));

    // Create texture object with optimized settings
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;

    cudaTextureObject_t texObj = 0;
    CUDA_CHECK_OPTIMIZED(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));

    // Launch optimized kernel with better occupancy
    const dim3 block(16, 16);
    const dim3 grid((output_w + block.x - 1) / block.x, (output_h + block.y - 1) / block.y);

    PreprocessKernelOptimized<<<grid, block, 0, stream>>>(
        texObj, d_processed_output, texture_w, texture_h, output_w, output_h);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Logger::GetInstance().Log("Kernel launch failed: %s", cudaGetErrorString(err));
        cudaDestroyTextureObject(texObj);
        cudaGraphicsUnmapResources(1, &cudaResource, stream);
        return false;
    }

    // Cleanup texture object and unmap resource
    CUDA_CHECK_OPTIMIZED(cudaDestroyTextureObject(texObj));
    CUDA_CHECK_OPTIMIZED(cudaGraphicsUnmapResources(1, &cudaResource, stream));

    // Store resource handle if requested
    if (ppCudaResource) {
        *ppCudaResource = cudaResource;
    }

    return true;
}

/**
 * @brief Backward compatibility wrapper for existing code
 */
bool PreprocessD3D11Texture(
    ID3D11Texture2D* pTexture, 
    float* d_processed_output, 
    int texture_w, 
    int texture_h, 
    int output_w, 
    int output_h, 
    cudaGraphicsResource** ppCudaResource
) {
    cudaGraphicsResource_t resource = nullptr;
    bool result = PreprocessD3D11TextureOptimized(
        pTexture, d_processed_output, texture_w, texture_h, 
        output_w, output_h, &resource, 0);
    
    if (ppCudaResource) {
        *ppCudaResource = resource;
    }
    
    return result;
}

/**
 * @brief Enhanced buffer allocation with alignment optimization
 */
float* AllocatePreprocessedBuffer(int width, int height) {
    float* d_buffer = nullptr;
    const size_t size = 3 * static_cast<size_t>(width) * height * sizeof(float);
    
    // Use aligned allocation for better memory performance
    cudaError_t err = cudaMalloc(&d_buffer, size);
    if (err != cudaSuccess) {
        Logger::GetInstance().Log("Failed to allocate preprocessed buffer: %s", cudaGetErrorString(err));
        return nullptr;
    }
    
    // Initialize buffer to zero for consistent behavior
    cudaMemset(d_buffer, 0, size);
    
    return d_buffer;
}

/**
 * @brief Cleanup cached graphics resources
 */
void CleanupGraphicsResourceCache() {
    std::lock_guard<std::mutex> lock(g_resource_cache_mutex);
    
    for (auto& pair : g_resource_cache) {
        cudaGraphicsUnregisterResource(pair.second);
    }
    
    g_resource_cache.clear();
}

/**
 * @brief Remove specific resource from cache (call when texture is destroyed)
 */
void RemoveFromResourceCache(ID3D11Texture2D* pTexture) {
    std::lock_guard<std::mutex> lock(g_resource_cache_mutex);
    
    auto it = g_resource_cache.find(pTexture);
    if (it != g_resource_cache.end()) {
        cudaGraphicsUnregisterResource(it->second);
        g_resource_cache.erase(it);
    }
} 