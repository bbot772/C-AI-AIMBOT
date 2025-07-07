#include "CudaKernels.h"
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include "Logger.h"

// Optimized CUDA kernels for better performance

/**
 * @brief Get optimal block size based on GPU architecture
 * Dynamically adjusts block size for different GPU generations
 */
__host__ dim3 GetOptimalBlockSize() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    
    // Optimize based on GPU architecture for maximum occupancy
    if (props.major >= 8) {        // Ampere+ (RTX 30xx/40xx series)
        return dim3(32, 32);
    } else if (props.major >= 7) { // Turing/Volta (RTX 20xx, GTX 16xx)
        return dim3(16, 16);
    } else if (props.major >= 6) { // Pascal (GTX 10xx)
        return dim3(16, 16);
    } else {                       // Older architectures
        return dim3(8, 8);
    }
}

/**
 * @brief Optimized preprocessing kernel with fusion
 * Combines multiple operations in a single kernel for better efficiency:
 * - Letterbox calculation
 * - Bilinear interpolation
 * - Color space conversion (BGRA -> RGB)
 * - Normalization
 * - Channel reordering (HWC -> CHW)
 */
__global__ void OptimizedPreprocessKernel(
    cudaTextureObject_t inputTexture, 
    float* output, 
    int img_w, int img_h, 
    int output_w, int output_h,
    float scale_factor = 1.0f / 255.0f)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= output_w || y >= output_h) {
        return;
    }

    // Calculate letterbox parameters once per thread
    const float scale = fminf(static_cast<float>(output_w) / img_w, 
                             static_cast<float>(output_h) / img_h);
    const int new_w = __float2int_rn(img_w * scale);
    const int new_h = __float2int_rn(img_h * scale);
    const int pad_x = (output_w - new_w) >> 1; // Bit shift division by 2
    const int pad_y = (output_h - new_h) >> 1;

    // Default to padding color (normalized grey: 0.5)
    float r_norm = 0.5f, g_norm = 0.5f, b_norm = 0.5f;

    // Check if pixel is within the scaled image bounds
    if (x >= pad_x && x < pad_x + new_w && y >= pad_y && y < pad_y + new_h) {
        // Map output coordinates back to input coordinates
        const float input_x = (x - pad_x) / scale;
        const float input_y = (y - pad_y) / scale;
        
        // Use hardware texture filtering for bilinear interpolation
        // tex2D automatically handles bounds checking and filtering
        const float4 pixel = tex2D<float4>(inputTexture, 
                                          (input_x + 0.5f) / img_w, 
                                          (input_y + 0.5f) / img_h);
        
        // Convert BGRA to RGB and apply normalization in one step
        r_norm = pixel.z * scale_factor; // R channel
        g_norm = pixel.y * scale_factor; // G channel  
        b_norm = pixel.x * scale_factor; // B channel
    }

    // Write to output in NCHW format with coalesced memory access
    const int pixel_idx = y * output_w + x;
    const int channel_size = output_h * output_w;
    
    output[pixel_idx] = r_norm;                      // R channel
    output[channel_size + pixel_idx] = g_norm;       // G channel
    output[2 * channel_size + pixel_idx] = b_norm;   // B channel
}

/**
 * @brief Optimized D3D11 texture preprocessing with dynamic block size
 */
bool OptimizedPreprocessD3D11Texture(
    ID3D11Texture2D* pTexture, 
    float* d_processed_output, 
    int texture_w, int texture_h, 
    int output_w, int output_h, 
    cudaGraphicsResource** ppCudaResource,
    cudaStream_t stream = 0)
{
    cudaError_t err;

    // Unregister previous resource if it exists
    if (*ppCudaResource != nullptr) {
        cudaGraphicsUnregisterResource(*ppCudaResource);
        *ppCudaResource = nullptr;
    }
    
    // Register the D3D11 texture as a CUDA graphics resource
    err = cudaGraphicsD3D11RegisterResource(ppCudaResource, pTexture, cudaGraphicsRegisterFlagsNone);
    if (err != cudaSuccess) {
        Logger::GetInstance().Log("cudaGraphicsD3D11RegisterResource failed: %s", cudaGetErrorString(err));
        return false;
    }

    // Map the resource to access it from CUDA
    err = cudaGraphicsMapResources(1, ppCudaResource, stream);
    if (err != cudaSuccess) {
        Logger::GetInstance().Log("cudaGraphicsMapResources failed: %s", cudaGetErrorString(err));
        return false;
    }

    cudaArray* cuArray;
    err = cudaGraphicsSubResourceGetMappedArray(&cuArray, *ppCudaResource, 0, 0);
    if (err != cudaSuccess) {
        Logger::GetInstance().Log("cudaGraphicsSubResourceGetMappedArray failed: %s", cudaGetErrorString(err));
        return false;
    }

    // Create texture object with optimal settings
    struct cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    struct cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;     // Hardware bilinear filtering
    texDesc.readMode = cudaReadModeNormalizedFloat; // Auto-normalize to [0,1]
    texDesc.normalizedCoords = 1;

    cudaTextureObject_t texObj = 0;
    err = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
    if (err != cudaSuccess) {
        Logger::GetInstance().Log("cudaCreateTextureObject failed: %s", cudaGetErrorString(err));
        cudaGraphicsUnmapResources(1, ppCudaResource, stream);
        return false;
    }

    // Use dynamic block size based on GPU architecture
    dim3 block = GetOptimalBlockSize();
    dim3 grid((output_w + block.x - 1) / block.x, 
              (output_h + block.y - 1) / block.y);

    // Launch optimized kernel with stream for async execution
    OptimizedPreprocessKernel<<<grid, block, 0, stream>>>(
        texObj, d_processed_output, texture_w, texture_h, output_w, output_h);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        Logger::GetInstance().Log("Kernel launch failed: %s", cudaGetErrorString(err));
    }

    // Cleanup
    cudaDestroyTextureObject(texObj);
    cudaGraphicsUnmapResources(1, ppCudaResource, stream);
    
    return err == cudaSuccess;
}

/**
 * @brief Allocate preprocessed buffer with proper alignment
 */
float* AllocateOptimizedPreprocessedBuffer(int width, int height) {
    const size_t size = 3 * width * height * sizeof(float);
    
    // Use buffer pool for better memory management
    void* buffer = nullptr;
    cudaError_t err = cudaMalloc(&buffer, size);
    
    if (err != cudaSuccess) {
        Logger::GetInstance().Log("Failed to allocate preprocessed buffer: %s", cudaGetErrorString(err));
        return nullptr;
    }
    
    return static_cast<float*>(buffer);
}