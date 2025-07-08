// CudaKernels.h
#pragma once

#include <d3d11.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>

// Error checking macro for CUDA calls
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while(0)

// Original preprocessing function (maintained for backward compatibility)
bool PreprocessD3D11Texture(
    ID3D11Texture2D* pTexture, 
    float* d_processed_output, 
    int texture_w, 
    int texture_h, 
    int output_w, 
    int output_h, 
    cudaGraphicsResource** ppCudaResource
);

// Optimized preprocessing function with stream support and persistent resource management
bool PreprocessD3D11TextureOptimized(
    ID3D11Texture2D* pTexture, 
    float* d_processed_output, 
    int texture_w, 
    int texture_h, 
    int output_w, 
    int output_h, 
    cudaGraphicsResource_t* ppCudaResource,
    cudaStream_t stream = 0
);

// Enhanced buffer allocation with better error handling
float* AllocatePreprocessedBuffer(int width, int height);

// Resource management functions for improved performance
void CleanupGraphicsResourceCache();
void RemoveFromResourceCache(ID3D11Texture2D* pTexture); 