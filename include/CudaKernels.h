// include/CudaKernels.h
#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <d3d11.h> // For ID3D11Texture2D

// Forward declaration for CUDA's struct
struct cudaGraphicsResource;

// Preprocesses a D3D11 texture on the GPU.
// - Registers the D3D11 texture with CUDA.
// - Performs letterboxing, color conversion (BGRA->RGB), and normalization.
// - Writes the result into the provided device buffer (d_processed_output).
// Returns true on success, false on failure.
bool PreprocessD3D11Texture(
    ID3D11Texture2D* pTexture, 
    float* d_processed_output,
    int texture_w, 
    int texture_h,
    int output_w, 
    int output_h,
    cudaGraphicsResource** ppCudaResource // Pass pointer to be managed
);

// Allocates memory on the GPU for the final preprocessed buffer.
// The caller is responsible for freeing this memory with cudaFree.
float* AllocatePreprocessedBuffer(int width, int height); 