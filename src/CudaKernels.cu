#include "CudaKernels.h"
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include "Logger.h"

/**
 * @brief CUDA kernel to preprocess a raw image for inference.
 *
 * This kernel is executed by a grid of thread blocks. Each thread is responsible
 * for computing the value of a single pixel in the destination (output) tensor.
 * It maps the destination pixel back to the source image, applies letterbox
 * padding if necessary, normalizes the pixel value, and writes it to the
 * correct position in the CHW-formatted output buffer.
 */
__global__ void PreprocessKernel(cudaTextureObject_t inputTexture, float* output, int img_w, int img_h, int output_w, int output_h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= output_w || y >= output_h) {
        return;
    }

    float scale = min(static_cast<float>(output_w) / img_w, static_cast<float>(output_h) / img_h);
    int new_w = static_cast<int>(img_w * scale);
    int new_h = static_cast<int>(img_h * scale);
    int pad_x = (output_w - new_w) / 2;
    int pad_y = (output_h - new_h) / 2;

    int input_x = (x - pad_x) / scale;
    int input_y = (y - pad_y) / scale;

    float r_norm = 0.5f, g_norm = 0.5f, b_norm = 0.5f; // Default to padding color (grey)

    if (x >= pad_x && x < pad_x + new_w && y >= pad_y && y < pad_y + new_h) {
        // tex2D with cudaReadModeNormalizedFloat gives us BGRA values in the [0.0, 1.0] range.
        float4 pixel = tex2D<float4>(inputTexture, (input_x + 0.5f) / img_w, (input_y + 0.5f) / img_h);
        
        // No need to divide by 255.0f again. Just swizzle from BGRA to RGB for the model.
        r_norm = pixel.z; // R channel
        g_norm = pixel.y; // G channel
        b_norm = pixel.x; // B channel
    }

    // NCHW format
    output[0 * output_h * output_w + y * output_w + x] = r_norm; // R channel
    output[1 * output_h * output_w + y * output_w + x] = g_norm; // G channel
    output[2 * output_h * output_w + y * output_w + x] = b_norm; // B channel
}

bool PreprocessD3D11Texture(ID3D11Texture2D* pTexture, float* d_processed_output, int texture_w, int texture_h, int output_w, int output_h, cudaGraphicsResource** ppCudaResource)
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
    err = cudaGraphicsMapResources(1, ppCudaResource, 0);
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

    // Create a texture object to read from the CUDA array
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;

    cudaTextureObject_t texObj = 0;
    err = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
     if (err != cudaSuccess) {
        Logger::GetInstance().Log("cudaCreateTextureObject failed: %s", cudaGetErrorString(err));
        cudaGraphicsUnmapResources(1, ppCudaResource, 0);
        return false;
    }

    dim3 block(16, 16);
    dim3 grid((output_w + block.x - 1) / block.x, (output_h + block.y - 1) / block.y);

    PreprocessKernel<<<grid, block>>>(texObj, d_processed_output, texture_w, texture_h, output_w, output_h);

    cudaDestroyTextureObject(texObj);
    cudaGraphicsUnmapResources(1, ppCudaResource, 0);
    
    // The resource should be unregistered when it's no longer needed, typically on cleanup.
    // We leave it registered for the lifetime of the texture.

    return true;
}

float* AllocatePreprocessedBuffer(int width, int height) {
    float* d_buffer;
    cudaMalloc(&d_buffer, 3 * width * height * sizeof(float));
    return d_buffer;
} 