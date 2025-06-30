// ScreenCapture.cpp
#include "ScreenCapture.h"
#include <iostream>

ScreenCapture::ScreenCapture() {
    // Constructor
}

ScreenCapture::~ScreenCapture() {
    Cleanup();
}

bool ScreenCapture::Init() {
    HRESULT hr;

    // Create the device and device context.
    UINT createDeviceFlags = 0;
#ifdef _DEBUG
    createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    // The following variable is intentionally unused in the current implementation,
    // but is kept for reference for future feature development.
    // D3D_DRIVER_TYPE driverTypes[] = { D3D_DRIVER_TYPE_HARDWARE };
    
    D3D_FEATURE_LEVEL featureLevels[] = { D3D_FEATURE_LEVEL_11_0 };
    UINT numFeatureLevels = ARRAYSIZE(featureLevels);

    // Create D3D11 device
    hr = D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        createDeviceFlags,
        featureLevels,
        numFeatureLevels,
        D3D11_SDK_VERSION,
        &m_d3d11_device,
        nullptr,
        &m_d3d11_context
    );
    if (FAILED(hr)) {
        std::cerr << "D3D11CreateDevice failed" << std::endl;
        return false;
    }

    // Get DXGI device
    IDXGIDevice* pDxgiDevice = nullptr;
    hr = m_d3d11_device->QueryInterface(__uuidof(IDXGIDevice), reinterpret_cast<void**>(&pDxgiDevice));
    if (FAILED(hr)) {
        std::cerr << "Failed to get DXGI device" << std::endl;
        return false;
    }

    // Get DXGI adapter
    IDXGIAdapter* pDxgiAdapter = nullptr;
    hr = pDxgiDevice->GetParent(__uuidof(IDXGIAdapter), reinterpret_cast<void**>(&pDxgiAdapter));
    pDxgiDevice->Release();
    if (FAILED(hr)) {
        std::cerr << "Failed to get DXGI adapter" << std::endl;
        return false;
    }

    // Get DXGI output
    IDXGIOutput* pDxgiOutput = nullptr;
    hr = pDxgiAdapter->EnumOutputs(0, &pDxgiOutput); // 0 for primary output
    pDxgiAdapter->Release();
    if (FAILED(hr)) {
        std::cerr << "Failed to get DXGI output" << std::endl;
        return false;
    }

    DXGI_OUTPUT_DESC outputDesc;
    pDxgiOutput->GetDesc(&outputDesc);
    m_desktop_width = outputDesc.DesktopCoordinates.right - outputDesc.DesktopCoordinates.left;
    m_desktop_height = outputDesc.DesktopCoordinates.bottom - outputDesc.DesktopCoordinates.top;

    // Query for IDXGIOutput1
    IDXGIOutput1* pDxgiOutput1 = nullptr;
    hr = pDxgiOutput->QueryInterface(__uuidof(IDXGIOutput1), reinterpret_cast<void**>(&pDxgiOutput1));
    pDxgiOutput->Release();
    if (FAILED(hr)) {
        std::cerr << "Failed to get DXGIOutput1" << std::endl;
        return false;
    }

    // Create desktop duplication
    hr = pDxgiOutput1->DuplicateOutput(m_d3d11_device, &m_duplication);
    pDxgiOutput1->Release();
    if (FAILED(hr)) {
        std::cerr << "DuplicateOutput failed" << std::endl;
        return false;
    }

    return true;
}

ID3D11Texture2D* ScreenCapture::CaptureScreen(int width, int height) {
    HRESULT hr;

    IDXGIResource* desktop_resource = nullptr;
    DXGI_OUTDUPL_FRAME_INFO frame_info;
    hr = m_duplication->AcquireNextFrame(5, &frame_info, &desktop_resource);

        if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
        return nullptr; // No new frame yet
    }
    if (FAILED(hr)) {
        std::cerr << "Failed to acquire next frame. Re-initializing..." << std::endl;
        // Handle error, maybe re-initialize
        return nullptr;
    }

    ID3D11Texture2D* acquired_texture = nullptr;
    hr = desktop_resource->QueryInterface(__uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&acquired_texture));
    desktop_resource->Release();
    if (FAILED(hr)) {
        std::cerr << "Failed to query interface for ID3D11Texture2D." << std::endl;
        return nullptr;
    }

    // If m_captured_texture hasn't been created or dimensions mismatch, create it.
    if (!m_captured_texture) {
        D3D11_TEXTURE2D_DESC desc;
        desc.Width = width;
        desc.Height = height;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET; // For CUDA
        desc.CPUAccessFlags = 0; // No CPU access needed
        desc.MiscFlags = 0;

        hr = m_d3d11_device->CreateTexture2D(&desc, nullptr, &m_captured_texture);
    if (FAILED(hr)) {
            std::cerr << "Failed to create the capture texture." << std::endl;
            acquired_texture->Release();
            return nullptr;
    }
    }
    
    // Define the source region to copy from the center of the desktop
    int cap_x = (m_desktop_width - width) / 2;
    int cap_y = (m_desktop_height - height) / 2;
    D3D11_BOX src_box;
    src_box.left = cap_x;
    src_box.right = cap_x + width;
    src_box.top = cap_y;
    src_box.bottom = cap_y + height;
    src_box.front = 0;
    src_box.back = 1;

    // Copy the specified region from the acquired texture to our persistent texture
    m_d3d11_context->CopySubresourceRegion(m_captured_texture, 0, 0, 0, 0, acquired_texture, 0, &src_box);

    acquired_texture->Release();
    m_duplication->ReleaseFrame();

    return m_captured_texture;
}

void ScreenCapture::Cleanup() {
    if (m_duplication) {
        m_duplication->Release();
        m_duplication = nullptr;
    }
    if (m_d3d11_context) {
        m_d3d11_context->Release();
        m_d3d11_context = nullptr;
    }
    if (m_d3d11_device) {
        m_d3d11_device->Release();
        m_d3d11_device = nullptr;
    }
    if (m_captured_texture) {
        m_captured_texture->Release();
        m_captured_texture = nullptr;
    }
}
