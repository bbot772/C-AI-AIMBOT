// ScreenCapture.h
#pragma once

#include <vector>
#include <d3d11.h>
#include <dxgi1_2.h>

class ScreenCapture {
public:
    ScreenCapture();
    ~ScreenCapture();

    bool Init();
    void Cleanup();

    // Returns a pointer to a D3D11 texture containing the captured screen region.
    // The caller does NOT own the returned pointer; its lifecycle is managed by this class.
    ID3D11Texture2D* CaptureScreen(int width, int height);

    int GetDesktopWidth() const { return m_desktop_width; }
    int GetDesktopHeight() const { return m_desktop_height; }
    ID3D11Device* GetDevice() const { return m_d3d11_device; } // Needed for CUDA interop

private:
    ID3D11Device* m_d3d11_device = nullptr;
    ID3D11DeviceContext* m_d3d11_context = nullptr;
    IDXGIOutputDuplication* m_duplication = nullptr;
    ID3D11Texture2D* m_captured_texture = nullptr;

    int m_desktop_width = 0;
    int m_desktop_height = 0;
};
