#include <iostream>
#include <string>

#include "ghub.h"
#include "../src/Logger.h" // Use our project's logger

GhubMouse::GhubMouse() : m_h_module(nullptr), m_move_r_ptr(nullptr), m_press_ptr(nullptr), m_release_ptr(nullptr), m_ok(false)
{
    m_h_module = LoadLibraryA("ghub_mouse.dll");
    if (m_h_module == nullptr)
    {
        Logger::GetInstance().Log("ERROR: ghub_mouse.dll not found. Please make sure it's in the same directory as the executable.");
        return;
}

    auto mouse_open_ptr = (p_mouse_open)GetProcAddress(m_h_module, "mouse_open");
    if (mouse_open_ptr == nullptr || !mouse_open_ptr())
    {
        Logger::GetInstance().Log("ERROR: Failed to open mouse via ghub_mouse.dll. The driver may not be running or the DLL may be incorrect.");
        FreeLibrary(m_h_module);
        m_h_module = nullptr;
        return;
    }

    m_move_r_ptr = (p_moveR)GetProcAddress(m_h_module, "moveR");
    m_press_ptr = (p_press)GetProcAddress(m_h_module, "press");
    m_release_ptr = (p_release)GetProcAddress(m_h_module, "release");

    if (m_move_r_ptr == nullptr || m_press_ptr == nullptr || m_release_ptr == nullptr)
        {
        Logger::GetInstance().Log("ERROR: Could not get function addresses (moveR, press, release) from ghub_mouse.dll.");
        FreeLibrary(m_h_module);
        m_h_module = nullptr;
        return;
    }

    Logger::GetInstance().Log("Logitech ghub_mouse.dll loaded and opened successfully.");
    m_ok = true;
}

GhubMouse::~GhubMouse()
{
    if (m_h_module != nullptr)
    {
        auto mouse_close_ptr = (p_mouse_close)GetProcAddress(m_h_module, "mouse_close");
        if (mouse_close_ptr)
        {
            mouse_close_ptr();
        }
        FreeLibrary(m_h_module);
    }
}

bool GhubMouse::is_valid()
        {
    return m_ok;
        }

bool GhubMouse::move_r(int dx, int dy)
{
    if (!m_ok) return false;
    return m_move_r_ptr(dx, dy);
}

bool GhubMouse::btn_down(int btn)
{
    if (!m_ok) return false;
    return m_press_ptr(btn);
}

bool GhubMouse::btn_up(int btn)
{
    if (!m_ok) return false;
    // This DLL's 'release' function seems to take no arguments.
    // We call it, but it likely just releases whatever button was last pressed.
    return m_release_ptr();
}