// UI.h
#pragma once

#include <d3d11.h>
#include <windows.h> // For HWND, LRESULT, etc.
#include "imgui.h"
// #include "Aimbot.h" // Replaced with forward declaration
#include "PerformanceMonitor.h" // Include the new header
#include <vector>
#include <string>
#include <memory> // For std::unique_ptr
#include <atomic>
#include <map>

// Forward-declare Aimbot to break circular include dependency
class Aimbot;

// Forward declare ImFont so we don't need to include imgui.h here
struct ImFont;

class UI {
public:
    UI(Aimbot& aimbot);
    ~UI();

    bool Init();
    void Render();
    void Cleanup();
    bool IsRunning();

    // The message handler for our window
    LRESULT HandleWndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

    void RenderPerformanceOverlay(bool* p_open);
    void RenderFileMenu();

private:
    enum class Tab { Model, Detection, AimAssist, Performance, Logging, Stealth, Personalization };
    Tab m_active_tab = Tab::Model;
    Tab m_previous_tab = Tab::Model;
    float m_tab_fade_alpha = 1.0f;
    
    Aimbot& aimbot;
    std::unique_ptr<PerformanceMonitor> perf_monitor; // Add instance
    std::vector<std::string> model_files;
    int selected_model = 0;
    
    // Window state
    HWND m_hwnd = nullptr;
    std::string m_random_window_title;
    bool m_is_window_visible = true;
    bool m_hide_key_pressed_last_frame = false;

    // Encapsulated UI state (moved from global static)
    int m_last_vk_pressed = 0;
    bool m_just_set_killswitch = false;
    bool m_just_set_hide_key = false;

    // Model settings
    char onnx_path[256] = "models/model.onnx";
    int batch_size = 1;

    // File browser state
    bool show_file_browser = false;

    // Loading Modal State
    std::string build_status_message;

    void FindModels();
    void FindFonts();
    void LoadFonts();
    void ApplyCustomStyle(int theme_index = -1); // Allow passing a theme preset index

    // --- Tab Rendering Functions ---
    void RenderTab(Tab tab);
    void RenderModelTab();
    void RenderAimingTab();
    void RenderPerformanceTab();
    void RenderLoggingTab();
    void RenderStealthTab();
    void RenderPersonalizationTab();

    char m_custom_window_title_buf[128];

    // --- Personalization Settings ---
    // These are now stored in Aimbot::m_personalization_settings
    // ImVec4 m_theme_color;
    // ImVec4 m_text_color;
    // float m_ui_scale = 1.0f;
    // float m_window_opacity = 1.0f;
    // char m_custom_window_title[128];
    std::vector<std::string> m_font_files;
    int m_selected_font_index = 0;
    std::map<std::string, ImFont*> m_loaded_fonts;
    const std::string m_default_font_name = "Default";

    // Win32 and D3D11 members
};
