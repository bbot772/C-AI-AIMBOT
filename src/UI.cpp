// UI.cpp
#define IMGUI_DEFINE_MATH_OPERATORS
#include "UI.h"
#include "Aimbot.h"
#include "Path.h"
#include "Logger.h"
#include "font_awesome.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_dx11.h"
#include <d3d11.h>
#include <tchar.h>
#include <windows.h> // For MessageBox
#include <windowsx.h> // For GET_X_LPARAM, GET_Y_LPARAM
#include <filesystem>
#include <iostream>
#include <map>
#include <cstdlib> // For rand, srand
#include <ctime>   // For time
#include <algorithm> // For std::min
#include <cmath> // For fabs
#include "ImPlot.h"

#include "imgui_internal.h"

namespace ImGui
{
    // A custom animated toggle switch widget, providing a more modern look than a standard checkbox.
    bool ToggleSwitch(const char* label, bool* v)
    {
        ImGuiWindow* window = GetCurrentWindow();
        if (window->SkipItems)
            return false;

        ImGuiContext& g = *GImGui;
        const ImGuiStyle& style = g.Style;
        const ImGuiID id = window->GetID(label);
        const ImVec2 label_size = CalcTextSize(label, NULL, true);

        const float height = GetFrameHeight();
        const float width = height * 1.8f;
        const float radius = height * 0.5f;

        const ImVec2 pos = window->DC.CursorPos;
        // The bounding box for the entire widget (switch + label).
        const ImRect total_bb(pos, ImVec2(pos.x + width + (label_size.x > 0.0f ? style.ItemInnerSpacing.x + label_size.x : 0.0f), pos.y + height));
        
        ItemSize(total_bb, style.FramePadding.y);
        if (!ItemAdd(total_bb, id))
            return false;

        // The whole widget should be interactive.
        bool hovered, held;
        bool pressed = ButtonBehavior(total_bb, id, &hovered, &held);
        if (pressed)
        {
            *v = !(*v);
            MarkItemEdited(id);
        }

        // Animation logic: smoothly transition between off and on states.
        float anim_ratio = window->StateStorage.GetFloat(id, *v ? 1.0f : 0.0f);
        if (*v && anim_ratio < 1.0f)
            anim_ratio += g.IO.DeltaTime * 8.f;
        else if (!*v && anim_ratio > 0.0f)
            anim_ratio -= g.IO.DeltaTime * 8.f;
        anim_ratio = ImClamp(anim_ratio, 0.0f, 1.0f);
        window->StateStorage.SetFloat(id, anim_ratio);

        // Rendering
        const ImRect switch_bb(pos, ImVec2(pos.x + width, pos.y + height));
        // The background color of the switch interpolates based on its state.
        const ImU32 col_bg = GetColorU32(ImLerp(ImVec4(0.20f, 0.21f, 0.22f, 0.54f), style.Colors[ImGuiCol_ButtonActive], anim_ratio));
        
        window->DrawList->AddRectFilled(switch_bb.Min, switch_bb.Max, col_bg, height * 0.5f);
        // The knob moves horizontally based on the animation ratio.
        window->DrawList->AddCircleFilled(ImVec2(pos.x + radius + anim_ratio * (width - 2 * radius), pos.y + radius), radius - 1.5f, GetColorU32(ImGuiCol_Text), 16);
        
        // Render the label to the right of the switch.
        if (label_size.x > 0.0f)
        {
            const ImVec2 label_pos = ImVec2(switch_bb.Max.x + style.ItemInnerSpacing.x, pos.y + (height - label_size.y) * 0.5f);
            RenderText(label_pos, label);
        }

        return pressed;
    }

    // Custom helper to render text rotated around its center.
    // This is a simplified implementation, assuming single-line ASCII text.
    void RenderTextRotated(const char* text, ImVec2 center, float angle, ImU32 col)
    {
        ImDrawList* draw_list = GetCurrentWindow()->DrawList;
        
        float s = sinf(angle);
        float c = cosf(angle);

        ImFont* font = GImGui->Font;
        // Correctly get the baked font data for the current size before looking up glyphs.
        // This resolves the compilation error as FindGlyph is a member of ImFontBaked.
        ImFontBaked* baked_font = font->GetFontBaked(font->LegacySize);
        if (!baked_font)
            return;

        const ImVec2 text_size = CalcTextSize(text);
        ImVec2 current_pos_local = -text_size * 0.5f;

        const char* text_end = text + strlen(text);
        while (text < text_end)
        {
            unsigned int current_char = (unsigned int)*text++;

            // Call FindGlyph on the baked font data, not the main font object.
            const ImFontGlyph* glyph = baked_font->FindGlyph((ImWchar)current_char);
            if (!glyph) continue;

            // Character quad points, relative to the text's center (0,0)
            ImVec2 p1 = current_pos_local + ImVec2(glyph->X0, glyph->Y0);
            ImVec2 p2 = current_pos_local + ImVec2(glyph->X1, glyph->Y0);
            ImVec2 p3 = current_pos_local + ImVec2(glyph->X1, glyph->Y1);
            ImVec2 p4 = current_pos_local + ImVec2(glyph->X0, glyph->Y1);

            // Rotate points around (0,0) and then translate to `center`
            ImVec2 r1 = ImVec2(p1.x * c - p1.y * s, p1.x * s + p1.y * c) + center;
            ImVec2 r2 = ImVec2(p2.x * c - p2.y * s, p2.x * s + p2.y * c) + center;
            ImVec2 r3 = ImVec2(p3.x * c - p3.y * s, p3.x * s + p3.y * c) + center;
            ImVec2 r4 = ImVec2(p4.x * c - p4.y * s, p4.x * s + p4.y * c) + center;
            
            draw_list->PrimReserve(6, 4);
            draw_list->PrimQuadUV(r1, r2, r3, r4, ImVec2(glyph->U0, glyph->V0), ImVec2(glyph->U1, glyph->V0), ImVec2(glyph->U1, glyph->V1), ImVec2(glyph->U0, glyph->V1), col);

            current_pos_local.x += glyph->AdvanceX;
        }
    }
}

namespace fs = std::filesystem;

// Helper to display a little (?) mark which shows a tooltip on hover.
static void HelpMarker(const char* desc)
{
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered())
    {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(desc);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

// Helper function to generate a random string for the window title
std::string GenerateRandomString(int length) {
    const char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::string result;
    for (int i = 0; i < length; ++i) {
        result += charset[rand() % (sizeof(charset) - 1)];
    }
    return result;
}

// Forward declaration for our helper function
std::string VKCodeToString(int vk_code);

// Data
static ID3D11Device*            g_pd3dDevice = NULL;
static ID3D11DeviceContext*     g_pd3dDeviceContext = NULL;
static IDXGISwapChain*          g_pSwapChain = NULL;
static ID3D11RenderTargetView*  g_mainRenderTargetView = NULL;
bool g_done = false;
// g_last_vk_pressed, g_just_set_killswitch, g_just_set_hide_key are now member variables of the UI class.

// Forward declarations of helper functions
bool CreateDeviceD3D(HWND hWnd);
void CleanupDeviceD3D();
void CreateRenderTarget();
void CleanupRenderTarget();
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

UI::UI(Aimbot& aimbot) : aimbot(aimbot), 
    perf_monitor(std::make_unique<PerformanceMonitor>())
{
    srand(static_cast<unsigned int>(time(nullptr))); // Seed the random number generator
    m_random_window_title = GenerateRandomString(16);
    // Initialize the window title from loaded settings
    strcpy_s(m_custom_window_title_buf, sizeof(m_custom_window_title_buf), aimbot.m_personalization_settings.custom_window_title.c_str());

    perf_monitor->Init();
    Logger::GetInstance().Log("UI constructor started.");
    
    // --- Load settings from Aimbot's config ---
    AimingSettings& settings = aimbot.m_aiming_settings; // Get a reference to the settings
    // fp16_mode and workspace_size are now directly part of the aimbot object
    FindModels(); // Find models before trying to select one
    
    // Find the last model in the list of available models
    std::string last_model = aimbot.GetLastLoadedModelPath();
    if (!last_model.empty()) {
        auto it = std::find(model_files.begin(), model_files.end(), fs::path(last_model).filename().string());
        if (it != model_files.end()) {
            selected_model = static_cast<int>(std::distance(model_files.begin(), it));
        }
    }
    // --- End loading settings ---

    fs::path models_path = get_models_path();
    Logger::GetInstance().Log("Attempting to use models path: " + models_path.string());
    
    if (!fs::exists(models_path)) {
        Logger::GetInstance().Log("Models directory does not exist. Attempting to create it...");
        std::error_code ec;
        if (!fs::create_directory(models_path, ec)) {
            std::string errmsg = "Failed to create models directory. Error: " + ec.message();
            Logger::GetInstance().Log(errmsg);
            std::string msg = "Failed to create models directory at: \n" + models_path.string() + "\n\nError: " + ec.message();
            MessageBoxA(NULL, msg.c_str(), "Fatal Error", MB_OK | MB_ICONERROR);
        } else {
            Logger::GetInstance().Log("Successfully created models directory.");
        }
    } else {
        Logger::GetInstance().Log("Models directory already exists.");
    }
    
    FindModels();
    FindFonts(); // Find available fonts
}

UI::~UI() {
    Cleanup();
}

void UI::FindModels() {
    try {
        fs::path models_path = get_models_path();
        for (const auto& entry : fs::directory_iterator(models_path)) {
            if (entry.is_regular_file()) {
                std::string extension = entry.path().extension().string();
                if (extension == ".onnx" || extension == ".engine") {
                model_files.push_back(entry.path().filename().string());
                }
            }
        }
    }
    catch (const fs::filesystem_error& e) {
        std::cerr << "Error accessing models directory: " << e.what() << std::endl;
    }
}

void UI::FindFonts() {
    try {
        fs::path fonts_path = get_fonts_path();
        if (!fs::exists(fonts_path) || !fs::is_directory(fonts_path)) {
            Logger::GetInstance().Log("Fonts directory not found at: " + fonts_path.string());
            return;
        }

        for (const auto& entry : fs::directory_iterator(fonts_path)) {
            if (entry.is_regular_file()) {
                std::string extension = entry.path().extension().string();
                if (extension == ".ttf" || extension == ".otf") {
                    m_font_files.push_back(entry.path().filename().string());
                }
            }
        }
    }
    catch (const fs::filesystem_error& e) {
        std::string error_msg = "Error accessing fonts directory: " + std::string(e.what());
        Logger::GetInstance().Log(error_msg);
        std::cerr << error_msg << std::endl;
    }
}

bool UI::Init() {
    // Create application window
    WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_CLASSDC, WndProc, 0L, 0L, GetModuleHandle(NULL), NULL, NULL, NULL, NULL, _T("ImGui Example"), NULL };
    ::RegisterClassEx(&wc);
    // Pass 'this' pointer to CreateWindow to be used in WndProc
    m_hwnd = ::CreateWindow(wc.lpszClassName, m_random_window_title.c_str(), WS_POPUP | WS_THICKFRAME, 100, 100, 1280, 600, NULL, NULL, wc.hInstance, this);

    // Initialize Direct3D
    if (!CreateDeviceD3D(m_hwnd)) {
        CleanupDeviceD3D();
        ::UnregisterClass(wc.lpszClassName, wc.hInstance);
        return false;
    }

    // Show the window
    ::ShowWindow(m_hwnd, SW_SHOWDEFAULT);
    ::UpdateWindow(m_hwnd);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Load Fonts before applying style
    LoadFonts();

    // Apply the custom style
    ApplyCustomStyle();

    // Setup Platform/Renderer backends
    ImGui_ImplWin32_Init(m_hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

    return true;
}

void UI::Render() {
    // --- Hide/Show Hotkey Check ---
    if (m_just_set_hide_key) {
        int hide_key = aimbot.hide_toggle_key.load();
        // Wait for the key to be released before arming the hotkey
        if (hide_key != 0 && (GetAsyncKeyState(hide_key) & 0x8000) == 0) {
            m_just_set_hide_key = false;
        }
    } else {
        int hide_key = aimbot.hide_toggle_key.load();
        if (hide_key != 0 && (GetAsyncKeyState(hide_key) & 0x8000)) {
            if (!m_hide_key_pressed_last_frame) {
                m_is_window_visible = !m_is_window_visible;
                ShowWindow(m_hwnd, m_is_window_visible ? SW_SHOW : SW_HIDE);
            }
            m_hide_key_pressed_last_frame = true;
        } else {
            m_hide_key_pressed_last_frame = false;
        }
    }

    // --- Killswitch Check ---
    // If we just set the key, we need to wait for it to be released before treating
    // subsequent presses as a kill signal. This prevents an immediate exit.
    if (m_just_set_killswitch) {
        int killswitch_vk = aimbot.killswitch_key.load();
        // Check if the key is UP (not pressed).
        if (killswitch_vk != 0 && (GetAsyncKeyState(killswitch_vk) & 0x8000) == 0) {
            m_just_set_killswitch = false; // Arm the killswitch now that the key is released.
        }
    }
    else {
        int killswitch_vk = aimbot.killswitch_key.load();
        if (killswitch_vk != 0 && GetAsyncKeyState(killswitch_vk) & 0x8000) {
            g_done = true;
        }
    }
    // --- End Killswitch Check ---

    MSG msg;
    while (::PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE)) {
        ::TranslateMessage(&msg);
        ::DispatchMessage(&msg);
        if (msg.message == WM_QUIT)
            g_done = true;
    }
    if (g_done) return;

    // Start the Dear ImGui frame
    ImGui_ImplDX11_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();

    // --- Font Selection ---
    // Push the selected font. It will be active for the entire frame.
    // Find the index for the loaded font name
    auto it = std::find(m_font_files.begin(), m_font_files.end(), aimbot.m_personalization_settings.font_name);
    if (it != m_font_files.end()) {
        m_selected_font_index = static_cast<int>(std::distance(m_font_files.begin(), it));
    } else {
        m_selected_font_index = 0; // Default to "Default" if not found
    }

    const std::string& selected_font_name = m_font_files.empty() ? m_default_font_name : m_font_files[m_selected_font_index];
    if (m_loaded_fonts.count(selected_font_name)) {
        ImGui::PushFont(m_loaded_fonts.at(selected_font_name));
    }

    // Apply global UI scale from settings
    ImGui::GetIO().FontGlobalScale = aimbot.m_personalization_settings.ui_scale;

    // Main Window
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize, ImGuiCond_Always);
    ImGui::Begin("Aimbot", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

    // --- Draw custom background gradient ---
    ImDrawList* draw_list_bg = ImGui::GetWindowDrawList();
    ImVec2 window_pos_bg = ImGui::GetWindowPos();
    ImVec2 window_size_bg = ImGui::GetWindowSize();

    PersonalizationSettings& ps = aimbot.m_personalization_settings;
    ImU32 col_top = ImGui::ColorConvertFloat4ToU32(ps.gradient_color_top);
    ImU32 col_bottom = ImGui::ColorConvertFloat4ToU32(ps.gradient_color_bottom);
    
    draw_list_bg->AddRectFilledMultiColor(window_pos_bg, ImVec2(window_pos_bg.x + window_size_bg.x, window_pos_bg.y + window_size_bg.y), col_top, col_top, col_bottom, col_bottom);

    // --- Draw Window Outline ---
    ImGuiStyle& style = ImGui::GetStyle();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 window_pos = ImGui::GetWindowPos();
    ImVec2 window_size = ImGui::GetWindowSize();
    ImU32 outline_color_u32 = ImGui::ColorConvertFloat4ToU32(aimbot.m_personalization_settings.outline_color);
    draw_list->AddRect(window_pos, ImVec2(window_pos.x + window_size.x, window_pos.y + window_size.y), outline_color_u32, style.WindowRounding, ImDrawFlags_None, 1.5f);

    // --- Custom Title Bar and Controls ---
    const float title_bar_height = 30.0f;
    // Use a child window for the title bar area to handle background color and layout
    ImGui::BeginChild("TitleBar", ImVec2(0, title_bar_height), false, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    // Buttons on the left
    ImGui::SetCursorPos(ImVec2(8, (title_bar_height - 20) / 2)); // Centered vertically
    if (ImGui::Button("X", ImVec2(20, 20))) {
        g_done = true;
        }
    ImGui::SameLine();
    if (ImGui::Button("_", ImVec2(20, 20))) {
        ::ShowWindow(m_hwnd, SW_MINIMIZE);
    }

    // Title text in the middle
    ImVec2 title_size = ImGui::CalcTextSize(aimbot.m_personalization_settings.custom_window_title.c_str());
    ImGui::SameLine(ImGui::GetWindowWidth() / 2 - title_size.x / 2);
    ImGui::Text("%s", aimbot.m_personalization_settings.custom_window_title.c_str());

    ImGui::EndChild();
    // --- End Custom Title Bar ---

    // Left Sidebar for Tabs
    ImGui::BeginChild("Sidebar", ImVec2(m_sidebar_anim_width, 0), true, ImGuiWindowFlags_NoScrollbar);
    
    // --- Collapse Button ---
    const char* collapse_icon = m_sidebar_collapsed ? ICON_FA_CHEVRON_RIGHT : ICON_FA_CHEVRON_LEFT;
    ImGui::SetCursorPosX((m_sidebar_anim_width - 30.0f) / 2.0f);
    if (ImGui::Button(collapse_icon, ImVec2(30, 30))) {
        m_sidebar_collapsed = !m_sidebar_collapsed;
    }
    ImGui::Separator();

    auto handle_tab_click = [&](Tab tab) {
        if (m_active_tab != tab) {
            m_previous_tab = m_active_tab;
            m_active_tab = tab;
            m_tab_fade_alpha = 0.0f;
    }
    };

    if (AnimatedTabButton(ICON_FA_CUBE, " Model", Tab::Model, m_sidebar_collapsed)) handle_tab_click(Tab::Model);
    if (AnimatedTabButton(ICON_FA_CROSSHAIRS, " Aiming", Tab::Detection, m_sidebar_collapsed)) handle_tab_click(Tab::Detection);
    if (AnimatedTabButton(ICON_FA_RETWEET, " Recoil", Tab::RecoilControl, m_sidebar_collapsed)) handle_tab_click(Tab::RecoilControl);
    if (AnimatedTabButton(ICON_FA_CHART_LINE, " Performance", Tab::Performance, m_sidebar_collapsed)) handle_tab_click(Tab::Performance);
    if (AnimatedTabButton(ICON_FA_BOOK, " Logging", Tab::Logging, m_sidebar_collapsed)) handle_tab_click(Tab::Logging);
    if (AnimatedTabButton(ICON_FA_SHIELD_ALT, " Stealth", Tab::Stealth, m_sidebar_collapsed)) handle_tab_click(Tab::Stealth);
    if (AnimatedTabButton(ICON_FA_PAINT_BRUSH, " Personalization", Tab::Personalization, m_sidebar_collapsed)) handle_tab_click(Tab::Personalization);
    
    ImGui::Separator();

    if (m_sidebar_collapsed) {
        ImGui::SetCursorPosX((m_sidebar_anim_width - 30.0f) / 2.0f);
        if (ImGui::Button(ICON_FA_SAVE, ImVec2(30, 30))) {
            aimbot.m_config_manager.Save();
            Logger::GetInstance().Log("Settings saved manually.");
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Save Settings");
        }
    } else {
        if (ImGui::Button(ICON_FA_SAVE " Save Settings", ImVec2(-1, 30))) {
            aimbot.m_config_manager.Save();
            Logger::GetInstance().Log("Settings saved manually.");
        }
    }
    
    ImGui::EndChild();

    ImGui::SameLine();

    // Right Content Area
    ImGui::BeginChild("Content", ImVec2(0, 0), false);

    // --- Animation state ---
    // Update linear alpha for the transition
    if (m_tab_fade_alpha < 1.0f) {
        m_tab_fade_alpha += ImGui::GetIO().DeltaTime * 6.0f; // Increased speed for a snappier feel
        m_tab_fade_alpha = (std::min)(m_tab_fade_alpha, 1.0f);
    }

    // Apply an easing function for a smooth curve
    auto ease_out_cubic = [](float t) {
        t -= 1.0f;
        return t * t * t + 1.0f;
    };
    float eased_alpha = ease_out_cubic(m_tab_fade_alpha);

    const float slide_distance = 30.0f;

    // --- Render tabs with transition ---
    if (m_tab_fade_alpha < 1.0f && m_previous_tab != m_active_tab) {
        ImVec2 original_pos = ImGui::GetCursorPos();

        // 1. Render previous tab (fading out, sliding up)
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 1.0f - eased_alpha);
        ImGui::SetCursorPosY(original_pos.y - (slide_distance * eased_alpha));
        RenderTab(m_previous_tab);
        ImGui::PopStyleVar();

        // 2. Render active tab (fading in, sliding up from below)
        ImGui::SetCursorPos(original_pos); // Reset cursor position for the new tab
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, eased_alpha);
        ImGui::SetCursorPosY(original_pos.y + (slide_distance * (1.0f - eased_alpha)));
        RenderTab(m_active_tab);
        ImGui::PopStyleVar();
    }
    else {
        // Just render the active tab when no transition is active
        RenderTab(m_active_tab);
    }

    ImGui::EndChild(); // End Content

    ImGui::End(); // End Main Window
    
    // Pop the selected font at the end of the frame
    if (m_loaded_fonts.count(selected_font_name)) {
        ImGui::PopFont();
    }
    
    // --- Sidebar Animation ---
    const float target_width = m_sidebar_collapsed ? 60.0f : 150.0f;
    if (fabs(m_sidebar_anim_width - target_width) > 0.1f) {
        m_sidebar_anim_width = ImLerp(m_sidebar_anim_width, target_width, ImGui::GetIO().DeltaTime * 10.0f);
    } else {
        m_sidebar_anim_width = target_width;
    }
    
    // Rendering
    ImGui::Render();
    const float clear_color_with_alpha[4] = { 0.12f, 0.12f, 0.12f, 1.00f }; // Darker background
    g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, NULL);
    g_pd3dDeviceContext->ClearRenderTargetView(g_mainRenderTargetView, clear_color_with_alpha);
    ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

    g_pSwapChain->Present(1, 0); // Present with vsync
}

void UI::Cleanup() {
    if (perf_monitor) {
        perf_monitor->Shutdown();
    }
    aimbot.m_config_manager.Save();
    ImPlot::DestroyContext();
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    CleanupDeviceD3D();
}

bool UI::IsRunning() {
    return !g_done;
}


// Helper functions (implementation is the same as before)
bool CreateDeviceD3D(HWND hWnd)
{
    // Setup swap chain
    DXGI_SWAP_CHAIN_DESC sd;
    ZeroMemory(&sd, sizeof(sd));
    sd.BufferCount = 2;
    sd.BufferDesc.Width = 0;
    sd.BufferDesc.Height = 0;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hWnd;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    UINT createDeviceFlags = 0;
    D3D_FEATURE_LEVEL featureLevel;
    const D3D_FEATURE_LEVEL featureLevelArray[2] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1, };
    if (D3D11CreateDeviceAndSwapChain(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, createDeviceFlags, featureLevelArray, 2, D3D11_SDK_VERSION, &sd, &g_pSwapChain, &g_pd3dDevice, &featureLevel, &g_pd3dDeviceContext) != S_OK)
        return false;

    CreateRenderTarget();
    return true;
}

void CleanupDeviceD3D()
{
    CleanupRenderTarget();
    if (g_pSwapChain) { g_pSwapChain->Release(); g_pSwapChain = NULL; }
    if (g_pd3dDeviceContext) { g_pd3dDeviceContext->Release(); g_pd3dDeviceContext = NULL; }
    if (g_pd3dDevice) { g_pd3dDevice->Release(); g_pd3dDevice = NULL; }
}

void CreateRenderTarget()
{
    ID3D11Texture2D* pBackBuffer;
    g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer));
    g_pd3dDevice->CreateRenderTargetView(pBackBuffer, NULL, &g_mainRenderTargetView);
    pBackBuffer->Release();
}

void CleanupRenderTarget()
{
    if (g_mainRenderTargetView) { g_mainRenderTargetView->Release(); g_mainRenderTargetView = NULL; }
}

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    // When the window is created, store the UI instance pointer
    if (msg == WM_CREATE) {
        CREATESTRUCT* pCreate = reinterpret_cast<CREATESTRUCT*>(lParam);
        SetWindowLongPtr(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pCreate->lpCreateParams));
        return 0;
    }

    // Retrieve the UI instance pointer
    UI* ui = reinterpret_cast<UI*>(GetWindowLongPtr(hWnd, GWLP_USERDATA));

    // If we have a valid pointer, delegate message handling to it
    if (ui) {
        return ui->HandleWndProc(hWnd, msg, wParam, lParam);
    }
    
    // Default handling if pointer is not valid yet
    return ::DefWindowProc(hWnd, msg, wParam, lParam);
}

// The actual WndProc implementation as a member of the UI class
LRESULT UI::HandleWndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return true;

    switch (msg)
    {
    case WM_NCCALCSIZE:
        if (wParam == TRUE) {
            // This effectively removes the standard window frame and title bar,
            // while keeping the resizing capabilities provided by WS_THICKFRAME.
            // The client area will now cover the entire window.
            return 0;
        }
        break;
    case WM_NCHITTEST:
    {
        // Let the default procedure handle resizing borders
        LRESULT hit = DefWindowProc(hWnd, msg, wParam, lParam);
        if (hit != HTCLIENT)
            return hit;

        // Check if the cursor is in our custom title bar area
        RECT window_rect;
        ::GetWindowRect(hWnd, &window_rect);
        POINT mouse_pos = { GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam) };

        // The title bar is the top 30 pixels of the window
        if (mouse_pos.y >= window_rect.top && mouse_pos.y < window_rect.top + 30)
        {
            // If ImGui wants to capture mouse, let it. This prevents our dragging
            // from interfering with ImGui widgets in the title bar.
            if (ImGui::GetIO().WantCaptureMouse)
                return HTCLIENT;
            
            return HTCAPTION; // This makes the area draggable
        }

        return HTCLIENT; // The rest of the window is client area
    }
    case WM_SIZE:
        if (g_pd3dDevice != NULL && wParam != SIZE_MINIMIZED)
        {
            CleanupRenderTarget();
            g_pSwapChain->ResizeBuffers(0, (UINT)LOWORD(lParam), (UINT)HIWORD(lParam), DXGI_FORMAT_UNKNOWN, 0);
            CreateRenderTarget();
        }
        return 0;
    case WM_SYSCOMMAND:
        if ((wParam & 0xfff0) == SC_KEYMENU) 
            return 0;
        break;
    case WM_DESTROY:
        ::PostQuitMessage(0);
        return 0;
    case WM_KEYDOWN:
    case WM_SYSKEYDOWN:
        m_last_vk_pressed = (int)wParam;
        break;
    case WM_KEYUP:
    case WM_SYSKEYUP:
        if (m_last_vk_pressed == (int)wParam) {
            m_last_vk_pressed = 0;
        }
        break;
    }
    return ::DefWindowProc(hWnd, msg, wParam, lParam);
}

// Helper to get a displayable name for a virtual-key code
std::string VKCodeToString(int vk_code) {
    // A simple map for common keys. This can be expanded.
    static const std::map<int, std::string> vk_map = {
        {VK_LBUTTON, "LMB"}, {VK_RBUTTON, "RMB"}, {VK_MBUTTON, "MMB"},
        {VK_XBUTTON1, "Mouse4"}, {VK_XBUTTON2, "Mouse5"},
        {VK_SHIFT, "Shift"}, {VK_CONTROL, "Ctrl"}, {VK_MENU, "Alt"},
        {VK_F1, "F1"}, {VK_F2, "F2"}, {VK_F3, "F3"}, {VK_F4, "F4"},
        {VK_F5, "F5"}, {VK_F6, "F6"}, {VK_F7, "F7"}, {VK_F8, "F8"},
        {VK_F9, "F9"}, {VK_F10, "F10"}, {VK_F11, "F11"}, {VK_F12, "F12"},
    };
    auto it = vk_map.find(vk_code);
    if (it != vk_map.end()) {
        return it->second;
    }
    // For regular ASCII keys
    if (vk_code >= 0x30 && vk_code <= 0x5A) {
        return std::string(1, static_cast<char>(vk_code));
    }
    return "Key " + std::to_string(vk_code);
}

void UI::ApplyCustomStyle(int theme_index) {
    PersonalizationSettings& ps = aimbot.m_personalization_settings;

    // Set base theme color based on preset
    if (theme_index != -1) {
        switch (theme_index) {
            case 1: ps.theme_color = ImVec4(0.26f, 0.59f, 0.98f, 1.00f); break; // Classic Dark (Default Blue)
            case 2: ps.theme_color = ImVec4(0.99f, 0.47f, 0.61f, 1.00f); break; // Dracula (Pink)
            case 3: ps.theme_color = ImVec4(0.44f, 0.38f, 0.91f, 1.00f); break; // Midnight Blue (Purple)
            case 4: ps.theme_color = ImVec4(0.31f, 0.69f, 0.33f, 1.00f); break; // Forest Green
            case 5: ps.theme_color = ImVec4(0.00f, 0.47f, 0.86f, 1.00f); break; // Light Mode (Bright Blue)
        }
    }

    ImGuiStyle& style = ImGui::GetStyle();

    // Global style settings
    style.WindowRounding = 5.0f;
    style.FrameRounding = 4.0f;
    style.ChildRounding = 4.0f;
    style.GrabRounding = 4.0f;
    style.PopupRounding = 4.0f;
    style.ScrollbarRounding = 4.0f;
    style.TabRounding = 4.0f;

    // Create color variations from the base theme color
    ImVec4 theme_col_bg = ps.theme_color;
    theme_col_bg.w = 0.5f;

    // Create a brighter, more vibrant hover color
    ImVec4 theme_col_hover;
    float h, s, v;
    ImGui::ColorConvertRGBtoHSV(ps.theme_color.x, ps.theme_color.y, ps.theme_color.z, h, s, v);
    v = (std::min)(v * 1.3f, 1.0f); // Increase brightness by 30% for a nice glow
    s = (std::min)(s * 1.1f, 1.0f); // Slightly increase saturation
    ImGui::ColorConvertHSVtoRGB(h, s, v, theme_col_hover.x, theme_col_hover.y, theme_col_hover.z);
    theme_col_hover.w = ps.theme_color.w * 0.9f; // Make it slightly less transparent than active

    ImVec4 theme_col_active = ps.theme_color;
    theme_col_active.w = 1.0f;

    // Apply colors
    ImVec4* colors = style.Colors;

    if (theme_index == 5) { // Light Mode
        colors[ImGuiCol_Text]                   = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
        ps.text_color = colors[ImGuiCol_Text]; // Sync text color
        colors[ImGuiCol_TextDisabled]           = ImVec4(0.60f, 0.60f, 0.60f, 1.00f);
        colors[ImGuiCol_WindowBg]               = ImVec4(0.94f, 0.94f, 0.94f, ps.window_opacity);
        colors[ImGuiCol_ChildBg]                = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        colors[ImGuiCol_PopupBg]                = ImVec4(1.00f, 1.00f, 1.00f, 0.98f);
        colors[ImGuiCol_Border]                 = ImVec4(0.00f, 0.00f, 0.00f, 0.30f);
        colors[ImGuiCol_BorderShadow]           = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        colors[ImGuiCol_FrameBg]                = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
        colors[ImGuiCol_FrameBgHovered]         = ImVec4(0.9f, 0.9f, 0.9f, 1.0f);
        colors[ImGuiCol_FrameBgActive]          = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
        colors[ImGuiCol_TitleBg]                = ImVec4(0.86f, 0.86f, 0.86f, 1.00f);
        colors[ImGuiCol_TitleBgActive]          = ImVec4(0.76f, 0.76f, 0.76f, 1.00f);
        colors[ImGuiCol_TitleBgCollapsed]       = ImVec4(0.86f, 0.86f, 0.86f, 0.7f);
        colors[ImGuiCol_MenuBarBg]              = ImVec4(0.86f, 0.86f, 0.86f, 1.00f);
        colors[ImGuiCol_ScrollbarBg]            = ImVec4(0.98f, 0.98f, 0.98f, 0.53f);
    } else { // Dark Modes
        colors[ImGuiCol_Text]                   = ps.text_color; // Use custom text color
        colors[ImGuiCol_TextDisabled]           = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
        colors[ImGuiCol_WindowBg]               = ImVec4(0.10f, 0.10f, 0.10f, ps.window_opacity);
        colors[ImGuiCol_ChildBg]                = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        colors[ImGuiCol_PopupBg]                = ImVec4(0.08f, 0.08f, 0.08f, 0.94f);
        colors[ImGuiCol_Border]                 = ImVec4(0.43f, 0.43f, 0.50f, 0.50f);
        colors[ImGuiCol_BorderShadow]           = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        colors[ImGuiCol_FrameBg]                = ImVec4(0.20f, 0.21f, 0.22f, 0.54f);
        colors[ImGuiCol_FrameBgHovered]         = ImVec4(0.40f, 0.40f, 0.40f, 0.40f);
        colors[ImGuiCol_FrameBgActive]          = ImVec4(0.18f, 0.18f, 0.18f, 0.67f);
        colors[ImGuiCol_TitleBg]                = ImVec4(0.04f, 0.04f, 0.04f, 1.00f);
        colors[ImGuiCol_TitleBgActive]          = ImVec4(0.29f, 0.29f, 0.29f, 1.00f);
        colors[ImGuiCol_TitleBgCollapsed]       = ImVec4(0.00f, 0.00f, 0.00f, 0.51f);
        colors[ImGuiCol_MenuBarBg]              = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
        colors[ImGuiCol_ScrollbarBg]            = ImVec4(0.02f, 0.02f, 0.02f, 0.53f);
    }
    
    // Set gradient colors based on the theme's window background, but only for presets.
    if (theme_index != 0 && theme_index != -1) { // If a preset is selected (not "Custom" or initial load)
        ps.gradient_color_top = style.Colors[ImGuiCol_WindowBg];
        
        ImGui::ColorConvertRGBtoHSV(ps.gradient_color_top.x, ps.gradient_color_top.y, ps.gradient_color_top.z, h, s, v);
        v = std::min(v + 0.15f, 1.0f);
        ImGui::ColorConvertHSVtoRGB(h, s, v, ps.gradient_color_bottom.x, ps.gradient_color_bottom.y, ps.gradient_color_bottom.z);
        ps.gradient_color_bottom.w = ps.gradient_color_top.w; // Match alpha
    }

    // Theme-agnostic colors (using the active theme color)
    colors[ImGuiCol_ScrollbarGrab]          = theme_col_hover;
    colors[ImGuiCol_ScrollbarGrabHovered]   = ImVec4(0.41f, 0.41f, 0.41f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabActive]    = ImVec4(0.51f, 0.51f, 0.51f, 1.00f);
    colors[ImGuiCol_CheckMark]              = ImVec4(0.94f, 0.94f, 0.94f, 1.00f);
    colors[ImGuiCol_SliderGrab]             = theme_col_hover;
    colors[ImGuiCol_SliderGrabActive]       = theme_col_active;
    colors[ImGuiCol_Button]                 = ImVec4(theme_col_bg.x, theme_col_bg.y, theme_col_bg.z, 0.40f);
    colors[ImGuiCol_ButtonHovered]          = theme_col_hover;
    colors[ImGuiCol_ButtonActive]           = theme_col_active;
    colors[ImGuiCol_Header]                 = ImVec4(theme_col_bg.x, theme_col_bg.y, theme_col_bg.z, 0.7f);
    colors[ImGuiCol_HeaderHovered]          = theme_col_hover;
    colors[ImGuiCol_HeaderActive]           = theme_col_active;
    colors[ImGuiCol_Separator]              = colors[ImGuiCol_Border];
    colors[ImGuiCol_SeparatorHovered]       = theme_col_hover;
    colors[ImGuiCol_SeparatorActive]        = theme_col_active;
    colors[ImGuiCol_ResizeGrip]             = ImVec4(theme_col_bg.x, theme_col_bg.y, theme_col_bg.z, 0.25f);
    colors[ImGuiCol_ResizeGripHovered]      = theme_col_hover;
    colors[ImGuiCol_ResizeGripActive]       = theme_col_active;
    colors[ImGuiCol_Tab]                    = ImVec4(theme_col_bg.x * 0.7f, theme_col_bg.y * 0.7f, theme_col_bg.z * 0.7f, 0.86f);
    colors[ImGuiCol_TabHovered]             = theme_col_hover;
    colors[ImGuiCol_TabActive]              = theme_col_active;
    colors[ImGuiCol_TabUnfocused]           = ImVec4(0.07f, 0.10f, 0.15f, 0.97f);
    colors[ImGuiCol_TabUnfocusedActive]     = ImVec4(theme_col_bg.x * 0.5f, theme_col_bg.y * 0.5f, theme_col_bg.z * 0.5f, 1.00f);
    colors[ImGuiCol_PlotLines]              = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
    colors[ImGuiCol_PlotLinesHovered]       = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
    colors[ImGuiCol_PlotHistogram]          = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
    colors[ImGuiCol_PlotHistogramHovered]   = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
    colors[ImGuiCol_TextSelectedBg]         = ImVec4(theme_col_bg.x, theme_col_bg.y, theme_col_bg.z, 0.35f);
    colors[ImGuiCol_DragDropTarget]         = ImVec4(1.00f, 1.00f, 0.00f, 0.90f);
    colors[ImGuiCol_NavHighlight]           = theme_col_active;
    colors[ImGuiCol_NavWindowingHighlight]  = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg]      = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
    colors[ImGuiCol_ModalWindowDimBg]       = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);
}

void UI::RenderTab(Tab tab) {
    switch (tab) {
        case Tab::Model: RenderModelTab(); break;
        case Tab::Detection: RenderAimingTab(); break;
        case Tab::RecoilControl: RenderRecoilTab(); break;
        case Tab::Performance: RenderPerformanceTab(); break;
        case Tab::Logging: RenderLoggingTab(); break;
        case Tab::Stealth: RenderStealthTab(); break;
        case Tab::Personalization: RenderPersonalizationTab(); break;
    }
}

void UI::RenderModelTab() {
    ImGui::Text(ICON_FA_CUBE " Model Settings");
    ImGui::Separator();
    
    // Model selection dropdown
    if (!model_files.empty()) {
        if (ImGui::Combo("Select Model", &selected_model, [](void* data, int idx, const char** out_text) {
            auto& files = *static_cast<std::vector<std::string>*>(data);
            if (idx < 0 || idx >= static_cast<int>(files.size())) return false;
            *out_text = files[idx].c_str();
            return true;
        }, &model_files, static_cast<int>(model_files.size()))) {
            // User made a selection
            aimbot.SetSelectedModel(model_files[selected_model]);
        }
    }
            
    // Build Button
    if (ImGui::Button("Build Engine")) {
        std::string model_path = (get_models_path() / model_files[selected_model]).string();
        aimbot.InitiateModelBuild(model_path);
    }
    ImGui::SameLine();
    // Load Button
    if (ImGui::Button("Load Engine")) {
        std::string engine_path = (get_models_path() / model_files[selected_model]).replace_extension(".engine").string();
        aimbot.LoadModel(engine_path);
    }
    ImGui::SameLine(); HelpMarker("Loads a pre-built .engine file for fast startup. The engine must match the selected model and your current GPU.");

    ImGui::ToggleSwitch(ICON_FA_BOLT " Use FP16 Precision", &aimbot.fp16_mode);
    ImGui::SameLine(); HelpMarker("Enables faster, half-precision floating-point calculations. May slightly decrease accuracy but significantly improves performance. Requires a compatible GPU.");
    
    ImGui::InputInt(ICON_FA_MEMORY " Max Workspace Size (MB)", &aimbot.workspace_size);
    ImGui::SameLine(); HelpMarker("The maximum amount of GPU memory (in megabytes) that TensorRT can use for building the engine. Larger values can lead to faster engines but require more VRAM.");
    
    ImGui::InputInt(ICON_FA_LAYER_GROUP " Batch Size", &aimbot.batch_size);
    ImGui::SameLine(); HelpMarker("The number of images to process in a single inference pass. Currently, only a batch size of 1 is fully supported.");
    
    ImGui::SliderFloat(ICON_FA_OBJECT_UNGROUP " IOU Threshold", &aimbot.iou_threshold, 0.1f, 1.0f, "%.2f");
    ImGui::SameLine(); HelpMarker("Intersection over Union (IOU) threshold for Non-Maximum Suppression (NMS). This value controls how overlapping detection boxes are filtered. Lower values result in fewer, less-overlapping boxes, while higher values may allow more overlapping boxes.");
    
    // Display build status
    ImGui::Text("Status:");
    ImGui::SameLine();
    ImGui::Text(aimbot.GetBuildStatus().c_str());
}

void UI::RenderAimingTab() {
    ImGui::Text(ICON_FA_CROSSHAIRS " Aiming Settings");
    ImGui::Separator();
    AimingSettings& settings = aimbot.m_aiming_settings;

    // Hotkey
    if (ImGui::Button("Set Aim Hotkey")) {
        ImGui::OpenPopup("hotkey_popup");
    }
    ImGui::SameLine();
    ImGui::Text("Current Hotkey: %s", settings.hotkey.c_str());
    ImGui::SameLine(); HelpMarker("Set the key that must be held down to activate the aimbot.");

    if (ImGui::BeginPopup("hotkey_popup")) {
        ImGui::Text("Press any key...");
        // This part would need a more complex implementation to capture a key press.
        // For now, it's a placeholder.
        ImGui::EndPopup();
    }

    // Targeting
    ImGui::Combo(ICON_FA_STAR " Priority", reinterpret_cast<int*>(&settings.priority), "Highest Confidence\0Closest to Crosshair\0");
    ImGui::SameLine(); HelpMarker("Determines how to select a target when multiple are visible.\n'Highest Confidence': Prefers the target the model is most sure about.\n'Closest to Crosshair': Prefers the target nearest to the center of your screen.");

    ImGui::SliderFloat(ICON_FA_PERCENT " Confidence Threshold", &settings.confidence_threshold, 0.0f, 1.0f);
    ImGui::SameLine(); HelpMarker("The minimum confidence score a detection must have to be considered a valid target. Higher values reduce false positives but may miss some targets.");

    // Aiming behavior
    ImGui::SliderFloat(ICON_FA_TACHOMETER_ALT " Max Speed", &settings.max_speed, 100.0f, 5000.0f, "%.0f px/s");
    ImGui::SameLine(); HelpMarker("The maximum speed at which the mouse will move towards a target. This setting is primarily for non-PID aiming modes.");
    
    ImGui::SliderFloat(ICON_FA_DOT_CIRCLE " Offset X", &settings.target_offset_x, -1.0f, 1.0f);
    ImGui::SameLine(); HelpMarker("Adjusts the horizontal aim point. A value of 0 targets the center of the bounding box, -0.5 targets the left edge, and 0.5 targets the right edge.");

    ImGui::SliderFloat(ICON_FA_DOT_CIRCLE " Offset Y", &settings.target_offset_y, -1.0f, 1.0f);
    ImGui::SameLine(); HelpMarker("Adjusts the vertical aim point. A value of 0 targets the center, while negative values aim higher (e.g., towards the head) and positive values aim lower.");

    // Smoothing Section
    ImGui::Separator();
    ImGui::Text(ICON_FA_WAVE_SQUARE " Smoothing");
    ImGui::ToggleSwitch("Use EMA Smoothing", &settings.use_ema);
    ImGui::SameLine(); HelpMarker("Enables Exponential Moving Average (EMA) smoothing. This is a simple smoothing method that averages the mouse movement over time to reduce jitter.");

    ImGui::SliderFloat("EMA Smoothing Amount", &settings.ema_alpha, 0.0f, 1.0f);
    ImGui::SameLine(); HelpMarker("Controls the amount of EMA smoothing. Higher values result in smoother, but more delayed, mouse movement. A value of 0 disables smoothing.");

    // PID Controller Section
    ImGui::Separator();
    ImGui::Text(ICON_FA_COGS " PID Controller");
    ImGui::ToggleSwitch("Use PID Controller", &settings.use_pid);
    ImGui::SameLine(); HelpMarker("Enables a Proportional-Integral-Derivative (PID) controller for mouse movement. This is a more advanced and responsive method than simple smoothing, allowing for fine-tuned control over aiming behavior.");

    BeginFade(ImGui::GetID("PIDControls"), settings.use_pid);
    ImGui::SliderFloat(ICON_FA_ANGLE_DOUBLE_RIGHT " P Gain", &settings.pid_p, 0.0f, 0.5f, "%.3f");
    ImGui::SameLine(); HelpMarker("Proportional Gain: The primary driving force. This determines how strongly the aimbot reacts to the distance between your crosshair and the target. Higher values result in faster movement.");
    
    ImGui::SliderFloat(ICON_FA_HISTORY " I Gain", &settings.pid_i, 0.0f, 0.1f, "%.3f");
    ImGui::SameLine(); HelpMarker("Integral Gain: Corrects for small, persistent errors over time. This helps the aimbot 'settle' perfectly onto a target instead of stopping just short. Can cause instability if set too high.");

    ImGui::SliderFloat(ICON_FA_CHART_LINE " D Gain", &settings.pid_d, 0.0f, 0.2f, "%.3f");
    ImGui::SameLine(); HelpMarker("Derivative Gain: Dampens the movement to prevent overshooting the target. Acts like a brake, predicting future error and slowing down the aim as it approaches the target. Higher values provide more damping.");
    EndFade();

    // Dynamic PID Section
    ImGui::Separator();
    ImGui::Text(ICON_FA_EXPAND_ARROWS_ALT " Dynamic PID Scaling");
    ImGui::ToggleSwitch("Enable Dynamic PID", &settings.use_dynamic_pid);
    ImGui::SameLine(); HelpMarker("Scales the PID strength based on the target's distance (estimated from its bounding box size). This makes aiming smoother for distant targets and more responsive for closer ones.");
    
    BeginFade(ImGui::GetID("DynamicPIDControls"), settings.use_dynamic_pid);
    ImGui::SliderFloat(ICON_FA_RULER_HORIZONTAL " Max Distance (px)", &settings.dynamic_pid_distance_max, 10.0f, 1000.0f, "%.0f");
    ImGui::SameLine(); HelpMarker("The target bounding box width at which the PID strength will be at its MINIMUM. Targets with a bounding box wider than this will use the minimum scale factor.");
    
    ImGui::SliderFloat(ICON_FA_COMPRESS_ARROWS_ALT " Minimum Scale", &settings.dynamic_pid_scale_min, 0.1f, 1.0f, "%.2f");
    ImGui::SameLine(); HelpMarker("The minimum scaling factor for the P-Gain (e.g., a value of 0.4 means 40% strength). This is the strength applied when the target is at or beyond the 'Max Distance'.");
    EndFade();

    // Prediction Section
    ImGui::Separator();
    ImGui::Text(ICON_FA_BULLSEYE " Prediction");
    ImGui::ToggleSwitch("Enable Prediction", &settings.use_prediction);
    ImGui::SameLine(); HelpMarker("Enables target velocity prediction to compensate for latency and target movement, improving accuracy against moving targets.");

    BeginFade(ImGui::GetID("PredictionControls"), settings.use_prediction);
    ImGui::SliderFloat(ICON_FA_CROSSHAIRS " Prediction Strength X", &settings.prediction_strength_x, 0.0f, 5.0f, "%.2f");
    ImGui::SameLine(); HelpMarker("A multiplier for how far to predict the target's movement on the horizontal (X) axis.");

    ImGui::SliderFloat(ICON_FA_CROSSHAIRS " Prediction Strength Y", &settings.prediction_strength_y, 0.0f, 5.0f, "%.2f");
    ImGui::SameLine(); HelpMarker("A multiplier for how far to predict the target's movement on the vertical (Y) axis.");
    EndFade();

    // PID Controller Graph Section
    ImGui::Separator();
    if (ImGui::CollapsingHeader("PID Controller Live Graph")) {
        
        // --- Graph Controls ---
        if (ImGui::Button("Start Recording")) {
            aimbot.m_is_recording_pid = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Stop Recording")) {
            aimbot.m_is_recording_pid = false;
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear Data")) {
            aimbot.ClearPlotData();
        }
        ImGui::SameLine();
        ImGui::Text(aimbot.m_is_recording_pid ? "Status: Recording" : "Status: Stopped");
        
        // --- The Plot ---
        std::lock_guard<std::mutex> lock(aimbot.plot_data_mutex);
        
        if (ImPlot::BeginPlot("PID Controller Response", ImVec2(-1, 200))) {
            ImPlot::SetupAxes("Time (frames)", "Value");
            ImPlot::SetupAxesLimits(0, aimbot.pid_plot_data.max_size, 0, 100, ImGuiCond_Always);
            
            ImPlot::PlotLine("Error", aimbot.pid_plot_data.error.data(), aimbot.pid_plot_data.max_size, 1.0, 0, ImPlotLineFlags_None, aimbot.pid_plot_data.offset);
            ImPlot::PlotLine("Output", aimbot.pid_plot_data.output.data(), aimbot.pid_plot_data.max_size, 1.0, 0, ImPlotLineFlags_None, aimbot.pid_plot_data.offset);
            
            ImPlot::EndPlot();
        }
    }
}

void UI::RenderRecoilTab() {
    ImGui::Text(ICON_FA_RETWEET " Recoil Control Settings");
    ImGui::Separator();
    RecoilSettings& settings = aimbot.m_recoil_settings;

    ImGui::ToggleSwitch("Enable Recoil Control", &settings.enable_recoil_control);
    ImGui::SameLine(); HelpMarker("Activates the anti-recoil system when both left and right mouse buttons are pressed.");

    BeginFade(ImGui::GetID("RecoilControls"), settings.enable_recoil_control);
    ImGui::SliderFloat("Strength", &settings.recoil_strength, 0.0f, 20.0f, "%.1f");
    ImGui::SameLine(); HelpMarker("Overall strength of the recoil compensation.");

    ImGui::SliderFloat("Horizontal Pull", &settings.recoil_horizontal, -5.0f, 5.0f, "%.1f");
    ImGui::SameLine(); HelpMarker("Compensates for horizontal recoil. Negative values pull left, positive values pull right.");

    ImGui::SliderFloat("Vertical Pull", &settings.recoil_vertical, 0.0f, 10.0f, "%.1f");
    ImGui::SameLine(); HelpMarker("Compensates for vertical recoil (downward pull).");
    EndFade();
}

void UI::RenderPerformanceTab() {
    ImGui::Text(ICON_FA_CHART_LINE " GPU Performance Metrics");
    ImGui::Separator();

    if (perf_monitor) {
        GpuMetrics metrics = perf_monitor->GetMetrics(); // Get the latest thread-safe data

        ImGui::Text("GPU: %s", metrics.deviceName.c_str());
        ImGui::Separator();

        ImGui::Text("Temperature: %u C", metrics.temperature);
        
        ImGui::Text("GPU Utilization:");
        ImGui::SameLine();
        char gpu_buf[32];
        sprintf_s(gpu_buf, "%u %%", metrics.utilizationGpu);
        ImGui::ProgressBar(metrics.utilizationGpu / 100.0f, ImVec2(-1, 0), gpu_buf);

        ImGui::Text("Memory Usage:");
        ImGui::SameLine();
        float mem_percent = (metrics.memoryTotal > 0) ? (static_cast<float>(metrics.memoryUsed) / metrics.memoryTotal) : 0.0f;
        char mem_buf[32];
        sprintf_s(mem_buf, "%.1f / %.1f GB", (metrics.memoryUsed / 1024.0 / 1024.0 / 1024.0), (metrics.memoryTotal / 1024.0 / 1024.0 / 1024.0));
        ImGui::ProgressBar(mem_percent, ImVec2(-1, 0), mem_buf);

        ImGui::Text("Power Draw:");
        ImGui::SameLine();
        float power_percent = (metrics.powerLimit > 0) ? (static_cast<float>(metrics.powerUsage) / metrics.powerLimit) : 0.0f;
        char power_buf[32];
        sprintf_s(power_buf, "%u / %u W", (metrics.powerUsage / 1000), (metrics.powerLimit / 1000));
        ImGui::ProgressBar(power_percent, ImVec2(-1, 0), power_buf);
    }
    else {
        ImGui::Text("Performance monitor not available.");
    }
}

void UI::RenderLoggingTab() {
    ImGui::Text(ICON_FA_BOOK " Logs");
    ImGui::Separator();
    
    if (ImGui::Button("Clear Log")) {
        Logger::GetInstance().Clear();
    }
    ImGui::SameLine();
    ImGui::Text("Hold CTRL to drag the log window.");

    ImGui::Separator();
    
    ImGui::BeginChild("LogView", ImVec2(0, 0), true, ImGuiWindowFlags_HorizontalScrollbar);
    auto logs = Logger::GetInstance().GetMessages();
    for (const auto& log : logs) {
        ImGui::TextUnformatted(log.c_str());
    }
    // Auto-scroll to the bottom
    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) {
        ImGui::SetScrollHereY(1.0f);
    }
    ImGui::EndChild();
}

void UI::RenderStealthTab() {
    ImGui::Text(ICON_FA_SHIELD_ALT " Application Killswitch");
    ImGui::Separator();

    static bool is_listening = false;
    int current_key = aimbot.killswitch_key.load();
    ImGui::SameLine(); HelpMarker("Sets a global hotkey that will immediately and completely terminate the application. This is a safety feature.");

    // When listening, check if a key has been captured by WndProc
    if (is_listening) {
        ImGui::Button("Press any key...", ImVec2(150, 0));
        if (m_last_vk_pressed != 0) {
            // Exclude common keys
            if (m_last_vk_pressed != VK_ESCAPE && m_last_vk_pressed != VK_RETURN) {
                aimbot.killswitch_key.store(m_last_vk_pressed);
                m_just_set_killswitch = true; // Set flag to wait for key release
            }
            is_listening = false;
            m_last_vk_pressed = 0; // Consume the key press
        }
    } else {
        if (ImGui::Button("Set Killswitch Key", ImVec2(150, 0))) {
            m_last_vk_pressed = 0; // Clear any previous key press
            is_listening = true;
        }
    }

    // Display current key
    ImGui::SameLine(0.0f, 20.0f);
    if (current_key == 0) {
        ImGui::Text("Current Key: None");
    } else {
        ImGui::Text("Current Key: %s", VKCodeToString(current_key).c_str());
    }

    // "Clear Key" button
    if (ImGui::Button("Clear Killswitch", ImVec2(150, 0))) {
        aimbot.killswitch_key.store(0);
        is_listening = false; 
    }

    ImGui::Dummy(ImVec2(0, 20));
    ImGui::Separator();
    ImGui::Text(ICON_FA_EYE " Application Visibility Hotkey");
    ImGui::Separator();
    ImGui::SameLine(); HelpMarker("Sets a hotkey that will toggle the visibility of this UI window.");

    static bool is_listening_hide = false;
    int current_hide_key = aimbot.hide_toggle_key.load();

    if (is_listening_hide) {
        ImGui::Button("Press any key...", ImVec2(150, 0));
        if (m_last_vk_pressed != 0) {
            if (m_last_vk_pressed != VK_ESCAPE && m_last_vk_pressed != VK_RETURN) {
                aimbot.hide_toggle_key.store(m_last_vk_pressed);
                m_just_set_hide_key = true; // Set flag to wait for key release
            }
            is_listening_hide = false;
            m_last_vk_pressed = 0;
        }
    } else {
        if (ImGui::Button("Set Hide/Show Key", ImVec2(150, 0))) {
            m_last_vk_pressed = 0;
            is_listening_hide = true;
        }
    }

    ImGui::SameLine(0.0f, 20.0f);
    if (current_hide_key == 0) {
        ImGui::Text("Current Key: None");
    } else {
        ImGui::Text("Current Key: %s", VKCodeToString(current_hide_key).c_str());
    }

    if (ImGui::Button("Clear Hide/Show Key", ImVec2(150, 0))) {
        aimbot.hide_toggle_key.store(0);
        is_listening_hide = false;
    }
}

void UI::RenderPersonalizationTab() {
    ImGui::Text(ICON_FA_PAINT_BRUSH " Theme & Style Customization");
    ImGui::Separator();
    ImGui::SameLine(); HelpMarker("Customize the visual appearance of the application.");

    // --- Theme Presets ---
    const char* themes[] = { "Custom", "Classic Dark", "Dracula", "Midnight Blue", "Forest Green", "Light Mode" };
    static int current_theme_index = 0;
    if (ImGui::Combo(ICON_FA_PALETTE " Theme", &current_theme_index, themes, IM_ARRAYSIZE(themes))) {
        if (current_theme_index > 0) { // Index 0 is for "Custom"
            ApplyCustomStyle(current_theme_index);
        }
    }
    ImGui::Dummy(ImVec2(0, 10));

    // Let user customize the "Custom" theme
    if (current_theme_index == 0) {
        // Use a table to layout the color pickers side-by-side.
        // By removing the SizingStretchSame flag, the table columns will now wrap their content.
        if (ImGui::BeginTable("ColorPickersLayout", 2))
        {
            ImGui::TableNextColumn();
            ImGui::Text("Custom Accent Color:");
            if (ImGui::ColorEdit4("##ThemeColor", (float*)&aimbot.m_personalization_settings.theme_color, ImGuiColorEditFlags_NoInputs)) {
                 ApplyCustomStyle();
            }

            ImGui::TableNextColumn();
            ImGui::Text("Custom Text Color:");
            if (ImGui::ColorEdit4("##TextColor", (float*)&aimbot.m_personalization_settings.text_color, ImGuiColorEditFlags_NoInputs)) {
                ApplyCustomStyle();
            }
            
            ImGui::EndTable();
        }
        ImGui::Dummy(ImVec2(0, 10));

        ImGui::Text("Custom Outline Color:");
        if (ImGui::ColorEdit4("##OutlineColor", (float*)&aimbot.m_personalization_settings.outline_color, ImGuiColorEditFlags_NoInputs)) {
            // No need to call ApplyCustomStyle() here unless the outline is part of the style
        }
    }

    // --- Background Gradient ---
    ImGui::Dummy(ImVec2(0, 10));
    ImGui::Text("Background Gradient");
    ImGui::Separator();
    
    // Use a table to layout the color pickers side-by-side.
    if (ImGui::BeginTable("GradientColorLayout", 2))
    {
        ImGui::TableNextColumn();
        ImGui::Text("Top Color:");
        if (ImGui::ColorEdit4("##GradientTop", (float*)&aimbot.m_personalization_settings.gradient_color_top, ImGuiColorEditFlags_NoInputs)) {
            if (current_theme_index != 0) {
                current_theme_index = 0; // Switch to custom if a preset was active
            }
        }

        ImGui::TableNextColumn();
        ImGui::Text("Bottom Color:");
        if (ImGui::ColorEdit4("##GradientBottom", (float*)&aimbot.m_personalization_settings.gradient_color_bottom, ImGuiColorEditFlags_NoInputs)) {
            if (current_theme_index != 0) {
                current_theme_index = 0; // Switch to custom if a preset was active
            }
        }
        ImGui::EndTable();
    }

    // --- UI Scaling ---
    ImGui::Text("UI Scale");
    ImGui::SameLine(); HelpMarker("Adjusts the overall size of all UI elements, including text and widgets.");
    ImGui::SliderFloat(ICON_FA_EXPAND " UI Scale", &aimbot.m_personalization_settings.ui_scale, 0.5f, 2.0f, "%.2f");
    ImGui::SameLine();
    if (ImGui::Button("Reset")) { aimbot.m_personalization_settings.ui_scale = 1.0f; }

    ImGui::Dummy(ImVec2(0, 10));

    // --- Font Selection ---
    ImGui::Text("Font");
    ImGui::SameLine(); HelpMarker("Select a custom font for the UI from the 'include/Fonts' directory. Requires application restart to load new fonts.");
    if (ImGui::Combo(ICON_FA_FONT " Font", &m_selected_font_index, [](void* data, int idx, const char** out_text) {
        auto& files = *static_cast<std::vector<std::string>*>(data);
        if (idx < 0 || idx >= static_cast<int>(files.size())) return false;
        *out_text = files[idx].c_str();
        return true;
        }, &m_font_files, static_cast<int>(m_font_files.size()))) {
        // The PushFont/PopFont in the main Render loop will handle the change.
        if (!m_font_files.empty()) {
            aimbot.m_personalization_settings.font_name = m_font_files[m_selected_font_index];
        }
    }

    ImGui::Dummy(ImVec2(0, 10));

    // --- Custom Window Title ---
    ImGui::Text("Window Title");
    ImGui::SameLine(); HelpMarker("Set a custom title for this window. This can help with OBS capture settings or for general stealth.");
    if (ImGui::InputText(ICON_FA_WINDOW_RESTORE " Window Title", m_custom_window_title_buf, sizeof(m_custom_window_title_buf))) {
        aimbot.m_personalization_settings.custom_window_title = m_custom_window_title_buf;
        ::SetWindowText(m_hwnd, aimbot.m_personalization_settings.custom_window_title.c_str());
    }

    ImGui::Dummy(ImVec2(0, 10));

    // --- Advanced Style Editor ---
    ImGuiStyle& style = ImGui::GetStyle();
    if (ImGui::TreeNode(ICON_FA_COG " Advanced Style Settings")) {
        ImGui::SliderFloat(ICON_FA_ADJUST " Window Opacity", &aimbot.m_personalization_settings.window_opacity, 0.1f, 1.0f, "%.2f");
        style.Colors[ImGuiCol_WindowBg].w = aimbot.m_personalization_settings.window_opacity;

        ImGui::SliderFloat(ICON_FA_BORDER_STYLE " Rounding", &style.WindowRounding, 0.0f, 12.0f, "%.0f");
        style.ChildRounding = style.FrameRounding = style.GrabRounding = style.PopupRounding = style.ScrollbarRounding = style.TabRounding = style.WindowRounding;
        
        ImGui::SliderFloat2(ICON_FA_COMPRESS " Window Padding", (float*)&style.WindowPadding, 0.0f, 20.0f, "%.0f");
        ImGui::SliderFloat2(ICON_FA_EXPAND " Frame Padding", (float*)&style.FramePadding, 0.0f, 20.0f, "%.0f");
        ImGui::SliderFloat2(ICON_FA_ARROWS_ALT_H " Item Spacing", (float*)&style.ItemSpacing, 0.0f, 20.0f, "%.0f");
        ImGui::TreePop();
    }
}

void UI::LoadFonts() {
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->Clear(); // Clear any existing fonts
    
    // 1. Load the default font.
    ImFont* default_font = io.Fonts->AddFontDefault();
    m_loaded_fonts[m_default_font_name] = default_font;
    m_font_files.insert(m_font_files.begin(), m_default_font_name);

    // 2. Define configuration for Font Awesome. We need to merge it.
    static const ImWchar icons_ranges[] = { ICON_MIN_FA, ICON_MAX_FA, 0 };
    ImFontConfig icons_config;
    icons_config.MergeMode = true;
    icons_config.PixelSnapH = true;

    fs::path fa_font_path = get_executable_dir() / "Icons" / "Font Awesome 6 Free-Solid-900.otf";

    // 3. Load custom fonts and merge Font Awesome into each one.
    for (const auto& font_filename : m_font_files) {
        if (font_filename == m_default_font_name) {
            // Merge into default font
            if(fs::exists(fa_font_path)) {
                io.Fonts->AddFontFromFileTTF(fa_font_path.string().c_str(), 13.0f, &icons_config, icons_ranges);
            }
            continue;
        }

        fs::path font_path = get_fonts_path() / font_filename;
        if (fs::exists(font_path)) {
            ImFont* font = io.Fonts->AddFontFromFileTTF(font_path.string().c_str(), 16.0f);
            if (font) {
                // Merge FA into this custom font
                if(fs::exists(fa_font_path)) {
                     io.Fonts->AddFontFromFileTTF(fa_font_path.string().c_str(), 13.0f, &icons_config, icons_ranges);
                }
                m_loaded_fonts[font_filename] = font;
                Logger::GetInstance().Log("Successfully loaded font: " + font_filename);
            } else {
                Logger::GetInstance().Log("Failed to load font: " + font_filename);
            }
        } else {
            Logger::GetInstance().Log("Font file not found: " + font_path.string());
        }
    }
     // The font atlas is implicitly rebuilt by the renderer backend.
}

bool UI::AnimatedTabButton(const char* icon, const char* label, Tab tab_id, bool collapsed)
{
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    if (window->SkipItems)
        return false;

    ImGuiContext& g = *GImGui;
    const ImGuiStyle& style = g.Style;
    const ImGuiID id = window->GetID(label); // Use label for a consistent ID
    const ImVec2 label_size = ImGui::CalcTextSize(label, NULL, true);
    const ImVec2 icon_size = ImGui::CalcTextSize(icon, NULL, true);

    ImVec2 pos = window->DC.CursorPos;
    ImVec2 size = ImGui::CalcItemSize(ImVec2(-1, 30), 0.0f, 30.0f);

    const ImRect bb(pos, ImVec2(pos.x + size.x, pos.y + size.y));
    ImGui::ItemSize(size, style.FramePadding.y);
    if (!ImGui::ItemAdd(bb, id))
        return false;

    bool hovered, held;
    bool pressed = ImGui::ButtonBehavior(bb, id, &hovered, &held);

    // Update animation progress
    float& progress = m_tab_hover_progress[tab_id];
    progress = ImClamp(progress + g.IO.DeltaTime * 6.0f * (hovered ? 1.0f : -1.0f), 0.0f, 1.0f);

    // --- Render ---
    // Background color
    const ImU32 bg_col = ImGui::GetColorU32(ImLerp(style.Colors[ImGuiCol_Button], style.Colors[ImGuiCol_ButtonHovered], progress));
    window->DrawList->AddRectFilled(bb.Min, bb.Max, bg_col, style.TabRounding);

    // Icon and Text color
    const ImU32 text_col = ImGui::GetColorU32(ImLerp(style.Colors[ImGuiCol_Text], style.Colors[ImGuiCol_ButtonActive], progress));
    
    if (collapsed) {
        // --- Collapsed State ---
        // Handle the rotating icon case
        if (tab_id == Tab::Detection && progress > 0.01f) {
            float angle = -IM_PI * 0.5f * progress;
            ImVec2 center = ImVec2(pos.x + size.x * 0.5f, pos.y + size.y * 0.5f);
            window->DrawList->PushTextureID(g.Font->ContainerAtlas->TexID);
            ImGui::RenderTextRotated(icon, center, angle, text_col);
            window->DrawList->PopTextureID();
        } else {
            // Standard centered icon
            ImVec2 icon_pos = ImVec2(pos.x + (size.x - icon_size.x) / 2.0f, pos.y + (size.y - icon_size.y) / 2.0f);
            window->DrawList->AddText(icon_pos, text_col, icon);
        }

        // Show tooltip on hover
        if (hovered) {
            const char* tooltip_text = label;
            while (*tooltip_text == ' ') { // Trim leading spaces for a cleaner tooltip
                tooltip_text++;
            }
            ImGui::SetTooltip("%s", tooltip_text);
        }
    } else {
        // --- Expanded State ---
        // Handle the rotating icon case
        if (tab_id == Tab::Detection && progress > 0.01f) {
            float angle = -IM_PI * 0.5f * progress;
            ImVec2 center = ImVec2(pos.x + 20, pos.y + size.y / 2);
            window->DrawList->PushTextureID(g.Font->ContainerAtlas->TexID);
            ImGui::RenderTextRotated(icon, center, angle, text_col);
            window->DrawList->PopTextureID();
        } else {
            // Standard icon on the left
            ImVec2 icon_pos = ImVec2(pos.x + 10, pos.y + (size.y - icon_size.y) / 2.0f);
            window->DrawList->AddText(icon_pos, text_col, icon);
        }
        // Text label to the right
        ImVec2 text_pos = ImVec2(pos.x + 40, pos.y + (size.y - label_size.y) / 2.0f);
        window->DrawList->AddText(text_pos, text_col, label);
    }

    return pressed;
}

// Map to store animation progress for context-aware controls
static std::map<ImGuiID, float> g_fade_progress;

// A helper function to begin a group of controls that can be faded in/out.
// This function calculates the animation progress and sets the alpha for the controls.
void UI::BeginFade(ImGuiID id, bool show)
{
    float& progress = g_fade_progress[id];
    progress = ImClamp(progress + ImGui::GetIO().DeltaTime * 8.0f * (show ? 1.0f : -1.0f), 0.0f, 1.0f);

    // Start a child window with a calculated height to create a smooth collapse/expand animation.
    // The alpha is also applied to fade the content.
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, progress);
    ImGui::BeginChild(id, ImVec2(0, ImGui::GetTextLineHeightWithSpacing() * 3.5f * progress), false, ImGuiWindowFlags_NoScrollbar);
}

// Ends the fading group.
void UI::EndFade()
{
    ImGui::EndChild();
    ImGui::PopStyleVar();
}
