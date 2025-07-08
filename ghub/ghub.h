#ifndef GHUB_H
#define GHUB_H

#define NOMINMAX
#include <windows.h>
#include <filesystem>
#include <string>

// Define function pointer types from the user's specific DLL
typedef bool(WINAPI* p_mouse_open)();
typedef bool(WINAPI* p_mouse_close)();
typedef bool(WINAPI* p_moveR)(int dx, int dy);
typedef bool(WINAPI* p_press)(int btn);
typedef bool(WINAPI* p_release)(); 

class GhubMouse
{
public:
    GhubMouse();
    ~GhubMouse();
    bool move_r(int dx, int dy);
    bool btn_down(int btn);
    bool btn_up(int btn);
    bool is_valid();

private:
    HMODULE m_h_module;
    p_moveR m_move_r_ptr;
    p_press m_press_ptr;
    p_release m_release_ptr;
    bool m_ok;
};

#endif // GHUB_H