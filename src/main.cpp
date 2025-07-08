// main.cpp
#include "Aimbot.h"
#include "UI.h"
#include <iostream>
#include <windows.h>

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PSTR lpCmdLine, int nCmdShow) {
    Aimbot aimbot;
    UI ui(aimbot);

    if (!ui.Init()) {
        std::cerr << "Failed to initialize UI" << std::endl;
        return -1;
    }

    while(ui.IsRunning()) {
        ui.Render();
    }

    // ui.Cleanup(); // This is now handled by the UI object's destructor
    
    return 0;
}
