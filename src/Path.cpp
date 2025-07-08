#include "Path.h"
#include <windows.h>
#include <filesystem>

namespace fs = std::filesystem;

fs::path get_executable_path() {
    wchar_t path[MAX_PATH] = { 0 };
    GetModuleFileNameW(NULL, path, MAX_PATH);
    return fs::path(path);
}

fs::path get_executable_dir() {
    return get_executable_path().parent_path();
}

fs::path get_models_path() {
    return get_executable_path().parent_path() / "models";
}

fs::path get_fonts_path() {
    // Return a "Fonts" directory located next to the executable
    return get_executable_dir() / "Fonts";
} 