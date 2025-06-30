#pragma once

#include <filesystem>

// Returns the full path to the executable.
std::filesystem::path get_executable_path();

// Returns the directory containing the executable.
std::filesystem::path get_executable_dir();

// Returns the path to the models directory.
std::filesystem::path get_models_path();

// Gets the path to the fonts directory
std::filesystem::path get_fonts_path(); 