#pragma once

#include "nlohmann/json.hpp"
#include <string>

// Forward-declare the Aimbot class to avoid circular header dependencies.
class Aimbot;

class ConfigManager {
public:
    // The constructor takes a reference to the Aimbot instance to access its settings.
    ConfigManager(Aimbot& aimbot);

    // Saves the current application settings to config.json.
    void Save() const;

    // Loads settings from config.json and applies them.
    void Load();

private:
    // Gets the full path to the config.json file.
    std::string GetConfigPath() const;

    // A reference to the main Aimbot class.
    Aimbot& m_aimbot;
};
