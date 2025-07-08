#include "ConfigManager.h"
#include "Aimbot.h"
#include "RecoilSettings.h"
#include "Path.h"
#include "Logger.h"
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

// Define how to convert AimingSettings to and from JSON.
// This allows nlohmann::json to automatically handle the struct.
void to_json(nlohmann::json& j, const AimingSettings& s) {
    j = nlohmann::json{
        {"hotkey", s.hotkey},
        {"priority", s.priority},
        {"confidence_threshold", s.confidence_threshold},
        {"max_speed", s.max_speed},
        {"target_offset_x", s.target_offset_x},
        {"target_offset_y", s.target_offset_y},
        {"use_ema", s.use_ema},
        {"ema_alpha", s.ema_alpha},
        {"use_pid", s.use_pid},
        {"pid_p", s.pid_p},
        {"pid_i", s.pid_i},
        {"pid_d", s.pid_d},
        {"use_prediction", s.use_prediction},
        {"prediction_strength_x", s.prediction_strength_x},
        {"prediction_strength_y", s.prediction_strength_y},
        {"use_dynamic_pid", s.use_dynamic_pid},
        {"dynamic_pid_distance_max", s.dynamic_pid_distance_max},
        {"dynamic_pid_scale_min", s.dynamic_pid_scale_min}
    };
}

void from_json(const nlohmann::json& j, AimingSettings& s) {
    s.hotkey = j.value("hotkey", "RMB");
    s.priority = j.value("priority", TargetPriority::ClosestToCrosshair);
    s.confidence_threshold = j.value("confidence_threshold", 0.5f);
    s.max_speed = j.value("max_speed", 1500.0f);
    s.target_offset_x = j.value("target_offset_x", 0.0f);
    s.target_offset_y = j.value("target_offset_y", -0.1f);
    s.use_ema = j.value("use_ema", true);
    s.ema_alpha = j.value("ema_alpha", 0.3f);
    s.use_pid = j.value("use_pid", true);
    if (j.contains("pid_p")) j.at("pid_p").get_to(s.pid_p);
    if (j.contains("pid_i")) j.at("pid_i").get_to(s.pid_i);
    if (j.contains("pid_d")) j.at("pid_d").get_to(s.pid_d);
    if (j.contains("use_prediction")) j.at("use_prediction").get_to(s.use_prediction);
    if (j.contains("prediction_strength_x")) j.at("prediction_strength_x").get_to(s.prediction_strength_x);
    if (j.contains("prediction_strength_y")) j.at("prediction_strength_y").get_to(s.prediction_strength_y);
    if (j.contains("use_dynamic_pid")) j.at("use_dynamic_pid").get_to(s.use_dynamic_pid);
    if (j.contains("dynamic_pid_distance_max")) j.at("dynamic_pid_distance_max").get_to(s.dynamic_pid_distance_max);
    if (j.contains("dynamic_pid_scale_min")) j.at("dynamic_pid_scale_min").get_to(s.dynamic_pid_scale_min);
}

// Define how to convert RecoilSettings to and from JSON
void to_json(nlohmann::json& j, const RecoilSettings& s) {
    j = nlohmann::json{
        {"enable_recoil_control", s.enable_recoil_control},
        {"recoil_strength", s.recoil_strength},
        {"recoil_horizontal", s.recoil_horizontal},
        {"recoil_vertical", s.recoil_vertical}
    };
}

void from_json(const nlohmann::json& j, RecoilSettings& s) {
    j.at("enable_recoil_control").get_to(s.enable_recoil_control);
    j.at("recoil_strength").get_to(s.recoil_strength);
    j.at("recoil_horizontal").get_to(s.recoil_horizontal);
    j.at("recoil_vertical").get_to(s.recoil_vertical);
}


ConfigManager::ConfigManager(Aimbot& aimbot) : m_aimbot(aimbot) {}

std::string ConfigManager::GetConfigPath() const {
    fs::path exe_path = get_executable_path();
    return (exe_path.parent_path() / "config.json").string();
}

void ConfigManager::Save() const {
    nlohmann::json j;

    // Directly access the public member
    j["aiming_settings"] = m_aimbot.m_aiming_settings; 
    j["personalization_settings"] = m_aimbot.m_personalization_settings;
    j["recoil_settings"] = m_aimbot.m_recoil_settings;

    // Save other settings...
    j["last_loaded_model"] = m_aimbot.GetLastLoadedModelPath();
    j["fp16_mode"] = m_aimbot.fp16_mode;
    j["workspace_size"] = m_aimbot.workspace_size;
    j["killswitch_key"] = m_aimbot.killswitch_key.load();
    j["hide_toggle_key"] = m_aimbot.hide_toggle_key.load();

    try {
        std::ofstream file(GetConfigPath());
        file << j.dump(4); // pretty print
        file.close();
        Logger::GetInstance().Log("Configuration saved successfully.");
    } catch (const std::exception& e) {
        Logger::GetInstance().Log("Error saving configuration: " + std::string(e.what()));
    }
}

void ConfigManager::Load() {
    std::string config_path = GetConfigPath();
    if (!fs::exists(config_path)) {
        Logger::GetInstance().Log("Configuration file not found. Using default settings.");
        Save(); // Create a default config file
        return;
    }

    try {
        std::ifstream file(config_path);
        nlohmann::json j;
        file >> j;
        file.close();

        // Load general settings
        m_aimbot.fp16_mode = j.value("fp16_mode", true);
        m_aimbot.workspace_size = j.value("workspace_size", 256);
        m_aimbot.SetLastLoadedModelPath(j.value("last_loaded_model", ""));

        // Load aiming settings directly into the aimbot's settings object
        if (j.contains("aiming_settings")) {
            j.at("aiming_settings").get_to(m_aimbot.m_aiming_settings);
        }

        // Load personalization settings
        if (j.contains("personalization_settings")) {
            j.at("personalization_settings").get_to(m_aimbot.m_personalization_settings);
        }

        // Load recoil settings
        if (j.contains("recoil_settings")) {
            j.at("recoil_settings").get_to(m_aimbot.m_recoil_settings);
        }

        // Load hotkeys
        m_aimbot.killswitch_key.store(j.value("killswitch_key", 0));
        m_aimbot.hide_toggle_key.store(j.value("hide_toggle_key", 0));
        
        Logger::GetInstance().Log("Configuration loaded successfully.");

    } catch (const std::exception& e) {
        Logger::GetInstance().Log("Error loading configuration: " + std::string(e.what()) + ". Using defaults.");
    }
}
