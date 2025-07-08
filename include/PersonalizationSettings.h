#pragma once
#include <string>
#include "imgui.h" // For ImVec4
#include "nlohmann/json.hpp"

// Helper to allow nlohmann::json to serialize/deserialize ImVec4
namespace nlohmann {
    template <>
    struct adl_serializer<ImVec4> {
        static void to_json(json& j, const ImVec4& vec) {
            j = {vec.x, vec.y, vec.z, vec.w};
        }

        static void from_json(const json& j, ImVec4& vec) {
            j.at(0).get_to(vec.x);
            j.at(1).get_to(vec.y);
            j.at(2).get_to(vec.z);
            j.at(3).get_to(vec.w);
        }
    };
}

// A struct to hold all settings related to personalization and themes.
// This makes it easy to serialize and manage.
struct PersonalizationSettings {
    // --- Theme & Colors ---
    ImVec4 theme_color = ImVec4(0.26f, 0.59f, 0.98f, 1.00f); // Main accent color
    ImVec4 text_color = ImVec4(0.9f, 0.9f, 0.9f, 1.0f);   // Main text color
    ImVec4 outline_color = ImVec4(0.26f, 0.59f, 0.98f, 0.5f); // Window outline color
    ImVec4 gradient_color_top = ImVec4(0.10f, 0.10f, 0.10f, 1.0f); // Top color of background gradient
    ImVec4 gradient_color_bottom = ImVec4(0.14f, 0.14f, 0.14f, 1.0f); // Bottom color of background gradient
    float window_opacity = 1.0f;

    // --- Font & Scaling ---
    std::string font_name = "Default"; // Name of the selected font file
    float ui_scale = 1.0f;

    // --- Window ---
    std::string custom_window_title = "Aimbot";

    // JSON serialization
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(PersonalizationSettings,
        theme_color, text_color, outline_color, gradient_color_top, gradient_color_bottom, window_opacity,
        font_name, ui_scale, custom_window_title);
}; 