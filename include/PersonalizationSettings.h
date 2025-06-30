#pragma once
#include "nlohmann/json.hpp"
#include "imgui.h" // For ImVec4

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

struct PersonalizationSettings {
    ImVec4 theme_color = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
    ImVec4 text_color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
    float ui_scale = 1.0f;
    float window_opacity = 1.0f;
    std::string custom_window_title = "Aimbot";
    std::string font_name = "Default";

    // Macro to automatically handle serialization
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(PersonalizationSettings, theme_color, text_color, ui_scale, window_opacity, custom_window_title, font_name);
}; 