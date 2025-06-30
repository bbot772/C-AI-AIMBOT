#include "../include/Aiming.h"
#include "Logger.h"
#include <algorithm>
#include <cmath>
#include <windows.h>
#include <thread>
#include <chrono>
#include <map>

// A map of common key names to their virtual-key codes
const std::map<std::string, int> g_key_map = {
    {"LMB", VK_LBUTTON}, {"RMB", VK_RBUTTON}, {"MMB", VK_MBUTTON},
    {"Mouse4", VK_XBUTTON1}, {"Mouse5", VK_XBUTTON2},
    {"Shift", VK_SHIFT}, {"Ctrl", VK_CONTROL}, {"Alt", VK_MENU},
    {"F1", VK_F1}, {"F2", VK_F2}, {"F3", VK_F3}, {"F4", VK_F4},
    {"F5", VK_F5}, {"F6", VK_F6}, {"F7", VK_F7}, {"F8", VK_F8},
    {"F9", VK_F9}, {"F10", VK_F10}, {"F11", VK_F11}, {"F12", VK_F12},
    {"A", 0x41}, {"B", 0x42}, {"C", 0x43}, {"D", 0x44}, {"E", 0x45},
    {"F", 0x46}, {"G", 0x47}, {"H", 0x48}, {"I", 0x49}, {"J", 0x4A},
    {"K", 0x4B}, {"L", 0x4C}, {"M", 0x4D}, {"N", 0x4E}, {"O", 0x4F},
    {"P", 0x50}, {"Q", 0x51}, {"R", 0x52}, {"S", 0x53}, {"T", 0x54},
    {"U", 0x55}, {"V", 0x56}, {"W", 0x57}, {"X", 0x58}, {"Y", 0x59}, {"Z", 0x5A}
};

int Aiming::GetVKCodeForString(const std::string& key)
{
    auto it = g_key_map.find(key);
    if (it != g_key_map.end())
    {
        return it->second;
    }
    return 0; // Not found
}

Aiming::Aiming(AimingSettings& settings) : m_settings(settings)
{
    m_mouse = std::make_unique<GhubMouse>();
    Logger::GetInstance().Log("Aiming module initialized.");
}

void Aiming::ProcessDetections(const std::vector<Detection>& detections, int screen_w, int screen_h) {
    SelectTarget(detections, screen_w, screen_h);

    if (m_target_lost) {
        ResetPID();
        ResetPrediction();
        return;
    }

    Detection selected_target;
    for(const auto& det : detections) {
        if (det.class_id == m_current_target_id) {
            selected_target = det;
            break;
        }
    }

    // Calculate target position
    float target_x = selected_target.x_center + (m_settings.target_offset_x * selected_target.w);
    float target_y = selected_target.y_center + (m_settings.target_offset_y * selected_target.h);

    // --- Velocity Prediction ---
    if (m_settings.use_prediction) {
        auto now = std::chrono::steady_clock::now();
        if (m_last_target_pos.x != 0 && m_last_target_pos.y != 0) { 
            std::chrono::duration<float> dt = now - m_last_prediction_time;
            float dt_s = dt.count();
            if (dt_s > 0.001f) { 
                m_target_velocity.x = (target_x - m_last_target_pos.x) / dt_s;
                m_target_velocity.y = (target_y - m_last_target_pos.y) / dt_s;
            }
        }
        m_last_prediction_time = now;
        m_last_target_pos = {target_x, target_y};
        
        float prediction_factor = 1.0f / 144.0f; 
        target_x += m_target_velocity.x * prediction_factor * m_settings.prediction_strength_x;
        target_y += m_target_velocity.y * prediction_factor * m_settings.prediction_strength_y;
    }
    // --- End Prediction ---

    float error_x = target_x - (screen_w / 2.0f);
    float error_y = target_y - (screen_h / 2.0f);

    // --- Dynamic PID Scaling ---
    float p_gain = m_settings.pid_p;
    if (m_settings.use_dynamic_pid) {
        float max_dist = m_settings.dynamic_pid_distance_max;
        if (max_dist > 0) {
            float min_scale = m_settings.dynamic_pid_scale_min;
            float target_width = static_cast<float>(selected_target.w);

            // Clamp the width to the max distance to avoid scaling > 1
            float clamped_width = std::min(target_width, max_dist);

            // Interpolate the scale from 1.0 (at 0 width) down to min_scale (at max_dist width)
            float scale = 1.0f - (clamped_width / max_dist) * (1.0f - min_scale);
            p_gain *= scale;
        }
    }
    // --- End Dynamic PID ---

    int hotkey_vk = GetVKCodeForString(m_settings.hotkey);
    if (hotkey_vk == 0 || !(GetAsyncKeyState(hotkey_vk) & 0x8000))
    {
        m_is_first_target = true; 
        if (m_is_aiming)
        {
            m_is_aiming = false;
            m_pid_x = {};
            m_pid_y = {};
        }
        return; 
    }
    
    if (!m_is_aiming) {
        m_is_aiming = true;
    }

    int move_dx = 0;
    int move_dy = 0;
    
    // Apply PID controller for movement if enabled
    if (m_settings.use_pid) {
        float p_out_x = p_gain * error_x;
        float p_out_y = p_gain * error_y;

        m_pid_x.integral += error_x;
        m_pid_y.integral += error_y;
        
        float i_max = 20.0f;
        m_pid_x.integral = std::max(-i_max, std::min(i_max, m_pid_x.integral));
        m_pid_y.integral = std::max(-i_max, std::min(i_max, m_pid_y.integral));
        
        float i_out_x = m_settings.pid_i * m_pid_x.integral;
        float i_out_y = m_settings.pid_i * m_pid_y.integral;

        float derivative_x = error_x - m_pid_x.previous_error;
        float derivative_y = error_y - m_pid_y.previous_error;
        float d_out_x = m_settings.pid_d * derivative_x;
        float d_out_y = m_settings.pid_d * derivative_y;

        move_dx = static_cast<int>(p_out_x + i_out_x + d_out_x);
        move_dy = static_cast<int>(p_out_y + i_out_y + d_out_y);

        m_pid_x.previous_error = error_x;
        m_pid_y.previous_error = error_y;

    } else {
        float dx = target_x - (screen_w / 2.0f);
        float dy = target_y - (screen_h / 2.0f);

        float distance = std::sqrt(dx * dx + dy * dy);
        float move_speed = m_settings.max_speed / 100.0f;

        if (distance > 1.0f) {
            move_dx = static_cast<int>((dx / distance) * move_speed);
            move_dy = static_cast<int>((dy / distance) * move_speed);
        }
    }

    if (move_dx != 0 || move_dy != 0)
    {
        m_mouse->move_r(move_dx, move_dy);
    }
}

void Aiming::SelectTarget(const std::vector<Detection>& detections, int screen_w, int screen_h) {
    m_target_lost = true;
    if (detections.empty()) {
        m_current_target_id = -1;
        return;
    }

    int best_target_id = -1;
    float best_metric = -1.0f;

    for (const auto& det : detections)
    {
        if (det.confidence < m_settings.confidence_threshold)
        {
            continue;
        }

        if (m_settings.priority == TargetPriority::HighestConfidence)
        {
            if (det.confidence > best_metric)
            {
                best_metric = det.confidence;
                best_target_id = det.class_id;
            }
        }
        else 
        {
            float dx = det.x_center - screen_w / 2.0f;
            float dy = det.y_center - screen_h / 2.0f;
            float distance_squared = dx * dx + dy * dy;

            if (best_metric < 0 || distance_squared < best_metric)
            {
                best_metric = distance_squared;
                best_target_id = det.class_id;
            }
        }
    }

    if (best_target_id != -1) {
        if (m_current_target_id != best_target_id) {
            ResetPrediction(); 
            m_current_target_id = best_target_id;
        }
        m_target_lost = false;
    } else {
        m_current_target_id = -1;
    }
}

void Aiming::Reset() {
    ResetPID();
    ResetPrediction();
}

void Aiming::ResetPID() {
    m_pid_x.integral = 0.0f;
    m_pid_y.integral = 0.0f;
    m_pid_x.previous_error = 0.0f;
    m_pid_y.previous_error = 0.0f;
}

void Aiming::ResetPrediction() {
    m_last_target_pos = {0, 0};
    m_target_velocity = {0, 0};
    m_last_prediction_time = std::chrono::steady_clock::now();
}
 