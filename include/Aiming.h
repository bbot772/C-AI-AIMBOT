#ifndef AIMING_H
#define AIMING_H

#include "AimingSettings.h"
#include "RecoilSettings.h"
#include "../ghub/ghub.h"
#include "types.h" // Use the new types header
#include "nlohmann/json.hpp"
#include <vector>
#include <memory>
#include <atomic>
#include <map>
#include <chrono>

// Forward declarations for JSON serialization
void to_json(nlohmann::json& j, const AimingSettings& s);
void from_json(const nlohmann::json& j, AimingSettings& s);

// Forward declaration
class GhubMouse;

// This struct holds the smoothed position for EMA
struct SmoothedPosition {
    float x = 0.0f;
    float y = 0.0f;
};

// This struct holds the state for a single PID controller axis.
struct PIDState {
    float previous_error = 0.0f;
    float integral = 0.0f;
};

class Aiming
{
public:
    Aiming(AimingSettings& settings);
    // Make the class non-copyable because it holds a reference
    Aiming(const Aiming&) = delete;
    Aiming& operator=(const Aiming&) = delete;

    std::pair<int, int> ProcessDetections(const std::vector<Detection>& detections, int screen_width, int screen_height);
    std::pair<int, int> ControlRecoil(const RecoilSettings& recoil_settings);
    void ApplyMouseMovement(int dx, int dy);
    void Reset();

private:
    void SelectTarget(const std::vector<Detection>& detections, int screen_width, int screen_height);
    int GetVKCodeForString(const std::string& key);
    void MoveMouse(float dx, float dy);
    void ResetPID();
    void ResetPrediction();

    std::unique_ptr<GhubMouse> m_mouse;
    AimingSettings& m_settings;

    // State for smoothing
    bool m_is_first_target = true;
    SmoothedPosition m_smoothed_target_pos;

    // State for PID controller
    bool m_is_aiming = false;
    PIDState m_pid_x;
    PIDState m_pid_y;

    bool m_target_lost = true;
    int m_current_target_id = -1;
    float m_pid_error_sum_x = 0.0f;
    float m_pid_error_sum_y = 0.0f;
    float m_pid_last_error_x = 0.0f;
    float m_pid_last_error_y = 0.0f;

    // State for velocity prediction
    Vec2 m_last_target_pos{0, 0};
    std::chrono::steady_clock::time_point m_last_prediction_time;
    Vec2 m_target_velocity{0, 0};
};

#endif // AIMING_H 