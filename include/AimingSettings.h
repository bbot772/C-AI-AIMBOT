#ifndef AIMING_SETTINGS_H
#define AIMING_SETTINGS_H

#include <string>

// Define the target selection priority
enum class TargetPriority
{
    HighestConfidence,
    ClosestToCrosshair
};

// All configurable settings for the Aimbot
struct AimingSettings
{
    // Activation
    std::string hotkey = "RMB";

    // Targeting
    TargetPriority priority = TargetPriority::ClosestToCrosshair;
    float confidence_threshold = 0.5f;

    // Aiming behavior
    float max_speed = 1500.0f; // Pixels per second
    float target_offset_x = 0.0f; // Percentage of bounding box width
    float target_offset_y = -0.1f; // Percentage of bounding box height (e.g., -0.1 for neck/head)

    // Smoothing
    bool use_ema = true;
    float ema_alpha = 0.3f; // Smoothing Amount: 0.0f = off, 1.0f = max smoothing. Higher values are smoother.

    // PID Controller for smooth, responsive aiming
    bool use_pid = true;
    float pid_p = 2.0f;      // Proportional gain: How strongly to react to the current error.
    float pid_i = 0.01f;     // Integral gain: How much to correct for past, persistent error.
    float pid_d = 0.25f;     // Derivative gain: How much to dampen the movement to prevent overshooting.

    // Target Velocity Prediction
    bool use_prediction = true;
    float prediction_strength_x = 1.0f; // Multiplier for how far to predict on the X-axis
    float prediction_strength_y = 1.0f; // Multiplier for how far to predict on the Y-axis

    // Dynamic PID Adjustment based on target distance
    bool use_dynamic_pid = true;
    float dynamic_pid_distance_max = 400.0f; // The bounding box width at which PID strength is at its minimum
    float dynamic_pid_scale_min = 0.4f;      // The minimum scaling factor for P-gain (e.g., 40% at max distance)
};

#endif // AIMING_SETTINGS_H 