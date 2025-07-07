#pragma once

// Minimal Font Awesome icon definitions - only icons actually used in the UI
// Replaces the massive 97KB font_awesome.h file for significant size reduction

#define FONT_ICON_FILE_NAME_FAS "fa-solid-900.ttf"

// Only include icons that are actually used in the UI
#define ICON_FA_CUBE "\xef\x99\x80"               // Model tab
#define ICON_FA_CROSSHAIRS "\xef\x81\x9b"         // Aiming tab  
#define ICON_FA_CHART_LINE "\xef\x88\x81"         // Performance tab
#define ICON_FA_BOOK "\xef\x80\xad"               // Logging tab
#define ICON_FA_SHIELD_ALT "\xef\x9d\xb5"         // Stealth tab
#define ICON_FA_PAINT_BRUSH "\xef\x95\x9d"        // Personalization tab
#define ICON_FA_SAVE "\xef\x80\x87"               // Save button
#define ICON_FA_PLAY "\xef\x81\x8b"               // Play/Start button
#define ICON_FA_STOP "\xef\x81\x8d"               // Stop button
#define ICON_FA_FOLDER_OPEN "\xef\x81\xbc"        // Folder operations
#define ICON_FA_COG "\xef\x80\x93"                // Settings/Config
#define ICON_FA_EYE "\xef\x81\xae"                // Visibility toggle
#define ICON_FA_EYE_SLASH "\xef\x81\xb0"          // Hide toggle

// Performance and monitoring icons
#define ICON_FA_TACHOMETER_ALT "\xef\x8c\xa2"     // Performance metrics
#define ICON_FA_MEMORY "\xef\x94\x98"             // Memory usage
#define ICON_FA_MICROCHIP "\xef\x8b\x9b"          // GPU/CPU usage
#define ICON_FA_THERMOMETER_HALF "\xef\x88\x92"   // Temperature

// UI control icons  
#define ICON_FA_TIMES "\xef\x80\x8d"              // Close/X button
#define ICON_FA_MINUS "\xef\x81\xa8"              // Minimize button
#define ICON_FA_SQUARE "\xef\x80\x88"             // Window controls
#define ICON_FA_CHECK "\xef\x80\x8c"              // Checkmarks
#define ICON_FA_EXCLAMATION_TRIANGLE "\xef\x81\xb1" // Warnings

// File and model management
#define ICON_FA_FILE "\xef\x85\x9b"               // File icons
#define ICON_FA_UPLOAD "\xef\x83\x93"             // Upload/Load
#define ICON_FA_DOWNLOAD "\xef\x83\x99"           // Download/Save

// Size reduction: From 1,861 icon definitions to ~25 actually used icons
// Expected binary size reduction: ~450KB (97KB header + 378KB implementation)