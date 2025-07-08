#include "Logger.h"
#include "Path.h" // To get executable path
#include <iostream>
#include <filesystem>
#include <cstdarg> // For va_list, va_start, vsnprintf, va_end

namespace fs = std::filesystem;

Logger::Logger() {
    fs::path log_path = get_executable_dir() / "Aimbot_Log.txt";
    // Open the file in append mode.
    log_file.open(log_path, std::ios_base::app);
    if (!log_file.is_open()) {
        // This is a tricky situation. We can't use our logger to log this.
        // We'll output to the console as a fallback.
        std::cerr << "FATAL: Could not open log file at: " << log_path << std::endl;
    }
}

Logger::~Logger() {
    if (log_file.is_open()) {
        log_file.close();
    }
}

Logger& Logger::GetInstance() {
    static Logger instance;
    return instance;
}

void Logger::Log(const std::string& message) {
    std::lock_guard<std::mutex> lock(mtx);
    
    // Get current time
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
	#pragma warning(suppress : 4996)
    ss << std::put_time(std::localtime(&in_time_t), "[%Y-%m-%d %X] ");
    
    std::string final_message = ss.str() + message;
    messages.push_back(final_message);
    
    // Also write to the log file
    if (log_file.is_open()) {
        log_file << final_message << std::endl;
    }

    // Optional: limit the number of messages to prevent memory issues
    if (messages.size() > 200) {
        messages.erase(messages.begin(), messages.begin() + 50); // Erase a chunk to be more efficient
    }
}

void Logger::Log(const char* format, ...) {
    char buffer[1024];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    Log(std::string(buffer)); // Call the original Log function
}

std::vector<std::string> Logger::GetMessages() {
    std::lock_guard<std::mutex> lock(mtx);
    return messages;
}

void Logger::Clear() {
    std::lock_guard<std::mutex> lock(mtx);
    messages.clear();
}
