#pragma once

#include <vector>
#include <string>
#include <mutex>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <fstream>

class Logger {
public:
    static Logger& GetInstance();
    void Log(const std::string& message);
    void Log(const char* format, ...);
    std::vector<std::string> GetMessages();
    void Clear();

    // Disable copy and assign
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

private:
    Logger();
    ~Logger();

    std::vector<std::string> messages;
    std::mutex mtx;
    std::ofstream log_file;
};
