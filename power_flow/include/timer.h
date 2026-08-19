#ifndef POWER_FLOW_TIMER_H
#define POWER_FLOW_TIMER_H

#include <chrono>
#include <string>
#include <cstdio>

namespace powerflow {

// High-resolution RAII-style timer for performance measurement.
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}

    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    // Elapsed time in seconds (double).
    double elapsed_seconds() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(now - start_).count();
    }

    // Elapsed time in milliseconds.
    double elapsed_ms() const {
        return elapsed_seconds() * 1000.0;
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

// Scoped timer that prints the elapsed time when destroyed.
class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& label)
        : label_(label) {}

    ~ScopedTimer() {
        std::printf("[timer] %-32s %.3f ms\n", label_.c_str(), t_.elapsed_ms());
    }

private:
    std::string label_;
    Timer t_;
};

} // namespace powerflow

#endif // POWER_FLOW_TIMER_H
