// include/runtime/core/timing_helper.h
#pragma once

#include <chrono>
#include <string>

#include "runtime/core/metrics.h"

namespace ptk::core
{

    class ScopedTimer
    {
    public:
        explicit ScopedTimer(const std::string &component_name)
            : component_name_(component_name), start_(std::chrono::steady_clock::now()) {}

        ~ScopedTimer()
        {
            auto end = std::chrono::steady_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(end - start_).count();
            MetricsCollector::Instance().RecordStageLatency(component_name_, elapsed_ms);
        }

    private:
        std::string component_name_;
        std::chrono::steady_clock::time_point start_;
    };

} // namespace ptk::core