#include "runtime/components/metrics_reporter.h"
#include "runtime/core/metrics.h"
#include <iostream>

namespace ptk::components
{

    MetricsReporter::MetricsReporter(const rclcpp::NodeOptions &options, int interval_ms)
        : ComponentInterface("metrics_reporter", options), context_(nullptr), report_interval_ms_(interval_ms), last_report_(std::chrono::steady_clock::now())
    {
    }

    core::Status MetricsReporter::Init(core::RuntimeContext *context)
    {
        context_ = context;
        std::cout << "[MetricsReporter] Initialized with " << report_interval_ms_ << " ms reporting interval" << std::endl;
        return core::Status::Ok();
    }

    core::Status MetricsReporter::Start()
    {
        last_report_ = std::chrono::steady_clock::now();
        std::cout << "[MetricsReporter] Started" << std::endl;
        return core::Status::Ok();
    }

    core::Status MetricsReporter::Stop()
    {
        std::cout << "[MetricsReporter] Final metrics report:" << std::endl;
        core::MetricsCollector::Instance().LogMetrics();
        return core::Status::Ok();
    }

    void MetricsReporter::Tick()
    {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           now - last_report_)
                           .count();

        if (elapsed >= report_interval_ms_)
        {
            core::MetricsCollector::Instance().LogMetrics();

            core::MetricsCollector::Instance().ResetPeriod();

            last_report_ = now;
        }
    }

} // namespace ptk::components
