#pragma once

#include "runtime/components/component_interface.h"
#include "runtime/core/runtime_context.h"
#include <chrono>

namespace ptk::components
{

    class MetricsReporter : public ComponentInterface
    {
    public:
        explicit MetricsReporter(
            const rclcpp::NodeOptions &options = rclcpp::NodeOptions(),
            int report_interval_ms = 1000);

        core::Status Init(core::RuntimeContext *context) override;
        core::Status Start() override;
        core::Status Stop() override;
        void Tick() override;

    private:
        core::RuntimeContext *context_;
        int report_interval_ms_;
        std::chrono::steady_clock::time_point last_report_;
    };

} // namespace ptk::components
