#pragma once

#include "runtime/core/status.h"
#include <rclcpp/rclcpp.hpp>

namespace ptk::core
{
    class RuntimeContext;
    class Scheduler;
}

namespace ptk::components
{

        class ComponentInterface : public rclcpp::Node
        {

        public:
            explicit ComponentInterface(
                const std::string& node_name,
                const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
                : rclcpp::Node(node_name, options), scheduler_(nullptr) {}

            virtual ~ComponentInterface() = default;

            void SetScheduler(core::Scheduler* scheduler) { scheduler_ = scheduler; }

            virtual core::Status Init(core::RuntimeContext *context) = 0; // called once before start

            virtual core::Status Start() = 0; // called once before the first ticker

            virtual core::Status Stop() = 0; // called once after the lasttick

            virtual void Tick() = 0; // called repeatedly by scheduler or external driver

        protected:
            core::Scheduler* scheduler_;
        };

} // namespace ptk::components