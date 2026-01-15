#pragma once

#include "runtime/components/component_interface.h"

namespace ptk::components
{

        class Counter : public ComponentInterface
        {
        public:
            explicit Counter(const rclcpp::NodeOptions &options = rclcpp::NodeOptions()); //passes config info for the node when initializing it
            ~Counter() override = default;

            core::Status Init(core::RuntimeContext *context) override;
            core::Status Start() override;
            core::Status Stop() override;
            void Tick() override;

        private:
            core::RuntimeContext *context_;
            int count_;
        };

} // namespace ptk::components