#pragma once

#include "runtime/components/component_interface.h"
#include "runtime/core/port.h"
#include "runtime/data/frame.h"
#include <vector>
#include <cstdint>

namespace ptk::components
{

        class SyntheticCamera : public ComponentInterface
        {

        public:
            explicit SyntheticCamera(const rclcpp::NodeOptions &options = rclcpp::NodeOptions()); //passes config info for the node when initializing it
            ~SyntheticCamera() override = default;

            // The pipeline or app calls this to connect a Frame sink.
            void BindOutput(core::OutputPort<data::Frame> *port);

            core::Status Init(core::RuntimeContext *context) override;
            core::Status Start() override;
            core::Status Stop() override;
            void Tick() override;

        private:
            core::RuntimeContext *context_;
            core::OutputPort<data::Frame> *output_;
            int frame_index_;
            
            std::vector<uint8_t> frame_buffer_[2]; // double buffering = no dangling pointers during resize
            int current_buffer_index_;
        };

} // namespace ptk::components